# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import logging

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
from flag_attn.FLA.utils import exp, make_tensor_descriptor
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril import (
    FLA_TRIL_PRECISION,
    _store_gated_inverse_block,
)

from flag_attn.runtime.backend._thead.FLA.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from flag_attn.runtime.backend._thead.FLA.chunk_o import chunk_fwd_o
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.chunk_fused_tail_vblock import (
    can_use_fused_tail_vblock,
    chunk_gated_delta_rule_fused_tail_vblock,
)
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.wy_fast import (
    recompute_w_u_fwd,
)
from flag_attn.FLA.utils import SUPPRESS_LEVEL

logger = logging.getLogger(__name__)


def _chunk_size_for_sequence(T: int, is_varlen: bool) -> int:
    if is_varlen:
        return 64
    return min(64, max(16, 1 << (T - 1).bit_length()))


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    logger.debug("GEMS CHUNK GATED DELTA RULE FWD")
    q_contiguous = q.is_contiguous()
    k_contiguous = k.is_contiguous()
    v_contiguous = v.is_contiguous()
    g_contiguous = g.is_contiguous()
    beta_contiguous = beta.is_contiguous()
    initial_state_contiguous = initial_state is None or initial_state.is_contiguous()
    cu_seqlens_contiguous = cu_seqlens is None or cu_seqlens.is_contiguous()
    if not (
        q_contiguous
        and k_contiguous
        and v_contiguous
        and g_contiguous
        and beta_contiguous
        and initial_state_contiguous
        and cu_seqlens_contiguous
    ):
        if not q_contiguous:
            q = q.contiguous()
        if not k_contiguous:
            k = k.contiguous()
        if not v_contiguous:
            v = v.contiguous()
        if not g_contiguous:
            g = g.contiguous()
        if not beta_contiguous:
            beta = beta.contiguous()
        if not initial_state_contiguous:
            initial_state = initial_state.contiguous()
        if not cu_seqlens_contiguous:
            cu_seqlens = cu_seqlens.contiguous()

    chunk_size = _chunk_size_for_sequence(q.shape[1], cu_seqlens is not None)

    solve = (
        _gdn_solve_reduced
        if chunk_size == 64
        else chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril
    )
    g, A = solve(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
    )
    # Compute the correction in the state recurrence directly for supported K.
    # The existing path remains available for debug output and larger K.
    if SUPPRESS_LEVEL < 3 and k.shape[-1] <= 256:
        h, v_new, final_state = _gdn_state_direct(
            k, v, beta, A, g, initial_state, output_final_state,
            chunk_size, cu_seqlens,
        )
        o = chunk_fwd_o(
            q=q, k=k, v=v_new, h=h, g=g, scale=scale,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        )
        return g, o, A, final_state, None, None, None
    block_v = 128 if v.shape[-1] == 128 else 64
    w, u = _gdn_recompute_w_u_jobs(
        k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=cu_seqlens, block_v=block_v,
    )
    if SUPPRESS_LEVEL < 3 and can_use_fused_tail_vblock(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    ):
        o, final_state = chunk_gated_delta_rule_fused_tail_vblock(
            q=q, k=k, w=w, u=u, g=g, initial_state=initial_state, scale=scale,
        )
        return g, o, A, final_state, None, None, None
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


# The solver below is adapted from fused_cumsum_kkt_solve_tril.py.
# Copyright (c) 2025 FlagGems. All rights reserved.
# License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
# Modification: use an FP32 multiply/reduction for the 16x16 IEEE products.
#
# Keep the 16x16 inverse-block products in FP32. An explicit reduction avoids
# forcing the scalar IEEE dot through a matrix-operand layout on this backend.
@triton.jit
def _gdn_dot16(a, b, input_precision: tl.constexpr):
    if input_precision == "ieee":
        return tl.sum(a[:, :, None] * b[None, :, :], axis=1)
    return tl.dot(a, b, input_precision=input_precision)


@triton.jit(do_not_specialize=["T"])
def _gdn_solve_reduced_kernel(g_in, g_out, k, beta, A, A_inv, A_g_inv, cu_seqlens, chunk_indices, T, H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr, USE_G: tl.constexpr, OUTPUT_GATED: tl.constexpr, USE_TMA: tl.constexpr, DOT_PRECISION: tl.constexpr):
    (i_t, i_bh) = (tl.program_id(0), tl.program_id(1))
    (i_b, i_h) = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        (i_n, i_t) = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        (bos, eos) = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        (bos, eos) = (i_b * T, i_b * T + T)
    p_g_in = tl.make_block_ptr(g_in + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g_out = tl.make_block_ptr(g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g_in, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    tl.store(p_g_out, b_g.to(p_g_out.dtype.element_ty), boundary_check=(0,))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * Hg + i_h // (H // Hg)) * K, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
    if USE_G:
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)
    m_A_kkt = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A_kkt, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_base = A + (bos * H + i_h) * BT
    A_inv_base = A_inv + (bos * H + i_h) * BT
    A_g_inv_base = A_g_inv + (bos * H + i_h) * BT
    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_A_33 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_A_44 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A_base, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)
    b_Ai_11 = _gdn_invert16_blocks(-tl.where(m_A, b_Ai_11, 0))
    b_Ai_22 = _gdn_invert16_blocks(-tl.where(m_A, b_Ai_22, 0))
    b_Ai_33 = _gdn_invert16_blocks(-tl.where(m_A, b_Ai_33, 0))
    b_Ai_44 = _gdn_invert16_blocks(-tl.where(m_A, b_Ai_44, 0))
    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_A_31 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_A_32 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_A_41 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_A_42 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_A_43 = tl.make_block_ptr(A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)
    b_Ai_21 = -_gdn_dot16(_gdn_dot16(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -_gdn_dot16(_gdn_dot16(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -_gdn_dot16(_gdn_dot16(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)
    b_Ai_31 = -_gdn_dot16(b_Ai_33, _gdn_dot16(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + _gdn_dot16(b_A_32, b_Ai_21, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    b_Ai_42 = -_gdn_dot16(b_Ai_44, _gdn_dot16(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) + _gdn_dot16(b_A_43, b_Ai_32, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    b_Ai_41 = -_gdn_dot16(b_Ai_44, _gdn_dot16(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) + _gdn_dot16(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) + _gdn_dot16(b_A_43, b_Ai_31, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_Ai_33 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_31 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_Ai_32 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_Ai_41 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_Ai_42 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_Ai_43 = tl.make_block_ptr(A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    else:
        desc_o = make_tensor_descriptor(A_inv_base, [T, BT], [H * BT, 1], [16, 16])
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding='rtne'))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding='rtne'))
    if OUTPUT_GATED:
        g_base = g_out + (bos + i_t * BT) * H + i_h
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_11, T, H, BT, 0, 0)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_22, T, H, BT, 16, 16)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_33, T, H, BT, 32, 32)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_44, T, H, BT, 48, 48)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_21, T, H, BT, 16, 0)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_31, T, H, BT, 32, 0)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_32, T, H, BT, 32, 16)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_41, T, H, BT, 48, 0)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_42, T, H, BT, 48, 16)
        _store_gated_inverse_block(A_g_inv_base, g_base, b_Ai_43, T, H, BT, 48, 32)


def _gdn_solve_reduced(g, k, beta, cu_seqlens, chunk_size, output_dtype):
    """810E solver for 64-token chunks; smaller chunks use the original solver."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    nt = triton.cdiv(T, chunk_size) if indices is None else len(indices)
    gc = torch.empty_like(g)
    temp = torch.empty((B, T, H, chunk_size), device=k.device, dtype=torch.float32)
    inverse = torch.zeros((B, T, H, chunk_size), device=k.device, dtype=output_dtype)
    _gdn_solve_reduced_kernel[(nt, H if indices is not None else B * H)](
        g,
        gc,
        k,
        beta,
        temp,
        inverse,
        inverse,
        cu_seqlens,
        indices,
        T,
        H=H,
        Hg=Hg,
        K=K,
        BT=chunk_size,
        BK=64,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=True,
        OUTPUT_GATED=False,
        USE_TMA=False,
        DOT_PRECISION=FLA_TRIL_PRECISION,
        num_warps=2,
        num_stages=4,
    )
    return gc, inverse


@triton.jit
def _gdn_batch_mm(a,b):
    return tl.sum(a[:,:,:,None]*b[:,None,:,:],axis=2)


@triton.jit
def _gdn_invert16_blocks(negative_lower):
    flat = negative_lower.reshape((256,))
    group = tl.arange(0, 8)
    i = tl.arange(0, 2)
    idx = (group[:, None, None] * 2 + i[None, :, None]) * 16 + group[:, None, None] * 2 + i[None, None, :]
    inv = tl.gather(flat, idx.reshape((32,)), 0).reshape((8, 2, 2))
    inv = inv + (i[:, None] == i[None, :])[None, :, :]
    inv = _gdn_inverse16_merge(inv, flat, 2)
    inv = _gdn_inverse16_merge(inv, flat, 4)
    inv = _gdn_inverse16_merge(inv, flat, 8)
    return inv.reshape((16, 16))


@triton.jit
def _gdn_inverse16_merge(inv, flat, HALF: tl.constexpr):
    GROUPS: tl.constexpr = 16 // (2 * HALF)
    (left, right) = tl.split(inv.reshape((GROUPS, 2, HALF, HALF)).permute((0, 2, 3, 1)))
    gr = tl.arange(0, GROUPS)
    ri = tl.arange(0, HALF)
    indices = (gr[:, None, None] * (2 * HALF) + HALF + ri[None, :, None]) * 16 + gr[:, None, None] * (2 * HALF) + ri[None, None, :]
    off = tl.gather(flat, indices.reshape((GROUPS * HALF * HALF,)), 0).reshape((GROUPS, HALF, HALF))
    cross = _gdn_batch_mm(_gdn_batch_mm(right, off), left)
    top = tl.join(left, tl.zeros((GROUPS, HALF, HALF), tl.float32)).permute((0, 1, 3, 2)).reshape((GROUPS, HALF, 2 * HALF))
    bottom = tl.join(cross, right).permute((0, 1, 3, 2)).reshape((GROUPS, HALF, 2 * HALF))
    inv = tl.join(top, bottom).permute((0, 3, 1, 2)).reshape((GROUPS, 2 * HALF, 2 * HALF))
    return inv


# W and U share the same triangular factor; adjacent jobs reuse its cache lines.
@triton.jit(do_not_specialize=['T'])
def wu_jobs(k,v,beta,w,u,A,g,cu_seqlens,chunk_indices,T,H:tl.constexpr,Hg:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BT:tl.constexpr,BN:tl.constexpr,IS_VARLEN:tl.constexpr):
    NV:tl.constexpr=((V+BN-1)//BN)
    NK:tl.constexpr=((K+BN-1)//BN)
    NJ:tl.constexpr=NV+NK
    it=tl.program_id(0)//NJ
    job=tl.program_id(0)%NJ
    bh=tl.program_id(1)
    b,h=bh//H,bh%H
    if IS_VARLEN:
        seq=tl.load(chunk_indices+it*2).to(tl.int32)
        it=tl.load(chunk_indices+it*2+1).to(tl.int32)
        bos=tl.load(cu_seqlens+seq).to(tl.int32)
        eos=tl.load(cu_seqlens+seq+1).to(tl.int32)
        T=eos-bos
    else:
        bos=b*T
    pb=tl.make_block_ptr(beta+bos*H+h,(T,),(H,),(it*BT,),(BT,),(0,))
    bb=tl.load(pb,boundary_check=(0,))
    pa=tl.make_block_ptr(A+(bos*H+h)*BT,(T,BT),(H*BT,1),(it*BT,0),(BT,BT),(1,0))
    ba=tl.load(pa,boundary_check=(0,1))
    if job<NV:
        pv=tl.make_block_ptr(v+(bos*H+h)*V,(T,V),(H*V,1),(it*BT,job*BN),(BT,BN),(1,0))
        bv=tl.load(pv,boundary_check=(0,1))
        bv=(bv*bb[:,None]).to(bv.dtype)
        out=tl.dot(ba,bv,allow_tf32=False)
        pu=tl.make_block_ptr(u+(bos*H+h)*V,(T,V),(H*V,1),(it*BT,job*BN),(BT,BN),(1,0))
        tl.store(pu,out.to(pu.dtype.element_ty),boundary_check=(0,1))
    else:
        pg=tl.make_block_ptr(g+bos*H+h,(T,),(H,),(it*BT,),(BT,),(0,))
        bg=tl.exp(tl.load(pg,boundary_check=(0,)).to(tl.float32))
        pk=tl.make_block_ptr(k+(bos*Hg+h//(H//Hg))*K,(T,K),(Hg*K,1),(it*BT,(job-NV)*BN),(BT,BN),(1,0))
        bk=tl.load(pk,boundary_check=(0,1))
        bk=(bk*bb[:,None]*bg[:,None]).to(bk.dtype)
        out=tl.dot(ba,bk)
        pw=tl.make_block_ptr(w+(bos*H+h)*K,(T,K),(H*K,1),(it*BT,(job-NV)*BN),(BT,BN),(1,0))
        tl.store(pw,out.to(pw.dtype.element_ty),boundary_check=(0,1))


def _gdn_recompute_w_u_jobs(k, v, beta, g_cumsum, A, cu_seqlens, block_v=None):
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    nt = triton.cdiv(T, BT) if indices is None else len(indices)
    nh = B * H if indices is None else H
    bn = 128
    jobs = triton.cdiv(V, bn) + triton.cdiv(K, bn)
    w = k.new_empty((B, T, H, K))
    u = torch.empty_like(v)
    wu_jobs[(nt * jobs, nh)](
        k, v, beta, w, u, A, g_cumsum, cu_seqlens, indices, T,
        H=H, Hg=Hg, K=K, V=V, BT=BT, BN=bn, IS_VARLEN=indices is not None,
        num_warps=2, num_stages=2,
    )
    return w, u


# Direct state recurrence adapted from the existing T-Head state kernel.
from flag_attn.FLA.compat import libentry, libtuner
from flag_attn.FLA.utils import use_cuda_graph
from flag_attn.runtime.backend._thead.FLA.chunk_delta_h import _autotune_configs as _state_autotune_configs


@libentry()
@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'USE_GK': lambda args: args['gk'] is not None, 'USE_INITIAL_STATE': lambda args: args['h0'] is not None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None, 'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@libtuner(configs=_state_autotune_configs('chunk_gated_delta_rule_fwd_kernel_h_blockdim64', [triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages) for num_warps in [2, 4] for num_stages in [2, 3, 4] for BV in [32, 64]]), cache_results=True, key=['H', 'K', 'V', 'BT', 'IS_VARLEN', 'USE_INITIAL_STATE', 'STORE_FINAL_STATE'], use_cuda_graph=use_cuda_graph)
@triton.jit(do_not_specialize=['T'])
def _gdn_state_direct_kernel(k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr, IS_VARLEN: tl.constexpr, beta):
    (i_v, i_nh) = (tl.program_id(0), tl.program_id(1))
    (i_n, i_h) = (i_nh // H, i_nh % H)
    if IS_VARLEN:
        (bos, eos) = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        (bos, eos) = (i_n * T, i_n * T + T)
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * BT).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_proj_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        proj_exp = tl.exp(tl.load(p_proj_g, boundary_check=(0,)).to(tl.float32))
        rhs = (tl.load(p_v, boundary_check=(0, 1)) - proj_exp[:, None] * b_v).to(v.dtype.element_ty)
        p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
        p_inv = tl.make_block_ptr(w, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        # Scale the right-hand side once per value tile before the A matmul.
        rhs_scaled = (rhs.to(tl.float32) * b_beta[:, None]).to(v.dtype.element_ty)
        b_v = tl.dot(tl.load(p_inv, boundary_check=(0, 1)), rhs_scaled)
        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = i_t * BT + tl.arange(0, BT) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=o_k1 < K, other=0.0).to(tl.float32)
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=o_k2 < K, other=0.0).to(tl.float32)
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=o_k3 < K, other=0.0).to(tl.float32)
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=o_k4 < K, other=0.0).to(tl.float32)
                b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def _gdn_state_direct(k,v,beta,A,g,initial_state,output_final_state,chunk_size,cu_seqlens):
    B,T,Hg,K=k.shape;H,V=v.shape[2:];BT=chunk_size
    assert K<=256, "current kernel does not support head dimension larger than 256."
    indices=prepare_chunk_indices(cu_seqlens,BT) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N,NT,offsets=B,triton.cdiv(T,BT),None
    else:
        N,NT,offsets=len(cu_seqlens)-1,len(indices),prepare_chunk_offsets(cu_seqlens,BT)
    h=k.new_empty((B,NT,H,K,V));vn=torch.empty_like(v)
    ht=k.new_empty((N,H,K,V),dtype=torch.float32) if output_final_state else None
    def grid(meta):return triton.cdiv(V,meta["BV"]),N*H
    _gdn_state_direct_kernel[grid](k=k,v=v,w=A,v_new=vn,g=g,gk=None,h=h,h0=initial_state,ht=ht,
        cu_seqlens=cu_seqlens,chunk_offsets=offsets,T=T,H=H,Hg=Hg,K=K,V=V,BT=BT,beta=beta)
    return h,vn,ht
