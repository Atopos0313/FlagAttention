# Copyright 2026 FlagOS Contributors
# Copyright 2023 BAAI
# SPDX-License-Identifier: Apache-2.0

"""Self-contained T-Head Forgetting Attention forward.

Port the optimized parallel implementation's fused prefix/pruning preparation,
grouped-query addressing, online softmax, decode and compiled-launch reuse.
PPU tensor-cell dot products and Triton software pipelining replace Hopper
TMA, barriers and WGMMA. The stride-aware ordinary path is included here and
preserves head_first, seq_start and optional/automatic/tensor thresholds.

The ordinary forward kernel derives from the bundled Forgetting Transformer
reference (commit 883f260c636e87971339749b0d8310004794a5cc), originally adapted
from FlagAttention. Both paths are forward-only; backward is not implemented.
No operator implementation is imported from another module.
"""

import math
from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import knobs
from triton.language.extra import libdevice
from triton.runtime import jit as runtime_jit


class _LaunchCache:
    """Cache compiled launch metadata, never caller tensors or outputs."""

    def __init__(self, kernel, limit=256):
        self.kernel = kernel
        self.limit = limit
        self.plans = OrderedDict()
        self.distributed_context = getattr(runtime_jit, "DistributedRtContext", None)

    def __call__(self, key, grid, args, kwargs):
        kernel = self.kernel
        allowed = not kernel.pre_run_hooks and not knobs.runtime.debug and not knobs.compilation.instrumentation_mode
        if self.distributed_context is not None and self.distributed_context().is_lite_mode:
            allowed = False
        plan = self.plans.get(key) if allowed else None
        if plan is None:
            compiled = kernel[grid](*args, **kwargs)
            if allowed:
                tail = tuple(kwargs[name] for name in kernel.arg_names[len(args) :])
                canonical_grid = tuple(grid) + (1,) * (3 - len(grid))
                self.plans[key] = (compiled[canonical_grid], tail)
                if len(self.plans) > self.limit:
                    self.plans.popitem(last=False)
        else:
            runner, tail = plan
            runner(*args, *tail)


@lru_cache(maxsize=128)
def _scalar_threshold(device_index, batch, heads, value):
    scalar = torch.tensor(value, dtype=torch.float32, device=torch.device("cuda", device_index))
    return scalar.broadcast_to((batch, heads))


@triton.jit
def _prepare_kernel(X, L, XB: tl.constexpr, XH: tl.constexpr, XT: tl.constexpr,
                    H: tl.constexpr, N: tl.constexpr, C: tl.constexpr, LOGC: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, C)
    carry = tl.full((), 0, tl.float32)
    for chunk in range(tl.cdiv(N, C)):
        t = chunk * C + i
        x = tl.load(X + b * XB + h * XH + t * XT, t < N, other=0)
        x = tl.where(i == 0, libdevice.add_rn(x, carry), x)
        for m in tl.static_range(LOGC):
            src = i // (2 << m) * (2 << m) + (1 << m) - 1
            left = tl.gather(x, src, axis=0)
            x = tl.where(i & 1 << m != 0, libdevice.add_rn(x, left), x)
        tl.store(L + (b * H + h) * N + t, x, t < N)
        carry = tl.sum(tl.where(i == C - 1, x, 0), 0)


@triton.jit
def _prune_start(P, T, anchor, N: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    j = tl.arange(0, NK)
    ends = tl.load(P + tl.minimum((j + 1) * BN - 1, N - 1), j < tl.cdiv(N, BN), other=0)
    threshold = tl.load(T)
    old = (j < tl.cdiv(N, BN)) & (anchor - ends < threshold)
    return tl.sum(old.to(tl.int32), 0) * BN


@triton.jit
def _parallel_kernel(
    Q,
    K,
    V,
    PREFIX,
    STARTS,
    O,
    sm_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    NK: tl.constexpr,
    QM: tl.constexpr,
):
    dtype = O.dtype.element_ty
    h, b, mblock = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    hk = h // (H // HK)
    rm = mblock * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    pp = PREFIX + (b * H + h) * N
    gq = tl.load(pp + N - M + rm, rm < M, other=0, cache_modifier=".ca")
    anchor = tl.load(pp + N - M + mblock * BM // QM * QM)
    lo = _prune_start(pp, STARTS, anchor, N, BN, NK)
    lo = lo // BN * BN
    hi = tl.minimum(N, N - M + (mblock + 1) * BM)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    for start in tl.range(lo, hi, BN, num_stages=STAGES, disable_licm=True):
        q = tl.load(Q + ((b * M + rm[:, None]) * H + h) * D + rd[None, :], rm[:, None] < M, other=0, cache_modifier=".ca")
        kn = start + rn
        k = tl.load(
            K + ((b * N + kn[None, :]) * HK + hk) * D + rd[:, None], kn[None, :] < N, other=0, cache_modifier=".ca"
        )
        v = tl.load(
            V + ((b * N + kn[:, None]) * HK + hk) * D + rd[None, :], kn[:, None] < N, other=0, cache_modifier=".ca"
        )
        gk = tl.load(pp + kn, kn < N, other=0, cache_modifier=".ca")
        s = tl.dot(q, k, input_precision="ieee") * scale
        s += (gq[:, None] - gk[None, :]) * log2e
        s = tl.where((kn[None, :] < N) & (N - M + rm[:, None] >= kn[None, :]), s, -float("inf"))
        mnew = tl.maximum(mi, tl.max(s, 1))
        alpha = tl.exp2(mi - mnew)
        p = tl.exp2(s - mnew[:, None])
        psum = tl.sum(p, 1)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(dtype), v, input_precision="ieee")
        li = li * alpha + psum
        mi = mnew
    acc *= 1.0 / li[:, None]
    tl.store(
        O + ((b * M + rm[:, None]) * H + h) * D + rd[None, :], acc.to(dtype), rm[:, None] < M, cache_modifier=".cg"
    )


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    PREFIX,
    STARTS,
    O,
    sm_scale,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    NK: tl.constexpr,
):
    dtype = O.dtype.element_ty
    h, b = (tl.program_id(0), tl.program_id(1))
    hk = h // (H // HK)
    rd = tl.arange(0, D)
    rn = tl.arange(0, BN)
    q = tl.load(Q + (b * H + h) * D + rd[None, :], cache_modifier=".ca")
    pp = PREFIX + (b * H + h) * N
    gq = tl.load(pp + N - 1, cache_modifier=".ca")
    lo = _prune_start(pp, STARTS, gq, N, BN, NK)
    mi = tl.full((1,), -float("inf"), tl.float32)
    li = tl.zeros((1,), tl.float32)
    acc = tl.zeros((1, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    for start in range(lo, N, BN):
        kn = start + rn
        k = tl.load(
            K + ((b * N + kn[None, :]) * HK + hk) * D + rd[:, None], kn[None, :] < N, other=0, cache_modifier=".ca"
        )
        v = tl.load(
            V + ((b * N + kn[:, None]) * HK + hk) * D + rd[None, :], kn[:, None] < N, other=0, cache_modifier=".ca"
        )
        gk = tl.load(pp + kn, kn < N, other=0, cache_modifier=".ca")
        s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * scale
        s += (gq - gk[None, :]) * log2e
        s = tl.where(kn[None, :] < N, s, -float("inf"))
        mnew = tl.maximum(mi, tl.max(s, 1))
        alpha = tl.exp2(mi - mnew)
        p = tl.exp2(s - mnew[:, None])
        psum = tl.sum(p, 1)
        acc *= alpha[:, None]
        acc += tl.sum(p.T * v, axis=0, keep_dims=True)
        li = li * alpha + psum
        mi = mnew
    tl.store(O + (b * H + h) * D + rd[None, :], (acc * (1.0 / li[:, None])).to(dtype), cache_modifier=".cg")


@triton.jit
def _ordinary_kernel(
    Q,
    K,
    V,
    LOG_LAMBDA,
    SEQ_START,
    START_INDEX,
    sm_scale,
    O,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_log_lambda_z,
    stride_log_lambda_h,
    stride_log_lambda_n,
    stride_start_index_z,
    stride_start_index_h,
    stride_start_index_mb,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".ca")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".ca")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".ca")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".ca")
    hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0
    if IS_ADAPTIVE:
        START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h + start_m * stride_start_index_mb
        start_index = tl.load(START_INDEX)
        lo = tl.maximum(start_index, lo)
    lo = lo // BLOCK_N * BLOCK_N
    offs_n_init += lo
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    log_lambda_in_ptrs = LOG_LAMBDA + offs_n_init * stride_log_lambda_n
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".ca")
            v = tl.load(v_ptrs, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".ca")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".ca")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".ca")
        if BLOCK_M > 1:
            s = tl.dot(q, k, input_precision="ieee") * qk_scale
        else:
            s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, float("-inf"))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(s - m_i_new[:, None])
        p_sum = tl.sum(p, 1)
        acc *= alpha[:, None]
        if BLOCK_M > 1:
            acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
        else:
            acc += tl.sum(p.T * v, axis=0, keep_dims=True)
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n
    if False or HAS_SEQ_START:
        is_empty_line = offs_m + P_SEQ < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
    else:
        acc = acc * (1.0 / l_i[:, None])
    if DIVISIBLE_M:
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


@triton.jit
def _find_starts_kernel(
    LOG_LAMBDA,
    START_INDEX,
    THRESHOLD,
    stride_log_lambda_z,
    stride_log_lambda_h,
    stride_log_lambda_n,
    stride_start_index_z,
    stride_start_index_h,
    stride_start_index_mb,
    stride_threshold_z,
    stride_threshold_h,
    M,
    N,
    P_SEQ,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h
    start_index = 0
    log_lambda_out_ptr = LOG_LAMBDA + P_SEQ * stride_log_lambda_n
    start_index_ptr = START_INDEX
    threshold = tl.load(THRESHOLD)
    for start_m in range(0, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        log_lambda_out = tl.load(log_lambda_out_ptr)
        offset_n = start_index + BLOCK_N - 1
        if not DIVISIBLE_N:
            offset_n = tl.minimum(N - 1, offset_n)
        log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
        decay = log_lambda_out - log_lambda_in
        while (decay < threshold) & (start_index + BLOCK_N < N):
            start_index += BLOCK_N
            offset_n = start_index + BLOCK_N - 1
            if not DIVISIBLE_N:
                offset_n = tl.minimum(N - 1, offset_n)
            log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
            decay = log_lambda_out - log_lambda_in
        tl.store(start_index_ptr, start_index.to(START_INDEX.dtype.element_ty))
        start_index_ptr += stride_start_index_mb
        log_lambda_out_ptr += stride_log_lambda_n * BLOCK_M


def _ceil_div(x, y):
    return (x + y - 1) // y


_prepare_launch = _LaunchCache(_prepare_kernel)
_parallel_launch = _LaunchCache(_parallel_kernel)
_decode_launch = _LaunchCache(_decode_kernel)


def _prepare_prefix(gate, threshold, m, bn, bm):
    b, h, n = gate.shape
    if b * h > 1 and torch.__version__.startswith("2.10."):
        ln, lr = (n - 1).bit_length(), (b * h - 1).bit_length()
        chunk = 2 * (1 << min(9, max(4, (9 + ln - lr) // 2)))
        prefix = torch.empty((b, h, n), device=gate.device, dtype=torch.float32)
        args = (gate, prefix, *gate.stride(), h, n, chunk, chunk.bit_length() - 1)
        key = (gate.device.index, b, gate.data_ptr() % 16, args[2:])
        _prepare_launch(key, (h, b), args, dict(num_warps=4))
    else:
        prefix = torch.cumsum(gate, dim=-1, dtype=torch.float32)
    return prefix, threshold


def _supports_optimized(q, k, v, gate, head_first, seq_start, scale, threshold):
    if head_first or seq_start is not None or not torch.is_inference_mode_enabled():
        return False
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold > 0:
        return False
    if scale is not None and (not isinstance(scale, (int, float)) or not math.isfinite(scale)):
        return False
    return all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v, gate))


def _optimized_forward(q, k, v, gate, sm_scale, threshold):
    b, m, h, real_d = q.shape
    n, hk = k.shape[1:3]
    scale = 1.0 / math.sqrt(real_d) if sm_scale is None else sm_scale
    # Match the PPU reference's reduction grouping, including decode rounding.
    qm = 1 if m == 1 else (64 if real_d <= 64 else 128)
    bm = 1 if m == 1 else 64
    bn = min(128, max(16, triton.next_power_of_2(n))) if m == 1 else 32
    stages = 1 if m == 1 else 2
    thr = _scalar_threshold(q.device.index, b, h, float(threshold))
    prefix, starts = _prepare_prefix(gate.transpose(1, 2), thr, m, bn, bm)
    d = triton.next_power_of_2(real_d)
    if d != real_d:
        q, k, v = (F.pad(t, (0, d - real_d)) for t in (q, k, v))
    out = torch.empty_like(q)
    args = (q, k, v, prefix, starts, out, scale)
    key = (q.device.index, q.dtype, b, m, n, h, hk, d, bm, bn, type(scale), float(scale))
    if m == 1:
        _decode_launch(key, (h, b), args, dict(N=n, H=h, HK=hk, D=d, BN=bn, NK=triton.next_power_of_2(_ceil_div(n, bn)), num_warps=4, num_stages=1))
    else:
        _parallel_launch(
            key,
            (h, b, _ceil_div(m, bm)),
            args,
            dict(M=m, N=n, H=h, HK=hk, D=d, BM=bm, BN=bn, NK=triton.next_power_of_2(_ceil_div(n, bn)), STAGES=stages, QM=qm, num_warps=4, num_stages=stages),
        )
    return out if d == real_d else out[..., :real_d].contiguous()


class _OrdinaryForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gate, seq_start, scale, threshold):
        b, h, m, d = q.shape
        n = k.shape[2]
        has_seq_start = seq_start is not None
        if has_seq_start:
            gate = gate.clone()
            mask = torch.arange(n, device=q.device)[None, None, :] < seq_start[:, None, None]
            gate.masked_fill_(mask, 0.0)
        prefix = torch.cumsum(gate, dim=-1, dtype=torch.float32)
        if threshold is not None:
            if isinstance(threshold, str):
                if threshold != "auto":
                    raise ValueError('adaptive_threshold must be "auto", a number, a tensor or None')
                max_q = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
                max_k = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
                logit_upper_bound = max_q * max_k * scale
                threshold = -(2 * logit_upper_bound + math.log(n)) - 10
            threshold = torch.as_tensor(threshold, dtype=torch.float32, device=q.device)
            try:
                threshold = torch.broadcast_to(threshold, (b, h))
            except RuntimeError as error:
                raise ValueError(f"adaptive_threshold must broadcast to ({b}, {h})") from error
        q, k, v = (t.contiguous() if t.stride(-1) != 1 else t for t in (q, k, v))
        bm = 1 if m == 1 else (64 if d <= 64 else 128)
        bn = min(128, max(16, triton.next_power_of_2(n))) if m == 1 else 32
        stages = 1 if m == 1 else (3 if d <= 64 else 4)
        starts = None
        start_strides = (0, 0, 0)
        if threshold is not None:
            starts = torch.empty((b, h, _ceil_div(m, bm)), device=q.device, dtype=torch.long)
            start_strides = starts.stride()
            _find_starts_kernel[(h, b)](
                prefix,
                starts,
                threshold,
                *prefix.stride(),
                *start_strides,
                *threshold.stride(),
                m,
                n,
                n - m,
                BLOCK_M=bm,
                BLOCK_N=bn,
                DIVISIBLE_N=n % bn == 0,
                num_warps=1,
            )
        out = torch.empty_like(q)
        _ordinary_kernel[(_ceil_div(m, bm), h, b)](
            q,
            k,
            v,
            prefix,
            seq_start,
            starts,
            scale,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *prefix.stride(),
            *start_strides,
            *out.stride(),
            m,
            n,
            n - m,
            1,
            BLOCK_M=bm,
            BLOCK_DMODEL=d,
            BLOCK_N=bn,
            HAS_SEQ_START=has_seq_start,
            IS_ADAPTIVE=threshold is not None,
            DIVISIBLE_M=m % bm == 0,
            DIVISIBLE_N=n % bn == 0,
            num_warps=4,
            num_stages=stages,
        )
        return out

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("Forgetting Attention is forward-only, as in the bundled reference")


def _ordinary_forward(q, k, v, gate, head_first, seq_start, sm_scale, threshold):
    head_axis = 1 if head_first else 2
    dimension = q.shape[-1]
    scale = 1.0 / math.sqrt(dimension) if sm_scale is None else sm_scale
    padded = triton.next_power_of_2(dimension)
    if padded != dimension:
        q, k, v = (F.pad(t, (0, padded - dimension)) for t in (q, k, v))
    groups = q.shape[head_axis] // k.shape[head_axis]
    if groups != 1:
        k, v = (t.repeat_interleave(groups, dim=head_axis) for t in (k, v))
    if not head_first:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        gate = gate.transpose(1, 2)
    out = _OrdinaryForward.apply(q, k, v, gate, seq_start, scale, threshold)
    if not head_first:
        out = out.transpose(1, 2)
    return out if dimension == padded else out[..., :dimension].contiguous()


def parallel_forgetting_attn(
    q,
    k,
    v,
    log_fgate,
    *,
    head_first=False,
    seq_start=None,
    sm_scale=None,
    adaptive_threshold=None,
):
    """Causal Forgetting Attention, preserving the generic forward interface.

    Q/K/V are FP16 or BF16 BTHD (BHTD with head_first=True); log_fgate is
    FP32 BNH (BHN with head_first=True). Query heads may group KV heads.
    D is one of 16, 32, 60, 64, 100, 128, and 0 < M <= N. Gates must be
    finite nonpositive log-forget values. seq_start is an optional batch-sized
    left-padding index tensor. adaptive_threshold accepts None, "auto", a
    nonpositive scalar or a tensor broadcastable to (batch, query_heads).
    Default scaling is 1/sqrt(the original head dimension).

    Optimized contiguous inference uses fused prefix/ACP preparation; other
    supported inputs use the ordinary kernels in this file. Selected-kernel
    errors propagate. Gradients are not implemented by the original operator.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have rank four")
    head_axis, time_axis = (1, 2) if head_first else (2, 1)
    b, d = q.shape[0], q.shape[-1]
    m, n = q.shape[time_axis], k.shape[time_axis]
    h, hk = q.shape[head_axis], k.shape[head_axis]
    if not (b > 0 and h > 0 and hk > 0 and 0 < m <= n and h % hk == 0):
        raise ValueError("require 0 < M <= N and query heads a positive multiple of KV heads")
    if d not in (16, 32, 60, 64, 100, 128) or k.shape[0] != b or k.shape[-1] != d or v.shape != k.shape:
        raise ValueError("q/k/v must have matching batch/dimension and k/v must have equal shapes")
    if not (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and all(t.dtype == q.dtype and t.device == q.device for t in (q, k, v))
    ):
        raise ValueError("require CUDA-compatible FP16/BF16 q/k/v on the same device")
    gate_shape = (b, h, n) if head_first else (b, n, h)
    if log_fgate.shape != gate_shape or log_fgate.dtype != torch.float32 or log_fgate.device != q.device:
        raise ValueError("log_fgate must be FP32 and match batch, query heads and KV length")
    if seq_start is not None and (
        seq_start.shape != (b,) or seq_start.device != q.device or seq_start.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("seq_start must be an integer tensor of shape (batch,) on the input device")
    with torch.cuda.device(q.device):
        if _supports_optimized(q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold):
            return _optimized_forward(q, k, v, log_fgate, sm_scale, adaptive_threshold)
        return _ordinary_forward(q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold)


forgetting_attention = parallel_forgetting_attn
