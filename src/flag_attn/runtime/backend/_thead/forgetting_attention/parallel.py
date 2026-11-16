# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head Forgetting Attention forward with optional adaptive pruning.

Use the portable Triton implementation, without NVIDIA TMA/WGMMA kernels.
Both BTHD and BHTD layouts, grouped-query heads, and left padding are supported.
Backward is not implemented.
"""

import importlib
import math

import torch
import torch.nn.functional as F


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    """Run the portable Triton forward kernel on T-Head PPU.

    The adapter supports grouped heads and dimensions 60/100 without changing
    attention scaling or pruning. Runtime errors from a selected kernel propagate.
    """
    dimension = q.shape[-1]
    if dimension not in (16, 32, 60, 64, 100, 128):
        raise ValueError("head dimension must be one of 16, 32, 60, 64, 100, 128")
    if k.shape[-1] != dimension or v.shape[-1] != dimension:
        raise ValueError("q, k and v must have the same head dimension")
    head_axis = 1 if head_first else 2
    heads, kv_heads = q.shape[head_axis], k.shape[head_axis]
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads or v.shape[head_axis] != kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    # Keep the scale of the original dimension when zero-padding q/k/v.
    scale = 1.0 / math.sqrt(dimension) if sm_scale is None else sm_scale
    padded = 1 << (dimension - 1).bit_length()
    if dimension != padded:
        q, k, v = (F.pad(t, (0, padded - dimension)) for t in (q, k, v))
    groups = heads // kv_heads
    if groups != 1:
        k, v = (t.repeat_interleave(groups, dim=head_axis) for t in (k, v))
    entry = importlib.import_module(".naive", __package__).forgetting_attention
    with torch.cuda.device(q.device):
        output = entry(
            q, k, v, log_fgate, head_first=head_first, seq_start=seq_start,
            sm_scale=scale, adaptive_threshold=adaptive_threshold,
        )
    return output if dimension == padded else output[..., :dimension].contiguous()


forgetting_attention = parallel_forgetting_attn
