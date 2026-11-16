# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head port of the optimized parallel Forgetting Attention forward.

Retain the fused prefix/ACP preparation, direct grouped-query addressing,
online softmax, separate decode reductions, and compiled-launch reuse.
PPU pipelined loads and tensor-cell dot products replace Hopper TMA/WGMMA.
The supported optimized contract is contiguous BTHD FP16/BF16 inference,
without seq_start, with a finite nonpositive scalar pruning threshold.
"""

import math
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.runtime import jit as runtime_jit
from triton.language.extra import libdevice

_trace=ContextVar('v76_launch_trace',default=None)
@contextmanager
def capture():
    records=[]
    token=_trace.set(records)
    try:
        yield records
    finally:
        _trace.reset(token)
def cdiv(x,y):return (x+y-1)//y
def pow2(x):return 1<<(x-1).bit_length()
class LaunchCache:
    def __init__(self,fn,limit=256):
        self.fn=fn
        self.limit=limit
        self.plans=OrderedDict()
        self.hits=0
        self.misses=0
        self.dist_context=getattr(runtime_jit,'DistributedRtContext',None)
    def __call__(self,key,grid,args,kwargs,fast=True):
        fn=self.fn
        # Preserve ordinary JIT behavior for debug/instrumented/hooked/distributed use.
        allowed=(fast and not fn.pre_run_hooks and not knobs.runtime.debug
                 and not knobs.compilation.instrumentation_mode)
        if self.dist_context is not None and self.dist_context().is_lite_mode:
            allowed=False
        plan=self.plans.get(key) if allowed else None
        if plan is None:
            compiled=fn[grid](*args,**kwargs)
            self.misses+=1
            if allowed:
                tail=tuple(kwargs[n] for n in fn.arg_names[len(args):])
                assert all(isinstance(x,(int,float,bool,str)) for x in tail)
                canonical=tuple(grid)+(1,)*(3-len(grid))
                # Runner retains code/grid metadata only, not this call's arguments.
                self.plans[key]=(compiled,compiled[canonical],tail)
                if len(self.plans)>self.limit:
                    self.plans.popitem(last=False)
        else:
            compiled,runner,tail=plan
            runner(*args,*tail)
            self.hits+=1
        records=_trace.get()
        if records is not None:
            records.append((fn,compiled,grid,dict(kwargs),plan is not None))
        return compiled

@lru_cache(maxsize=128)
def _get_cached_scalar_adaptive_threshold(
    device_index: int,
    B: int,
    H: int,
    value: float,
):
    """Create one reusable broadcast threshold view.

    v5.7a: cached scalar adaptive threshold.

    The returned tensor has shape (B, H), dtype FP32 and zero
    strides, matching torch.as_tensor(value).broadcast_to((B, H)).
    """
    scalar = torch.tensor(
        value,
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )

    return torch.broadcast_to(
        scalar,
        (B, H),
    )

@triton.jit
def _prepare_kernel(X,L,S,THR, XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,
                   TB:tl.constexpr,TH:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
                   C:tl.constexpr,LOGC:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,M:tl.constexpr,
                   BN:tl.constexpr,QM:tl.constexpr):
    h=tl.program_id(0)
    b=tl.program_id(1)
    i=tl.arange(0,C)
    qi=tl.arange(0,NQ)
    ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32)
    ends=tl.full((NK,),0,tl.float32)
    carry=tl.full((),0,tl.float32)
    pos_q=N-M+qi*QM
    pos_k=tl.minimum(ki*BN+BN-1,N-1)
    for chunk in range(tl.cdiv(N,C)):
        t=chunk*C+i
        x=tl.load(X+b*XB+h*XH+t*XT,t<N,other=0)
        x=tl.where(i==0,libdevice.add_rn(x,carry),x)
        for m in tl.static_range(LOGC):
            src=(i//(2<<m))*(2<<m)+(1<<m)-1
            left=tl.gather(x,src,axis=0)
            x=tl.where((i&(1<<m))!=0,libdevice.add_rn(x,left),x)
        tl.store(L+(b*H+h)*N+t,x,t<N)
        carry=tl.sum(tl.where(i==C-1,x,0),0)
        av=tl.gather(x,pos_q%C,axis=0)
        ev=tl.gather(x,pos_k%C,axis=0)
        anchor=tl.where(pos_q//C==chunk,av,anchor)
        ends=tl.where(pos_k//C==chunk,ev,ends)
    threshold=tl.load(THR+b*TB+h*TH)
    old=(ki[None,:]<tl.cdiv(N,BN))&((anchor[:,None]-ends[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+qi,starts,qi<tl.cdiv(M,QM))

@triton.jit
def _general_starts(P, T, S, M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
                    BN:tl.constexpr,QM:tl.constexpr,NK:tl.constexpr):
    h,b,group=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    r=group*16+tl.arange(0,16)
    j=tl.arange(0,NK)
    pp=P+(b*H+h)*N
    a=tl.load(pp+N-M+r*QM,r<tl.cdiv(M,QM),other=0)
    end=tl.minimum(j*BN+BN-1,N-1)
    e=tl.load(pp+end,j<tl.cdiv(N,BN),other=0)
    threshold=tl.load(T)
    old=(j[None,:]<tl.cdiv(N,BN))&((a[:,None]-e[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+r,starts,r<tl.cdiv(M,QM))


@triton.jit
def _parallel_kernel(
    Q, K, V, PREFIX, STARTS, O, sm_scale,
    M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
):
    dtype = O.dtype.element_ty
    h, b, mblock = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = h // (H // HK)
    rm = mblock * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    q = tl.load(Q + ((b * M + rm[:, None]) * H + h) * D + rd[None, :],
                rm[:, None] < M, other=0, cache_modifier=".ca")
    pp = PREFIX + (b * H + h) * N
    gq = tl.load(pp + N - M + rm, rm < M, other=0, cache_modifier=".ca")
    lo = tl.load(STARTS + (b * H + h) * tl.cdiv(M, BM) + mblock)
    lo = lo // BN * BN
    hi = tl.minimum(N, N - M + (mblock + 1) * BM)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    for start in tle.gpu.pipeline(lo, hi, BN, num_stages=STAGES):
        kn = start + rn
        k = tl.load(K + ((b * N + kn[None, :]) * HK + hk) * D + rd[:, None],
                    kn[None, :] < N, other=0, cache_modifier=".ca")
        v = tl.load(V + ((b * N + kn[:, None]) * HK + hk) * D + rd[None, :],
                    kn[:, None] < N, other=0, cache_modifier=".ca")
        gk = tl.load(pp + kn, kn < N, other=0, cache_modifier=".ca")
        s = tl.dot(q, k, input_precision="ieee") * scale
        s += (gq[:, None] - gk[None, :]) * log2e
        s = tl.where((kn[None, :] < N) & (N - M + rm[:, None] >= kn[None, :]),
                     s, -float("inf"))
        mnew = tl.maximum(mi, tl.max(s, 1))
        alpha = tl.exp2(mi - mnew)
        p = tl.exp2(s - mnew[:, None])
        psum = tl.sum(p, 1)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(dtype), v, input_precision="ieee")
        li = li * alpha + psum
        mi = mnew
    acc *= 1.0 / li[:, None]
    tl.store(O + ((b * M + rm[:, None]) * H + h) * D + rd[None, :],
             acc.to(dtype), rm[:, None] < M, cache_modifier=".cg")


@triton.jit
def _decode_kernel(
    Q, K, V, PREFIX, STARTS, O, sm_scale,
    N: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, BN: tl.constexpr,
):
    dtype = O.dtype.element_ty
    h, b = tl.program_id(0), tl.program_id(1)
    hk = h // (H // HK)
    rd = tl.arange(0, D)
    rn = tl.arange(0, BN)
    q = tl.load(Q + (b * H + h) * D + rd[None, :], cache_modifier=".ca")
    pp = PREFIX + (b * H + h) * N
    gq = tl.load(pp + N - 1, cache_modifier=".ca")
    lo = tl.load(STARTS + b * H + h)
    mi = tl.full((1,), -float("inf"), tl.float32)
    li = tl.zeros((1,), tl.float32)
    acc = tl.zeros((1, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    for start in range(lo, N, BN):
        kn = start + rn
        k = tl.load(K + ((b * N + kn[None, :]) * HK + hk) * D + rd[:, None],
                    kn[None, :] < N, other=0, cache_modifier=".ca")
        v = tl.load(V + ((b * N + kn[:, None]) * HK + hk) * D + rd[None, :],
                    kn[:, None] < N, other=0, cache_modifier=".ca")
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
    tl.store(O + (b * H + h) * D + rd[None, :],
             (acc * (1.0 / li[:, None])).to(dtype), cache_modifier=".cg")


_prepare_launch = LaunchCache(_prepare_kernel)
_starts_launch = LaunchCache(_general_starts)
_parallel_launch = LaunchCache(_parallel_kernel)
_decode_launch = LaunchCache(_decode_kernel)


def exact(gate, threshold, m, bn, qm):
    # Retain the Torch 2.10 accumulation tree from the original optimized path.
    b, h, n = gate.shape
    ln, lr = (n - 1).bit_length(), (b * h - 1).bit_length()
    chunk = 2 * (1 << min(9, max(4, (9 + ln - lr) // 2)))
    prefix = torch.empty((b, h, n), device=gate.device, dtype=torch.float32)
    starts = torch.empty((b, h, cdiv(m, qm)), device=gate.device, dtype=torch.int32)
    args = (gate, prefix, starts, threshold, *gate.stride(), *threshold.stride(),
            h, n, chunk, chunk.bit_length() - 1, pow2(cdiv(m, qm)),
            pow2(cdiv(n, bn)), m, bn, qm)
    key = (gate.device.index, b, gate.data_ptr() % 16, args[4:])
    _prepare_launch(key, (h, b), args, dict(num_warps=4))
    return prefix, starts


def prepare(q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold):
    if head_first or seq_start is not None or not torch.is_inference_mode_enabled():
        raise NotImplementedError("optimized PPU path requires BTHD inference without seq_start")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("require rank-4 Q and matching rank-4 K/V")
    b, m, h, d = q.shape
    n, hk = k.shape[1:3]
    if d not in (16, 32, 60, 64, 100, 128) or k.shape != (b, n, hk, d) or hk <= 0 or h % hk:
        raise ValueError("require supported dimensions and Q heads divisible by KV heads")
    if not (0 < m <= n and b > 0 and h > 0):
        raise ValueError("require 0 < M <= N and positive batch/heads")
    if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and all(
        t.dtype == q.dtype and t.device == q.device and t.is_contiguous() for t in (q, k, v)
    )):
        raise ValueError("require contiguous CUDA-compatible FP16/BF16 Q/K/V on the same device")
    if gate.shape != (b, n, h) or gate.dtype != torch.float32 or gate.device != q.device or not gate.is_contiguous():
        raise ValueError("require contiguous FP32 BNH gates on the Q device")
    if not isinstance(adaptive_threshold, (int, float)) or not math.isfinite(adaptive_threshold) or adaptive_threshold > 0:
        raise ValueError("require a finite nonpositive scalar ACP threshold")
    scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite scalar")
    # Preserve the unchanged PPU reference's reduction grouping for bitwise checks.
    bm = 1 if m == 1 else (64 if d <= 64 else 128)
    bn = min(128, max(16, pow2(n))) if m == 1 else 32
    stages = 1 if m == 1 else (3 if d <= 64 else 4)
    threshold = _get_cached_scalar_adaptive_threshold(q.device.index, b, h, float(adaptive_threshold))
    if b * h > 1 and torch.__version__.startswith('2.10.'):
        prefix, starts = exact(gate.transpose(1, 2), threshold, m, bn, bm)
    else:
        prefix = torch.cumsum(gate.transpose(1, 2), dim=-1, dtype=torch.float32)
        starts = torch.empty((b, h, cdiv(m, bm)), device=q.device, dtype=torch.int32)
        args = (prefix, threshold, starts, m, n, h, bn, bm, pow2(cdiv(n, bn)))
        _starts_launch((q.device.index, b, args[3:]), (h, b, cdiv(cdiv(m, bm), 16)), args, dict(num_warps=4))
    return prefix, starts, scale, bm, bn, stages


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    """Run optimized parallel ACP directly; the naive reference is never called."""
    with torch.cuda.device(q.device):
        b, m, h, real_d = q.shape
        n, hk = k.shape[1:3]
        prefix, starts, scale, bm, bn, stages = prepare(
            q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold,
        )
        d = pow2(real_d)
        if d != real_d:
            q, k, v = (F.pad(t, (0, d - real_d)) for t in (q, k, v))
        out = torch.empty_like(q)
        args = (q, k, v, prefix, starts, out, scale)
        key = (q.device.index, q.dtype, b, m, n, h, hk, d, bm, bn, type(scale), float(scale))
        if m == 1:
            _decode_launch(key, (h, b), args, dict(N=n, H=h, HK=hk, D=d, BN=bn, num_warps=4, num_stages=1))
        else:
            _parallel_launch(key, (h, b, cdiv(m, bm)), args,
                             dict(M=m, N=n, H=h, HK=hk, D=d, BM=bm, BN=bn, STAGES=stages,
                                  num_warps=4, num_stages=stages))
        return out if d == real_d else out[..., :real_d].contiguous()


forgetting_attention = parallel_forgetting_attn
