"""Pointer-only Triton kernels: no CUDA TMA or unsigned-dot compiler dependency.

SOL algorithm: Li et al., arXiv:2607.24027, Eqs. 9-11 and 15.
BF16 pooling and BF16 probability operands are explicit numerical choices.
"""

import math
import torch
import triton
import triton.language as tl

BLOCK = 64
GROUP = 16


@triton.jit
def _pool(X, Y, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
          NB: tl.constexpr, BLOCK: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    x = tl.load(X + batch * S0 + rows[:, None] * S1 + head * S2 + dims[None, :],
                rows[:, None] < T, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / tl.minimum(BLOCK, T - block * BLOCK)
    tl.store(Y + ((batch * NB + block) * H + head) * D + dims, mean)


@triton.jit
def _moments(KC, MEAN, VAR, H: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
             CHUNK: tl.constexpr, FD: tl.constexpr):
    # Diagonal population covariance of pooled BF16 keys, without proxy map.
    bh, feature_tile = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    features = feature_tile * FD + tl.arange(0, FD)
    indices = tl.arange(0, CHUNK)
    sums = tl.zeros((FD,), tl.float32)
    squares = tl.zeros((FD,), tl.float32)
    for start in range(tl.cdiv(NB, CHUNK)):
        blocks = start * CHUNK + indices
        x = tl.load(KC + ((batch * NB + blocks[:, None]) * H + head) * D
                    + features[None, :], blocks[:, None] < NB, other=0).to(tl.float32)
        sums += tl.sum(x, 0)
        squares += tl.sum(x * x, 0)
    mean = sums / NB
    variance = tl.maximum(squares / NB - mean * mean, 0.0)
    tl.store(MEAN + bh * D + features, mean)
    tl.store(VAR + bh * D + features, variance)


@triton.jit
def _threshold(QC, MEAN, VAR, THRESHOLD, H: tl.constexpr, D: tl.constexpr,
               NB: tl.constexpr, SCALE: tl.constexpr, BETA: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    dims = tl.arange(0, D)
    q = tl.load(QC + ((batch * NB + block) * H + head) * D + dims)
    mean = tl.load(MEAN + bh * D + dims)
    variance = tl.load(VAR + bh * D + dims)
    mu = tl.sum(q * mean, 0)
    sigma = tl.sqrt(tl.sum(q * q * variance, 0))
    cutoff = (mu + BETA * sigma) * SCALE * 1.4426950408889634
    tl.store(THRESHOLD + (batch * NB + block) * H + head, cutoff)


@triton.jit
def _dense(Q, K, V, O, NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
           D: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
           K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
           V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
           SCALE: tl.constexpr, BLOCK: tl.constexpr):
    qb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    qr = qb * BLOCK + tl.arange(0, BLOCK)
    kr = tl.arange(0, BLOCK)
    ds = tl.arange(0, D)
    q = tl.load(Q + batch * Q0 + qr[:, None] * Q1 + head * Q2 + ds[None, :],
                qr[:, None] < NQ, other=0)
    maximum = tl.full((BLOCK,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK, D), tl.float32)
    for kb in range(tl.cdiv(NK, BLOCK)):
        rows = kb * BLOCK + kr
        k = tl.load(K + batch * K0 + rows[None, :] * K1 + head * K2 + ds[:, None],
                    rows[None, :] < NK, other=0)
        s = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        s = tl.where(rows[None, :] < NK, s, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(s, 1))
        alpha = tl.exp2(maximum - next_max)
        p = tl.exp2(s - next_max[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        v = tl.load(V + batch * V0 + rows[:, None] * V1 + head * V2 + ds[None, :],
                    rows[:, None] < NK, other=0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        maximum = next_max
    tl.store(O + ((batch * NQ + qr[:, None]) * H + head) * D + ds[None, :],
             acc / denominator[:, None], qr[:, None] < NQ)


@triton.jit
def _sol(Q, K, V, KC, VC, THRESHOLD, O, COUNTS,
         NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
         K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
         V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
         SCALE: tl.constexpr, LOCAL: tl.constexpr, KV_SINK: tl.constexpr,
         Q_SINK: tl.constexpr, FORCE: tl.constexpr, STATS: tl.constexpr,
         BLOCK: tl.constexpr, GROUP: tl.constexpr):
    qb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    nqb, nkb = tl.cdiv(NQ, BLOCK), tl.cdiv(NK, BLOCK)
    qr = qb * BLOCK + tl.arange(0, BLOCK)
    r = tl.arange(0, BLOCK)
    ds = tl.arange(0, D)
    g = tl.arange(0, GROUP)
    q = tl.load(Q + batch * Q0 + qr[:, None] * Q1 + head * Q2 + ds[None, :],
                qr[:, None] < NQ, other=0)
    if FORCE == 0:
        cutoff = tl.load(THRESHOLD + (batch * nqb + qb) * H + head)
    else:
        cutoff = float("inf")
    maximum = tl.full((BLOCK,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK, D), tl.float32)
    count = 0
    qlength = tl.minimum(BLOCK, NQ - qb * BLOCK)
    for group in range(tl.cdiv(nkb, GROUP)):
        blocks = group * GROUP + g
        kc = tl.load(KC + ((batch * nkb + blocks[None, :]) * H + head) * D
                     + ds[:, None], blocks[None, :] < nkb, other=0)
        scores = tl.dot(q, kc) * (SCALE * 1.4426950408889634)
        proxy = tl.sum(tl.where(qr[:, None] < NQ, scores, 0.0), 0) / qlength
        selected = (proxy > cutoff)
        if FORCE == 1:
            selected = tl.full((GROUP,), True, tl.int1)
        selected = (selected | (tl.abs(blocks - qb) <= LOCAL)
                    | (blocks < KV_SINK) | (qb < Q_SINK)) & (blocks < nkb)
        # local_blocks=0 deliberately protects the same-index block only.
        approximate = (blocks < nkb) & ~selected
        approximate_scores = tl.where(approximate[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(approximate_scores, 1))
        # No approximate mass in a group is valid; avoid -inf - -inf NaNs.
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp2(maximum - safe_max)
        approximate_probability = tl.exp2(approximate_scores - safe_max[:, None])
        lengths = tl.minimum(BLOCK, NK - blocks * BLOCK).to(tl.float32)
        weighted = approximate_probability * tl.where(approximate, lengths, 0.0)[None, :]
        vc = tl.load(VC + ((batch * nkb + blocks[:, None]) * H + head) * D
                     + ds[None, :], blocks[:, None] < nkb, other=0)
        acc = tl.dot(weighted.to(tl.bfloat16), vc, acc * alpha[:, None])
        denominator = denominator * alpha + tl.sum(weighted, 1)
        maximum = next_max
        offsets = tl.where(selected, g, GROUP)
        nselected = tl.sum(selected.to(tl.int32), 0)
        count += nselected
        # Reductions yield block-uniform scalars; no lane-dependent loop bounds.
        for _ in range(nselected):
            offset = tl.min(offsets, 0)
            offsets = tl.where(g == offset, GROUP, offsets)
            rows = (group * GROUP + offset) * BLOCK + r
            k = tl.load(K + batch * K0 + rows[None, :] * K1 + head * K2 + ds[:, None],
                        rows[None, :] < NK, other=0)
            s = tl.dot(q, k) * (SCALE * 1.4426950408889634)
            s = tl.where(rows[None, :] < NK, s, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(s, 1))
            alpha = tl.exp2(maximum - next_max)
            p = tl.exp2(s - next_max[:, None])
            v = tl.load(V + batch * V0 + rows[:, None] * V1 + head * V2 + ds[None, :],
                        rows[:, None] < NK, other=0)
            acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = next_max
    tl.store(O + ((batch * NQ + qr[:, None]) * H + head) * D + ds[None, :],
             acc / denominator[:, None], qr[:, None] < NQ)
    if STATS:
        tl.store(COUNTS + (batch * nqb + qb) * H + head, count)


def _pooled(x, dtype):
    b, t, h, d = x.shape
    out = torch.empty((b, triton.cdiv(t, BLOCK), h, d), device=x.device, dtype=dtype)
    _pool[(out.shape[1], b * h)](x, out, t, h, d, *x.stride()[:3], out.shape[1],
                               BLOCK, num_warps=4)
    return out


def launch_dense(q, k, v, scale):
    b, nq, h, d = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _dense[(triton.cdiv(nq, BLOCK), b * h)](
        q, k, v, out, nq, k.shape[1], h, d, *q.stride()[:3],
        *k.stride()[:3], *v.stride()[:3], scale, BLOCK,
        num_warps=4, num_stages=1)
    return out


def launch_sol(q, k, v, scale, beta, local, kv_sink, q_sink, stats):
    b, nq, h, d = q.shape
    nqb = triton.cdiv(nq, BLOCK)
    kc, vc = _pooled(k, torch.bfloat16), _pooled(v, torch.bfloat16)
    threshold = torch.empty((b, nqb, h), device=q.device, dtype=torch.float32)
    force = 1 if beta == -math.inf else (2 if beta == math.inf else 0)
    if force == 0:
        qc = _pooled(q, torch.float32)
        mean = torch.empty((b * h, d), device=q.device, dtype=torch.float32)
        var = torch.empty_like(mean)
        _moments[(b * h, d // 16)](kc, mean, var, h, d, kc.shape[1], 64, 16, num_warps=4)
        _threshold[(nqb, b * h)](qc, mean, var, threshold, h, d, nqb,
                               scale, beta, num_warps=4)
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    counts = torch.empty((b, nqb, h), device=q.device, dtype=torch.int32)
    _sol[(nqb, b * h)](
        q, k, v, kc, vc, threshold, out, counts, nq, k.shape[1], h, d,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], scale, local,
        triton.cdiv(kv_sink, BLOCK), triton.cdiv(q_sink, BLOCK), force, stats,
        BLOCK, GROUP, num_warps=4, num_stages=1)
    return out, counts
