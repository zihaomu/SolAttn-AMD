"""Explicit equal-length indexed windows; no arbitrary per-query mask fallback."""

from dataclasses import dataclass
import math

import torch

from .attention import SolStats, _validate, _nonnegative_integer


@dataclass(frozen=True)
class IndexedPlan:
    """Each batch has independent, sorted Q rows and an identical allowed KV set.

    Construct on CPU and transfer once, outside repeated operator timing. This
    descriptor is application-owned and immutable by convention. Its tensors
    contain only indices/protection, never model activations. No masked-out key
    enters pooling. Blocks touching protected keys or the original-index local
    neighborhood are exact; protection rounds outward to whole packed blocks.
    """

    rows: torch.Tensor
    keys: torch.Tensor
    protected_blocks: torch.Tensor
    sequence_length: int

    @classmethod
    def create(cls, rows, keys, sequence_length, *, protected_ranges=(), local_tokens=64):
        _nonnegative_integer("local_tokens", local_tokens)
        _nonnegative_integer("sequence_length", sequence_length)
        if sequence_length == 0:
            raise ValueError("sequence_length must be positive")
        for name, x in (("rows", rows), ("keys", keys)):
            if (x.device.type != "cpu" or x.dtype != torch.int64 or x.ndim != 2
                    or any(n == 0 for n in x.shape)):
                raise ValueError(f"{name} must be a nonempty CPU int64 matrix")
            if bool((x < 0).any() or (x >= sequence_length).any()):
                raise ValueError(f"{name} index outside sequence")
            if bool((x[:, 1:] <= x[:, :-1]).any()):
                raise ValueError(f"{name} must be strictly increasing within each batch")
        if rows.shape[0] != keys.shape[0]:
            raise ValueError("rows and keys must have the same batch size")
        ranges = tuple(protected_ranges)
        for a, b in ranges:
            if not (isinstance(a, int) and isinstance(b, int)
                    and 0 <= a < b <= sequence_length):
                raise ValueError("Invalid protected key range")
        batch, nq = rows.shape
        nk = keys.shape[1]
        protection = torch.zeros((batch, (nq + 63) // 64, (nk + 63) // 64), dtype=torch.bool)
        # Metadata only: no S*S mask and no activation-dependent routing map.
        for kb, start in enumerate(range(0, nk, 64)):
            ki = keys[:, start:start + 64]
            sink = torch.zeros(batch, dtype=torch.bool)
            for a, b in ranges:
                sink |= ((ki >= a) & (ki < b)).any(1)
            for qb, qs in enumerate(range(0, nq, 64)):
                qi = rows[:, qs:qs + 64]
                lo, hi = qi[:, :1] - local_tokens, qi[:, -1:] + local_tokens
                local = ((ki >= lo) & (ki <= hi)).any(1)
                protection[:, qb, kb] = sink | local
        return cls(rows.clone().contiguous(), keys.clone().contiguous(), protection, sequence_length)

    def to(self, device):
        return IndexedPlan(self.rows.to(device), self.keys.to(device),
                           self.protected_blocks.to(device), self.sequence_length)


def sol_attention_indexed(q, k, v, plan, *, beta=1.0, scale=None, return_stats=False):
    """BF16 NHD inputs -> [windows, query_rows, heads, D] output.

    Includes all Q/K/V gathers, pooling, thresholds and attention. Plan creation
    and transfer are explicit one-time setup costs. Does not scatter output or
    infer VDN masks. Callers must not put queries with different allowed key sets
    in the same window. Arbitrary masks/dropout/training are not implemented.
    """
    if any(t.ndim != 3 for t in (q, k, v)) or q.shape != k.shape or k.shape != v.shape:
        raise ValueError("Indexed Q/K/V must have matching NHD shapes")
    scale = _validate(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale)
    if not isinstance(plan, IndexedPlan):
        raise ValueError("An explicit IndexedPlan is required")
    if plan.sequence_length != q.shape[0]:
        raise ValueError("Plan sequence length does not match Q/K/V")
    if any(x.device != q.device for x in (plan.rows, plan.keys, plan.protected_blocks)):
        raise ValueError("Transfer the plan to the Q/K/V device first")
    b, nq = plan.rows.shape
    if (plan.rows.ndim != 2 or plan.keys.ndim != 2 or plan.keys.shape[0] != b
            or b <= 0 or nq <= 0 or plan.keys.shape[1] <= 0
            or plan.rows.dtype != torch.int64 or plan.keys.dtype != torch.int64
            or plan.protected_blocks.dtype != torch.bool
            or plan.protected_blocks.shape != (b, (nq + 63) // 64, (plan.keys.shape[1] + 63) // 64)
            or any(not x.is_contiguous() for x in (plan.rows, plan.keys, plan.protected_blocks))):
        raise ValueError("Invalid indexed descriptor; use IndexedPlan.create")
    beta = float(beta)
    if math.isnan(beta):
        raise ValueError("beta must not be NaN")
    from .kernels import launch_sol
    with torch.no_grad(), torch.cuda.device(q.device):
        qw, kw, vw = q[plan.rows], k[plan.keys], v[plan.keys]
        out, counts = launch_sol(qw, kw, vw, scale, beta, -1, 0, 0,
                                 return_stats, plan.protected_blocks)
    if return_stats:
        return out, SolStats(counts, (plan.keys.shape[1] + 63) // 64)
    return out
