"""Public BTHD interface; deliberately no model monkey-patching or fallback."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SolStats:
    # Device tensor [batch, query_blocks, heads]; obtaining a Python density syncs.
    exact_blocks: torch.Tensor
    key_blocks: int

    def density(self) -> float:
        return self.exact_blocks.float().mean().item() / self.key_blocks


def _validate(q, k, v, scale):
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Q/K/V must have BTHD rank-4 layout")
    if (k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[2:] != k.shape[2:] or any(n <= 0 for n in q.shape + k.shape)):
        raise ValueError("Q/K/V batch, head and dimension shapes must agree and be nonempty")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("Only BF16 Q/K/V are implemented")
    if q.shape[-1] not in (64, 128):
        raise ValueError("Only head dimensions 64 and 128 are implemented")
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError("Inference only: Q/K/V must not require gradients")
    if any(t.stride(-1) != 1 or any(s <= 0 for s in t.stride()) for t in (q, k, v)):
        raise ValueError("Q/K/V need positive strides and contiguous head dimension")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if (torch.version.hip is None or q.device.type != "cuda"
            or k.device != q.device or v.device != q.device):
        raise ValueError("The Triton backend requires Q/K/V on the same AMD ROCm GPU")
    return scale


def _nonnegative_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def dense_attention(q, k, v, *, scale=None):
    """Noncausal BF16 attention over all keys, FP32 softmax/accumulation.

    This is a same-project dense control, not an optimized vendor baseline.
    """
    scale = _validate(q, k, v, scale)
    from .kernels import launch_dense
    with torch.no_grad(), torch.cuda.device(q.device):
        return launch_dense(q, k, v, scale)


def sol_attention(q, k, v, *, beta=1.0, scale=None, local_blocks=1,
                  kv_sink_tokens=0, query_sink_tokens=0, return_stats=False):
    """SOL-style noncausal attention; no semantic mask support in v0.1.

    Physical blocks contain 64 tokens. Selected blocks use BF16 QK/PV with FP32
    softmax/accumulation; other blocks contribute pooled-key/value correction.
    Thresholds use the documented diagonal-covariance estimator (paper Eq. 15).
    Prefix protection rounds OUTWARD to whole blocks. beta=-inf forces dense;
    beta=+inf leaves only protected blocks exact. No automatic approximation
    fallback and no quantized Q/K/V path are present.
    """
    scale = _validate(q, k, v, scale)
    beta = float(beta)
    if math.isnan(beta):
        raise ValueError("beta must not be NaN")
    for name, value in (("local_blocks", local_blocks),
                        ("kv_sink_tokens", kv_sink_tokens),
                        ("query_sink_tokens", query_sink_tokens)):
        _nonnegative_integer(name, value)
    if kv_sink_tokens > k.shape[1] or query_sink_tokens > q.shape[1]:
        raise ValueError("Sink token counts must not exceed their sequence lengths")
    from .kernels import launch_sol
    with torch.no_grad(), torch.cuda.device(q.device):
        out, counts = launch_sol(q, k, v, scale, beta, local_blocks,
                                 kv_sink_tokens, query_sink_tokens, return_stats)
    if return_stats:
        return out, SolStats(counts, (k.shape[1] + 63) // 64)
    return out
