"""Inference-only SOL-style BF16 attention on AMD ROCm."""

from .attention import SolStats, dense_attention, sol_attention

__all__ = ["SolStats", "dense_attention", "sol_attention"]
