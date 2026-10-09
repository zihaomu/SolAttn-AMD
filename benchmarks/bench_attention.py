"""All-in kernel benchmark; synthetic inputs are never model-quality evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import triton

from sol_attn_amd import dense_attention, sol_attention


def timed(fn, repeats, trials):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "min_ms": min(samples), "max_ms": max(samples)}


def make_inputs(tokens, heads, dimension, seed, kind):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    def rand(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    q, k, v = [rand((1, tokens, heads, dimension)) for _ in range(3)]
    if kind == "block-smooth":
        # Controlled synthetic case where pooled blocks are useful; not real H3.
        for x in (q, k, v):
            centers = rand((1, (tokens + 63) // 64, heads, dimension))
            x.copy_(centers.repeat_interleave(64, dim=1)[:, :tokens] + 0.05 * x)
    return q, k, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--input", choices=["random", "block-smooth"], default="block-smooth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.tokens, args.heads, args.repeats, args.trials) <= 0:
        parser.error("shape and repetition counts must be positive")
    if not math.isfinite(args.beta):
        parser.error("Benchmark beta must be finite; dense-limit cases are covered by tests")
    if args.out.exists():
        parser.error("Use a new output path to preserve experiment receipts")
    device = torch.cuda.get_device_properties(0)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src" / "sol_attn_amd").glob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {"state": "running", "scope": "All-in GPU-event latency, warm JIT; synthetic data only",
              "started_unix": time.time(), "torch": torch.__version__, "hip": torch.version.hip,
              "triton": triton.__version__, "device": str(device),
              "gpu_uuid": os.environ.get("ROCR_VISIBLE_DEVICES"),
              "image": os.environ.get("SOL_ROCM_IMAGE"), "configuration": vars(args).copy(),
              "threshold_estimator": "diagonal_covariance", "source_sha256": hashes, "cases": []}
    report["configuration"]["out"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        for n in args.tokens:
            q, k, v = make_inputs(n, args.heads, args.head_dim, args.seed, args.input)
            sdpa = lambda: F.scaled_dot_product_attention(q.transpose(1, 2),
                k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            functions = {"torch_sdpa_auto": sdpa,
                         "project_dense": lambda: dense_attention(q, k, v),
                         "sol_all_in": lambda: sol_attention(q, k, v, beta=args.beta)}
            # Rotate order between trials; compile/warm-up before recording.
            for fn in functions.values():
                for _ in range(3):
                    fn()
            samples = {name: [] for name in functions}
            names = list(functions)
            for trial in range(args.trials):
                for name in names[trial % len(names):] + names[:trial % len(names)]:
                    samples[name].extend(timed(functions[name], args.repeats, 1)["samples_ms"])
            timings = {name: {"median_ms": statistics.median(xs), "samples_ms": xs,
                              "min_ms": min(xs), "max_ms": max(xs),
                              "cv_percent": statistics.pstdev(xs) / statistics.mean(xs) * 100}
                       for name, xs in samples.items()}
            actual, stats = sol_attention(q, k, v, beta=args.beta, return_stats=True)
            reference = sdpa().float()
            delta = actual.float() - reference
            case = {"shape": list(q.shape), "timings": timings, "exact_block_density": stats.density(),
                    "finite": bool(torch.isfinite(actual).all()),
                    "relative_l2_vs_sdpa": (delta.norm() / reference.norm().clamp_min(1e-8)).item(),
                    "max_abs_vs_sdpa": delta.abs().max().item(),
                    "cosine_vs_sdpa": F.cosine_similarity(actual.float().flatten(), reference.flatten(), dim=0).item(),
                    "speedup_vs_sdpa": timings["torch_sdpa_auto"]["median_ms"] / timings["sol_all_in"]["median_ms"],
                    "speedup_vs_project_dense": timings["project_dense"]["median_ms"] / timings["sol_all_in"]["median_ms"]}
            report["cases"].append(case); save()
            print(json.dumps(case), flush=True)
        report["state"] = "complete"; save()
    except Exception as exc:
        report.update(state="failed", error=repr(exc)); save()
        raise


if __name__ == "__main__":
    main()
