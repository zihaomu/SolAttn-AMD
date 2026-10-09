# Initial R9700 validation — 2026-10-09

This is a standalone operator milestone, **not** FreeVideo integration or a
production-quality/model-speedup result. No real model activations were used.

## Environment and checks

- Radeon AI PRO R9700, gfx1201, GPU UUID `GPU-b5ab24f2dc5d81ac`, PCI 03:00.0.
- PyTorch `2.12.0+rocm10.0.0`, HIP `7.15.26333`, Triton `3.8.0`, Python 3.12.3.
- Immutable Docker image
  `sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9`.
- Device isolated with ROCR_VISIBLE_DEVICES and its render node; no other work
  observed on this device before execution. Clocks were not locked, and no
  hardware-counter profiling was performed.
- `pytest -s --tb=short -p no:cacheprovider`: **30 passed**, including 25 GPU
  cases and 5 CPU mathematical-reference/API cases; no skipped tests.
- Separate container without GPU devices: CPU suite **5 passed**, 25 deselected.
- Wheel build with no dependencies/build isolation, isolated installation and
  imports passed. Initial wheel SHA-256:
  `c8dd637bb392a6a4bb03b8df1522416f09253eb134aa8071a19b57a8aa411054`.
- Python AST parsing, shell syntax and `git diff --check` passed.
- Generated `_sol.amdgcn` contains `v_wmma_f32_16x16x16_bf16`, confirming
  native BF16 WMMA code generation on this device; this is not utilization data.
- Initial compile attempt failed because one variable name was reused for
  differently shaped approximate/exact probability tiles. Distinct tensor names
  fixed it; the full suite was rerun. Initial and final receipts remain in the
  local ignored `results/` directory.

GPU coverage includes D64/D128, B1/B2, heads, positive non-contiguous strides,
ragged Q/K lengths, cross-attention, multiple routing groups, rows with no exact
blocks, dense routing, zero inputs, whole-block prefix protection, deterministic
repeatability, unchanged input tensors and unsupported-mask rejection.

Operator tolerances against the independent FP32 oracle are atol 0.012,
rtol 0.025 and relative L2 <0.008. This oracle implements the selected sparse
formula, not dense attention; matching it is not a dense/model-quality gate.

## All-in synthetic benchmarks

Both fixtures use `[B=1,T,H=4,D=128]`, BF16, seed 42, beta 1.0,
diagonal-covariance thresholds, physical block 64 and group 16. Reported latency
includes pooling, moment/threshold construction, allocations/dispatch gaps in
the GPU event window, and the attention kernel. It is warm-JIT, **not** first
request or video-generation wall time.

Each number is the median of seven trials, each averaging 30 calls. Comparison
order rotates by trial. Baselines are PyTorch's automatic SDPA (backend not
forced) and the same-project dense BF16 control. Full samples, CV, environment
and source SHA-256 identities are in the JSON receipts:

- [Constructed block-smooth fixture](benchmarks/r9700-synthetic-smooth.json)
- [Unstructured random stress fixture](benchmarks/r9700-synthetic-random.json)

| Fixture | T | Torch SDPA ms | Project dense ms | SOL all-in ms | Exact block density | Relative L2 vs SDPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Block-smooth | 1024 | 0.05308 | 0.03379 | 0.09360 | 31.25% | 0.211% |
| Block-smooth | 4096 | 0.58003 | 0.37488 | 0.13369 | 19.70% | 0.232% |
| Block-smooth | 8192 | 2.12930 | 1.49284 | 0.39879 | 17.83% | 0.230% |
| Random | 1024 | 0.05204 | 0.03427 | 0.09070 | 30.57% | 70.59% |
| Random | 4096 | 0.58502 | 0.38068 | 0.13336 | 19.64% | 73.88% |
| Random | 8192 | 2.16564 | 1.52595 | 0.41112 | 17.85% | 74.88% |

**Interpretation:** block-smooth is deliberately favorable: each 64-token block
shares a randomly drawn centroid plus small noise. At 8192 tokens, this fixture
shows 5.34x over automatic SDPA and 3.74x over the project dense control, with
0.23% relative L2. It does not establish this ratio on H3 or masked VDN inputs.
The 1024-token case regresses due to overhead. The random fixture is fast at
long lengths **but numerically unacceptable as a dense substitute**, showing
why speed and operator correctness cannot serve as model-quality evidence.

No automatic production promotion, universal accuracy guarantee, optimal
baseline claim or hardware-stall explanation follows from these measurements.
Clock/host variation is visible in the retained short-sequence samples.

## Reproduce and next gate

```bash
bash scripts/run_rocm.sh -m pytest -s --tb=short -p no:cacheprovider
bash scripts/run_rocm.sh benchmarks/bench_attention.py \
  --tokens 1024 4096 8192 --heads 4 --input block-smooth \
  --trials 7 --repeats 30 --out /results/smooth-new.json
bash scripts/run_rocm.sh benchmarks/bench_attention.py \
  --tokens 1024 4096 8192 --heads 4 --input random \
  --trials 7 --repeats 30 --out /results/random-new.json
```

Next: real first-pass/refinement Q/K/V, stronger dense baselines and explicit
VDN allowed-key/frame/anchor/global semantics. Only after those gates should
the official 8+3-step two-pass video/audio pipeline be evaluated. Quantization
and full-covariance threshold experiments remain separate branches of work.
