# Initial R9700 validation — 2026-10-09

This is a standalone operator milestone, **not** FreeVideo integration or a
production-quality/model-speedup result. No real model activations were used.
This opening section records the initial historical milestone; the subsequent
real-VDN experiments are recorded below, not inferred from the synthetic data.

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

## Subsequent real-VDN mask/operator gate

The explicit indexed-window API now gathers only declared allowed keys before
pooling. Its cached static protection descriptor rounds text/audio/anchor and
original-index local protection outward to whole packed 64-token blocks.
Windows stay independent, with one normalization over each actual allowed set.
The optional research adapter keeps global/anchor-row queries dense and leaves
VDN's learned linear attention branch/gates/projections unchanged. No runtime
sources, installed packages, production defaults or Triton compiler are edited.

Real captures come from the first 16-head group of the first model block:
base initial denoising step and enhancement initial step using a retained real
refinement input. Shapes are `[19003,16,128]` and `[73435,16,128]`, BF16; text/audio
prefix length 859, 72 latent frames, 252/1008 tokens per latent frame, anchor
policy `both`. Every query's allowed-key intervals were compared against both
the captured task descriptors and the upstream VDN plan. Global/anchor-query
outputs remain bitwise equal to the exact baseline.

The [initial real-QKV receipt](benchmarks/r9700-vdn-qkv-initial.json) retains all
nine warm GPU-event samples per candidate with rotating order. Timings include
gathers, pooling, moments/thresholds, attention, exact global queries and scatter;
static metadata construction/transfer is separately recorded. The base baseline
is exact gathered-window attention, enhancement baseline is the existing exact
indexed-window optimization, not the standalone project's untuned dense control.

| Canvas | Beta | Exact window-block density | Baseline ms | SOL all-in ms | Relative L2 vs baseline | Worst-head relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 672×384 | 0 | 82.71% | 14.14 | 15.15 | 2.87% | 10.75% |
| 672×384 | 1 | 65.60% | 14.14 | 13.72 | 7.56% | 27.20% |
| 1344×768 | 0 | 67.72% | 142.56 | 129.66 | 1.82% | 5.11% |
| 1344×768 | 1 | 42.46% | 142.56 | 99.84 | 6.10% | 21.69% |

This shows an enhancement-only opportunity, not a 5.34x video speedup. Beta 1
has large per-head deviations hidden by aggregate error. The conservative full
case therefore retains all eight base steps exactly and tests beta 0 only in
the three enhancement steps. Audio is retained from the exact first pass.
Single first-block/first-step captures do not represent later blocks/timesteps.
The independent FP32 virtual-token oracle checks the sparse formula separately
from approximation error; matching it is not a multi-step/perceptual quality gate.

The [same-card follow-up receipt](benchmarks/r9700-vdn-qkv-final.json) repeats
beta 0/1 on GPU1 (`GPU-b11d6bcf3a61a551`), also including each candidate's
worst-error head in the oracle samples and window-only errors. Enhancement
baseline/SOL medians are 139.66/127.45 ms at beta 0 (1.096x) and 139.66/98.20 ms
at beta 1 (1.422x). Do not average these GPU1 timings with the initial GPU0 data.
Captured Q/K/V strides are all `[2048,128,1]`, matching benchmark layouts.

Additional regression checks:

- Full ROCm suite: **41 passed**, including true strided indexed NHD inputs,
  distinct batched masks, ragged indexed tails, dense limit and approximate
  oracle, protection and invariance to mutations of excluded K/V.
- No-device CPU container: **13 passed**, 28 GPU tests deselected.
- [Independent none/rows/columns/both anchor checks](benchmarks/r9700-vdn-mask-modes.json):
  every query's allowed set equals the upstream plan; synthetic globals, suffix
  audio, unaligned 65-token frames and grouped windows are covered. Dense and
  sparse outputs pass separate FP32 checks. No model weights needed.
- Indexed wheel build, isolated installation/import and plan creation passed.
  Final indexed wheel SHA-256:
  `e7c711a13a0a848770b4a11d03bcc3c93b628348368cdda9b2512993ebee295c`.

## Native full 8+3 case

Configuration: same frozen FreeVideo source `e09d4bd7a954363fe668712ba2ff4b6d9d16854f`,
model weights, retained official toy-car prompt, seed 2026090901, 8 steps at
672×384, unchanged learned latent upscaler, seed +1 and community sigma schedule
for 3 steps at 1344×768. Output is 243 frames, 24 FPS, 10.125 s, with audio
retained from the exact first pass. The existing exact indexed-attention/FP8-FF
composition is enabled identically on baseline and candidate. SOL changes only
the enhancement window-softmax leg, beta 0; no approximation in the first pass.

| Native baseline/SOL/baseline bracket | Full request s | Video worker s | Base 8 s | Enhance 3 s |
| --- | ---: | ---: | ---: | ---: |
| Exact baseline | 342.131 | 333.372 | 100.968 | 169.533 |
| Enhancement-only SOL, beta 0 | 330.505 | 321.953 | 101.075 | 157.739 |
| Exact baseline repeat | 341.737 | 332.967 | 100.868 | 169.319 |

Against the two-baseline median 341.934 s, SOL saves **11.429 s / 3.342%**
(1.03458x). Baseline range is only 0.394 s; both baselines produce bitwise-identical
latents, raw RGB, raw audio and the same MP4 SHA-256. The
[public compact receipt](benchmarks/r9700-official-8plus3.json) retains
provenance, both baseline samples and the candidate's independent gates.
This is not 5x and nowhere near a 100 s
generation result. This is measured native generation without profiler overhead,
including a fresh isolated text encoder process, model load, both sampling passes,
upscale, VAE/video/audio decode, artifact save and MP4 mux. Disk JIT/model caches
are reused; parent Python startup and post-generation validation are excluded.
The baseline reproduces the remembered ~340 s workload. No RTX 5090 measurement
is performed here, and no same-quality/hardware comparison to its advertised
timing is established.

The candidate's freshly encoded conditioning is tensor-exact with baseline;
frozen runtime sources and SOL sources are unchanged during execution. Static
SOL plan setup costs 1.395 s in the full run (included), with 600 enhancement
head-group calls. Host full AV decoding, 1344×768/243-frame/24-FPS geometry and
audio/video duration checks pass. All 243 raw frames are distinct and latents/
decoded audio are finite; no blank-video/silent-audio substitution.

Numerical comparisons against exact baseline:

- Video latent relative L2 **8.673%**, cosine 0.996236, max absolute error 2.838
  (independent FP64 CPU recheck).
- Raw RGB relative L2 **5.559%**, PSNR **31.872 dB**, max channel error 255.
- Audio latents, decoder input and decoded waveform: **bitwise exact**.
- Predetermined diagnostic gates (video-latent relative L2 ≤10%, audio relative
  L2 ≤10% and cosine ≥0.99, finite outputs) pass. Exact-media gate fails as
  expected. These are diagnostic tolerances, **not perceptual acceptance**.

Single-prompt/single-candidate evidence does not certify motion, fine detail,
prompt fidelity, multi-prompt quality or a statistically stable speedup. Thumbnail
spot checks are not a blinded evaluation. Approximation remains opt-in and is
not enabled in production. Further tuning must treat the observed accumulated
video error seriously; the first-layer 1.82% error does not bound the final result.
The original receipts retain their FP32 diagnostic reductions; large CPU-vector
self-cosines can exceed 1 due to accumulation error. Separate FP64 latent receipts
correct this diagnostic, without altering generated data, timings or thresholds.

Preserved experiment incidents: GPU0 became occupied by another Qwen server
between operator timing and the first encoder launch, causing an encoder OOM
before any SOL/model sampling; the baseline/candidate were moved to idle GPU1
without stopping that service. The first successful baseline generation's receipt
records a post-generation `ffprobe`-missing error in the container; independent
host verification passed. Subsequent runners explicitly separate host AV checks.
No receipt was overwritten, and the encoder-OOM generation attempt is not a speed
sample. The completed baseline generation with the tool-only post-validation
failure is counted only after independent successful host verification.

### Research reproduction

The development runner requires the optional existing frozen FreeVideo/VDN
runtime and local model/capture storage, **not dependencies of the kernel package**.
Use fresh output paths, and override GPU UUID/render together:

```bash
export SOL_GPU_UUID=GPU-b11d6bcf3a61a551
export SOL_RENDER_DEVICE=/dev/dri/renderD130
exp=/data/experiments/freevideo-r9700
control="$exp/optimization-tasks/official-8plus3-asmevo-20261007"
request="$exp/outputs/official-8plus3-s2-v1/run-00/video.artifacts"

bash scripts/run_vdn_rocm.sh /sol/integrations/check_vdn_masks.py \
  --out /results/masks-new.json
bash scripts/run_vdn_rocm.sh /sol/benchmarks/bench_vdn_qkv.py \
  --data "$exp/optimization-tasks/sageattention-amd-20261007/results" \
  --composition "$control/07-composition/final/optimized.py" \
  --beta 0 1 --out /results/qkv-new.json
bash scripts/run_vdn_rocm.sh /sol/integrations/run_vdn_case.py \
  --out /results/baseline-new --request "$request/video.json" \
  --encode-request "$request/encode.json" \
  --composition "$control/07-composition/final/optimized.py" \
  --frozen-manifest "$control/shared/frozen-source.json"
# Candidate: repeat the preceding command with a fresh --out and add:
# --beta 0 --reference /results/baseline-new
python3 integrations/verify_vdn_av.py \
  --run results/baseline-new --out results/baseline-new/host-av-verification.json
# Run host AV verification for every generated baseline/candidate.
```
