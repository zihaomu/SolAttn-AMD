# SolAttn-AMD

Experimental, standalone **BF16 SOL-style attention kernels for AMD ROCm**.
The initial target is Radeon AI PRO R9700 (`gfx1201`). This project does not
require the Triton `uint8 dot` draft PR, ComfyUI, FreeVideo or model weights.

Status: initial BF16 implementation validated on one R9700: **30 tests passed**,
plus wheel build/import and CPU-only reference checks. Initial all-in synthetic
benchmarks and limitations are retained in [the validation report](docs/VALIDATION.md).
No model-level acceleration or video/audio-quality claim is made. In particular,
unstructured random inputs have large approximation error; see the report.

## Install and run

Use an existing ROCm-compatible PyTorch/Triton environment:

```bash
pip install -e '.[test]'
pytest -s --tb=short
python benchmarks/bench_attention.py --tokens 1024 4096 --out results/first.json
```

On the development machine, the isolated Docker runner pins one R9700 and keeps
all writable caches/results under this checkout's ignored `results/` directory:

```bash
bash scripts/run_rocm.sh -m pytest -s --tb=short -p no:cacheprovider
bash scripts/run_rocm.sh benchmarks/bench_attention.py \
  --tokens 1024 4096 --out /results/first.json
```

Override **both** `SOL_GPU_UUID` and `SOL_RENDER_DEVICE` for another card. The
runner's image/UUID/render defaults are development-machine-specific. Other
AMD devices are not considered validated until their tests and measurements
are retained. Do not run benchmarks on a contended GPU.

```python
from sol_attn_amd import sol_attention

# Q/K/V: BF16 [batch, tokens, heads, dimension] on one ROCm GPU.
output, stats = sol_attention(
    q, k, v, beta=1.0, kv_sink_tokens=128, query_sink_tokens=128,
    return_stats=True,
)
print(stats.density())  # explicit CPU/GPU synchronization for diagnostics
```

## Numerical contract

- Inference only; BF16 inputs/output, head dimension 64 or 128, noncausal BTHD.
  Unequal Q/K lengths and positive-strided tensors with contiguous D are allowed.
  Inputs must be finite. No semantic mask, dropout, backward or KV cache API.
- Physical blocks contain 64 tokens; the outer loop scans groups of 16 centroids.
- K/V block means are rounded to BF16. Query means and key moments are FP32.
  Thresholds use the **diagonal pooled-key covariance estimator**, paper Eq. 15:
  `cutoff = scale * (q_mean · k_mean + beta * sqrt(sum(q_mean² * key_variance)))`.
  This differs from the full covariance estimator in Eq. 5.
- Selected blocks use ordinary BF16 QK/PV with FP32 online softmax/accumulation.
  Unselected blocks contribute `exp(q·k_mean) * length` to the denominator and
  the corresponding weighted V mean to the numerator. Ragged lengths are actual
  valid-token counts, not always 64. BF16 probability operands introduce rounding.
- Routing, correction and exact computation share one online-softmax state.
  No full quadratic proxy map or routing-index tensor is materialized.
- `local_blocks=1` protects same-index and neighboring blocks; zero protects
  only same-index. Prefix sink protection rounds outward to whole blocks.
  This is conservative conditioning protection, not a VDN frame/anchor mask.
- `beta=-inf` routes every block exactly; `beta=+inf` routes only protected
  blocks exactly. `dense_attention` supplies a simpler all-key dense control.
- Sparse output is approximate, **not equivalent to dense attention**. Passing
  the independent operator oracle does not establish multi-step model quality.

## Validation and benchmarks

Tests compare against an independent FP32 Torch oracle: exact blocks become
original tokens, approximate blocks become virtual centroid tokens with
`log(valid_length)` added to their logits, followed by one ordinary softmax.
Tests cover dense limits, zero inputs, ragged sequences, cross-attention,
strides, multiple batches/heads, sinks and deterministic repeatability.

Benchmarks include pooling, threshold construction and attention. They rotate
comparison order, retain all warm GPU-event samples, report density and error,
and compare with both PyTorch automatic SDPA and the same-project dense control.
`block-smooth` is an explicitly constructed favorable synthetic fixture;
`random` is a stress fixture. Neither is a captured model activation. Model
speedup claims require separately measured real activation and end-to-end gates.

## Scope and next steps

See [ROADMAP.md](ROADMAP.md). The official FreeVideo 672×384/8-step → latent
upscale → 1344×768/3-step workload remains the intended later integration case.
Its VDN window/anchor/global semantics must be implemented explicitly before
integration; replacing its hybrid architecture with dense SOL is not permitted.

## References

- Li et al., [Sol-Attn: Accelerating Video Generation Inference via On-the-Fly
  Attention Sparsification](https://arxiv.org/abs/2607.24027).
- [Community Triton implementation](https://github.com/kijai/ComfyUI-SolAttn_triton)
  and [comfy-kitchen HIP implementation](https://github.com/Comfy-Org/comfy-kitchen/tree/main/comfy_kitchen/backends/hip/sage_attention)
  provide useful implementation comparisons; neither is vendored here.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
