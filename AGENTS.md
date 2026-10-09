# SolAttn-AMD development

- Keep this project independent of compiler patches, ComfyUI and model runtimes.
- Do not enable approximate kernels in production applications from this repo.
- Preserve the Triton block programming model: scalar loop indices and branches
  are block-uniform; represent per-element variation with tensors. No hidden
  lane-varying scalar values or hardware-ID assembly tricks.
- Validate kernels against the independent mathematical reference, and report
  approximation error against dense attention separately. Never call sparse
  attention mathematically equivalent to dense attention.
- Fail explicitly on unsupported masks, training, devices, dtypes and layouts.
- Keep benchmarks all-in (pooling, threshold generation, attention and required
  copies). Separate diagnostic forward-only timings if added.
- Report hardware, software, data provenance, density, error, median and samples.
  Synthetic results are not evidence of video/audio quality or model speedup.
- Use `pytest -s --tb=short`; preserve failed experiment receipts.
- Do not commit real model activations, models, caches or large generated files.
- No native compiler changes are required by the initial BF16 implementation.
