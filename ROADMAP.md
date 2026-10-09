# Correctness-gated roadmap

1. **Standalone BF16 foundation**: online routing/correction, explicit dense
   control, independent oracle, ragged/stride/sink tests, reproducible all-in
   benchmark. Validate on one isolated R9700 before claiming device support.
2. **Performance characterization**: retained real Q/K/V from both stages,
   multiple shapes and route densities, timing variance and GPU resource/stall
   analysis. Do not treat the project's untuned dense control as the only baseline.
3. **Semantic masks**: explicit allowed-key intervals and actual VDN frame,
   anchor, global and text/audio rules. Pool only valid keys; preserve query-
   dependent multiplicities. The initial unmasked API is not a window adapter.
   The indexed-window API now pools only explicitly allowed keys and protects
   conditioning/anchor/local blocks; the VDN adapter remains research-only.
4. **Numerical experiments**: full covariance thresholds and separately measured
   INT8 QK, FP8/BF16 PV, or mixed-signedness integer PV. Preserve a BF16 control.
   The uint8×uint8 compiler PR does not supply uint8×int8 PV support.
5. **Model integration gate**: unchanged weights/seed/8+3 steps/latent upscaler,
   paired warm wall time, multiple prompts, video/temporal/audio checks. Keep
   production defaults unchanged until acceptance evidence exists.
6. **Portability**: characterize other RDNA/CDNA devices independently. HIP
   compilation alone does not establish hardware support or performance.
