"""Independent FP64 CPU reductions for large latent-vector diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(); p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    torch.set_num_threads(8)
    x = torch.load(a.actual / "video.artifacts/latents.pt", map_location="cpu", weights_only=True)
    y = torch.load(a.reference / "video.artifacts/latents.pt", map_location="cpu", weights_only=True)
    checks = {}
    for key in ("video", "audio"):
        u, v = x[key].double(), y[key].double()
        if u.shape != v.shape: raise RuntimeError("Latent shape mismatch")
        delta = u - v
        checks[key] = dict(shape=list(u.shape), finite=bool(u.isfinite().all()), exact=torch.equal(x[key], y[key]),
                           max_abs=float(delta.abs().max()), relative_l2=float(delta.norm()/v.norm().clamp_min(1e-30)),
                           cosine=max(-1.0, min(1.0, float((u.flatten() @ v.flatten())/(u.norm()*v.norm()).clamp_min(1e-30)))))
    record = dict(state="complete", precision="FP64 CPU dot/norm/subtraction; independent post-generation check",
                  reference=a.reference.name, checks=checks,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.out.write_text(json.dumps(record, indent=2) + "\n"); print(json.dumps(record, indent=2))


if __name__ == "__main__": main()
