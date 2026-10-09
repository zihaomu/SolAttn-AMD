"""Small independent mask checks against installed upstream VDN planning.

Requires the optional FreeVideo research environment, not part of package CI.
No model weights or real activations are loaded.
"""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.bench_vdn_qkv import metric, sha
from integrations.vdn import VDNWindowSOL
from sol_attn_amd.reference import sol_reference


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    from freevideo_engine.attention import WindowAttention
    from freevideo_engine.locking import runtime_lock, device_lock_path
    torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32 = False
    f, tpf, vs, suffix = 7, 65, 11, 19
    ve = vs + f * tpf; s = ve + suffix
    layout = SimpleNamespace(seq_len=s, video_start=vs, video_end=ve, num_frames=f, tokens_per_frame=tpf)
    bounds = tuple((max(0, n // 2 * 2 - 1), min(f - 1, n // 2 * 2 + 2)) for n in range(f))
    gen = torch.Generator().manual_seed(1919)
    cpu = [torch.randn((s, 2, 128), generator=gen, dtype=torch.bfloat16) for _ in range(3)]
    record = dict(state="running", gpu_uuid=os.environ["SOL_GPU_UUID"], script_sha256=sha(__file__),
                  scope="synthetic independent per-token frame/window/anchor mask semantics, not quality/performance", cases=[])
    def save(): a.out.write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        with ExitStack() as lease:
            lease.enter_context(runtime_lock(shared=True))
            lease.enter_context(runtime_lock(device_lock_path(os.environ["SOL_GPU_UUID"]), inherit=False))
            q, k, v = [x.cuda() for x in cpu]
            for anchor in ("none", "rows", "columns", "both"):
                allowed = torch.zeros((s, s), dtype=torch.bool)
                for row in range(s):
                    if row < vs or row >= ve:
                        allowed[row] = True; continue
                    frame = (row - vs) // tpf
                    if anchor in ("rows", "both") and frame in (0, f - 1):
                        allowed[row] = True; continue
                    allowed[row, :vs] = True; allowed[row, ve:] = True
                    lo, hi = bounds[frame]
                    frames = set(range(lo, hi + 1))
                    if anchor in ("columns", "both"): frames |= {0, f - 1}
                    for n in frames: allowed[row, vs + n * tpf:vs + (n + 1) * tpf] = True
                dense = torch.nn.functional.scaled_dot_product_attention(
                    cpu[0].float().transpose(0, 1), cpu[1].float().transpose(0, 1),
                    cpu[2].float().transpose(0, 1), attn_mask=allowed).transpose(0, 1)
                policy = WindowAttention("torch-flash/torch-flash", window_batch=2)
                adapter = VDNWindowSOL(policy, beta=-math.inf)
                plan, windows = adapter.prepare(layout, bounds, q.device, anchor)
                # Compare every query's entire allowed set, including boundaries.
                actual_mask = torch.zeros_like(allowed)
                actual_mask[plan.dense_q.cpu()] = True
                for indexed, _ in windows:
                    for rows, keys in zip(indexed.rows, indexed.keys):
                        actual_mask[rows[:, None], keys[None, :]] = True
                if not torch.equal(actual_mask, allowed): raise RuntimeError("Independent mask mismatch")
                exact = adapter(q, k, v, layout, bounds, 128 ** -0.5, anchor)
                error = metric(exact.cpu(), dense)
                if not error["finite"] or error["relative_l2"] >= 0.008: raise RuntimeError("Dense mask numerical mismatch")
                adapter.beta = adapter.base_beta = 1.0
                sparse = adapter(q, k, v, layout, bounds, 128 ** -0.5, anchor)
                oracles = []
                for indexed, _ in windows:
                    ref, _ = sol_reference(cpu[0][indexed.rows], cpu[1][indexed.keys], cpu[2][indexed.keys],
                        beta=1.0, local_blocks=-1, protected_blocks=indexed.protected_blocks)
                    result = metric(sparse[indexed.rows].cpu(), ref)
                    if not result["finite"] or result["relative_l2"] >= 0.008: raise RuntimeError("Sparse oracle mismatch")
                    oracles.append(result)
                row = dict(anchor=anchor, every_query_mask_equal=True, dense_vs_fp32=error, sparse_oracle=oracles)
                record["cases"].append(row); save(); print(json.dumps(row), flush=True)
        record["state"] = "complete"; save()
    except BaseException as e: record.update(state="failed", error=repr(e)); save(); raise


if __name__ == "__main__": main()
