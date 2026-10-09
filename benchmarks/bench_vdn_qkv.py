"""Real activation, identical-mask benchmark. Never stores activations in git."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import shutil
import sys
import time
from types import SimpleNamespace

import torch

from integrations.vdn import VDNWindowSOL
from sol_attn_amd.reference import sol_reference


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def metric(actual, expected):
    # Large CPU vector reductions in FP32 can give self-cosine >1 and a
    # visibly biased norm. Use FP64 for post-generation CPU diagnostics.
    dtype = torch.float64 if actual.device.type == "cpu" else torch.float32
    a, b = actual.to(dtype), expected.to(dtype)
    delta = a - b
    return dict(finite=bool(a.isfinite().all()), max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / b.norm().clamp_min(1e-30)),
                cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))


def intervals(values):
    result = []
    for n in values:
        if result and result[-1][1] == n:
            result[-1] = (result[-1][0], n + 1)
        else:
            result.append((n, n + 1))
    return tuple(result)


def verify_mask(tasks, plan, windows, s):
    # Compare every actual captured query against upstream plan. Only metadata.
    expected = [None] * s
    for q in plan.dense_q.cpu().tolist():
        expected[q] = ((0, s),)
    for cpu, _ in windows:
        for rows, keys in zip(cpu.rows.tolist(), cpu.keys.tolist()):
            allowed = intervals(keys)
            for q in rows:
                if expected[q] is not None:
                    raise RuntimeError("Duplicate query row")
                expected[q] = allowed
    covered = 0
    for task in tasks.tolist():
        start, n, count = map(int, task[:3])
        allowed = tuple((int(task[3 + 2 * j]), int(task[4 + 2 * j])) for j in range(count))
        if start != covered or any(expected[q] != allowed for q in range(start, start + n)):
            raise RuntimeError("Mask differs from captured VDN mask")
        covered += n
    if covered != s or any(x is None for x in expected):
        raise RuntimeError("Incomplete mask coverage")


def oracle_samples(q, k, v, windows, beta, scale, output, heads=None):
    # Full pooled-key moments/routing context, then check first/middle/tail Q
    # blocks for first/last heads. CPU FP32 virtual-token softmax independent
    # from Triton recurrence. Restrict heads, not the allowed key set.
    cases = []
    for wi in sorted(set((0, len(windows) // 2, len(windows) - 1))):
        cpu, _ = windows[wi]
        batch = 0
        qi, ki = cpu.rows[batch], cpu.keys[batch]
        heads = sorted(set([0, q.shape[1] - 1] if heads is None else heads))
        qw, kw, vw = (x[idx][:, heads][None] for x, idx in ((q, qi), (k, ki), (v, ki)))
        # This oracle deliberately covers the full selected window so Q-block
        # origin and threshold statistics cannot be changed by subsampling.
        qb = sorted(set((0, qw.shape[1] // 128, (qw.shape[1] - 1) // 64)))
        ref, counts = sol_reference(qw, kw, vw, beta=beta, scale=scale, local_blocks=-1,
                                    protected_blocks=cpu.protected_blocks[batch:batch + 1], query_blocks=qb)
        selected = torch.cat([torch.arange(n * 64, min((n + 1) * 64, len(qi))) for n in qb])
        cases.append(dict(window=wi, heads=heads, query_rows=len(selected),
                          error=metric(output[qi[selected]][:, heads].cpu(), ref[0, selected]),
                          oracle_density=float(counts[:, qb].float().mean() / ((len(ki) + 63) // 64))))
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--composition", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--beta", nargs="+", type=float, default=[-2, -1, 0, 1])
    p.add_argument("--trials", type=int, default=9)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, a.out.with_suffix(".py"))
    torch.set_grad_enabled(False); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    from freevideo_engine.attention import WindowAttention
    from freevideo_engine.locking import runtime_lock, device_lock_path
    import os
    record = dict(state="running", scope="warm real first-head-group operator, all-in gathers/pooling/threshold/attention/scatter; cached static metadata setup excluded and separately recorded; not model quality evidence",
                  gates=dict(operator_relative_l2=0.10, operator_cosine=0.99,
                             oracle_relative_l2=0.012), cases=[],
                  hardware=torch.cuda.get_device_name(0), gpu_uuid=os.environ["SOL_GPU_UUID"],
                  torch=torch.__version__, hip=torch.version.hip,
                  composition_sha256=sha(a.composition), script_sha256=sha(__file__),
                  source_sha256={str(x):sha(x)
                                 for x in list(Path("src/sol_attn_amd").glob("*.py")) + [Path("integrations/vdn.py")]})
    def save(): a.out.write_text(json.dumps(record, indent=2) + "\n")
    save()
    spec = importlib.util.spec_from_file_location("freevideo_engine.sol_control", a.composition)
    composition = importlib.util.module_from_spec(spec); sys.modules[spec.name] = composition; spec.loader.exec_module(composition)
    try:
        with ExitStack() as lease:
            lease.enter_context(runtime_lock(shared=True))
            lease.enter_context(runtime_lock(device_lock_path(os.environ["SOL_GPU_UUID"]), inherit=False))
            for width in (672, 1344):
                path = a.data / f"actual-qkv-{width}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=True)
                meta = payload["metadata"]; layout = SimpleNamespace(**meta["layout"])
                bounds = tuple(map(tuple, meta["bounds"])); anchor = meta["anchor"]; scale = meta["scale"]
                cpu = [payload[x] for x in ("q", "k", "v")]
                # torch.save/.cpu may compact a sliced head group. Reconstruct
                # the captured model strides, not just its numerical values.
                gpu = []
                for x, strides in zip(cpu, meta["qkv_strides"]):
                    target = torch.empty_strided(x.shape, tuple(strides), device="cuda", dtype=x.dtype)
                    target.copy_(x); gpu.append(target)
                q, k, v = gpu
                policy = WindowAttention("torch-flash/torch-flash", window_batch=4)
                restore = composition.activate_index(None, policy)
                baseline = lambda: policy(q, k, v, layout, bounds, scale, anchor)
                adapter = VDNWindowSOL(policy)
                plan, windows = adapter.prepare(layout, bounds, q.device, anchor)
                verify_mask(payload["tasks"], plan, windows, layout.seq_len)
                row = dict(canvas=meta["canvas"], shape=list(q.shape), layout=meta["layout"], anchor=anchor,
                           captured_qkv_strides=meta["qkv_strides"], measured_qkv_strides=[list(x.stride()) for x in gpu],
                           activation_sha256=sha(path), mask_all_queries_equal=True,
                           metadata_setup_seconds=adapter.setup_seconds, candidates={})
                record["cases"].append(row); save()
                control = baseline()
                candidates = [-math.inf] + a.beta
                functions = {"baseline": baseline}
                for beta in candidates:
                    name = str(beta)
                    adapter.beta = adapter.base_beta = beta; adapter.stats = True
                    output = adapter(q, k, v, layout, bounds, scale, anchor)
                    error = metric(output, control)
                    # Per-head and global-query checks prevent aggregate hiding.
                    per_head = [metric(output[:, h], control[:, h]) for h in range(q.shape[1])]
                    error["worst_head_relative_l2"] = max(x["relative_l2"] for x in per_head)
                    error["global_queries_exact"] = torch.equal(output[plan.dense_q], control[plan.dense_q])
                    error["window_only"] = metric(output[plan.win_q], control[plan.win_q])
                    worst_head = max(range(q.shape[1]), key=lambda h: per_head[h]["relative_l2"])
                    checks = oracle_samples(*cpu, windows, beta, scale, output,
                                             heads=[0, q.shape[1] - 1, worst_head])
                    oracle_pass = all(x["error"]["relative_l2"] < 0.012 for x in checks)
                    if not oracle_pass or not error["finite"] or not error["global_queries_exact"]:
                        raise RuntimeError(f"Kernel/mask correctness gate failed at {width}, beta={beta}")
                    adapter.stats = False
                    row["candidates"][name] = dict(vs_dense=error, oracle_checks=checks, oracle_pass=oracle_pass,
                        window_block_density=adapter.density_samples[-1]["window_block_density"],
                        operator_error_gate=error["relative_l2"] <= 0.10 and error["cosine"] >= 0.99)
                    functions[name] = lambda beta=beta: run(beta)
                    save(); print(json.dumps(dict(canvas=width, beta=name, **row["candidates"][name])), flush=True)
                    del output
                def run(beta):
                    adapter.beta = adapter.base_beta = beta
                    return adapter(q, k, v, layout, bounds, scale, anchor)
                for fn in functions.values():
                    for _ in range(2): output = fn(); del output
                samples = {n: [] for n in functions}; names = list(functions)
                for trial in range(a.trials):
                    for name in names[trial % len(names):] + names[:trial % len(names)]:
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(); output = functions[name](); end.record(); end.synchronize()
                        samples[name].append(begin.elapsed_time(end)); del output
                row["timings"] = {n:dict(samples_ms=x, median_ms=statistics.median(x), cv=statistics.pstdev(x)/statistics.mean(x)) for n,x in samples.items()}
                for name, candidate in row["candidates"].items():
                    candidate["speedup"] = row["timings"]["baseline"]["median_ms"] / row["timings"][name]["median_ms"]
                # Warm the other real head-group width; no activation recapture.
                run(a.beta[0]); adapter(q[:, :8], k[:, :8], v[:, :8], layout, bounds, scale, anchor)
                restore(); save(); print(json.dumps(dict(canvas=width, timings=row["timings"])), flush=True)
                del control, q, k, v, cpu, payload, gpu, target; torch.cuda.empty_cache()
        record["state"] = "complete"; save()
    except BaseException as e:
        record.update(state="failed", error=repr(e)); save(); raise


if __name__ == "__main__": main()
