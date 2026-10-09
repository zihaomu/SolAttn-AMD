"""Explicit full 8+3 request, optional research SOL; fresh encoder included.

No installed/runtime sources or defaults are edited. All temporary hooks are
restored on close/error. A paired single trial is diagnostic, not a stable
speedup or perceptual-quality certification.
"""

import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from benchmarks.bench_vdn_qkv import sha, metric
from integrations import vdn


def summarize_media(artifacts):
    import hashlib
    import numpy as np
    import torch
    latents = torch.load(artifacts / "latents.pt", map_location="cpu", weights_only=True)
    summary = {"latent_" + k:dict(shape=list(latents[k].shape), finite=bool(latents[k].isfinite().all()))
               for k in ("video", "audio")}
    frames = np.load(artifacts / "rgb.npy", mmap_mode="r")
    hashes = {hashlib.sha256(frames[n].tobytes()).hexdigest() for n in range(len(frames))}
    summary["rgb"] = dict(shape=list(frames.shape), unique_frames=len(hashes),
                           minimum=int(frames.min()), maximum=int(frames.max()))
    for name in ("audio.npy", "audio_decoder_input.npy"):
        x = np.load(artifacts / name, mmap_mode="r")
        summary[name] = dict(shape=list(x.shape), finite=bool(np.isfinite(x).all()),
                             rms=float(np.sqrt(np.square(x.astype(np.float64)).mean())),
                             max_abs=float(np.abs(x).max()))
    summary["artifact_sanity_gate"] = (summary["rgb"]["unique_frames"] > 1
        and summary["rgb"]["maximum"] > summary["rgb"]["minimum"]
        and summary["audio.npy"]["rms"] > 1e-5
        and all(x["finite"] for x in summary.values() if isinstance(x, dict) and "finite" in x))
    return summary


def compare_media(actual, reference):
    import numpy as np
    import torch
    checks = {}
    observed = torch.load(actual / "latents.pt", map_location="cpu", weights_only=True)
    expected = torch.load(reference / "latents.pt", map_location="cpu", weights_only=True)
    for key in ("video", "audio"):
        x, y = observed[key], expected[key]
        if x.shape != y.shape:
            raise RuntimeError("Latent shape mismatch: " + key)
        checks["latent_" + key] = dict(shape=list(x.shape), exact=torch.equal(x, y), **metric(x, y))
    for name in ("audio.npy", "audio_decoder_input.npy", "rgb.npy"):
        x = np.load(actual / name, mmap_mode="r")
        y = np.load(reference / name, mmap_mode="r")
        if x.shape != y.shape or x.dtype != y.dtype:
            raise RuntimeError("Decoded media shape/dtype mismatch: " + name)
        dot = xx = yy = error = 0.0
        maximum = 0.0; exact = finite = True
        for i in range(len(x)):
            a, b = np.asarray(x[i], dtype=np.float64), np.asarray(y[i], dtype=np.float64)
            d = a - b
            dot += float((a * b).sum()); xx += float((a * a).sum()); yy += float((b * b).sum())
            error += float((d * d).sum()); maximum = max(maximum, float(np.abs(d).max()))
            exact &= bool(np.array_equal(x[i], y[i])); finite &= bool(np.isfinite(a).all())
        checks[name] = dict(shape=list(x.shape), dtype=str(x.dtype), finite=finite, exact=exact,
                            max_abs=maximum, relative_l2=(error / max(yy, 1e-30)) ** 0.5,
                            cosine=dot / max((xx * yy) ** 0.5, 1e-30), sha256=sha(actual / name))
        if name == "rgb.npy":
            import math
            checks[name]["psnr_db"] = 10 * math.log10(255 ** 2 / max(error / x.size, 1e-30))
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--encode-request", type=Path, required=True)
    p.add_argument("--composition", type=Path, required=True)
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--beta", type=float)
    p.add_argument("--base-beta", type=float)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, a.out / "runner.py")
    import torch
    torch.set_num_threads(8); torch.set_grad_enabled(False)
    from freevideo_engine.locking import runtime_lock, device_lock_path
    from freevideo_engine.runtime import Engine
    from freevideo_engine.worker import generate
    frozen = json.loads(a.frozen_manifest.read_text())
    frozen_root = a.frozen_manifest.parent / "source"
    def verify_source():
        for row in frozen["files"]:
            if sha(frozen_root / row["path"]) != row["sha256"]:
                raise RuntimeError("Frozen runtime source drift: " + row["path"])
    verify_source()
    req = json.loads(a.request.read_text())
    sampling = req["sampling_plan"]
    if (sampling["base_steps"], sampling["refine_steps"], sampling["first"]["width"], sampling["first"]["height"], sampling["second"]["width"], sampling["second"]["height"], req["geometry"]["frames"]) != (8, 3, 672, 384, 1344, 768, 243):
        raise ValueError("Not the official target 8+3 case")
    artifacts = a.out / "video.artifacts"; artifacts.mkdir()
    req.update(output=str(a.out / "video.mp4"), metrics=str(a.out / "engine.json"),
               artifacts=str(artifacts), conditioning=str(artifacts / "conditioning.pt"),
               input_cache_dir=None, resource_attempt=a.out.name)
    enc = json.loads(a.encode_request.read_text())
    enc.update(output=req["conditioning"], metrics=str(a.out / "encoding.json"),
               diagnostic_output=req["output"])
    (a.out / "request.json").write_text(json.dumps(req, indent=2) + "\n")
    (a.out / "encode.json").write_text(json.dumps(enc, indent=2) + "\n")
    record = dict(state="running", scope="one fresh full request including encoder process, model load, low-resolution 8 steps, learned latent upscale, refinement 3 steps, video/audio decode and mux; disk caches reused; no profiler; not stable speedup or perceptual certification",
                  gpu_uuid=os.environ["SOL_GPU_UUID"], gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, hip=torch.version.hip, source_commit=frozen["source_commit"],
                  request_sha256=sha(a.request), encode_request_sha256=sha(a.encode_request),
                  script_sha256=sha(__file__), composition_sha256=sha(a.composition),
                  sol_source_sha256={str(x):sha(x) for x in list(Path("src/sol_attn_amd").glob("*.py")) + [Path("integrations/vdn.py")]},
                  beta=a.beta, base_beta=a.base_beta, base_SOL_enabled=a.base_beta is not None,
                  phases=[], adapters=[], gates=dict(audio_cosine=0.99, audio_relative_l2=0.10,
                  latent_video_relative_l2=0.10), perceptual_quality_gate="not evaluated")
    def save(): (a.out / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    spec = importlib.util.spec_from_file_location("freevideo_engine.sol_case_control", a.composition)
    composition = importlib.util.module_from_spec(spec); sys.modules[spec.name] = composition; spec.loader.exec_module(composition)
    active = {}
    try:
        with ExitStack() as stack:
            stack.enter_context(runtime_lock(shared=True))
            stack.enter_context(runtime_lock(device_lock_path(os.environ["SOL_GPU_UUID"]), inherit=False))
            started = time.perf_counter()
            subprocess.run([sys.executable, "-m", "freevideo_engine.encode_worker", "--request", str(a.out / "encode.json")], check=True)
            record["encoder_process_wall_seconds"] = time.perf_counter() - started
            if a.reference:
                x = torch.load(Path(req["conditioning"]), map_location="cpu", weights_only=True)
                y = torch.load(a.reference / "video.artifacts/conditioning.pt", map_location="cpu", weights_only=True)
                def equal(x, y):
                    if isinstance(x, torch.Tensor): return isinstance(y, torch.Tensor) and torch.equal(x, y)
                    if isinstance(x, dict): return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
                    if isinstance(x, (tuple, list)): return len(x) == len(y) and all(equal(i, j) for i, j in zip(x, y))
                    return x == y
                record["conditioning_equal_reference"] = equal(x, y)
                if not record["conditioning_equal_reference"]:
                    raise RuntimeError("Fresh encoded conditioning differs; pair is invalid")
            record["conditioning_sha256"] = sha(req["conditioning"]); save()
            original_init, original_close, original_sample = Engine.__init__, Engine.close, Engine.sample
            def close(engine, *args, **kw):
                tokens = active.pop(id(engine), None)
                if tokens:
                    _, control, sol = tokens
                    if sol is not None:
                        adapter = sol[2]
                        record["adapters"].append(dict(calls=adapter.calls, setup_seconds=adapter.setup_seconds,
                                                        beta=adapter.beta, base_beta=adapter.base_beta,
                                                        base_enabled=adapter.base_enabled))
                        vdn.restore_engine(engine, sol)
                    composition.restore_engine(engine, control)
                return original_close(engine, *args, **kw)
            def init(engine, *args, **kw):
                original_init(engine, *args, **kw)
                control = composition.apply_engine(engine)
                active[id(engine)] = (engine, control, None)
                if a.beta is not None:
                    sol = vdn.apply_engine(engine, beta=a.beta, base_beta=a.base_beta,
                                           base_enabled=a.base_beta is not None, stats=False)
                    active[id(engine)] = (engine, control, sol)
            def sample(engine, *args, **kw):
                name = "refine_3" if kw.get("initial_latents") is not None else "base_8"
                tick = time.perf_counter(); value = original_sample(engine, *args, **kw); torch.cuda.synchronize()
                record["phases"].append(dict(name=name, wall_seconds=time.perf_counter() - tick)); save()
                return value
            for name, fn in (("__init__", init), ("close", close), ("sample", sample)):
                old = getattr(Engine, name); setattr(Engine, name, fn); stack.callback(setattr, Engine, name, old)
            def cleanup():
                for engine, _, _ in list(active.values()): close(engine)
            stack.callback(cleanup)
            tick = time.perf_counter(); generate(req); torch.cuda.synchronize()
            record["video_worker_wall_seconds"] = time.perf_counter() - tick
            record["full_request_wall_seconds"] = time.perf_counter() - started
        verify_source()
        record["frozen_source_unchanged"] = True
        record["sol_source_unchanged"] = all(sha(path) == digest for path, digest in record["sol_source_sha256"].items())
        if a.beta is not None and not record["sol_source_unchanged"]:
            raise RuntimeError("SOL source changed during candidate execution")
        record["engine_metrics"] = json.loads((a.out / "engine.json").read_text())
        record["encoding_metrics"] = json.loads((a.out / "encoding.json").read_text())
        record["artifact_summary"] = summarize_media(artifacts)
        record["av_verification"] = "pending explicit host verify_vdn_av.py; container has no ffprobe"
        if a.reference:
            checks = compare_media(artifacts, a.reference / "video.artifacts")
            record["checks"] = checks
            record["numeric_media_gate"] = (all(x["finite"] for x in checks.values())
                and checks["latent_video"]["relative_l2"] <= 0.10
                and checks["audio.npy"]["relative_l2"] <= 0.10 and checks["audio.npy"]["cosine"] >= 0.99)
            record["exact_media_gate"] = all(x["exact"] for x in checks.values())
        record["state"] = "generation_complete_av_pending"; save()
        print(json.dumps({k:v for k,v in record.items() if k not in ("engine_metrics", "encoding_metrics", "ffprobe")}, indent=2), flush=True)
    except BaseException as e:
        record.update(state="failed", error=repr(e)); save(); raise


if __name__ == "__main__": main()
