"""Compact public receipts: provenance/measurements only, never activations."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path):
    raw = json.loads((path / "validation.json").read_text())
    av = json.loads((path / "host-av-verification.json").read_text())
    engine = json.loads((path / "engine.json").read_text())
    if not (engine["success"] and av["av_shape_gate"] and av["av_duration_gate"] and av["full_av_decode_pass"]):
        raise RuntimeError("Generation/AV verification incomplete: " + str(path))
    names = ("gpu_uuid", "gpu", "torch", "hip", "source_commit", "request_sha256",
             "encode_request_sha256", "script_sha256", "composition_sha256", "sol_source_sha256",
             "beta", "base_beta", "base_SOL_enabled", "phases", "adapters", "gates",
             "perceptual_quality_gate", "encoder_process_wall_seconds", "video_worker_wall_seconds",
             "full_request_wall_seconds", "conditioning_equal_reference", "frozen_source_unchanged",
             "sol_source_unchanged", "checks", "numeric_media_gate", "exact_media_gate", "artifact_summary")
    row = {k:raw[k] for k in names if k in raw}
    row.update(name=path.name, generation_and_av_validated=True,
               original_receipt_state=raw["state"], original_receipt_error=raw.get("error"),
               validation_receipt_sha256=digest(path / "validation.json"),
               av_receipt_sha256=digest(path / "host-av-verification.json"),
               mp4_sha256=av["mp4_sha256"], av_shape_gate=True, av_duration_gate=True, full_av_decode_pass=True)
    row["engine_phase_seconds"] = {k:engine[k] for k in (
        "load_seconds", "sample_seconds", "vae_load_seconds", "video_decode_seconds",
        "audio_load_decode_seconds", "decode_save_seconds", "work_seconds") if k in engine}
    recheck = path / "fp64-latent-verification.json"
    if recheck.is_file():
        row["fp64_latent_verification"] = json.loads(recheck.read_text())
        row["fp64_latent_receipt_sha256"] = digest(recheck)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    baselines = [read_run(x) for x in a.baseline]; candidate = read_run(a.candidate)
    if any(x["gpu_uuid"] != candidate["gpu_uuid"] or x["request_sha256"] != candidate["request_sha256"]
           or x["composition_sha256"] != candidate["composition_sha256"] for x in baselines):
        raise RuntimeError("Mismatched hardware/request/control composition")
    if not candidate["conditioning_equal_reference"] or not candidate["sol_source_unchanged"]:
        raise RuntimeError("Invalid conditioning/source provenance")
    seconds = [r["full_request_wall_seconds"] for r in baselines]
    workers = [r["video_worker_wall_seconds"] for r in baselines]
    median = statistics.median(seconds)
    r = dict(state="complete", scope="diagnostic baseline/candidate/baseline bracket on one GPU; two baseline trials and ONE candidate, not a statistically established speedup or perceptual certification; parent launcher startup and post-generation validation excluded",
             workload=dict(first=[672,384,8], second=[1344,768,3], learned_latent_upscale=True,
                           frames=243, fps=24, seconds=10.125, seed=2026090901,
                           refinement_seed=2026090902, audio_policy="exact first pass retained"),
             baselines=baselines, candidate=candidate,
             comparison=dict(baseline_full_samples_seconds=seconds, baseline_full_median_seconds=median,
                             baseline_full_range_seconds=max(seconds)-min(seconds),
                             candidate_full_seconds=candidate["full_request_wall_seconds"],
                             saved_seconds=median-candidate["full_request_wall_seconds"],
                             time_reduction_percent=100*(1-candidate["full_request_wall_seconds"]/median),
                             speedup=median/candidate["full_request_wall_seconds"],
                             baseline_worker_samples_seconds=workers,
                             candidate_worker_seconds=candidate["video_worker_wall_seconds"]),
             conclusions=dict(production_enabled=False, dense_equivalence=False,
                              multi_prompt_quality_gate_passed=False,
                              original_Triton_and_FreeVideo_sources_edited=False))
    a.out.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["comparison"], indent=2))


if __name__ == "__main__": main()
