"""Host-side full AV decode/shape gate; independent of GPU timing/container tools."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    video = a.run / "video.mp4"
    metrics = json.loads((a.run / "engine.json").read_text())
    if not metrics["success"]: raise RuntimeError("Generation failed")
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(video)], check=True, capture_output=True, text=True)
    data = json.loads(probe.stdout)
    v = [s for s in data["streams"] if s["codec_type"] == "video"]
    au = [s for s in data["streams"] if s["codec_type"] == "audio"]
    shape = (len(v) == len(au) == 1 and (v[0]["width"], v[0]["height"], int(v[0]["nb_read_frames"]), v[0]["avg_frame_rate"]) == (1344, 768, 243, "24/1"))
    sync = bool(shape and abs(float(v[0]["duration"]) - float(au[0]["duration"])) < 0.10)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    r = dict(state="complete", generation_success=True, av_shape_gate=shape, av_duration_gate=sync,
             full_av_decode_pass=True, mp4_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), ffprobe=data,
             scope="post-generation host verification, excluded from generation wall timing")
    a.out.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k:v for k,v in r.items() if k != "ffprobe"}, indent=2))
    if not (shape and sync): raise RuntimeError("AV geometry/duration gate failed")


if __name__ == "__main__": main()
