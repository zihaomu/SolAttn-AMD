#!/usr/bin/env bash
# Isolated device and writable results; never installs into the existing runtime.
set -euo pipefail
sol_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sol_image="${SOL_ROCM_IMAGE:-sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9}"
sol_render="${SOL_RENDER_DEVICE:-/dev/dri/renderD129}"
sol_uuid="${SOL_GPU_UUID:-GPU-b5ab24f2dc5d81ac}"
mkdir -p "$sol_root/results/cache"
exec docker run --rm --pull=never --read-only \
  --device=/dev/kfd --device="$sol_render" \
  --user="$(id -u):$(id -g)" \
  --group-add="$(stat -c '%g' "$sol_render")" --group-add="$(stat -c '%g' /dev/kfd)" \
  --tmpfs /tmp:rw,exec,size=1g --shm-size=1g \
  --env ROCR_VISIBLE_DEVICES="$sol_uuid" \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --env PYTHONPATH=/workspace/src --env TRITON_CACHE_DIR=/results/cache/triton \
  --env XDG_CACHE_HOME=/results/cache --env SOL_ROCM_IMAGE="$sol_image" \
  --mount "type=bind,src=$sol_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$sol_root/results,dst=/results" \
  --workdir /workspace --entrypoint python3 "$sol_image" "$@"
