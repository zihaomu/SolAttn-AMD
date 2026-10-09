#!/usr/bin/env bash
# Explicit research runtime: no install and no source/default changes to FreeVideo.
set -euo pipefail
sol_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sol_storage="${SOL_VDN_STORAGE:-/dc1/zihaomu/free_token_mapping}"
sol_frozen="$sol_storage/experiments/freevideo-r9700/optimization-tasks/official-8plus3-asmevo-20261007/shared/source"
sol_exp=/data/experiments/freevideo-r9700
sol_render="${SOL_RENDER_DEVICE:-/dev/dri/renderD129}"
sol_uuid="${SOL_GPU_UUID:-GPU-b5ab24f2dc5d81ac}"
sol_image="${SOL_ROCM_IMAGE:-sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9}"
mkdir -p "$sol_root/results/cache"
exec docker run --rm --pull=never --read-only --device=/dev/kfd --device="$sol_render" \
  --user="$(id -u):$(id -g)" --group-add="$(stat -c '%g' "$sol_render")" --group-add="$(stat -c '%g' /dev/kfd)" \
  --tmpfs /tmp:rw,exec,size=2g --shm-size=2g \
  --env ROCR_VISIBLE_DEVICES="$sol_uuid" --env SOL_GPU_UUID="$sol_uuid" \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env OMP_NUM_THREADS=8 --env MKL_NUM_THREADS=8 \
  --env TORCH_BLAS_PREFER_HIPBLASLT=0 --env ROCBLAS_USE_HIPBLASLT=0 \
  --env FREEVIDEO_ROCM_VIDEO_BLAS=cublaslt --env FREEVIDEO_ROCM_SPATIAL_CONV=triton \
  --env FREEVIDEO_ROCM_ATTENTION=triton-window --env FREEVIDEO_ROCM_AUDIO_CONV=native \
  --env FREEVIDEO_ROCM_GATE_BLAS=default --env FREEVIDEO_ROCM_STATE_BLAS=default \
  --env FV_STORAGE_ROOT=/data --env FREEVIDEO_HOME="$sol_exp/runtime" \
  --env FREEVIDEO_VDN_ROOT="$sol_exp/vendor/vdn" --env FREEVIDEO_COMFY_ROOT="$sol_exp/vendor/h3-text-encoder" \
  --env FREEVIDEO_MODEL_ROOT=/data/models/vdn --env HF_HOME=/data/models/.hf-cache \
  --env TRITON_CACHE_DIR="$sol_exp/kernel-cache/asmevo-20261007/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$sol_exp/kernel-cache/asmevo-20261007/inductor" \
  --env MIOPEN_USER_DB_PATH="$sol_exp/cache/miopen" \
  --env MIOPEN_CUSTOM_CACHE_DIR="$sol_exp/kernel-cache/asmevo-20261007/miopen" \
  --env XDG_CACHE_HOME="$sol_exp/cache" --env XDG_CONFIG_HOME="$sol_exp/cache/config" \
  --env PYTHONPATH=/sol/src:/sol:/workspace \
  --mount "type=bind,src=$sol_frozen,dst=/workspace,readonly" \
  --mount "type=bind,src=$sol_root,dst=/sol,readonly" \
  --mount "type=bind,src=$sol_root/results,dst=/results" \
  --mount "type=bind,src=$sol_storage,dst=/data" --mount "type=bind,src=$sol_storage,dst=$sol_storage" \
  --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly --mount type=bind,src=/etc/group,dst=/etc/group,readonly \
  --workdir /sol --entrypoint "$sol_exp/envs/rocm/bin/python" "$sol_image" "$@"
