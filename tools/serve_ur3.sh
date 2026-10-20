#!/usr/bin/env bash
set -euo pipefail

UR3_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${UR3_REPO_ROOT}"

source /data-teamA/miniforge3/etc/profile.d/conda.sh
conda activate lingbotvla

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-/data-teamA/cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN3VL_PATH="${QWEN3VL_PATH:-/data-teamA/models/Qwen3-VL-4B-Instruct}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python -u deploy/lingbot_vla_v2_policy.py \
  --model_path "${UR3_MODEL_PATH:-/data-teamA/outputs/ur3_20261016/train_two_views_640_last4_seed42/checkpoints/global_step_5000/hf_ckpt}" \
  --host "${UR3_HOST:-127.0.0.1}" \
  --port "${UR3_PORT:-8006}" \
  --use_bf16 true --use_fp32 false --use_compile false \
  --chunk_ret true --use_length 50 \
  "$@"
