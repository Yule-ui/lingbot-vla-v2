#!/usr/bin/env bash
set -euo pipefail
UR3_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${UR3_REPO_ROOT}"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo 'Activate the lingbotvla conda environment first.' >&2
  exit 1
fi
UR3_STAGE="${1:?Usage: bash tools/start_ur3.sh norm|check|preflight|smoke|train [CLI overrides]}"
shift
export PYTHONNOUSERSITE=1
export HF_HOME=/data-teamA/cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/data-teamA/tmp
export TRITON_CACHE_DIR=/data-teamA/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data-teamA/cache/torchinductor
export CUDA_CACHE_PATH=/data-teamA/cache/cuda
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=1
mkdir -p /data-teamA/outputs/ur3/logs "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"
UR3_STAGE_LOG="/data-teamA/outputs/ur3/logs/${UR3_STAGE}-$(date -u +%Y%m%dT%H%M%S).log"
case "${UR3_STAGE}" in
  norm) UR3_COMMAND=(python tools/run_ur3.py --stage norm configs/vla/ur3/norm.yaml) ;;
  check) UR3_COMMAND=(python tools/check_ur3_data.py) ;;
  preflight) UR3_COMMAND=(python -m torch.distributed.run --standalone --nproc_per_node=4 tools/run_ur3.py --stage train --preflight-only configs/vla/ur3/smoke.yaml) ;;
  smoke) UR3_COMMAND=(python -m torch.distributed.run --standalone --nproc_per_node=4 tools/run_ur3.py --stage train configs/vla/ur3/smoke.yaml) ;;
  train) UR3_COMMAND=(python -m torch.distributed.run --standalone --nproc_per_node=4 tools/run_ur3.py --stage train configs/vla/ur3/train.yaml) ;;
  *) echo "Unknown stage: ${UR3_STAGE}" >&2; exit 1 ;;
esac
"${UR3_COMMAND[@]}" "$@" 2>&1 | tee "${UR3_STAGE_LOG}"
echo "Log: ${UR3_STAGE_LOG}"
