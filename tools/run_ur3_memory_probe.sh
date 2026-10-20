#!/usr/bin/env bash
set -euo pipefail
cd /home/teamA/lingbot-vla-v2
source /data-teamA/miniforge3/etc/profile.d/conda.sh
conda activate lingbotvla
export PYTHONNOUSERSITE=1
export HF_HOME=/data-teamA/cache/huggingface
export HF_DATASETS_CACHE=/data-teamA/cache/huggingface/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/data-teamA/tmp
export TRITON_CACHE_DIR=/data-teamA/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data-teamA/cache/torchinductor
export CUDA_CACHE_PATH=/data-teamA/cache/cuda
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
UR3_PROBE_ARGS=("$@")
UR3_PROBE_OUTPUT=''
for ((UR3_PROBE_INDEX=0; UR3_PROBE_INDEX<${#UR3_PROBE_ARGS[@]}; UR3_PROBE_INDEX++)); do
  if [[ "${UR3_PROBE_ARGS[UR3_PROBE_INDEX]}" == '--output-dir' ]]; then
    UR3_PROBE_OUTPUT="${UR3_PROBE_ARGS[UR3_PROBE_INDEX+1]}"
  fi
done
if [[ -z "${UR3_PROBE_OUTPUT}" ]]; then
  echo '--output-dir is required' >&2
  exit 2
fi
if [[ -e "${UR3_PROBE_OUTPUT}/memory_rank0.json" || -e "${UR3_PROBE_OUTPUT}/train.log" ]]; then
  echo 'Choose a new output directory to preserve existing measurement results.' >&2
  exit 2
fi
mkdir -p "${UR3_PROBE_OUTPUT}"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "${UR3_PROBE_OUTPUT}/nvidia-smi.csv" &
UR3_PROBE_MONITOR_PID=$!
trap 'kill -TERM "${UR3_PROBE_MONITOR_PID}" 2>/dev/null || true' EXIT
timeout 1200 python -m torch.distributed.run --standalone --nproc_per_node=4 tools/measure_ur3_memory.py "$@" > "${UR3_PROBE_OUTPUT}/train.log" 2>&1
echo "Memory measurement reports: ${UR3_PROBE_OUTPUT}"
