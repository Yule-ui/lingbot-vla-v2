#!/usr/bin/env bash
# Run inside an activated Python 3.12 conda environment on the GPU server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data-teamA/cache/pip}"
export TMPDIR="${TMPDIR:-/data-teamA/tmp}"
export HF_HOME="${HF_HOME:-/data-teamA/cache/huggingface}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${HF_HOME}"

python - <<'PY'
import os, sys
from pathlib import Path
assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12; got {sys.version}"
prefix = os.environ.get('CONDA_PREFIX')
assert prefix and Path(sys.prefix).resolve() == Path(prefix).resolve(), 'Activate the target conda environment first.'
print('Installing into', sys.executable)
PY

python -m pip install --upgrade pip 'setuptools>=71,<81' wheel
# cu126 matches the server's existing nvcc toolkit, including source builds.
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r requirements-depth.txt \
  -c tools/constraints-cu126.txt
python -m pip install -c requirements.txt -c tools/constraints-cu126.txt \
  draccus==0.10.0 gymnasium==1.1.1 deepdiff==8.6.1 \
  termcolor==2.5.0 imageio-ffmpeg==0.6.0
python -m pip install numpydantic==1.9.0 --no-deps

python - <<'PY'
import torch
print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda)
assert torch.__version__.split('+')[0] == '2.8.0'
assert torch.cuda.is_available(), 'CUDA unavailable: run in the GPU-enabled server session.'
assert torch.cuda.device_count() == 4, f'Expected 4 visible GPUs, got {torch.cuda.device_count()}'
PY

if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
  test -f "${FLASH_ATTN_WHEEL}"
  python -m pip install --no-deps "${FLASH_ATTN_WHEEL}"
else
  python -m pip install --no-build-isolation --no-deps flash-attn==2.8.3
fi

# Preserve LingBot's core stack: LeRobot/MDM metadata requires other versions.
python -m pip install --no-deps \
  'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz'
python -m pip install --no-deps \
  'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
python -m pip install --no-deps -e . \
  -e lingbotvla/models/vla/vision_models/lingbot-depth \
  -e lingbotvla/models/vla/vision_models/MoGe

python - <<'PY'
import importlib.metadata as metadata
import torch
import av, cv2, pyarrow, transformers, flash_attn, utils3d
from torchcodec.decoders import VideoDecoder
from moge.model.v2 import MoGeModel
from mdm.model.v2 import MDMModel
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lingbotvla.data.vla_data.base_dataset import VLADataset

for name, expected in {
    'torch': '2.8.0', 'torchvision': '0.23.0', 'torchcodec': '0.6.0',
    'transformers': '4.57.3', 'huggingface-hub': '0.34.3',
    'numpy': '1.26.4', 'pyarrow': '21.0.0',
    'flash-attn': '2.8.3', 'lerobot': '0.4.2',
}.items():
    actual = metadata.version(name)
    print(name, actual)
    assert actual.split('+')[0] == expected, (name, actual, expected)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
for index in range(4):
    with torch.cuda.device(index):
        value = torch.ones(8, device='cuda')
        assert value.sum().item() == 8
        print('GPU', index, torch.cuda.get_device_name(index))
dataset = LeRobotDatasetMetadata('ur3_eef_relative', root='/data-teamA/datasets/ur3_eef_relative')
assert dataset.total_episodes == 90 and dataset.total_frames == 43141
video = VideoDecoder('/data-teamA/datasets/ur3_eef_relative/videos/observation.images.camera_global/chunk-000/file-000.mp4')
print('torchcodec video first frame', tuple(video[0].shape))
print('Dataset metadata, depth model imports and GPU checks passed.')
PY

mkdir -p /data-teamA/env-records
python -m pip freeze > /data-teamA/env-records/lingbotvla-pip-freeze.txt
conda list --explicit > /data-teamA/env-records/lingbotvla-conda-explicit.txt
echo "UR3 training dependencies installed and basic runtime checks passed."
