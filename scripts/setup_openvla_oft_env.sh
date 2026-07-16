#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-safeloop-openvla-oft}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Set CONDA_SH to the path of conda.sh." >&2
  exit 1
fi

source "${CONDA_SH}"
if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  conda create -n "${CONDA_ENV_NAME}" python=3.10.14 -y
fi
conda activate "${CONDA_ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -e "${PROJECT_ROOT}/third_party/openvla-oft"
python -m pip install "websockets>=14,<16" "msgpack>=1,<2" "hf-transfer>=0.1.6,<0.2"
python -m pip install \
  "setuptools==70.0.0" \
  "protobuf==3.20.3" \
  "tensorflow-metadata==1.14.0" \
  "wandb==0.16.3"

python - <<'PY'
import torch
import transformers
import websockets
import tensorflow_datasets
import wandb
import hf_transfer

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("websockets", websockets.__version__)
print("tensorflow_datasets", tensorflow_datasets.__version__)
print("wandb", wandb.__version__)
print("hf_transfer", hf_transfer.__version__)
print("cuda", torch.cuda.is_available())
PY
