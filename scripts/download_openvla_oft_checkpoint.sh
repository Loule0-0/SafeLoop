#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-safeloop-openvla-oft}"
CONDA_ROOT="${CONDA_ROOT:-/opt/liblibai-models/user-workspace/yangying/.miniconda}"
HF_CLI="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/huggingface-cli"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
REVISION="${REVISION:-638918f3d1c2e43a39a8a20772bdb8b91835e4b7}"

export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

"${HF_CLI}" download "${CHECKPOINT}" \
  --revision "${REVISION}" \
  --resume-download \
  --max-workers "${MAX_WORKERS:-8}"
