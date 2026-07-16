#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-safeloop-openvla-oft}"
CONDA_ROOT="${CONDA_ROOT:-/opt/liblibai-models/user-workspace/yangying/.miniconda}"
PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/outputs/openvla_oft_server.log}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
POLICY_PORT="${POLICY_PORT:-8001}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec >"${LOG_PATH}" 2>&1
exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PYTHONUNBUFFERED=1 \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}" \
  DISABLE_ADDMM_CUDA_LT="${DISABLE_ADDMM_CUDA_LT:-1}" \
  TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-0}" \
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}" \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/serve_openvla_oft.py" \
  --checkpoint "${CHECKPOINT}" \
  --inference-dtype "${OPENVLA_INFERENCE_DTYPE:-bfloat16}" \
  --port "${POLICY_PORT}" \
  "$@"
