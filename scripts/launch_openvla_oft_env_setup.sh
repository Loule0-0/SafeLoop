#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/outputs/openvla_oft_env_setup.log}"
STATUS_PATH="${STATUS_PATH:-${PROJECT_ROOT}/outputs/openvla_oft_env_setup.status}"

mkdir -p "$(dirname "${LOG_PATH}")"
rm -f "${STATUS_PATH}"
exec >"${LOG_PATH}" 2>&1

status=0
PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}" \
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=http.proxy \
GIT_CONFIG_VALUE_0="${GIT_HTTP_PROXY:-http://127.0.0.1:6111}" \
CONDA_SH="${CONDA_SH:-/opt/liblibai-models/user-workspace/yangying/.miniconda/etc/profile.d/conda.sh}" \
CONDA_ENV_NAME="${CONDA_ENV_NAME:-safeloop-openvla-oft}" \
bash "${PROJECT_ROOT}/scripts/setup_openvla_oft_env.sh" || status=$?

printf '%s\n' "${status}" >"${STATUS_PATH}"
exit "${status}"
