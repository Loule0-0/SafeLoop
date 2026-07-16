#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/outputs/openvla_oft_download.log}"
STATUS_PATH="${STATUS_PATH:-${PROJECT_ROOT}/outputs/openvla_oft_download.status}"

mkdir -p "$(dirname "${LOG_PATH}")"
echo "running" >"${STATUS_PATH}"
if bash "${PROJECT_ROOT}/scripts/download_openvla_oft_checkpoint.sh" >"${LOG_PATH}" 2>&1; then
  echo "complete" >"${STATUS_PATH}"
else
  status=$?
  echo "failed:${status}" >"${STATUS_PATH}"
  exit "${status}"
fi
