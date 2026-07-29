#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_ROOT="${LIBERO_ROOT:-${ROOT}/third_party/LIBERO}"
OPENPI_ROOT="${OPENPI_ROOT:-${ROOT}/third_party/openpi}"
SAFELOOP_WEIGHTS="${SAFELOOP_WEIGHTS:-${ROOT}/.artifacts/safeloop_weights}"
SAFELOOP_OUTPUT="${SAFELOOP_OUTPUT:-${ROOT}/outputs/quickstart}"
SAFELOOP_TASK="${SAFELOOP_TASK:-libero_10:6}"
SAFELOOP_SEED="${SAFELOOP_SEED:-389}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
POLICY_STARTUP_TIMEOUT="${POLICY_STARTUP_TIMEOUT:-600}"
SAFELOOP_START_POLICY_SERVER="${SAFELOOP_START_POLICY_SERVER:-1}"
SAFELOOP_SYNC_OPENPI="${SAFELOOP_SYNC_OPENPI:-1}"
POLICY_CUDA_VISIBLE_DEVICES="${POLICY_CUDA_VISIBLE_DEVICES:-0}"
SAFELOOP_CUDA_VISIBLE_DEVICES="${SAFELOOP_CUDA_VISIBLE_DEVICES:-0}"
POLICY_LOG="${SAFELOOP_OUTPUT}/pi0_policy_server.log"

: "${QWEN_MODEL:?Set QWEN_MODEL to Qwen2.5-VL-3B-Instruct}"
: "${PI0_CHECKPOINT_DIR:?Set PI0_CHECKPOINT_DIR to the pi0_libero checkpoint}"

export LIBERO_ROOT OPENPI_ROOT SAFELOOP_WEIGHTS SAFELOOP_OUTPUT
mkdir -p "${SAFELOOP_OUTPUT}"

python "${ROOT}/scripts/download_release_weights.py" \
  --output-dir "${SAFELOOP_WEIGHTS}"

policy_pid=""
cleanup() {
  if [[ -n "${policy_pid}" ]]; then
    kill -TERM -- "-${policy_pid}" 2>/dev/null || true
    wait "${policy_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${SAFELOOP_START_POLICY_SERVER}" == "1" ]]; then
  if [[ "${SAFELOOP_SYNC_OPENPI}" == "1" ]]; then
    (
      cd "${OPENPI_ROOT}"
      env -u VIRTUAL_ENV uv sync --frozen \
        --no-install-package lerobot \
        --no-install-package rerun-sdk \
        --no-install-package evdev \
        --no-install-package av
    )
  fi
  (
    cd "${OPENPI_ROOT}"
    exec setsid env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES="${POLICY_CUDA_VISIBLE_DEVICES}" \
      uv run --no-sync python "${ROOT}/scripts/serve_pi0_policy.py" \
        --port="${POLICY_PORT}" policy:checkpoint \
        --policy.config=pi0_libero \
        --policy.dir="${PI0_CHECKPOINT_DIR}"
  ) >"${POLICY_LOG}" 2>&1 &
  policy_pid="$!"
fi

for _ in $(seq 1 "${POLICY_STARTUP_TIMEOUT}"); do
  if python - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
import sys

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

try:
    with connect(
        f"ws://{sys.argv[1]}:{int(sys.argv[2])}",
        compression=None,
        max_size=None,
        open_timeout=0.5,
        close_timeout=0.5,
    ) as connection:
        connection.recv(timeout=0.5)
except (OSError, TimeoutError, WebSocketException):
    raise SystemExit(1)
PY
  then
    break
  fi
  if [[ -n "${policy_pid}" ]] && ! kill -0 "${policy_pid}" 2>/dev/null; then
    echo "Pi0 policy server stopped during startup. See ${POLICY_LOG}" >&2
    exit 1
  fi
  sleep 1
done

python "${ROOT}/scripts/check_release_environment.py" \
  --scope eval \
  --check-policy-server \
  --policy-host "${POLICY_HOST}" \
  --policy-port "${POLICY_PORT}"

CUDA_VISIBLE_DEVICES="${SAFELOOP_CUDA_VISIBLE_DEVICES}" \
python "${ROOT}/scripts/run_release_24task_eval.py" \
  --model-dir "${QWEN_MODEL}" \
  --weights-dir "${SAFELOOP_WEIGHTS}" \
  --output-root "${SAFELOOP_OUTPUT}/rollout" \
  --libero-root "${LIBERO_ROOT}" \
  --openpi-root "${OPENPI_ROOT}" \
  --checkpoint-dir "${PI0_CHECKPOINT_DIR}" \
  --policy-host "${POLICY_HOST}" \
  --policy-port "${POLICY_PORT}" \
  --task "${SAFELOOP_TASK}" \
  --seeds "${SAFELOOP_SEED}"

echo "SafeLoop quickstart completed: ${SAFELOOP_OUTPUT}/rollout"
