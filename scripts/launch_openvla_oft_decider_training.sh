#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/release/openvla_oft_decider_training.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/openvla_oft}"
RUN_NAME="${RUN_NAME:-openvla_oft_decider}"
LOG_PATH="${LOG_PATH:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
STATUS_PATH="${STATUS_PATH:-${OUTPUT_ROOT}/${RUN_NAME}.status}"
UPDATES_VALUE="${UPDATES:-8}"
UPDATE_START_INDEX_VALUE="${UPDATE_START_INDEX:-0}"
TOTAL_SCHEDULE_UPDATES_VALUE="${TOTAL_SCHEDULE_UPDATES:-$((UPDATE_START_INDEX_VALUE + UPDATES_VALUE))}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

: "${QWEN_MODEL:?Set QWEN_MODEL to the Qwen2.5-VL checkpoint directory}"
: "${SAFELOOP_WEIGHTS:?Set SAFELOOP_WEIGHTS to the released SafeLoop weights directory}"
: "${LIBERO_ROOT:?Set LIBERO_ROOT to the LIBERO checkout}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/decision_heads/${RUN_NAME}"
echo "running" >"${STATUS_PATH}"

command=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_release_decider_training.py"
  --config "${CONFIG}"
  --model-dir "${QWEN_MODEL}"
  --weights-dir "${SAFELOOP_WEIGHTS}"
  --output-root "${OUTPUT_ROOT}"
  --libero-root "${LIBERO_ROOT}"
  --policy-host "${POLICY_HOST:-127.0.0.1}"
  --policy-port "${POLICY_PORT:-8001}"
  --set "seed=${TRAIN_SEED:-1701}"
  --set "init-state-start=${INIT_STATE_START:-0}"
  --set "qwen-dtype=${QWEN_DTYPE:-bfloat16}"
  --set "updates=${UPDATES_VALUE}"
  --set "update-start-index=${UPDATE_START_INDEX_VALUE}"
  --set "total-schedule-updates=${TOTAL_SCHEDULE_UPDATES_VALUE}"
  --set "task-ids=${TASK_IDS_JSON:-[0,1,2,3,4,5,6,7,8,9]}"
  --set "max-rollout-steps=${MAX_ROLLOUT_STEPS:-520}"
  --set "object-hazard-penalty=${OBJECT_HAZARD_PENALTY:--0.18}"
  --set "out-dir=${OUTPUT_ROOT}/decision_heads/${RUN_NAME}"
)

if [[ -n "${INITIAL_DECISION_CHECKPOINT:-}" ]]; then
  command+=(--set "initial-decision-checkpoint=${INITIAL_DECISION_CHECKPOINT}")
fi

if "${command[@]}" >"${LOG_PATH}" 2>&1; then
  echo "complete" >"${STATUS_PATH}"
else
  status=$?
  echo "failed:${status}" >"${STATUS_PATH}"
  exit "${status}"
fi
