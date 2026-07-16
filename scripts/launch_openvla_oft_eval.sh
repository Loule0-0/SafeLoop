#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/openvla_oft/eval}"
RUN_NAME="${RUN_NAME:-openvla_oft_eval}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}.log}"
STATUS_PATH="${STATUS_PATH:-${RUN_DIR}.status}"
MODE="${MODE:-rl}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

: "${LIBERO_ROOT:?Set LIBERO_ROOT to the LIBERO checkout}"

if [[ "${MODE}" != "baseline" && "${MODE}" != "rl" ]]; then
  echo "MODE must be baseline or rl, got: ${MODE}" >&2
  exit 2
fi

read -r -a TASK_IDS <<<"${TASK_IDS:-8 6 9}"
if [[ "${#TASK_IDS[@]}" -eq 0 ]]; then
  echo "TASK_IDS must contain at least one integer task id" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/videos"
echo "running" >"${STATUS_PATH}"

command=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_pi0_safeguard_closed_loop.py"
  --project-root "${PROJECT_ROOT}"
  --libero-root "${LIBERO_ROOT}"
  --policy-backend openvla-oft
  --policy-mode websocket
  --policy-host "${POLICY_HOST:-127.0.0.1}"
  --policy-port "${POLICY_PORT:-8001}"
  --benchmark "${BENCHMARK:-libero_10}"
  --task-ids "${TASK_IDS[@]}"
  --episodes "${EPISODES:-1}"
  --init-state-start "${INIT_STATE_START:-0}"
  --seed "${SEED:-1701}"
  --max-rollout-steps "${MAX_ROLLOUT_STEPS:-520}"
  --replan-steps "${REPLAN_STEPS:-8}"
  --save-video-episodes "${SAVE_VIDEO_EPISODES:-1}"
  --video-dir "${RUN_DIR}/videos"
  --out "${RUN_DIR}/summary.json"
  --trace-out "${RUN_DIR}/trace.jsonl"
  --manual-hazard-labels
)

if [[ "${MODE}" == "baseline" ]]; then
  command+=(--mode baseline --disable-safeloop --predictor constant)
else
  : "${QWEN_MODEL:?Set QWEN_MODEL for SafeLoop evaluation}"
  : "${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT for SafeLoop evaluation}"
  : "${DECISION_CHECKPOINT:?Set DECISION_CHECKPOINT for SafeLoop evaluation}"
  command+=(
    --mode rl
    --predictor qwen-multitask
    --qwen-model-path "${QWEN_MODEL}"
    --qwen-head-path "${PREDICTOR_CHECKPOINT}"
    --qwen-device "${QWEN_DEVICE:-cuda}"
    --qwen-dtype "${QWEN_DTYPE:-bfloat16}"
    --future-window "${FUTURE_WINDOW:-100}"
    --decision-checkpoint "${DECISION_CHECKPOINT}"
    --decision-device "${DECISION_DEVICE:-cpu}"
    --decision-interval "${DECISION_INTERVAL:-20}"
    --record-probability "${RECORD_PROBABILITY:-0.42}"
    --record-max-risk-score "${RECORD_MAX_RISK_SCORE:-1.01}"
    --record-max-current-body-probability "${RECORD_MAX_CURRENT_BODY_PROBABILITY:-0.44}"
    --record-max-current-object-probability "${RECORD_MAX_CURRENT_OBJECT_PROBABILITY:-1.01}"
    --min-record-interval "${MIN_RECORD_INTERVAL:-20}"
    --rollback-cooldown "${ROLLBACK_COOLDOWN:-75}"
    --min-rollback-step "${MIN_ROLLBACK_STEP:-50}"
    --max-rollbacks-per-episode "${MAX_ROLLBACKS_PER_EPISODE:-2}"
    --stuck-window-steps "${STUCK_WINDOW_STEPS:-70}"
    --stuck-min-low-motion-steps "${STUCK_MIN_LOW_MOTION_STEPS:-50}"
    --stuck-max-step-displacement "${STUCK_MAX_STEP_DISPLACEMENT:-0.0025}"
    --stuck-max-window-displacement "${STUCK_MAX_WINDOW_DISPLACEMENT:-0.025}"
    --stuck-override-min-rollback-probability "${STUCK_OVERRIDE_MIN_ROLLBACK_PROBABILITY:-0.95}"
    --stuck-rollback-target-max-age "${STUCK_ROLLBACK_TARGET_MAX_AGE:-320}"
    --rollback-gate-current-threshold "${ROLLBACK_GATE_CURRENT_THRESHOLD:-1.01}"
    --rollback-gate-future-probability "${ROLLBACK_GATE_FUTURE_PROBABILITY:-1.01}"
    --rollback-gate-current-body-threshold "${ROLLBACK_GATE_CURRENT_BODY_THRESHOLD:-0.46}"
    --rollback-gate-future-body-probability "${ROLLBACK_GATE_FUTURE_BODY_PROBABILITY:-0.72}"
    --rollback-gate-future-object-probability "${ROLLBACK_GATE_FUTURE_OBJECT_PROBABILITY:-0.92}"
    --rollback-gate-max-current-object-probability "${ROLLBACK_GATE_MAX_CURRENT_OBJECT_PROBABILITY:-0.3}"
    --rollback-gate-future-tth "${ROLLBACK_GATE_FUTURE_TTH:-0.50}"
    --rollback-gate-future-object-tth "${ROLLBACK_GATE_FUTURE_OBJECT_TTH:-0.18}"
    --rollback-target-safe-score-threshold "${ROLLBACK_TARGET_SAFE_SCORE_THRESHOLD:-0.7}"
    --rollback-target-min-age "${ROLLBACK_TARGET_MIN_AGE:-45}"
    --rollback-target-max-age "${ROLLBACK_TARGET_MAX_AGE:-160}"
    --rollback-target-require-safe
    --auto-record-safe-anchors
    --auto-record-min-interval "${AUTO_RECORD_MIN_INTERVAL:-40}"
    --auto-record-max-risk-score "${AUTO_RECORD_MAX_RISK_SCORE:-0.72}"
    --auto-record-max-current-body-probability "${AUTO_RECORD_MAX_CURRENT_BODY_PROBABILITY:-0.44}"
    --auto-record-max-current-object-probability "${AUTO_RECORD_MAX_CURRENT_OBJECT_PROBABILITY:-0.58}"
    --rollback-mode motion-plan
    --motion-execution-mode kinematic
    --max-joint-step "${MAX_JOINT_STEP:-0.08}"
    --max-steps-per-waypoint "${MAX_STEPS_PER_WAYPOINT:-2}"
    --kinematic-substeps-per-waypoint "${KINEMATIC_SUBSTEPS_PER_WAYPOINT:-2}"
    --allow-restore-fallback
  )
  if [[ "${STUCK_FALLBACK:-1}" == "1" ]]; then
    command+=(--stuck-fallback)
  elif [[ "${STUCK_FALLBACK}" != "0" ]]; then
    echo "STUCK_FALLBACK must be 0 or 1, got: ${STUCK_FALLBACK}" >&2
    exit 2
  fi
  if [[ "${STUCK_OBJECT_OVERRIDE:-1}" == "1" ]]; then
    command+=(--rollback-gate-allow-stuck-object-override)
  elif [[ "${STUCK_OBJECT_OVERRIDE}" != "0" ]]; then
    echo "STUCK_OBJECT_OVERRIDE must be 0 or 1, got: ${STUCK_OBJECT_OVERRIDE}" >&2
    exit 2
  fi
fi

if [[ -n "${FIXED_INIT_STATE_INDEX:-}" ]]; then
  command+=(--fixed-init-state-index "${FIXED_INIT_STATE_INDEX}")
fi

command+=("$@")
if "${command[@]}" >"${LOG_PATH}" 2>&1; then
  echo "complete" >"${STATUS_PATH}"
else
  status=$?
  echo "failed:${status}" >"${STATUS_PATH}"
  exit "${status}"
fi
