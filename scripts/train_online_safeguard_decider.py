from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard.controller import (
    ProprioceptiveStuckMonitor,
    apply_stuck_fallback,
    capture_env_state,
    execute_rollback,
    extract_robot_joint_vector,
    normalize_risk_prediction,
)
from safety_guard.decider import Intervention, RuleBasedDecider
from safety_guard.libero_motion import MotionPlanningRollbackExecutor
from safety_guard.libero_oracle import LiberoHazardOracle
from safety_guard.memory import WaypointMemory, _waypoint_risk_score, initial_anchor_rollback_allowed
from safety_guard.online_rl import (
    AsymmetricPPOConfig,
    AsymmetricPPOTrainer,
    OnlineRewardConfig,
    OnlineStepSignals,
    actor_checkpoint_to_asymmetric,
    batch_from_online_records,
    export_actor_checkpoint,
    mark_online_episode_terminal,
    online_hazard_score,
    online_safeloop_reward,
    noop_outcome_credit,
    privileged_feature_dim,
    rollback_outcome_credit,
    rollback_terminal_credit,
    sample_action_with_intervention_exploration,
)
from safety_guard.predictors import ActionNormRiskPredictor, ConstantRiskPredictor
from safety_guard.qwen_multitask import (
    QWEN_IMAGE_TOKEN,
    QwenMultitaskSafetyPredictor,
    build_live_qwen_prompt,
    live_observation_to_frame,
)
from safety_guard.rl_decider import ThreeAction
from safety_guard.rl_decision_data import build_action_mask, build_decision_features
from safety_guard.rl_policy_decider import is_high_confidence_rollback
from safety_guard.rollout_sampling import append_prehazard_rollout_samples

from scripts.libero_policy_utils import (
    LIBERO_DUMMY_ACTION,
    configure_paths,
    env_path,
    make_env,
    make_policy,
    max_steps_for_suite,
    policy_observation,
)
from scripts.evaluate_pi0_safeguard_closed_loop import make_rollback_executor
from scripts.evaluate_pi0_safeguard_closed_loop import write_qwen_rollout_samples


ACTION_TO_INTERVENTION = {
    ThreeAction.NOOP: Intervention.NOOP,
    ThreeAction.RECORD: Intervention.RECORD,
    ThreeAction.ROLLBACK: Intervention.ROLLBACK,
}
INTERVENTION_TO_ACTION = {value: key for key, value in ACTION_TO_INTERVENTION.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online asymmetric PPO training for SafeLoop's three-action decider.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--openpi-root", type=Path, default=env_path("OPENPI_ROOT"))
    parser.add_argument("--libero-root", type=Path, default=env_path("LIBERO_ROOT"))
    parser.add_argument("--checkpoint-dir", type=Path, default=env_path("PI0_CHECKPOINT_DIR"))
    parser.add_argument("--config-name", default="pi0_libero")
    parser.add_argument("--policy-backend", choices=["pi0"], default="pi0")
    parser.add_argument("--policy-mode", choices=["inprocess", "websocket"], default="inprocess")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--benchmark", default="libero_10", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"])
    parser.add_argument("--task-ids", nargs="+", default=[0])
    parser.add_argument("--init-state-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-rollout-steps", "--max-steps", dest="max_rollout_steps", type=int, default=120)
    parser.add_argument("--allow-extended-rollout", action="store_true")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--predictor", choices=["constant", "action-norm", "qwen-multitask"], default="action-norm")
    parser.add_argument("--constant-risk", nargs=4, type=float, default=[0.0, 1.0, 0.0, 1.0])
    parser.add_argument("--qwen-model-dir", "--qwen-model-path", dest="qwen_model_dir", type=Path, default=env_path("QWEN_MODEL"))
    parser.add_argument("--qwen-checkpoint", "--qwen-head-path", dest="qwen_checkpoint", type=Path)
    parser.add_argument("--qwen-lora-adapter", type=Path)
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-dtype", default="float32")
    parser.add_argument("--qwen-tau", "--future-window", dest="qwen_tau", type=int, default=50)
    parser.add_argument("--initial-decision-checkpoint", "--init-checkpoint", dest="initial_decision_checkpoint", type=Path, required=True)
    parser.add_argument("--initial-actor-logit-scale", type=float, default=1.0)
    parser.add_argument("--decision-device", default="cpu")
    parser.add_argument("--decision-period", "--decision-interval", dest="decision_period", type=int, default=5)
    parser.add_argument("--record-probability", type=float, default=0.12)
    parser.add_argument("--record-max-risk-score", type=float, default=1.01)
    parser.add_argument("--record-max-current-body-probability", type=float)
    parser.add_argument("--record-max-current-object-probability", type=float)
    parser.add_argument("--record-initial-safe-anchor", action="store_true")
    parser.add_argument("--initial-anchor-rollback-min-current-body-probability", type=float)
    parser.add_argument("--initial-anchor-rollback-min-current-object-probability", type=float)
    parser.add_argument("--rollback-probability", type=float, default=0.55)
    parser.add_argument("--rollback-tth", type=float, default=0.35)
    parser.add_argument("--min-record-interval", type=int, default=30)
    parser.add_argument("--rollback-cooldown", type=int, default=15)
    parser.add_argument("--rollback-exploration-probability", type=float, default=0.0)
    parser.add_argument("--noop-exploration-probability", type=float, default=0.0)
    parser.add_argument("--noop-probe-blocks-rollback", action="store_true")
    parser.add_argument("--record-exploration-probability", type=float, default=0.0)
    parser.add_argument("--min-rollback-step", type=int, default=0)
    parser.add_argument("--max-rollbacks-per-episode", type=int)
    parser.add_argument("--stuck-fallback", action="store_true")
    parser.add_argument("--stuck-window-steps", type=int, default=70)
    parser.add_argument("--stuck-min-low-motion-steps", type=int, default=50)
    parser.add_argument("--stuck-max-step-displacement", type=float, default=0.0025)
    parser.add_argument("--stuck-max-window-displacement", type=float, default=0.025)
    parser.add_argument("--rollback-gate-allow-stuck-object-override", action="store_true")
    parser.add_argument("--stuck-rollback-target-max-age", type=int)
    parser.add_argument("--rollback-gate-current-threshold", type=float, default=1.01)
    parser.add_argument("--rollback-gate-current-body-threshold", type=float)
    parser.add_argument("--rollback-gate-current-object-threshold", type=float)
    parser.add_argument("--rollback-gate-max-current-object-probability", type=float)
    parser.add_argument("--rollback-gate-future-probability", type=float, default=1.01)
    parser.add_argument("--rollback-gate-future-body-probability", type=float)
    parser.add_argument("--rollback-gate-future-object-probability", type=float)
    parser.add_argument("--rollback-gate-future-tth", type=float, default=0.0)
    parser.add_argument("--rollback-gate-future-object-tth", type=float)
    parser.add_argument("--rollback-mode", choices=["motion-plan", "restore"], default="motion-plan")
    parser.add_argument("--motion-execution-mode", choices=["pd", "kinematic"], default="kinematic")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-steps-per-waypoint", type=int, default=120)
    parser.add_argument("--kinematic-substeps-per-waypoint", type=int, default=2)
    parser.add_argument("--allow-restore-fallback", action="store_true")
    parser.add_argument("--render-camera", default="agentview")
    parser.add_argument("--rollback-target-safe-score-threshold", type=float, default=0.6)
    parser.add_argument("--rollback-target-min-age", type=int, default=30)
    parser.add_argument("--rollback-target-max-age", type=int, default=120)
    parser.add_argument("--rollback-target-prefer-recent-safe", action="store_true")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--update-start-index", type=int, default=0)
    parser.add_argument("--total-schedule-updates", type=int)
    parser.add_argument("--episodes-per-update", "--episodes-per-task", dest="episodes_per_update", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--bc-coef", type=float, default=1.0)
    parser.add_argument("--class-balanced-bc", action="store_true")
    parser.add_argument("--bc-anneal-fraction", type=float, default=0.3)
    parser.add_argument("--hazard-penalty", type=float, default=-1.0)
    parser.add_argument("--hazard-onset-penalty", type=float, default=0.0)
    parser.add_argument("--predictor-onset-penalty", type=float, default=0.0)
    parser.add_argument("--predictor-realized-body-threshold", type=float, default=0.99)
    parser.add_argument("--predictor-realized-object-threshold", type=float, default=0.99)
    parser.add_argument("--body-hazard-penalty", type=float, default=0.0)
    parser.add_argument("--object-hazard-penalty", type=float, default=0.0)
    parser.add_argument("--stuck-hazard-penalty", type=float, default=0.0)
    parser.add_argument("--rollback-penalty", type=float, default=-0.005)
    parser.add_argument("--rollback-failure-penalty", type=float, default=-0.25)
    parser.add_argument("--rollback-rendered-frame-penalty", type=float, default=0.0)
    parser.add_argument("--rollback-resolved-bonus", type=float, default=0.0)
    parser.add_argument("--rollback-preemptive-bonus", type=float, default=0.0)
    parser.add_argument("--rollback-unresolved-penalty", type=float, default=0.0)
    parser.add_argument("--rollback-episode-success-bonus", type=float, default=0.0)
    parser.add_argument("--rollback-episode-failure-penalty", type=float, default=0.0)
    parser.add_argument("--rollback-effect-horizon", type=int, default=0)
    parser.add_argument("--noop-effect-horizon", type=int, default=0)
    parser.add_argument("--noop-safe-bonus", type=float, default=0.0)
    parser.add_argument("--noop-hazard-penalty", type=float, default=0.0)
    parser.add_argument("--rollback-count-scale", type=float, default=0.25)
    parser.add_argument("--record-penalty", type=float, default=-0.0005)
    parser.add_argument("--step-penalty", type=float, default=-1e-5)
    parser.add_argument("--completion-reward", type=float, default=1.0)
    parser.add_argument("--task-reward-scale", type=float, default=0.0)
    parser.add_argument("--qwen-rollout-jsonl-out", type=Path)
    parser.add_argument("--qwen-rollout-image-root", type=Path)
    parser.add_argument("--qwen-rollout-stride", type=int, default=40)
    parser.add_argument("--qwen-rollout-prehazard-stride", type=int, default=0)
    parser.add_argument("--qwen-rollout-run-tag")
    parser.add_argument("--decision-debug-jsonl-out", type=Path)
    parser.add_argument("--save-online-records", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.task_ids = _parse_task_ids(args.task_ids)
    return args


def _parse_task_ids(values) -> list[int]:
    task_ids: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                task_ids.append(int(part))
    if not task_ids:
        raise ValueError("--task-ids must contain at least one id")
    return task_ids


def make_predictor(args: argparse.Namespace):
    if args.predictor == "constant":
        return ConstantRiskPredictor(args.constant_risk)
    if args.predictor == "action-norm":
        return ActionNormRiskPredictor()
    if not args.qwen_checkpoint:
        raise ValueError("--qwen-checkpoint is required for qwen-multitask predictor")
    if args.qwen_model_dir is None:
        raise ValueError("--qwen-model-dir or QWEN_MODEL is required for qwen-multitask predictor")
    return QwenMultitaskSafetyPredictor(
        model_dir=args.qwen_model_dir,
        checkpoint_path=args.qwen_checkpoint,
        lora_adapter_path=args.qwen_lora_adapter,
        device=args.qwen_device,
        torch_dtype=args.qwen_dtype,
        history_length=3,
        tau=args.qwen_tau,
    )


def cool_down_elapsed(step_index: int, previous_step: int | None, interval: int) -> bool:
    if previous_step is None:
        return True
    return int(step_index) - int(previous_step) >= int(interval)


def reward_config_from_args(args: argparse.Namespace) -> OnlineRewardConfig:
    return OnlineRewardConfig(
        hazard_penalty=args.hazard_penalty,
        hazard_onset_penalty=args.hazard_onset_penalty,
        body_hazard_penalty=args.body_hazard_penalty,
        object_hazard_penalty=args.object_hazard_penalty,
        stuck_hazard_penalty=args.stuck_hazard_penalty,
        rollback_penalty=args.rollback_penalty,
        rollback_failure_penalty=args.rollback_failure_penalty,
        rollback_rendered_frame_penalty=args.rollback_rendered_frame_penalty,
        rollback_resolved_bonus=args.rollback_resolved_bonus,
        rollback_preemptive_bonus=args.rollback_preemptive_bonus,
        rollback_unresolved_penalty=args.rollback_unresolved_penalty,
        rollback_episode_success_bonus=args.rollback_episode_success_bonus,
        rollback_episode_failure_penalty=args.rollback_episode_failure_penalty,
        noop_safe_bonus=args.noop_safe_bonus,
        noop_hazard_penalty=args.noop_hazard_penalty,
        rollback_count_scale=args.rollback_count_scale,
        record_penalty=args.record_penalty,
        step_penalty=args.step_penalty,
        completion_reward=args.completion_reward,
        task_reward_scale=args.task_reward_scale,
    )


def augment_training_signals(
    signals: OnlineStepSignals,
    *,
    monitor_stuck: bool,
    previous_any_hazard: bool,
) -> OnlineStepSignals:
    stuck_hazard = bool(signals.stuck_hazard or monitor_stuck)
    any_hazard = bool(signals.body_hazard or signals.object_hazard or stuck_hazard)
    return dataclasses.replace(
        signals,
        stuck_hazard=stuck_hazard,
        hazard_event=bool(any_hazard and not previous_any_hazard),
    )


def scale_actor_logits(decision_policy, scale: float) -> None:
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("--initial-actor-logit-scale must be positive")
    if scale == 1.0:
        return
    with torch.no_grad():
        decision_policy.actor.weight.mul_(scale)
        decision_policy.actor.bias.mul_(scale)


def predictor_proxy_hazard(
    current_body: float,
    current_object: float,
    stuck_detected: bool,
    *,
    body_threshold: float = 0.99,
    object_threshold: float = 0.99,
) -> bool:
    return bool(
        stuck_detected
        or float(current_body) >= float(body_threshold)
        or float(current_object) >= float(object_threshold)
    )


def rollback_gate_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.rollback_gate_current_threshold <= 1.0
        or (
            args.rollback_gate_current_body_threshold is not None
            and args.rollback_gate_current_body_threshold <= 1.0
        )
        or (
            args.rollback_gate_current_object_threshold is not None
            and args.rollback_gate_current_object_threshold <= 1.0
        )
        or args.rollback_gate_future_probability <= 1.0
        or (
            args.rollback_gate_future_body_probability is not None
            and args.rollback_gate_future_body_probability <= 1.0
        )
        or (
            args.rollback_gate_future_object_probability is not None
            and args.rollback_gate_future_object_probability <= 1.0
        )
    )


def rollback_allowed_by_gate(
    args: argparse.Namespace,
    risk,
    current_body: float,
    current_object: float,
    rollback_count: int,
    step_index: int,
    stuck_detected: bool = False,
) -> bool:
    if int(step_index) < int(args.min_rollback_step):
        return False
    if args.max_rollbacks_per_episode is not None and rollback_count >= args.max_rollbacks_per_episode:
        return False
    if not rollback_gate_enabled(args):
        return True
    max_current_object_probability = args.rollback_gate_max_current_object_probability
    if bool(getattr(args, "rollback_gate_allow_stuck_object_override", False)) and bool(stuck_detected):
        max_current_object_probability = None
    return is_high_confidence_rollback(
        risk,
        current_body_probability=current_body,
        current_object_probability=current_object,
        current_hazard_threshold=args.rollback_gate_current_threshold,
        current_body_threshold=args.rollback_gate_current_body_threshold,
        current_object_threshold=args.rollback_gate_current_object_threshold,
        max_current_object_probability=max_current_object_probability,
        future_probability_threshold=args.rollback_gate_future_probability,
        future_body_probability_threshold=args.rollback_gate_future_body_probability,
        future_object_probability_threshold=args.rollback_gate_future_object_probability,
        future_tth_threshold=args.rollback_gate_future_tth,
        future_object_tth_threshold=args.rollback_gate_future_object_tth,
    )


def record_allowed_by_gate(args: argparse.Namespace, risk, current_body: float, current_object: float) -> bool:
    max_risk_score = getattr(args, "record_max_risk_score", None)
    if max_risk_score is not None and _candidate_risk_score(risk, current_body, current_object) > float(max_risk_score):
        return False
    max_current_body = getattr(args, "record_max_current_body_probability", None)
    if max_current_body is not None and float(current_body) > float(max_current_body):
        return False
    max_current_object = getattr(args, "record_max_current_object_probability", None)
    if max_current_object is not None and float(current_object) > float(max_current_object):
        return False
    return True


def _candidate_risk_score(risk, current_body: float, current_object: float) -> float:
    scores = [
        float(getattr(risk, "max_probability", 0.0)),
        float(max(0.0, 1.0 - float(getattr(risk, "min_tth", 1.0))) * 0.25),
        float(current_body),
        float(current_object),
    ]
    return float(max(scores))


def qwen_outcome_labels(body_hazard: bool, object_hazard: bool, stuck_hazard: bool = False) -> dict[str, float]:
    body_or_stuck = bool(body_hazard or stuck_hazard)
    return {
        "current_body": float(body_or_stuck),
        "current_object": float(object_hazard),
        "future_body": float(body_or_stuck),
        "future_body_tth": 0.0 if body_or_stuck else 1.0,
        "future_object": float(object_hazard),
        "future_object_tth": 0.0 if object_hazard else 1.0,
    }


def append_decision_debug(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def masked_action_probabilities(decision_policy, actor_features: np.ndarray, action_mask) -> list[float]:
    try:
        device = next(decision_policy.parameters()).device
        tensor = torch.as_tensor(actor_features, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if hasattr(decision_policy, "forward_actor"):
                logits = decision_policy.forward_actor(tensor, action_mask=action_mask)
            else:
                logits, _ = decision_policy(tensor, action_mask=action_mask)
            return [float(item) for item in torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()]
    except Exception as exc:
        return [float("nan"), float("nan"), float("nan")]


def make_qwen_rollout_sample(
    frame_history,
    instruction: str,
    proposed_action,
    step_index: int,
    signals,
    metadata: dict | None = None,
    labels: dict | None = None,
    tau: int = 50,
) -> dict:
    prompt_text, prompt_images = build_live_qwen_prompt(
        history=list(frame_history),
        instruction=instruction,
        proposed_action=proposed_action,
        current_global_step=step_index,
        history_length=3,
        tau=tau,
    )
    return {
        "sample_index": int(step_index),
        "text": prompt_text.replace(QWEN_IMAGE_TOKEN, "<image>"),
        "images": [image.copy() for image in prompt_images],
        "current_body": float(signals.body_hazard or getattr(signals, "stuck_hazard", False)),
        "current_object": float(signals.object_hazard),
        "labels": labels or {},
        "metadata": metadata or {},
    }


def rollout_episode(env, task, init_states, base_policy, predictor, decision_policy, rollback_executor, args, episode_id: int) -> tuple[list[dict], dict]:
    if hasattr(predictor, "reset"):
        predictor.reset()
    np.random.seed(args.seed + episode_id)
    torch.manual_seed(args.seed + episode_id)
    env.seed(args.seed + episode_id)
    env.reset()
    init_index = (args.init_state_start + episode_id) % len(init_states)
    obs = env.set_init_state(init_states[init_index])
    oracle = LiberoHazardOracle(env.sim)
    memory = WaypointMemory()
    stuck_monitor = ProprioceptiveStuckMonitor(
        window_steps=args.stuck_window_steps,
        min_low_motion_steps=args.stuck_min_low_motion_steps,
        max_step_displacement=args.stuck_max_step_displacement,
        max_window_displacement=args.stuck_max_window_displacement,
    )
    teacher = RuleBasedDecider(
        record_probability=args.record_probability,
        rollback_probability=args.rollback_probability,
        rollback_tth=args.rollback_tth,
        min_record_interval=args.min_record_interval,
        rollback_cooldown=args.rollback_cooldown,
    )
    action_plan: collections.deque[np.ndarray] = collections.deque()
    joint_history: collections.deque[np.ndarray] = collections.deque(maxlen=3)
    risk_history: collections.deque = collections.deque(maxlen=3)
    current_history: collections.deque[tuple[float, float]] = collections.deque(maxlen=3)
    qwen_frame_history: collections.deque = collections.deque(maxlen=3)
    qwen_rollout_samples: list[dict] = []
    qwen_step_candidates: dict[int, dict] = {}
    qwen_hazard_timeline = []
    records: list[dict] = []
    interventions = {"noop": 0, "record": 0, "rollback": 0}
    hazards = {"body": 0, "object": 0, "stuck": 0, "any": 0}
    last_record_step: int | None = None
    last_rollback_step: int | None = None
    rollback_count = 0
    rollback_failures = 0
    rollback_rendered_frames = 0
    rollback_opportunities = 0
    rollback_explorations = 0
    noop_explorations = 0
    noop_probe_blocked_opportunities = 0
    record_explorations = 0
    initial_safe_anchors = 0
    rollback_outcome_credits = 0.0
    rollback_preemptive_credits = 0.0
    rollback_terminal_credits = 0.0
    noop_outcome_credits = 0.0
    noop_safe_labels = 0
    noop_hazard_labels = 0
    noop_censored_labels = 0
    predictor_onset_events = 0
    predictor_onset_credits = 0.0
    rollback_record_indices: list[int] = []
    rollback_events: list[dict] = []
    pending_rollback_record_index: int | None = None
    pending_rollback_hazard_steps = 0
    pending_rollback_observed_steps = 0
    pending_rollback_pre_hazard_steps = 0
    pending_rollback_pre_observed_steps = 0
    pending_rollback_preemptive = False
    pending_noop_outcomes: list[dict] = []
    recent_hazard_outcomes: collections.deque[float] = collections.deque(
        maxlen=max(1, int(args.rollback_effect_horizon))
    )
    reward_sum = 0.0
    credited_interval_reward = 0.0
    credited_interval_hazard_steps = 0
    success = False
    previous_training_hazard = False
    previous_predictor_hazard = False
    suite_max_steps = max_steps_for_suite(args.benchmark)
    max_steps = int(args.max_rollout_steps) if args.allow_extended_rollout else min(args.max_rollout_steps, suite_max_steps)
    reward_config = reward_config_from_args(args)

    def finish_pending_rollback_credit() -> None:
        nonlocal pending_rollback_record_index
        nonlocal pending_rollback_hazard_steps
        nonlocal pending_rollback_observed_steps
        nonlocal pending_rollback_pre_hazard_steps
        nonlocal pending_rollback_pre_observed_steps
        nonlocal pending_rollback_preemptive
        nonlocal rollback_outcome_credits
        nonlocal rollback_preemptive_credits
        nonlocal reward_sum
        if pending_rollback_record_index is None:
            return
        credit = rollback_outcome_credit(
            post_rollback_hazard_steps=pending_rollback_hazard_steps,
            observed_steps=pending_rollback_observed_steps,
            pre_rollback_hazard_steps=pending_rollback_pre_hazard_steps,
            pre_observed_steps=pending_rollback_pre_observed_steps,
            preemptive_warning=pending_rollback_preemptive,
            config=reward_config,
        )
        if credit and 0 <= pending_rollback_record_index < len(records):
            records[pending_rollback_record_index]["reward"] = (
                float(records[pending_rollback_record_index]["reward"]) + float(credit)
            )
            rollback_outcome_credits += float(credit)
            if pending_rollback_preemptive:
                rollback_preemptive_credits += float(credit)
            reward_sum += float(credit)
        if 0 <= pending_rollback_record_index < len(records):
            outcome_teacher_action = None
            if credit > 0.0:
                outcome_teacher_action = ThreeAction.ROLLBACK
            elif credit < 0.0:
                outcome_teacher_action = ThreeAction.NOOP
            if outcome_teacher_action is not None:
                records[pending_rollback_record_index]["teacher_action"] = int(outcome_teacher_action)
                records[pending_rollback_record_index]["outcome_teacher_action"] = int(outcome_teacher_action)
                records[pending_rollback_record_index]["rollback_outcome_credit"] = float(credit)
                records[pending_rollback_record_index]["rollback_pre_hazard_steps"] = float(
                    pending_rollback_pre_hazard_steps
                )
                records[pending_rollback_record_index]["rollback_post_hazard_steps"] = float(
                    pending_rollback_hazard_steps
                )
                records[pending_rollback_record_index]["rollback_preemptive"] = bool(
                    pending_rollback_preemptive
                )
        pending_rollback_record_index = None
        pending_rollback_hazard_steps = 0
        pending_rollback_observed_steps = 0
        pending_rollback_pre_hazard_steps = 0
        pending_rollback_pre_observed_steps = 0
        pending_rollback_preemptive = False

    def prepare_reward_signals(signals: OnlineStepSignals, *, monitor_stuck: bool) -> OnlineStepSignals:
        nonlocal previous_training_hazard
        prepared = augment_training_signals(
            signals,
            monitor_stuck=monitor_stuck,
            previous_any_hazard=previous_training_hazard,
        )
        previous_training_hazard = prepared.any_hazard
        return prepared

    def finish_noop_outcome(item: dict, *, censored: bool = False) -> None:
        nonlocal noop_outcome_credits
        nonlocal noop_safe_labels
        nonlocal noop_hazard_labels
        nonlocal noop_censored_labels
        nonlocal reward_sum
        record_index = int(item["record_index"])
        if not 0 <= record_index < len(records):
            return
        if censored:
            records[record_index]["teacher_action"] = -1
            records[record_index]["noop_outcome_censored"] = True
            noop_censored_labels += 1
            return
        hazard_steps = float(item["hazard_steps"])
        observed_steps = int(item["observed_steps"])
        credit = noop_outcome_credit(hazard_steps, observed_steps, config=reward_config)
        records[record_index]["reward"] = float(records[record_index]["reward"]) + float(credit)
        records[record_index]["teacher_action"] = int(
            ThreeAction.ROLLBACK if hazard_steps > 0.0 else ThreeAction.NOOP
        )
        records[record_index]["noop_outcome_credit"] = float(credit)
        records[record_index]["noop_outcome_hazard_steps"] = float(hazard_steps)
        records[record_index]["noop_outcome_observed_steps"] = int(observed_steps)
        noop_outcome_credits += float(credit)
        noop_hazard_labels += int(hazard_steps > 0.0)
        noop_safe_labels += int(hazard_steps <= 0.0)
        reward_sum += float(credit)

    def observe_pending_noops(signals: OnlineStepSignals) -> None:
        if not pending_noop_outcomes:
            return
        horizon = max(1, int(args.noop_effect_horizon))
        hazard_score = float(online_hazard_score(signals, reward_config))
        completed = []
        for item in pending_noop_outcomes:
            item["observed_steps"] += 1
            item["hazard_steps"] += hazard_score
            if item["hazard_steps"] > 0.0 or item["observed_steps"] >= horizon or signals.success:
                completed.append(item)
        for item in completed:
            finish_noop_outcome(item)
            pending_noop_outcomes.remove(item)

    def censor_pending_noops() -> None:
        for item in list(pending_noop_outcomes):
            finish_noop_outcome(item, censored=True)
            pending_noop_outcomes.remove(item)

    def finish_pending_noops() -> None:
        horizon = max(1, int(args.noop_effect_horizon))
        for item in list(pending_noop_outcomes):
            complete = (
                float(item["hazard_steps"]) > 0.0
                or int(item["observed_steps"]) >= horizon
                or success
            )
            finish_noop_outcome(item, censored=not complete)
            pending_noop_outcomes.remove(item)

    def observe_post_rollback(signals) -> None:
        nonlocal pending_rollback_hazard_steps
        nonlocal pending_rollback_observed_steps
        if pending_rollback_record_index is None or int(args.rollback_effect_horizon) <= 0:
            return
        pending_rollback_observed_steps += 1
        pending_rollback_hazard_steps += float(online_hazard_score(signals, reward_config))
        if pending_rollback_observed_steps >= int(args.rollback_effect_horizon):
            finish_pending_rollback_credit()

    def remember_hazard(signals) -> None:
        recent_hazard_outcomes.append(float(online_hazard_score(signals, reward_config)))

    for step in range(max_steps + args.num_steps_wait):
        if step < args.num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        step_index = step - args.num_steps_wait
        if not action_plan:
            policy_input = policy_observation(
                obs,
                task.language,
                args.resize_size,
                policy_backend=args.policy_backend,
                benchmark=args.benchmark,
            )
            action_chunk = base_policy.infer(policy_input)["actions"]
            if len(action_chunk) < args.replan_steps:
                raise RuntimeError(f"policy returned {len(action_chunk)} actions, need {args.replan_steps}")
            action_plan.extend(np.asarray(action_chunk[: args.replan_steps]))
        proposed_action = action_plan.popleft()
        if args.record_initial_safe_anchor and step_index == 0 and len(memory) == 0:
            memory.record(
                step_index=0,
                state=capture_env_state(env),
                risk=None,
                metadata={
                    "source": "initial_safe_anchor",
                    "task_id": int(args.task_id),
                    "episode_id": int(episode_id),
                },
            )
            last_record_step = 0
            initial_safe_anchors += 1
        stuck_detected = bool(args.stuck_fallback and stuck_monitor.update(obs))
        pre_action_signals = None
        if args.qwen_rollout_jsonl_out is not None:
            pre_action_signals = oracle.read(success=False, task_reward=0.0)
            qwen_hazard_timeline.append(pre_action_signals)
            qwen_frame_history.append(live_observation_to_frame(obs))
            stride_due = step_index % max(1, int(args.qwen_rollout_stride)) == 0
            dense_enabled = int(args.qwen_rollout_prehazard_stride) > 0
            if stride_due or dense_enabled:
                sample = make_qwen_rollout_sample(
                    qwen_frame_history,
                    instruction=task.language,
                    proposed_action=proposed_action.tolist(),
                    step_index=step_index,
                    signals=pre_action_signals,
                    metadata={"source": "online_stride" if stride_due else "online_candidate"},
                    tau=args.qwen_tau,
                )
                if dense_enabled:
                    qwen_step_candidates[int(step_index)] = sample
                if stride_due:
                    qwen_rollout_samples.append(sample)
        decision_due = step_index % max(1, args.decision_period) == 0
        if not decision_due:
            obs, task_reward, done, _ = env.step(proposed_action.tolist())
            success = bool(done or env.check_success())
            signals = oracle.read(success=success, task_reward=float(task_reward))
            signals = prepare_reward_signals(signals, monitor_stuck=stuck_detected)
            observe_post_rollback(signals)
            observe_pending_noops(signals)
            step_reward = online_safeloop_reward(
                signals,
                Intervention.NOOP,
                rollback_count=rollback_count,
                config=reward_config,
            )
            reward_sum += float(step_reward)
            if records:
                records[-1]["reward"] = float(records[-1]["reward"]) + float(step_reward)
                records[-1]["done"] = bool(records[-1]["done"] or success)
                credited_interval_reward += float(step_reward)
                credited_interval_hazard_steps += int(signals.any_hazard)
            interventions["noop"] += 1
            hazards["body"] += int(signals.body_hazard)
            hazards["object"] += int(signals.object_hazard)
            hazards["stuck"] += int(getattr(signals, "stuck_hazard", False))
            hazards["any"] += int(signals.any_hazard)
            remember_hazard(signals)
            if success:
                break
            continue

        prediction = predictor.predict(obs, proposed_action.tolist(), instruction=task.language)
        risk, current_body, current_object = normalize_risk_prediction(prediction)
        risk, current_body = apply_stuck_fallback(risk, current_body, stuck_detected)
        proxy_hazard = predictor_proxy_hazard(
            current_body,
            current_object,
            stuck_detected,
            body_threshold=args.predictor_realized_body_threshold,
            object_threshold=args.predictor_realized_object_threshold,
        )
        proxy_onset = bool(proxy_hazard and not previous_predictor_hazard)
        previous_predictor_hazard = proxy_hazard
        if proxy_onset:
            predictor_onset_events += 1
            if records and float(args.predictor_onset_penalty) != 0.0:
                credit = float(args.predictor_onset_penalty)
                records[-1]["reward"] = float(records[-1]["reward"]) + credit
                records[-1]["predictor_onset_credit"] = credit
                reward_sum += credit
                predictor_onset_credits += credit
        joint_history.append(extract_robot_joint_vector(obs, proposed_action))
        risk_history.append(risk)
        current_history.append((current_body, current_object))

        can_record = (
            cool_down_elapsed(step_index, last_record_step, args.min_record_interval)
            and record_allowed_by_gate(args, risk, current_body, current_object)
        )
        rollback_waypoint = None
        raw_rollback_gate_open = (
            len(memory) > 0
            and cool_down_elapsed(step_index, last_rollback_step, args.rollback_cooldown)
            and rollback_allowed_by_gate(
                args,
                risk,
                current_body,
                current_object,
                rollback_count,
                step_index,
                stuck_detected=stuck_detected,
            )
        )
        noop_probe_active = bool(args.noop_probe_blocks_rollback and pending_noop_outcomes)
        rollback_gate_open = bool(raw_rollback_gate_open and not noop_probe_active)
        noop_probe_blocked_opportunities += int(raw_rollback_gate_open and noop_probe_active)
        if rollback_gate_open:
            try:
                rollback_target_max_age = (
                    args.stuck_rollback_target_max_age
                    if stuck_detected and args.stuck_rollback_target_max_age is not None
                    else args.rollback_target_max_age
                )
                rollback_waypoint = memory.select_rollback(
                    current_step_index=step_index,
                    safe_score_threshold=args.rollback_target_safe_score_threshold,
                    min_safe_age=args.rollback_target_min_age,
                    max_safe_age=rollback_target_max_age,
                    require_safe=True,
                    prefer_recent_safe=args.rollback_target_prefer_recent_safe,
                )
                if not initial_anchor_rollback_allowed(
                    rollback_waypoint,
                    current_body_probability=current_body,
                    current_object_probability=current_object,
                    stuck_detected=stuck_detected,
                    min_current_body_probability=(
                        args.initial_anchor_rollback_min_current_body_probability
                    ),
                    min_current_object_probability=(
                        args.initial_anchor_rollback_min_current_object_probability
                    ),
                ):
                    rollback_waypoint = None
            except IndexError:
                rollback_waypoint = None
        can_rollback = rollback_waypoint is not None
        rollback_opportunities += int(can_rollback)
        action_mask = build_action_mask(can_record=can_record, can_rollback=can_rollback)
        context_values = [
            float(can_record),
            float(can_rollback),
            float(np.clip(len(memory) / 10.0, 0.0, 1.0)),
            float(np.clip(step_index / max(max_steps, 1), 0.0, 1.0)),
        ]
        actor_features = build_decision_features(
            joint_history=list(joint_history),
            risk_history=list(risk_history),
            current_hazard_history=list(current_history),
            context_values=context_values,
            history_length=3,
        )
        action_probabilities = masked_action_probabilities(decision_policy, actor_features, action_mask)
        action, old_logprob, exploration = sample_action_with_intervention_exploration(
            decision_policy,
            actor_features,
            action_mask=action_mask,
            rollback_exploration_probability=args.rollback_exploration_probability,
            noop_exploration_probability=args.noop_exploration_probability,
        )
        explored_rollback = exploration == "rollback"
        explored_noop = exploration == "noop"
        explored_record = False
        record_probability = min(max(float(args.record_exploration_probability), 0.0), 1.0)
        if can_record and record_probability > 0.0 and float(np.random.random()) < record_probability:
            tensor = torch.as_tensor(
                actor_features,
                dtype=torch.float32,
                device=next(decision_policy.parameters()).device,
            ).unsqueeze(0)
            with torch.no_grad():
                if hasattr(decision_policy, "forward_actor"):
                    logits = decision_policy.forward_actor(tensor, action_mask=action_mask)
                else:
                    logits, _ = decision_policy(tensor, action_mask=action_mask)
                distribution = torch.distributions.Categorical(logits=logits)
                record_tensor = torch.as_tensor([int(ThreeAction.RECORD)], dtype=torch.long, device=logits.device)
                old_logprob = distribution.log_prob(record_tensor).squeeze(0)
            action = ThreeAction.RECORD
            explored_record = True
            record_explorations += 1
            explored_rollback = False
            explored_noop = False
        rollback_explorations += int(explored_rollback)
        noop_explorations += int(explored_noop)
        teacher_intervention = teacher.decide(
            risk=risk,
            memory=memory,
            step_index=step_index,
            last_record_step=last_record_step,
            last_rollback_step=last_rollback_step,
        )
        intervention = ACTION_TO_INTERVENTION[action]
        append_decision_debug(
            args.decision_debug_jsonl_out,
            {
                "task_id": int(args.task_id),
                "episode_id": int(episode_id),
                "step_index": int(step_index),
                "memory_size": int(len(memory)),
                "rollback_count": int(rollback_count),
                "can_record": bool(can_record),
                "can_rollback": bool(can_rollback),
                "noop_probe_active": bool(noop_probe_active),
                "noop_probe_blocked_rollback": bool(raw_rollback_gate_open and noop_probe_active),
                "action_mask": [bool(item) for item in action_mask],
                "action": int(action),
                "intervention": intervention.value,
                "executed_rollback": bool(intervention == Intervention.ROLLBACK and can_rollback),
                "stuck_fallback": bool(stuck_detected),
                "explored_rollback": bool(explored_rollback),
                "explored_noop": bool(explored_noop),
                "explored_record": bool(explored_record),
                "action_probabilities": action_probabilities,
                "teacher_intervention": teacher_intervention.value,
                "risk": {
                    "body_probability": float(risk.body_probability),
                    "body_tth": float(risk.body_tth),
                    "object_probability": float(risk.object_probability),
                    "object_tth": float(risk.object_tth),
                    "current_body_probability": float(current_body),
                    "current_object_probability": float(current_object),
                },
            },
        )
        rollback_failed = False
        rendered_frames = 0
        rollback_target_step = None
        rollback_target_age = None
        rollback_target_risk_score = None
        rollback_motion_info: dict = {}
        rollback_pre_hazard_steps = float(sum(recent_hazard_outcomes))
        rollback_pre_observed_steps = int(len(recent_hazard_outcomes))
        rollback_preemptive = bool(not stuck_detected and not previous_training_hazard)

        if intervention == Intervention.ROLLBACK and can_rollback:
            if args.qwen_rollout_jsonl_out is not None:
                if pre_action_signals is None:
                    pre_action_signals = oracle.read(success=False, task_reward=0.0)
                qwen_rollout_samples.append(
                    make_qwen_rollout_sample(
                        qwen_frame_history,
                        instruction=task.language,
                        proposed_action=proposed_action.tolist(),
                        step_index=step_index,
                        signals=pre_action_signals,
                        metadata={"source": "rollback_pre"},
                        tau=args.qwen_tau,
                    )
                )
            finish_pending_rollback_credit()
            censor_pending_noops()
            waypoint = rollback_waypoint
            if waypoint is None:
                waypoint = memory.select_rollback(
                    current_step_index=step_index,
                    safe_score_threshold=args.rollback_target_safe_score_threshold,
                    min_safe_age=args.rollback_target_min_age,
                    max_safe_age=args.rollback_target_max_age,
                    require_safe=True,
                    prefer_recent_safe=args.rollback_target_prefer_recent_safe,
                )
            rollback_target_step = int(waypoint.step_index)
            rollback_target_age = int(step_index) - int(waypoint.step_index)
            rollback_target_risk_score = float(_waypoint_risk_score(waypoint))
            obs, rollback_info = execute_rollback(env, waypoint, rollback_executor)
            stuck_monitor.reset()
            rollback_motion_info = dict(rollback_info)
            rollback_failed = bool(rollback_info.get("safe") is False or rollback_info.get("reached") is False)
            if not rollback_failed:
                anchor_metadata = dict(waypoint.metadata)
                anchor_metadata["source"] = "rollback_anchor"
                memory.record(
                    step_index=step_index,
                    state=capture_env_state(env),
                    risk=waypoint.risk,
                    metadata=anchor_metadata,
                )
                last_record_step = step_index
            rendered_frames = int(rollback_info.get("rendered_frames") or 0)
            rollback_rendered_frames += rendered_frames
            rollback_events.append(
                {
                    "step_index": int(step_index),
                    "target_step": int(rollback_target_step),
                    "target_age": int(rollback_target_age),
                    "target_risk_score": float(rollback_target_risk_score),
                    "failed": bool(rollback_failed),
                    "rendered_frames": int(rendered_frames),
                    "motion": rollback_motion_info,
                    "risk": {
                        "body_probability": float(risk.body_probability),
                        "body_tth": float(risk.body_tth),
                        "object_probability": float(risk.object_probability),
                        "object_tth": float(risk.object_tth),
                        "current_body_probability": float(current_body),
                        "current_object_probability": float(current_object),
                    },
                }
            )
            action_plan.clear()
            last_rollback_step = step_index
            rollback_count += 1
            rollback_failures += int(rollback_failed)
            task_reward = 0.0
            done = False
        else:
            if intervention == Intervention.RECORD and can_record:
                memory.record(
                    step_index=step_index,
                    state=capture_env_state(env),
                    risk=risk,
                    metadata={
                        "current_body_probability": current_body,
                        "current_object_probability": current_object,
                    },
                )
                last_record_step = step_index
            obs, task_reward, done, _ = env.step(proposed_action.tolist())

        success = bool(done or env.check_success())
        signals = oracle.read(success=success, task_reward=float(task_reward))
        executed_rollback = bool(intervention == Intervention.ROLLBACK and can_rollback)
        signals = prepare_reward_signals(
            signals,
            monitor_stuck=bool(stuck_detected and not executed_rollback),
        )
        observe_pending_noops(signals)
        if args.qwen_rollout_jsonl_out is not None and intervention == Intervention.ROLLBACK and can_rollback:
            qwen_frame_history.append(live_observation_to_frame(obs))
            qwen_rollout_samples.append(
                make_qwen_rollout_sample(
                    qwen_frame_history,
                    instruction=task.language,
                    proposed_action=None,
                    step_index=step_index,
                    signals=signals,
                    metadata={
                        "source": "rollback_post",
                        "rollback_failed": bool(rollback_failed),
                    },
                    labels=qwen_outcome_labels(
                        signals.body_hazard,
                        signals.object_hazard,
                        getattr(signals, "stuck_hazard", False),
                    ),
                    tau=args.qwen_tau,
                )
            )
        if not (intervention == Intervention.ROLLBACK and can_rollback):
            observe_post_rollback(signals)
        reward = online_safeloop_reward(
            signals,
            intervention,
            rollback_count=rollback_count,
            rollback_failed=rollback_failed,
            rollback_rendered_frames=rendered_frames,
            config=reward_config,
        )
        critic_features = np.concatenate([actor_features, signals.privileged_features()]).astype(np.float32)
        records.append(
            {
                "actor_features": actor_features,
                "critic_features": critic_features,
                "action": int(action),
                "old_logprob": float(old_logprob.detach().cpu()) if hasattr(old_logprob, "detach") else float(old_logprob),
                "reward": float(reward),
                "done": bool(success),
                "action_mask": action_mask,
                "teacher_action": int(INTERVENTION_TO_ACTION[teacher_intervention]),
                "ppo_weight": 0.0 if (explored_rollback or explored_noop or explored_record) else 1.0,
                "rollback_target_step": rollback_target_step,
                "rollback_target_age": rollback_target_age,
                "rollback_target_risk_score": rollback_target_risk_score,
                "rollback_rendered_frames": int(rendered_frames),
                "rollback_failed": bool(rollback_failed),
                "task_id": int(args.task_id),
                "episode_id": int(episode_id),
                "init_state_index": int(init_index),
                "step_index": int(step_index),
                "can_record": bool(can_record),
                "can_rollback": bool(can_rollback),
                "explored_rollback": bool(explored_rollback),
                "explored_noop": bool(explored_noop),
                "explored_record": bool(explored_record),
            }
        )
        if (
            intervention == Intervention.NOOP
            and can_rollback
            and int(args.noop_effect_horizon) > 0
        ):
            item = {
                "record_index": len(records) - 1,
                "hazard_steps": float(online_hazard_score(signals, reward_config)),
                "observed_steps": 1,
            }
            if item["hazard_steps"] > 0.0 or signals.success or int(args.noop_effect_horizon) <= 1:
                finish_noop_outcome(item)
            else:
                pending_noop_outcomes.append(item)
        if (
            intervention == Intervention.ROLLBACK
            and can_rollback
            and not rollback_failed
            and int(args.rollback_effect_horizon) > 0
        ):
            pending_rollback_record_index = len(records) - 1
            pending_rollback_hazard_steps = 0
            pending_rollback_observed_steps = 0
            pending_rollback_pre_hazard_steps = rollback_pre_hazard_steps
            pending_rollback_pre_observed_steps = rollback_pre_observed_steps
            pending_rollback_preemptive = rollback_preemptive
        if intervention == Intervention.ROLLBACK and can_rollback:
            rollback_record_indices.append(len(records) - 1)
        interventions[intervention.value] += 1
        hazards["body"] += int(signals.body_hazard)
        hazards["object"] += int(signals.object_hazard)
        hazards["stuck"] += int(getattr(signals, "stuck_hazard", False))
        hazards["any"] += int(signals.any_hazard)
        remember_hazard(signals)
        reward_sum += float(reward)
        if success:
            break

    finish_pending_rollback_credit()
    finish_pending_noops()
    if args.qwen_rollout_jsonl_out is not None and qwen_rollout_samples:
        append_prehazard_rollout_samples(
            qwen_rollout_samples,
            qwen_step_candidates,
            qwen_hazard_timeline,
            tau=args.qwen_tau,
            stride=args.qwen_rollout_prehazard_stride,
        )
        write_qwen_rollout_samples(
            args=args,
            task=task,
            episode_index=episode_id,
            samples=qwen_rollout_samples,
            hazard_timeline=qwen_hazard_timeline,
        )
    if rollback_record_indices:
        terminal_credit = rollback_terminal_credit(
            success=success,
            rollback_count=len(rollback_record_indices),
            config=reward_config,
        )
        if terminal_credit:
            for record_index in rollback_record_indices:
                if 0 <= record_index < len(records):
                    records[record_index]["reward"] = float(records[record_index]["reward"]) + float(terminal_credit)
                    if "outcome_teacher_action" not in records[record_index]:
                        if terminal_credit > 0.0:
                            records[record_index]["teacher_action"] = int(ThreeAction.ROLLBACK)
                        elif terminal_credit < 0.0:
                            records[record_index]["teacher_action"] = int(ThreeAction.NOOP)
                    rollback_terminal_credits += float(terminal_credit)
                    reward_sum += float(terminal_credit)
    mark_online_episode_terminal(records)
    return records, {
        "task": task.name,
        "task_id": args.task_id,
        "init_state_index": init_index,
        "steps": len(records),
        "max_steps": int(max_steps),
        "suite_max_steps": int(suite_max_steps),
        "allow_extended_rollout": bool(args.allow_extended_rollout),
        "success": bool(success),
        "reward_sum": float(reward_sum),
        "credited_interval_reward": float(credited_interval_reward),
        "credited_interval_hazard_steps": int(credited_interval_hazard_steps),
        "interventions": interventions,
        "hazard_steps": hazards,
        "rollback_failures": int(rollback_failures),
        "rollback_rendered_frames": int(rollback_rendered_frames),
        "rollback_opportunities": int(rollback_opportunities),
        "rollback_explorations": int(rollback_explorations),
        "noop_explorations": int(noop_explorations),
        "noop_probe_blocked_opportunities": int(noop_probe_blocked_opportunities),
        "record_explorations": int(record_explorations),
        "initial_safe_anchors": int(initial_safe_anchors),
        "rollback_outcome_credits": float(rollback_outcome_credits),
        "rollback_preemptive_credits": float(rollback_preemptive_credits),
        "rollback_terminal_credits": float(rollback_terminal_credits),
        "noop_outcome_credits": float(noop_outcome_credits),
        "noop_safe_labels": int(noop_safe_labels),
        "noop_hazard_labels": int(noop_hazard_labels),
        "noop_censored_labels": int(noop_censored_labels),
        "predictor_onset_events": int(predictor_onset_events),
        "predictor_onset_credits": float(predictor_onset_credits),
        "rollback_events": rollback_events,
    }


def bc_coef_for_update(update_index: int, total_updates: int, initial: float, fraction: float) -> float:
    anneal_updates = max(1.0, float(total_updates) * max(float(fraction), 1e-6))
    return float(initial) * max(0.0, 1.0 - float(update_index) / anneal_updates)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_paths(args)
    base_policy = make_policy(args)
    predictor = make_predictor(args)
    rollback_executor = make_rollback_executor(args)
    actor_dim = 49
    critic_dim = actor_dim + privileged_feature_dim()
    decision_policy = actor_checkpoint_to_asymmetric(
        args.initial_decision_checkpoint,
        critic_dim=critic_dim,
        device=args.decision_device,
    )
    scale_actor_logits(decision_policy, args.initial_actor_logit_scale)
    trainer = AsymmetricPPOTrainer(
        decision_policy,
        AsymmetricPPOConfig(
            learning_rate=args.lr,
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            entropy_coef=args.entropy_coef,
            bc_coef=args.bc_coef,
            class_balanced_bc=args.class_balanced_bc,
        ),
    )
    update_end_index = args.update_start_index + args.updates
    total_schedule_updates = args.total_schedule_updates or update_end_index
    if args.update_start_index < 0 or args.updates <= 0:
        raise ValueError("update-start-index must be non-negative and updates must be positive")
    if total_schedule_updates < update_end_index:
        raise ValueError("total-schedule-updates must cover every requested update")

    all_metrics = []
    global_episode = 0
    for update_index in range(args.update_start_index, update_end_index):
        update_records: list[dict] = []
        update_reports: list[dict] = []
        for task_id in args.task_ids:
            args.task_id = int(task_id)
            env, task, init_states = make_env(args)
            try:
                for _ in range(args.episodes_per_update):
                    records, report = rollout_episode(
                        env,
                        task,
                        init_states,
                        base_policy,
                        predictor,
                        decision_policy,
                        rollback_executor,
                        args,
                        global_episode,
                    )
                    global_episode += 1
                    update_records.extend(records)
                    update_reports.append(report)
            finally:
                env.close()
        if args.save_online_records:
            torch.save(update_records, args.out_dir / f"online_records_update{update_index:03d}.pt")
        batch = batch_from_online_records(update_records)
        coef = bc_coef_for_update(update_index, total_schedule_updates, args.bc_coef, args.bc_anneal_fraction)
        history = trainer.update(batch, epochs=args.ppo_epochs, bc_coef=coef)
        checkpoint = args.out_dir / f"online_decider_update{update_index:03d}.pt"
        export_actor_checkpoint(decision_policy, checkpoint)
        metrics = {
            "update": update_index,
            "records": len(update_records),
            "episode_boundaries": int(sum(bool(item.get("done", False)) for item in update_records)),
            "bc_coef": coef,
            "checkpoint": str(checkpoint),
            "ppo_last": history[-1] if history else {},
            "reports": update_reports,
            "success_rate": float(np.mean([item["success"] for item in update_reports])) if update_reports else 0.0,
            "mean_reward_sum": float(np.mean([item["reward_sum"] for item in update_reports])) if update_reports else 0.0,
            "credited_interval_reward": float(sum(item["credited_interval_reward"] for item in update_reports)),
            "credited_interval_hazard_steps": int(sum(item["credited_interval_hazard_steps"] for item in update_reports)),
            "hazard_steps": {
                key: int(sum(item["hazard_steps"].get(key, 0) for item in update_reports))
                for key in ("body", "object", "stuck", "any")
            },
            "rollback_failures": int(sum(item["rollback_failures"] for item in update_reports)),
            "rollback_rendered_frames": int(sum(item["rollback_rendered_frames"] for item in update_reports)),
            "rollback_opportunities": int(sum(item["rollback_opportunities"] for item in update_reports)),
            "rollback_explorations": int(sum(item["rollback_explorations"] for item in update_reports)),
            "noop_explorations": int(sum(item.get("noop_explorations", 0) for item in update_reports)),
            "noop_probe_blocked_opportunities": int(
                sum(item.get("noop_probe_blocked_opportunities", 0) for item in update_reports)
            ),
            "record_explorations": int(sum(item.get("record_explorations", 0) for item in update_reports)),
            "initial_safe_anchors": int(sum(item.get("initial_safe_anchors", 0) for item in update_reports)),
            "rollback_outcome_credits": float(sum(item["rollback_outcome_credits"] for item in update_reports)),
            "rollback_preemptive_credits": float(
                sum(item.get("rollback_preemptive_credits", 0.0) for item in update_reports)
            ),
            "rollback_terminal_credits": float(sum(item["rollback_terminal_credits"] for item in update_reports)),
            "noop_outcome_credits": float(sum(item.get("noop_outcome_credits", 0.0) for item in update_reports)),
            "noop_safe_labels": int(sum(item.get("noop_safe_labels", 0) for item in update_reports)),
            "noop_hazard_labels": int(sum(item.get("noop_hazard_labels", 0) for item in update_reports)),
            "noop_censored_labels": int(sum(item.get("noop_censored_labels", 0) for item in update_reports)),
            "predictor_onset_events": int(sum(item.get("predictor_onset_events", 0) for item in update_reports)),
            "predictor_onset_credits": float(sum(item.get("predictor_onset_credits", 0.0) for item in update_reports)),
            "interventions": {
                key: int(sum(item["interventions"][key] for item in update_reports))
                for key in ("noop", "record", "rollback")
            },
        }
        all_metrics.append(metrics)
        (args.out_dir / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    k: metrics[k]
                    for k in (
                        "update",
                        "records",
                        "success_rate",
                        "mean_reward_sum",
                        "hazard_steps",
                        "interventions",
                        "rollback_explorations",
                        "noop_explorations",
                        "rollback_opportunities",
                        "checkpoint",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
