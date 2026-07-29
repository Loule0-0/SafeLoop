from __future__ import annotations

import argparse
import collections
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard import RuleBasedDecider, SafeLoopController
from safety_guard.decider import Intervention
from safety_guard.libero_motion import MotionPlanningRollbackExecutor
from safety_guard.libero_oracle import LiberoHazardOracle
from safety_guard.online_rl import OnlineStepSignals
from safety_guard.predictors import ActionNormRiskPredictor, ConstantRiskPredictor
from safety_guard.qwen_multitask import (
    QWEN_IMAGE_TOKEN,
    QwenMultitaskSafetyPredictor,
    build_live_qwen_prompt,
    live_observation_to_frame,
)
from safety_guard.rollout_sampling import append_prehazard_rollout_samples
from safety_guard.rl_policy_decider import RLPolicyDecider, RollbackGateDecider
from safety_guard.video import write_frames

from scripts.libero_policy_utils import (
    LIBERO_DUMMY_ACTION,
    configure_paths,
    current_observation,
    env_path,
    make_env,
    make_policy,
    max_steps_for_suite,
    policy_observation,
    render_frame,
)


@dataclass
class PeriodicDecider:
    inner: object
    period: int = 1
    last_decision_info: dict | None = None

    def decide(self, step_index: int, **kwargs):
        if self.period > 1 and int(step_index) % int(self.period) != 0:
            self.last_decision_info = {
                "decider": "periodic",
                "period": int(self.period),
                "due": False,
                "selected_action": Intervention.NOOP.value,
            }
            return Intervention.NOOP
        decide = self.inner.decide
        signature = inspect.signature(decide)
        merged = {"step_index": step_index, **kwargs}
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            action = decide(**merged)
        else:
            accepted = {key: value for key, value in merged.items() if key in signature.parameters}
            action = decide(**accepted)
        self.last_decision_info = {
            "decider": "periodic",
            "period": int(self.period),
            "due": True,
            "selected_action": action.value,
            "inner": getattr(self.inner, "last_decision_info", None),
        }
        return action


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO VLA evaluation for SafeLoop.")
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
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--episodes", "--episodes-per-task", dest="episodes", type=int, default=1)
    parser.add_argument("--init-state-start", type=int, default=0)
    parser.add_argument(
        "--fixed-init-state-index",
        type=int,
        help="Reuse one LIBERO initial state while episode seeds advance; intended for paired stress tests.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-rollout-steps", "--max-steps", dest="max_rollout_steps", type=int, default=120)
    parser.add_argument("--allow-extended-rollout", action="store_true")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mode", choices=["baseline", "teacher", "rl"], default="baseline")
    parser.add_argument("--disable-safeloop", action="store_true")
    parser.add_argument(
        "--manual-hazard-labels",
        action="store_true",
        help="Disable automatic hazard counting in evaluation summaries.",
    )
    parser.add_argument("--predictor", choices=["constant", "action-norm", "qwen-multitask"], default="action-norm")
    parser.add_argument("--constant-risk", nargs=4, type=float, default=[0.0, 1.0, 0.0, 1.0])
    parser.add_argument("--qwen-model-dir", "--qwen-model-path", dest="qwen_model_dir", type=Path, default=env_path("QWEN_MODEL"))
    parser.add_argument("--qwen-checkpoint", "--qwen-head-path", dest="qwen_checkpoint", type=Path)
    parser.add_argument("--qwen-lora-adapter", type=Path)
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-dtype", default="float32")
    parser.add_argument("--qwen-tau", "--future-window", dest="qwen_tau", type=int, default=50)
    parser.add_argument("--decision-checkpoint", "--decider-checkpoint", dest="decision_checkpoint", type=Path)
    parser.add_argument("--decision-device", default="cpu")
    parser.add_argument("--decision-period", "--decision-interval", dest="decision_period", type=int, default=5)
    parser.add_argument("--record-probability", type=float, default=0.12)
    parser.add_argument("--record-max-risk-score", type=float)
    parser.add_argument("--record-max-current-body-probability", type=float)
    parser.add_argument("--record-max-current-object-probability", type=float)
    parser.add_argument("--record-initial-safe-anchor", action="store_true")
    parser.add_argument("--initial-anchor-rollback-min-current-body-probability", type=float)
    parser.add_argument("--initial-anchor-rollback-min-current-object-probability", type=float)
    parser.add_argument("--auto-record-safe-anchors", action="store_true")
    parser.add_argument("--auto-record-min-interval", type=int, default=20)
    parser.add_argument("--auto-record-max-risk-score", type=float, default=0.4)
    parser.add_argument("--auto-record-max-current-body-probability", type=float, default=0.3)
    parser.add_argument("--auto-record-max-current-object-probability", type=float, default=0.45)
    parser.add_argument("--stuck-fallback", action="store_true")
    parser.add_argument("--stuck-window-steps", type=int, default=70)
    parser.add_argument("--stuck-min-low-motion-steps", type=int, default=50)
    parser.add_argument("--stuck-max-step-displacement", type=float, default=0.0025)
    parser.add_argument("--stuck-max-window-displacement", type=float, default=0.025)
    parser.add_argument("--rollback-probability", type=float, default=0.55)
    parser.add_argument("--rollback-tth", type=float, default=0.35)
    parser.add_argument("--min-record-interval", type=int, default=30)
    parser.add_argument("--rollback-cooldown", type=int, default=15)
    parser.add_argument("--min-rollback-step", type=int, default=0)
    parser.add_argument("--max-rollbacks-per-episode", type=int)
    parser.add_argument("--rollback-gate-current-threshold", type=float, default=1.01)
    parser.add_argument("--rollback-gate-current-body-threshold", type=float)
    parser.add_argument("--rollback-gate-current-object-threshold", type=float)
    parser.add_argument("--rollback-gate-max-current-object-probability", type=float)
    parser.add_argument("--rollback-gate-future-probability", type=float, default=1.01)
    parser.add_argument("--rollback-gate-future-body-probability", type=float)
    parser.add_argument("--rollback-gate-future-object-probability", type=float)
    parser.add_argument("--rollback-gate-future-tth", type=float, default=0.0)
    parser.add_argument("--rollback-gate-future-object-tth", type=float)
    parser.add_argument("--rollback-gate-allow-risk-override", action="store_true")
    parser.add_argument("--rollback-gate-allow-stuck-object-override", action="store_true")
    parser.add_argument("--stuck-override-min-rollback-probability", type=float, default=0.95)
    parser.add_argument("--stuck-rollback-target-max-age", type=int)
    parser.add_argument("--rollback-target-safe-score-threshold", type=float, default=0.6)
    parser.add_argument("--rollback-target-min-age", type=int, default=30)
    parser.add_argument("--rollback-target-max-age", type=int, default=120)
    parser.add_argument("--rollback-target-require-safe", action="store_true")
    parser.add_argument("--rollback-target-prefer-recent-safe", action="store_true")
    parser.add_argument("--rollback-mode", choices=["motion-plan", "restore"], default="motion-plan")
    parser.add_argument("--motion-execution-mode", choices=["pd", "kinematic"], default="kinematic")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-steps-per-waypoint", type=int, default=120)
    parser.add_argument("--kinematic-substeps-per-waypoint", type=int, default=2)
    parser.add_argument("--allow-restore-fallback", action="store_true")
    parser.add_argument("--render-camera", default="agentview")
    parser.add_argument("--save-video-episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--trace-out", "--decision-debug-jsonl-out", dest="trace_out", type=Path)
    parser.add_argument("--qwen-rollout-jsonl-out", type=Path)
    parser.add_argument("--qwen-rollout-image-root", type=Path)
    parser.add_argument("--qwen-rollout-stride", type=int, default=40)
    parser.add_argument("--qwen-rollout-prehazard-stride", type=int, default=0)
    parser.add_argument("--qwen-rollout-run-tag")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.disable_safeloop:
        args.mode = "baseline"
    args.task_ids = _parse_task_ids(args.task_ids) if args.task_ids else [int(args.task_id)]
    if args.out is None:
        if args.output_dir is None:
            parser.error("one of --out or --output-dir is required")
        args.out = args.output_dir / "summary.json"
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


def make_decider(args: argparse.Namespace):
    if args.mode == "teacher":
        decider = RuleBasedDecider(
            record_probability=args.record_probability,
            rollback_probability=args.rollback_probability,
            rollback_tth=args.rollback_tth,
            min_record_interval=args.min_record_interval,
            rollback_cooldown=args.rollback_cooldown,
        )
    elif args.mode == "rl":
        if not args.decision_checkpoint:
            raise ValueError("--decision-checkpoint is required for --mode rl")
        decider = RLPolicyDecider(
            checkpoint_path=args.decision_checkpoint,
            device=args.decision_device,
            history_length=3,
            min_record_interval=args.min_record_interval,
            rollback_cooldown=args.rollback_cooldown,
            max_steps=args.max_rollout_steps,
        )
    else:
        raise ValueError("baseline mode does not use a decider")
    if (
        args.max_rollbacks_per_episode is not None
        or args.rollback_gate_current_threshold <= 1.0
        or (args.rollback_gate_current_body_threshold is not None and args.rollback_gate_current_body_threshold <= 1.0)
        or (args.rollback_gate_current_object_threshold is not None and args.rollback_gate_current_object_threshold <= 1.0)
        or args.rollback_gate_future_probability <= 1.0
        or (
            args.rollback_gate_future_body_probability is not None
            and args.rollback_gate_future_body_probability <= 1.0
        )
        or (
            args.rollback_gate_future_object_probability is not None
            and args.rollback_gate_future_object_probability <= 1.0
        )
    ):
        decider = RollbackGateDecider(
            decider,
            max_rollbacks_per_episode=args.max_rollbacks_per_episode,
            current_hazard_threshold=args.rollback_gate_current_threshold,
            current_body_threshold=args.rollback_gate_current_body_threshold,
            current_object_threshold=args.rollback_gate_current_object_threshold,
            max_current_object_probability=args.rollback_gate_max_current_object_probability,
            future_probability_threshold=args.rollback_gate_future_probability,
            future_body_probability_threshold=args.rollback_gate_future_body_probability,
            future_object_probability_threshold=args.rollback_gate_future_object_probability,
            future_tth_threshold=args.rollback_gate_future_tth,
            future_object_tth_threshold=args.rollback_gate_future_object_tth,
            allow_risk_override=args.rollback_gate_allow_risk_override,
            allow_stuck_object_override=args.rollback_gate_allow_stuck_object_override,
            stuck_override_min_rollback_probability=args.stuck_override_min_rollback_probability,
            min_rollback_step=args.min_rollback_step,
        )
    return PeriodicDecider(decider, period=max(1, args.decision_period))


def make_rollback_executor(args: argparse.Namespace, frames: list[np.ndarray] | None = None):
    if args.rollback_mode == "restore":
        return None
    return MotionPlanningRollbackExecutor(
        max_joint_step=args.max_joint_step,
        motion_execution_mode=args.motion_execution_mode,
        max_steps_per_waypoint=args.max_steps_per_waypoint,
        kinematic_substeps_per_waypoint=args.kinematic_substeps_per_waypoint,
        render_height=args.resize_size,
        render_width=args.resize_size,
        render_camera=args.render_camera,
        allow_restore_fallback=args.allow_restore_fallback,
        frames=frames,
    )


def run_episode(
    env,
    task,
    init_states,
    policy,
    args: argparse.Namespace,
    episode_index: int,
    predictor=None,
) -> dict:
    np.random.seed(args.seed + episode_index)
    env.seed(args.seed + episode_index)
    env.reset()
    if args.fixed_init_state_index is None:
        init_index = (args.init_state_start + episode_index) % len(init_states)
    else:
        init_index = int(args.fixed_init_state_index) % len(init_states)
    obs = env.set_init_state(init_states[init_index])
    use_hazard_oracle = (not args.manual_hazard_labels) or args.qwen_rollout_jsonl_out is not None
    oracle = LiberoHazardOracle(env.sim) if use_hazard_oracle else None
    action_plan: collections.deque[np.ndarray] = collections.deque()
    frames: list[np.ndarray] = []
    interventions = {"noop": 0, "record": 0, "rollback": 0}
    rollback_planned = 0
    rollback_failed = 0
    rollback_rendered_frames = 0
    nominal_after_rollback = 0
    saw_rollback = False
    hazard_steps = {"body": 0, "object": 0, "stuck": 0, "any": 0}
    hazard_events = {"body": 0, "object": 0, "stuck": 0, "any": 0}
    trace_records: list[dict] = []
    qwen_rollout_samples: list[dict] = []
    qwen_step_candidates: dict[int, dict] = {}
    qwen_hazard_timeline = []
    qwen_frame_history: collections.deque = collections.deque(maxlen=3)
    prev_body = False
    prev_object = False
    prev_stuck = False
    reward = 0.0
    done = False
    info = {}
    controller = None
    if args.mode != "baseline":
        predictor = predictor if predictor is not None else make_predictor(args)
        if hasattr(predictor, "reset"):
            predictor.reset()
        controller = SafeLoopController(
            predictor=predictor,
            decider=make_decider(args),
            rollback_executor=make_rollback_executor(
                args,
                frames=frames if episode_index < args.save_video_episodes else None,
            ),
            max_history=3,
            max_steps=args.max_rollout_steps,
            prediction_period=args.decision_period,
            rollback_target_safe_score_threshold=args.rollback_target_safe_score_threshold,
            rollback_target_min_age=args.rollback_target_min_age,
            rollback_target_max_age=args.rollback_target_max_age,
            rollback_target_require_safe=args.rollback_target_require_safe,
            rollback_target_prefer_recent_safe=args.rollback_target_prefer_recent_safe,
            stuck_rollback_target_max_age=args.stuck_rollback_target_max_age,
            record_max_risk_score=args.record_max_risk_score,
            record_max_current_body_probability=args.record_max_current_body_probability,
            record_max_current_object_probability=args.record_max_current_object_probability,
            record_initial_safe_anchor=args.record_initial_safe_anchor,
            initial_anchor_rollback_min_current_body_probability=(
                args.initial_anchor_rollback_min_current_body_probability
            ),
            initial_anchor_rollback_min_current_object_probability=(
                args.initial_anchor_rollback_min_current_object_probability
            ),
            auto_record_safe_anchors=args.auto_record_safe_anchors,
            auto_record_min_interval=args.auto_record_min_interval,
            auto_record_max_risk_score=args.auto_record_max_risk_score,
            auto_record_max_current_body_probability=args.auto_record_max_current_body_probability,
            auto_record_max_current_object_probability=args.auto_record_max_current_object_probability,
            stuck_fallback=args.stuck_fallback,
            stuck_window_steps=args.stuck_window_steps,
            stuck_min_low_motion_steps=args.stuck_min_low_motion_steps,
            stuck_max_step_displacement=args.stuck_max_step_displacement,
            stuck_max_window_displacement=args.stuck_max_window_displacement,
        )

    suite_max_steps = max_steps_for_suite(args.benchmark)
    max_steps = int(args.max_rollout_steps) if args.allow_extended_rollout else min(args.max_rollout_steps, suite_max_steps)
    total_steps = 0
    for step in range(max_steps + args.num_steps_wait):
        save_video = episode_index < args.save_video_episodes
        if save_video:
            frames.append(render_frame(env, args.render_camera, args.resize_size))

        if step < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            continue

        if not action_plan:
            policy_input = policy_observation(
                obs,
                task.language,
                args.resize_size,
                policy_backend=args.policy_backend,
                benchmark=args.benchmark,
            )
            action_chunk = np.asarray(policy.infer(policy_input)["actions"])
            if len(action_chunk) < args.replan_steps:
                raise RuntimeError(f"policy returned {len(action_chunk)} actions, need {args.replan_steps}")
            action_chunk = action_chunk[: args.replan_steps]
            action_plan.extend(action_chunk)
        action = action_plan.popleft()
        pre_action_signals = (
            oracle.read(success=False, task_reward=0.0)
            if oracle is not None
            else OnlineStepSignals(success=False, task_reward=0.0)
        )
        qwen_hazard_timeline.append(pre_action_signals)
        if args.qwen_rollout_jsonl_out is not None:
            qwen_frame_history.append(live_observation_to_frame(obs))
            stride_due = total_steps % max(1, int(args.qwen_rollout_stride)) == 0
            dense_enabled = int(args.qwen_rollout_prehazard_stride) > 0
            if stride_due or dense_enabled:
                prompt_text, prompt_images = build_live_qwen_prompt(
                    history=list(qwen_frame_history),
                    instruction=task.language,
                    proposed_action=action.tolist(),
                    current_global_step=total_steps,
                    history_length=3,
                    tau=args.qwen_tau,
                )
                sample = {
                    "sample_index": int(total_steps),
                    "text": prompt_text.replace(QWEN_IMAGE_TOKEN, "<image>"),
                    "images": [image.copy() for image in prompt_images],
                    "current_body": float(pre_action_signals.body_hazard or pre_action_signals.stuck_hazard),
                    "current_object": float(pre_action_signals.object_hazard),
                    "metadata": {
                        "source": "online_stride" if stride_due else "online_candidate",
                        "current_body_collision": bool(pre_action_signals.body_hazard),
                        "current_stuck": bool(pre_action_signals.stuck_hazard),
                    },
                }
                if dense_enabled:
                    qwen_step_candidates[int(total_steps)] = sample
                if stride_due:
                    qwen_rollout_samples.append(sample)

        if controller is None:
            obs, reward, done, info = env.step(action.tolist())
            intervention = Intervention.NOOP
            executed_nominal = True
            safeloop_info = {"intervention": intervention.value}
        else:
            result = controller.step(env, obs, action.tolist(), instruction=task.language)
            obs, reward, done, info = result.observation, result.reward, result.done, result.info
            intervention = result.intervention
            executed_nominal = result.executed_nominal_action
            safeloop_info = dict((info or {}).get("safeloop") or {})
            if intervention == Intervention.ROLLBACK:
                action_plan.clear()
                saw_rollback = True
                if safeloop_info.get("planned"):
                    rollback_planned += 1
                if safeloop_info.get("safe") is False or safeloop_info.get("reached") is False:
                    rollback_failed += 1
                rollback_rendered_frames += int(safeloop_info.get("rendered_frames") or 0)
            elif saw_rollback and executed_nominal:
                nominal_after_rollback += 1

        success = bool(env.check_success())
        signals = (
            oracle.read(success=success, task_reward=float(reward))
            if oracle is not None
            else OnlineStepSignals(success=success, task_reward=float(reward))
        )
        hazard_steps["body"] += int(signals.body_hazard)
        hazard_steps["object"] += int(signals.object_hazard)
        hazard_steps["stuck"] += int(signals.stuck_hazard)
        hazard_steps["any"] += int(signals.any_hazard)
        hazard_events["body"] += int(signals.body_hazard and not prev_body)
        hazard_events["object"] += int(signals.object_hazard and not prev_object)
        hazard_events["stuck"] += int(signals.stuck_hazard and not prev_stuck)
        hazard_events["any"] += int(signals.hazard_event)
        prev_body = signals.body_hazard
        prev_object = signals.object_hazard
        prev_stuck = signals.stuck_hazard
        interventions[intervention.value] += 1
        total_steps += 1
        if args.trace_out is not None:
            trace_records.append(
                {
                    "episode": int(episode_index),
                    "init_state_index": int(init_index),
                    "benchmark": str(args.benchmark),
                    "task_id": int(args.task_id),
                    "task": str(task.name),
                    "instruction": str(task.language),
                    "step_index": int(total_steps - 1),
                    "intervention": intervention.value,
                    "executed_nominal": bool(executed_nominal),
                    "success": bool(success),
                    "reward": float(reward),
                    "body_hazard": bool(signals.body_hazard),
                    "object_hazard": bool(signals.object_hazard),
                    "stuck_hazard": bool(signals.stuck_hazard),
                    "hazard_event": bool(signals.hazard_event),
                    "eef_speed": float(signals.eef_speed),
                    "arm_contact_count": int(signals.arm_contact_count),
                    "object_hazard_reasons": oracle.last_object_hazard_reasons() if oracle is not None else [],
                    "memory_size": int(len(controller.memory)) if controller is not None else 0,
                    "rollback_planned_total": int(rollback_planned),
                    "rollback_failed_total": int(rollback_failed),
                    "rollback_rendered_frames_total": int(rollback_rendered_frames),
                    "nominal_after_rollback_total": int(nominal_after_rollback),
                    "safeloop": safeloop_info,
                }
            )
        if done or success:
            done = True
            break

    if args.trace_out is not None and trace_records:
        with args.trace_out.open("a", encoding="utf-8") as trace_file:
            for record in trace_records:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
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
            episode_index=episode_index,
            samples=qwen_rollout_samples,
            hazard_timeline=qwen_hazard_timeline,
        )

    video_path = None
    if frames:
        video_dir = args.video_dir or args.out.parent
        video_dir.mkdir(parents=True, exist_ok=True)
        video = write_frames(
            frames,
            video_dir / f"{args.out.stem}_task{int(args.task_id):02d}_ep{episode_index:03d}.mp4",
            fps=args.fps,
        )
        video_path = str(video.path)
    return {
        "episode": episode_index,
        "init_state_index": init_index,
        "task": task.name,
        "instruction": task.language,
        "success": bool(done),
        "reward": float(reward),
        "steps": int(total_steps),
        "max_steps": int(max_steps),
        "suite_max_steps": int(suite_max_steps),
        "allow_extended_rollout": bool(args.allow_extended_rollout),
        "effective_control_steps": int(total_steps + rollback_rendered_frames),
        "hazard_steps": hazard_steps,
        "hazard_events": hazard_events,
        "hazard_label_source": "libero_oracle" if oracle is not None else "manual_review_required",
        "events_per_1k": float(1000.0 * hazard_events["any"] / max(total_steps, 1)),
        "interventions": interventions,
        "rollback_planned": int(rollback_planned),
        "rollback_failed": int(rollback_failed),
        "rollback_rendered_frames": int(rollback_rendered_frames),
        "nominal_after_rollback": int(nominal_after_rollback),
        "video": video_path,
        "last_info": info,
    }


def write_qwen_rollout_samples(args: argparse.Namespace, task, episode_index: int, samples: list[dict], hazard_timeline) -> None:
    image_root = args.qwen_rollout_image_root or args.qwen_rollout_jsonl_out.parent
    tau = int(getattr(args, "qwen_tau", 50))
    run_tag = _safe_filename_component(getattr(args, "qwen_rollout_run_tag", None) or "")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    run_id = f"online_rollout/{args.benchmark}_task{int(args.task_id):02d}_seed{int(args.seed)}_ep{int(episode_index):03d}{tag_suffix}"
    image_dir = image_root / run_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    args.qwen_rollout_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    sample_index_counts = collections.Counter(int(sample["sample_index"]) for sample in samples)
    duplicate_sample_indices = {
        sample_index for sample_index, count in sample_index_counts.items() if count > 1
    }
    source_occurrences: dict[tuple[int, str], int] = collections.defaultdict(int)
    with args.qwen_rollout_jsonl_out.open("a", encoding="utf-8") as handle:
        for sample in samples:
            sample_index = int(sample["sample_index"])
            labels = _future_labels_from_timeline(hazard_timeline, sample_index, tau=tau)
            labels.update(sample.get("labels") or {})
            labels["current_body"] = float(sample["current_body"])
            labels["current_object"] = float(sample["current_object"])
            image_stem = _sample_image_stem(
                sample,
                sample_index=sample_index,
                duplicate_sample_indices=duplicate_sample_indices,
                source_occurrences=source_occurrences,
            )
            image_paths = []
            for image_index, image in enumerate(sample["images"]):
                relative_path = f"{run_id}/images/{image_stem}_{image_index:02d}.jpg"
                image.save(image_root / relative_path, quality=92)
                image_paths.append(relative_path)
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": sample["text"]},
                            {"role": "assistant", "content": _assistant_label_text(labels, tau=tau)},
                        ],
                        "images": image_paths,
                        "labels": labels,
                        "metadata": {
                            "task": getattr(task, "name", ""),
                            "instruction": getattr(task, "language", ""),
                            "benchmark": args.benchmark,
                            "task_id": int(args.task_id),
                            "episode": int(episode_index),
                            "sample_index": sample_index,
                            **(sample.get("metadata") or {}),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _safe_filename_component(value: object) -> str:
    text = str(value or "sample")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe[:48] or "sample"


def _sample_image_stem(
    sample: dict,
    *,
    sample_index: int,
    duplicate_sample_indices: set[int],
    source_occurrences: dict[tuple[int, str], int],
) -> str:
    if int(sample_index) not in duplicate_sample_indices:
        return f"{int(sample_index):05d}"
    metadata = sample.get("metadata") or {}
    source = _safe_filename_component(metadata.get("source") or "sample")
    key = (int(sample_index), source)
    occurrence = int(source_occurrences[key])
    source_occurrences[key] = occurrence + 1
    if occurrence > 0:
        source = f"{source}_{occurrence:02d}"
    return f"{int(sample_index):05d}_{source}"


def _future_labels_from_timeline(hazard_timeline, start_index: int, tau: int) -> dict[str, float]:
    end = min(len(hazard_timeline), int(start_index) + int(tau) + 1)
    body_tth = None
    object_tth = None
    for offset, signals in enumerate(hazard_timeline[int(start_index) + 1 : end], start=1):
        body_or_stuck = bool(signals.body_hazard or signals.stuck_hazard)
        if body_tth is None and body_or_stuck:
            body_tth = offset
        if object_tth is None and bool(signals.object_hazard):
            object_tth = offset
    return {
        "future_body": float(body_tth is not None),
        "future_body_tth": float(body_tth / max(int(tau), 1)) if body_tth is not None else 1.0,
        "future_object": float(object_tth is not None),
        "future_object_tth": float(object_tth / max(int(tau), 1)) if object_tth is not None else 1.0,
    }


def _assistant_label_text(labels: dict[str, float], tau: int) -> str:
    body_tth = int(round(float(labels["future_body_tth"]) * int(tau))) if labels["future_body"] else -1
    object_tth = int(round(float(labels["future_object_tth"]) * int(tau))) if labels["future_object"] else -1
    return f"{int(labels['future_body'])} {body_tth}\n{int(labels['future_object'])} {object_tth}"


def summarize(reports: list[dict]) -> dict:
    steps = sum(int(item["steps"]) for item in reports)
    aggregate_events = {
        key: sum(int(item["hazard_events"].get(key, 0)) for item in reports)
        for key in ("body", "object", "stuck", "any")
    }
    aggregate_steps = {
        key: sum(int(item["hazard_steps"].get(key, 0)) for item in reports)
        for key in ("body", "object", "stuck", "any")
    }
    interventions = {
        key: sum(int(item["interventions"][key]) for item in reports)
        for key in ("noop", "record", "rollback")
    }
    rollback_rendered_frames = int(sum(int(item.get("rollback_rendered_frames", 0)) for item in reports))
    per_episode_effective_steps = [
        int(item.get("effective_control_steps", int(item["steps"]) + int(item.get("rollback_rendered_frames", 0))))
        for item in reports
    ]
    effective_control_steps = int(sum(per_episode_effective_steps))
    hazard_label_sources = sorted({str(item.get("hazard_label_source", "unknown")) for item in reports})
    return {
        "episodes": len(reports),
        "success_rate": float(np.mean([bool(item["success"]) for item in reports])) if reports else 0.0,
        "mean_steps": float(np.mean([int(item["steps"]) for item in reports])) if reports else 0.0,
        "mean_effective_control_steps": (
            float(np.mean(per_episode_effective_steps))
            if reports
            else 0.0
        ),
        "effective_control_steps": effective_control_steps,
        "hazard_events": aggregate_events,
        "hazard_steps": aggregate_steps,
        "hazard_label_source": hazard_label_sources[0] if len(hazard_label_sources) == 1 else hazard_label_sources,
        "manual_hazard_review_required": "manual_review_required" in hazard_label_sources,
        "events_per_1k": float(1000.0 * aggregate_events["any"] / max(steps, 1)),
        "body_event_rate": float(aggregate_events["body"] / max(len(reports), 1)),
        "object_event_rate": float(aggregate_events["object"] / max(len(reports), 1)),
        "stuck_event_rate": float(aggregate_events["stuck"] / max(len(reports), 1)),
        "interventions": interventions,
        "rollback_planned": int(sum(int(item["rollback_planned"]) for item in reports)),
        "rollback_failed": int(sum(int(item["rollback_failed"]) for item in reports)),
        "rollback_rendered_frames": rollback_rendered_frames,
        "nominal_after_rollback": int(sum(int(item["nominal_after_rollback"]) for item in reports)),
    }


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    if args.trace_out is not None:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text("", encoding="utf-8")
    configure_paths(args)
    policy = make_policy(args)
    predictor = make_predictor(args) if args.mode != "baseline" else None
    all_reports = []
    task_results = []
    for task_id in args.task_ids:
        args.task_id = int(task_id)
        env, task, init_states = make_env(args)
        try:
            reports = [
                run_episode(env, task, init_states, policy, args, idx, predictor=predictor)
                for idx in range(args.episodes)
            ]
        finally:
            env.close()
        all_reports.extend(reports)
        task_results.append(
            {
                "task_id": int(task_id),
                "task": task.name,
                "instruction": task.language,
                "summary": summarize(reports),
                "episodes": reports,
            }
        )
    result = {
        "mode": args.mode,
        "policy_backend": args.policy_backend,
        "predictor": args.predictor,
        "benchmark": args.benchmark,
        "task_id": args.task_ids[0] if len(args.task_ids) == 1 else None,
        "task_ids": args.task_ids,
        "seed": args.seed,
        "summary": summarize(all_reports),
        "tasks": task_results,
        "episodes": all_reports,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
