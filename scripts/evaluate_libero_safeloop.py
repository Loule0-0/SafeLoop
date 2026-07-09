from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard import RuleBasedDecider, SafeLoopController
from safety_guard.compat import torch_load_compat
from safety_guard.libero_motion import MotionPlanningRollbackExecutor
from safety_guard.predictors import ActionNormRiskPredictor, ConstantRiskPredictor
from safety_guard.qwen_multitask import QwenMultitaskSafetyPredictor
from safety_guard.rl_policy_decider import RLPolicyDecider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SafeLoop around a single LIBERO environment.")
    parser.add_argument("--libero-root", required=True, help="Path to the existing LIBERO checkout.")
    parser.add_argument(
        "--benchmark",
        default="libero_10",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"],
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--policy-callable", help="Optional module:function returning a 7-D action.")
    parser.add_argument("--predictor", choices=["constant", "action-norm", "qwen-multitask"], default="constant")
    parser.add_argument("--constant-risk", nargs=4, type=float, default=[0.0, 1.0, 0.0, 1.0])
    parser.add_argument("--qwen-model-dir")
    parser.add_argument("--qwen-checkpoint")
    parser.add_argument("--qwen-lora-adapter")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--decider", choices=["rule", "rl"], default="rule")
    parser.add_argument("--decision-checkpoint")
    parser.add_argument("--decision-device", default="cpu")
    parser.add_argument("--decision-history-length", type=int, default=3)
    parser.add_argument("--record-probability", type=float, default=0.05)
    parser.add_argument("--rollback-probability", type=float, default=0.8)
    parser.add_argument("--rollback-tth", type=float, default=0.25)
    parser.add_argument("--min-record-interval", type=int, default=1)
    parser.add_argument("--rollback-cooldown", type=int, default=0)
    parser.add_argument("--rollback-mode", choices=["motion-plan", "restore"], default="motion-plan")
    parser.add_argument("--motion-execution-mode", choices=["pd", "kinematic"], default="kinematic")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-steps-per-waypoint", type=int, default=120)
    parser.add_argument("--allow-restore-fallback", action="store_true")
    return parser.parse_args()


def import_policy_callable(spec: str):
    module_name, sep, function_name = spec.partition(":")
    if not sep:
        raise ValueError("--policy-callable must be formatted as module:function")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def make_base_policy(args: argparse.Namespace):
    if args.policy_callable:
        return import_policy_callable(args.policy_callable)

    rng = np.random.default_rng(args.seed)

    if args.policy == "zero":
        return lambda observation, instruction: np.zeros(7, dtype=np.float32)
    if args.policy == "random":
        return lambda observation, instruction: rng.uniform(-0.05, 0.05, size=7).astype(np.float32)
    raise ValueError(f"Unsupported policy {args.policy}")


def make_predictor(args: argparse.Namespace):
    if args.predictor == "constant":
        return ConstantRiskPredictor(args.constant_risk)
    if args.predictor == "action-norm":
        return ActionNormRiskPredictor()
    if args.predictor == "qwen-multitask":
        if not args.qwen_model_dir or not args.qwen_checkpoint:
            raise ValueError("--qwen-model-dir and --qwen-checkpoint are required for --predictor qwen-multitask")
        return QwenMultitaskSafetyPredictor(
            model_dir=args.qwen_model_dir,
            checkpoint_path=args.qwen_checkpoint,
            lora_adapter_path=args.qwen_lora_adapter,
            device=args.qwen_device,
            torch_dtype=args.qwen_dtype,
            history_length=args.decision_history_length,
        )
    raise ValueError(f"Unsupported predictor {args.predictor}")


def make_decider(args: argparse.Namespace):
    if args.decider == "rule":
        return RuleBasedDecider(
            record_probability=args.record_probability,
            rollback_probability=args.rollback_probability,
            rollback_tth=args.rollback_tth,
            min_record_interval=args.min_record_interval,
            rollback_cooldown=args.rollback_cooldown,
        )
    if args.decider == "rl":
        if not args.decision_checkpoint:
            raise ValueError("--decision-checkpoint is required for --decider rl")
        return RLPolicyDecider(
            checkpoint_path=args.decision_checkpoint,
            device=args.decision_device,
            history_length=args.decision_history_length,
            min_record_interval=args.min_record_interval,
            rollback_cooldown=args.rollback_cooldown,
            max_steps=args.max_steps,
        )
    raise ValueError(f"Unsupported decider {args.decider}")


def make_rollback_executor(args: argparse.Namespace):
    if args.rollback_mode == "restore":
        return None
    return MotionPlanningRollbackExecutor(
        max_joint_step=args.max_joint_step,
        motion_execution_mode=args.motion_execution_mode,
        max_steps_per_waypoint=args.max_steps_per_waypoint,
        render_height=args.camera_height,
        render_width=args.camera_width,
        allow_restore_fallback=args.allow_restore_fallback,
    )


def configure_libero(libero_root: str) -> None:
    abs_root = os.path.abspath(libero_root)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(abs_root)
    if abs_root not in sys.path:
        sys.path.insert(0, abs_root)


def make_env(args: argparse.Namespace):
    configure_libero(args.libero_root)
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import torch

    benchmark = get_benchmark(args.benchmark)()
    task = benchmark.get_task(args.task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    init_states = torch_load_compat(init_states_path, torch.load)
    return env, task, init_states


def run_episode(env, task, init_state, args: argparse.Namespace, episode_index: int) -> dict:
    env.seed(args.seed + episode_index)
    observation = env.reset()
    observation = env.set_init_state(init_state)
    policy = make_base_policy(args)
    controller = SafeLoopController(
        predictor=make_predictor(args),
        decider=make_decider(args),
        rollback_executor=make_rollback_executor(args),
        max_history=args.decision_history_length,
        max_steps=args.max_steps,
    )

    interventions = {"noop": 0, "record": 0, "rollback": 0}
    done = False
    reward = 0.0
    for _ in range(args.max_steps):
        action = policy(observation, task.language)
        result = controller.step(env, observation, action, instruction=task.language)
        observation = result.observation
        reward = result.reward
        done = result.done or bool(env.check_success())
        interventions[result.intervention.value] += 1
        if done:
            break

    return {
        "episode": episode_index,
        "task": task.name,
        "instruction": task.language,
        "success": bool(done),
        "reward": reward,
        "steps": controller.step_index,
        "memory_size": len(controller.memory),
        "interventions": interventions,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    env, task, init_states = make_env(args)
    try:
        reports = []
        for episode_index in range(args.episodes):
            init_state = init_states[episode_index % len(init_states)]
            reports.append(run_episode(env, task, init_state, args, episode_index))
    finally:
        env.close()

    summary = {
        "config": vars(args),
        "episodes": reports,
        "success_rate": float(np.mean([item["success"] for item in reports])) if reports else 0.0,
    }
    print(asdict(summary) if hasattr(summary, "__dataclass_fields__") else summary)


if __name__ == "__main__":
    main()
