from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard.compat import torch_load_compat
from safety_guard.libero_motion import (
    JointPathExecutor,
    LinearJointPlanner,
    build_robot_geom_sets,
    check_collision_filtered,
    find_arm_indices,
    make_libero_state_validator,
    render_joint_path_kinematic,
)
from safety_guard.video import write_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a collision-filtered LIBERO joint-space rollback trajectory.")
    parser.add_argument("--libero-root", type=Path, default=Path(os.environ["LIBERO_ROOT"]) if os.environ.get("LIBERO_ROOT") else None)
    parser.add_argument("--benchmark", default="libero_10", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"])
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--render-camera", default="agentview")
    parser.add_argument("--joint-offsets", nargs="*", type=float, help="Offsets applied to the safe arm qpos before planning back.")
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-steps-per-waypoint", type=int, default=120)
    parser.add_argument("--kp", type=float, default=50.0)
    parser.add_argument("--kd", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--velocity-tolerance", type=float, default=0.08)
    parser.add_argument("--execution-mode", choices=["kinematic", "pd"], default="kinematic")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("outputs/rollback_demo.mp4"))
    return parser.parse_args()


def configure_libero(libero_root: Path) -> None:
    if libero_root is None:
        raise ValueError("--libero-root or LIBERO_ROOT is required")
    abs_root = str(libero_root.resolve())
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(abs_root)
    if abs_root not in sys.path:
        sys.path.insert(0, abs_root)


def make_env(args: argparse.Namespace):
    configure_libero(args.libero_root)
    import torch
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark)()
    task = benchmark.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    init_states = torch_load_compat(str(init_states_path), torch.load)
    return env, task, init_states


def default_offsets(num_joints: int) -> np.ndarray:
    base = np.asarray([0.28, -0.22, 0.16, -0.12, 0.10, -0.08, 0.05], dtype=np.float32)
    if num_joints <= len(base):
        return base[:num_joints].copy()
    return np.pad(base, (0, num_joints - len(base))).astype(np.float32)


def render_initial_frame(env, size: int, camera: str) -> np.ndarray:
    frame = env.sim.render(height=size, width=size, camera_name=camera)
    return np.ascontiguousarray(frame[::-1, ::-1])


def run_rollback_render(args: argparse.Namespace) -> dict:
    env, task, init_states = make_env(args)
    frames: list[np.ndarray] = []
    try:
        env.seed(args.seed)
        env.reset()
        env.set_init_state(init_states[args.init_state_index % len(init_states)])
        arm_joint_indices, actuator_indices = find_arm_indices(env)
        safe_qpos = env.sim.data.qpos[arm_joint_indices].copy()

        offsets = np.asarray(args.joint_offsets, dtype=np.float32) if args.joint_offsets else default_offsets(len(arm_joint_indices))
        if offsets.shape[0] != len(arm_joint_indices):
            raise ValueError(f"expected {len(arm_joint_indices)} joint offsets, got {offsets.shape[0]}")
        start_qpos = safe_qpos + offsets
        env.sim.data.qpos[arm_joint_indices] = start_qpos
        env.sim.data.qvel[arm_joint_indices] = 0
        env.sim.forward()
        frames.append(render_initial_frame(env, args.render_size, args.render_camera))

        _, arm_geoms, gripper_geoms = build_robot_geom_sets(env.sim, "robot0")
        validator = make_libero_state_validator(
            env.sim,
            arm_joint_indices,
            lambda sim: check_collision_filtered(sim, arm_geoms, gripper_geoms),
        )
        planner = LinearJointPlanner(max_joint_step=args.max_joint_step)
        plan = planner.plan(start_qpos, safe_qpos, validator)
        if not plan.safe:
            raise RuntimeError(f"rollback plan rejected: {plan.reason}")

        if args.execution_mode == "kinematic":
            reached = render_joint_path_kinematic(
                env,
                plan.path,
                arm_joint_indices,
                frames=frames,
                render_height=args.render_size,
                render_width=args.render_size,
                render_camera=args.render_camera,
            )
        else:
            executor = JointPathExecutor(
                kp=args.kp,
                kd=args.kd,
                tolerance=args.tolerance,
                velocity_tolerance=args.velocity_tolerance,
                max_steps_per_waypoint=args.max_steps_per_waypoint,
                render_height=args.render_size,
                render_width=args.render_size,
                render_camera=args.render_camera,
            )
            reached = executor.execute(env, plan.path, arm_joint_indices, actuator_indices, frames)
        video = write_frames(frames, args.output, fps=args.fps)
        return {
            "task": task.name,
            "instruction": task.language,
            "safe": bool(plan.safe),
            "reached": bool(reached),
            "waypoints": len(plan.path),
            "frames": video.frame_count,
            "output": str(video.path),
            "output_mode": video.mode,
        }
    finally:
        env.close()


def main() -> None:
    print(json.dumps(run_rollback_render(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
