from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard.compat import torch_load_compat

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def configure_paths(args: argparse.Namespace) -> None:
    paths: list[Path] = [Path(args.project_root)]
    libero_root = getattr(args, "libero_root", None)
    openpi_root = getattr(args, "openpi_root", None)
    if libero_root is not None:
        paths.append(Path(libero_root))
    if openpi_root is not None:
        openpi_root = Path(openpi_root)
        paths.append(openpi_root / "packages" / "openpi-client" / "src")
        if getattr(args, "policy_mode", None) == "inprocess":
            paths.extend(
                [
                    openpi_root / "src",
                    openpi_root / ".venv" / "lib" / "python3.11" / "site-packages",
                ]
            )
    elif getattr(args, "policy_mode", None) == "inprocess":
        raise ValueError("--openpi-root or OPENPI_ROOT is required for in-process policy mode")

    for path in paths:
        abs_path = str(path.resolve())
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)


def max_steps_for_suite(name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[name]


def make_env(args: argparse.Namespace):
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
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    init_states = torch_load_compat(str(init_states_path), torch.load)
    return env, task, init_states


def make_policy(args: argparse.Namespace):
    if args.policy_mode == "websocket":
        from openpi_client import websocket_client_policy

        return websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)

    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir or PI0_CHECKPOINT_DIR is required for in-process policy mode")

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.config_name)
    return policy_config.create_trained_policy(train_config, args.checkpoint_dir)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def preprocess_policy_images(obs: dict, resize_size: int):
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def policy_observation(obs: dict, task_description: str, resize_size: int) -> dict:
    img, wrist_img = preprocess_policy_images(obs, resize_size)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def current_observation(env) -> dict:
    if hasattr(env, "regenerate_obs_from_state") and hasattr(env, "get_sim_state"):
        return env.regenerate_obs_from_state(env.get_sim_state())
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
        if hasattr(env, "_post_process"):
            env._post_process()
        if hasattr(env, "_update_observables"):
            env._update_observables(force=True)
        return env.env._get_observations()
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    raise AttributeError("LIBERO environment does not expose a current-observation method")


def render_frame(env, camera: str, size: int) -> np.ndarray:
    frame = env.sim.render(height=size, width=size, camera_name=camera)
    return np.ascontiguousarray(frame[::-1, ::-1])
