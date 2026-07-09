from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard.risk import RiskVector
from safety_guard.rl_decider import ThreeAction
from safety_guard.rl_decision_data import (
    DecisionContext,
    DecisionRecord,
    DecisionRewardModel,
    build_decision_features,
    decision_records_to_rollout_batch,
    save_decision_rollout_npz,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a three-action decision-head PPO rollout from pi0 LIBERO trajectory dumps.",
    )
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--tau", type=int, default=50, help="Future horizon, in simulator steps, for oracle risk.")
    parser.add_argument("--current-tolerance", type=int, default=5)
    parser.add_argument("--max-runs", type=int, help="Optional cap for quick debugging.")
    parser.add_argument("--min-record-interval", type=int, default=1)
    parser.add_argument("--rollback-cooldown", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records, summary = build_records_from_pi0_trajectories(
        trajectory_root=args.trajectory_root,
        history_length=args.history_length,
        tau=args.tau,
        current_tolerance=args.current_tolerance,
        max_runs=args.max_runs,
        min_record_interval=args.min_record_interval,
        rollback_cooldown=args.rollback_cooldown,
    )
    batch = decision_records_to_rollout_batch(records, DecisionRewardModel())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_decision_rollout_npz(batch, args.out)

    action_counts = Counter(int(record.action) for record in records)
    summary.update(
        {
            "out": str(args.out),
            "records": len(records),
            "input_dim": int(batch.observations.shape[1]),
            "action_counts": {
                "noop": int(action_counts[ThreeAction.NOOP]),
                "record": int(action_counts[ThreeAction.RECORD]),
                "rollback": int(action_counts[ThreeAction.ROLLBACK]),
            },
        }
    )
    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_records_from_pi0_trajectories(
    trajectory_root: Path,
    history_length: int = 3,
    tau: int = 50,
    current_tolerance: int = 5,
    max_runs: int | None = None,
    min_record_interval: int = 1,
    rollback_cooldown: int = 0,
) -> tuple[list[DecisionRecord], dict[str, object]]:
    trajectory_root = Path(trajectory_root)
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    run_dirs = _find_run_dirs(trajectory_root)
    if max_runs is not None:
        run_dirs = run_dirs[: int(max_runs)]
    if not run_dirs:
        raise ValueError(f"no trajectory_metadata.json files found under {trajectory_root}")

    reward_model = DecisionRewardModel()
    records: list[DecisionRecord] = []
    skipped_runs: list[str] = []
    bad_samples: list[str] = []
    per_run_counts: dict[str, int] = {}

    for run_dir in run_dirs:
        metadata = _read_json(run_dir / "trajectory_metadata.json")
        marked_steps = _parse_marked_steps(metadata.get("marked_steps") or [])
        samples, run_bad_samples = _load_samples(run_dir)
        bad_samples.extend(str(path) for path in run_bad_samples)
        if not samples:
            skipped_runs.append(str(run_dir))
            continue

        max_step = _infer_max_step(metadata, samples)
        memory_size = 0
        last_record_step = -10**9
        last_rollback_step = -10**9
        joint_history: list[np.ndarray] = []
        risk_history: list[RiskVector] = []
        current_history: list[tuple[float, float]] = []
        run_records = 0

        for sample_index, sample in enumerate(samples):
            current_step = int(sample.get("global_step", sample.get("sample_id", sample_index)))
            joints = _robot_state_to_joint_vector(sample.get("robot_state") or {})
            risk = _risk_from_marked_steps(current_step, marked_steps, tau=tau)
            current_body, current_object = _current_labels_from_marked_steps(
                current_step,
                marked_steps,
                tolerance=current_tolerance,
            )

            joint_history.append(joints)
            risk_history.append(risk)
            current_history.append((current_body, current_object))

            can_record = current_step - last_record_step >= min_record_interval
            can_rollback = (
                memory_size > 0
                and current_step - last_rollback_step >= rollback_cooldown
            )
            context_values = [
                float(can_record),
                float(can_rollback),
                min(memory_size, 10) / 10.0,
                _safe_ratio(current_step, max_step),
            ]
            features = build_decision_features(
                joint_history=joint_history,
                risk_history=risk_history,
                current_hazard_history=current_history,
                context_values=context_values,
                history_length=history_length,
            )
            context = DecisionContext(
                features=features,
                risk=risk,
                current_body_probability=current_body,
                current_object_probability=current_object,
                can_record=can_record,
                can_rollback=can_rollback,
                step_index=current_step,
                memory_size=memory_size,
                done=sample_index == len(samples) - 1,
            )
            action = reward_model.best_action(context)
            action_rewards = reward_model.action_rewards(context)
            records.append(
                DecisionRecord(
                    context=context,
                    action=action,
                    reward=float(action_rewards[action]),
                    done=context.done,
                    action_rewards=action_rewards,
                )
            )
            run_records += 1

            if action == ThreeAction.RECORD:
                memory_size += 1
                last_record_step = current_step
            elif action == ThreeAction.ROLLBACK:
                last_rollback_step = current_step

        per_run_counts[run_dir.name] = run_records

    if not records:
        raise ValueError(f"no usable samples found under {trajectory_root}")
    return records, {
        "trajectory_root": str(trajectory_root),
        "runs": len(run_dirs),
        "skipped_runs": skipped_runs,
        "bad_samples": bad_samples,
        "per_run_records": per_run_counts,
        "history_length": history_length,
        "tau": tau,
        "current_tolerance": current_tolerance,
        "min_record_interval": min_record_interval,
        "rollback_cooldown": rollback_cooldown,
    }


def _find_run_dirs(root: Path) -> list[Path]:
    if (root / "trajectory_metadata.json").exists():
        return [root]
    return sorted(path.parent for path in root.glob("**/trajectory_metadata.json"))


def _load_samples(run_dir: Path) -> tuple[list[dict], list[Path]]:
    samples: list[dict] = []
    bad_paths: list[Path] = []
    for path in sorted(run_dir.glob("sample_*.json"), key=_sample_sort_key):
        try:
            samples.append(_read_json(path))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            bad_paths.append(path)
    return (
        sorted(samples, key=lambda sample: int(sample.get("global_step", sample.get("sample_id", 0)))),
        bad_paths,
    )


def _sample_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.rsplit("_", 1)[-1]
    try:
        return int(stem), path.name
    except ValueError:
        return 0, path.name


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_marked_steps(raw_steps: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    marked_steps: list[tuple[int, int]] = []
    for item in raw_steps:
        if len(item) < 2:
            continue
        marked_steps.append((int(item[0]), int(item[1])))
    return sorted(marked_steps)


def _robot_state_to_joint_vector(robot_state: dict) -> np.ndarray:
    joint_pos = np.asarray(robot_state.get("robot0_joint_pos") or [], dtype=np.float32).reshape(-1)
    gripper = np.asarray(robot_state.get("robot0_gripper_qpos") or [], dtype=np.float32).reshape(-1)
    if joint_pos.size == 0:
        raise ValueError("sample robot_state is missing robot0_joint_pos")
    return np.concatenate([joint_pos, gripper]).astype(np.float32)


def _risk_from_marked_steps(current_step: int, marked_steps: Sequence[tuple[int, int]], tau: int) -> RiskVector:
    body_probability, body_tth = _future_hazard(current_step, marked_steps, hazard_type=0, tau=tau)
    object_probability, object_tth = _future_hazard(current_step, marked_steps, hazard_type=1, tau=tau)
    return RiskVector(
        body_probability=body_probability,
        body_tth=body_tth,
        object_probability=object_probability,
        object_tth=object_tth,
    )


def _future_hazard(
    current_step: int,
    marked_steps: Sequence[tuple[int, int]],
    hazard_type: int,
    tau: int,
) -> tuple[float, float]:
    future = [
        step
        for step, kind in marked_steps
        if int(kind) == int(hazard_type) and 0 <= int(step) - int(current_step) <= tau
    ]
    if not future:
        return 0.0, 1.0
    distance = max(0, min(future) - int(current_step))
    return 1.0, float(np.clip(distance / tau, 0.0, 1.0))


def _current_labels_from_marked_steps(
    current_step: int,
    marked_steps: Sequence[tuple[int, int]],
    tolerance: int = 5,
) -> tuple[float, float]:
    body = 0.0
    obj = 0.0
    for marked_step, hazard_type in marked_steps:
        if abs(int(marked_step) - int(current_step)) <= tolerance:
            if int(hazard_type) == 0:
                body = 1.0
            elif int(hazard_type) == 1:
                obj = 1.0
    return body, obj


def _infer_max_step(metadata: dict, samples: Sequence[dict]) -> int:
    if metadata.get("max_steps") is not None:
        return max(int(metadata["max_steps"]), 1)
    max_sample_step = max(int(sample.get("global_step", sample.get("sample_id", 0))) for sample in samples)
    return max(max_sample_step, 1)


def _safe_ratio(value: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(np.clip(float(value) / float(denominator), 0.0, 1.0))


if __name__ == "__main__":
    main()
