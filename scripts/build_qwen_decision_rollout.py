from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety_guard.qwen_multitask import (  # noqa: E402
    LiveQwenFrame,
    QwenMultitaskSafetyPredictor,
    build_live_qwen_prompt,
)
from safety_guard.risk import RiskVector  # noqa: E402
from safety_guard.rl_decider import ThreeAction  # noqa: E402
from safety_guard.rl_decision_data import (  # noqa: E402
    DecisionContext,
    DecisionRecord,
    DecisionRewardModel,
    build_decision_features,
    decision_records_to_rollout_batch,
    save_decision_rollout_npz,
)
from scripts.build_decision_rollout_from_pi0 import (  # noqa: E402
    _current_labels_from_marked_steps,
    _find_run_dirs,
    _infer_max_step,
    _load_samples,
    _parse_marked_steps,
    _read_json,
    _risk_from_marked_steps,
    _robot_state_to_joint_vector,
    _safe_ratio,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SafeLoop decision rollout whose observations use Qwen predicted risk.",
    )
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--lora-adapter-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--tau", type=int, default=50)
    parser.add_argument("--current-tolerance", type=int, default=20)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--run-shard-index", type=int, default=0)
    parser.add_argument("--run-shard-count", type=int, default=1)
    parser.add_argument("--min-record-interval", type=int, default=30)
    parser.add_argument("--rollback-cooldown", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cache = _PredictionCache(args.prediction_cache)
    predictor = QwenMultitaskSafetyPredictor(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint_path,
        lora_adapter_path=args.lora_adapter_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        history_length=args.history_length,
        tau=args.tau,
    )
    records, summary = build_records_with_qwen_predictions(
        trajectory_root=args.trajectory_root,
        predictor=predictor,
        cache=cache,
        history_length=args.history_length,
        tau=args.tau,
        current_tolerance=args.current_tolerance,
        max_runs=args.max_runs,
        run_shard_index=args.run_shard_index,
        run_shard_count=args.run_shard_count,
        min_record_interval=args.min_record_interval,
        rollback_cooldown=args.rollback_cooldown,
        progress_every=args.progress_every,
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
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_records_with_qwen_predictions(
    trajectory_root: Path,
    predictor: QwenMultitaskSafetyPredictor,
    cache: "_PredictionCache",
    history_length: int,
    tau: int,
    current_tolerance: int,
    max_runs: int | None,
    run_shard_index: int,
    run_shard_count: int,
    min_record_interval: int,
    rollback_cooldown: int,
    progress_every: int,
) -> tuple[list[DecisionRecord], dict[str, object]]:
    run_dirs = _find_run_dirs(Path(trajectory_root))
    if run_shard_count <= 0:
        raise ValueError("run_shard_count must be positive")
    if not 0 <= run_shard_index < run_shard_count:
        raise ValueError("run_shard_index must be in [0, run_shard_count)")
    run_dirs = [
        run_dir
        for index, run_dir in enumerate(run_dirs)
        if index % run_shard_count == run_shard_index
    ]
    if max_runs is not None:
        run_dirs = run_dirs[: int(max_runs)]
    if not run_dirs:
        raise ValueError(f"no trajectory_metadata.json files found under {trajectory_root}")

    reward_model = DecisionRewardModel()
    records: list[DecisionRecord] = []
    skipped_runs: list[str] = []
    bad_samples: list[str] = []
    per_run_counts: dict[str, int] = {}

    processed = 0
    for run_index, run_dir in enumerate(run_dirs, start=1):
        predictor.reset()
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
        qwen_frame_history: deque[LiveQwenFrame] = deque(maxlen=history_length)
        joint_history: list[np.ndarray] = []
        predicted_risk_history: list[RiskVector] = []
        predicted_current_history: list[tuple[float, float]] = []
        run_records = 0

        for sample_index, sample in enumerate(samples):
            current_step = int(sample.get("global_step", sample.get("sample_id", sample_index)))
            joints = _robot_state_to_joint_vector(sample.get("robot_state") or {})
            frame = _sample_to_qwen_frame(sample)
            qwen_frame_history.append(frame)
            cache_key = f"{run_dir.name}:{current_step}"
            prediction = cache.get(cache_key)
            if prediction is None:
                text, images = build_live_qwen_prompt(
                    history=list(qwen_frame_history),
                    current_global_step=current_step,
                    history_length=history_length,
                    tau=tau,
                )
                output = predictor.predict_prompt_images(text, images)
                prediction = {
                    "body_probability": output.risk.body_probability,
                    "body_tth": output.risk.body_tth,
                    "object_probability": output.risk.object_probability,
                    "object_tth": output.risk.object_tth,
                    "current_body_probability": output.current_body_probability,
                    "current_object_probability": output.current_object_probability,
                }
                cache.put(cache_key, prediction)

            predicted_risk = RiskVector(
                body_probability=float(prediction["body_probability"]),
                body_tth=float(prediction["body_tth"]),
                object_probability=float(prediction["object_probability"]),
                object_tth=float(prediction["object_tth"]),
            )
            predicted_current = (
                float(prediction["current_body_probability"]),
                float(prediction["current_object_probability"]),
            )
            oracle_risk = _risk_from_marked_steps(current_step, marked_steps, tau=tau)
            oracle_current = _current_labels_from_marked_steps(
                current_step,
                marked_steps,
                tolerance=current_tolerance,
            )

            joint_history.append(joints)
            predicted_risk_history.append(predicted_risk)
            predicted_current_history.append(predicted_current)

            can_record = current_step - last_record_step >= min_record_interval
            can_rollback = memory_size > 0 and current_step - last_rollback_step >= rollback_cooldown
            context_values = [
                float(can_record),
                float(can_rollback),
                min(memory_size, 10) / 10.0,
                _safe_ratio(current_step, max_step),
            ]
            predicted_features = build_decision_features(
                joint_history=joint_history,
                risk_history=predicted_risk_history,
                current_hazard_history=predicted_current_history,
                context_values=context_values,
                history_length=history_length,
            )
            predicted_context = DecisionContext(
                features=predicted_features,
                risk=predicted_risk,
                current_body_probability=predicted_current[0],
                current_object_probability=predicted_current[1],
                can_record=can_record,
                can_rollback=can_rollback,
                step_index=current_step,
                memory_size=memory_size,
                done=sample_index == len(samples) - 1,
            )
            oracle_context = DecisionContext(
                features=predicted_features,
                risk=oracle_risk,
                current_body_probability=oracle_current[0],
                current_object_probability=oracle_current[1],
                can_record=can_record,
                can_rollback=can_rollback,
                step_index=current_step,
                memory_size=memory_size,
                done=predicted_context.done,
            )
            action_rewards = reward_model.action_rewards(oracle_context)
            action = ThreeAction(int(np.argmax(action_rewards)))
            records.append(
                DecisionRecord(
                    context=predicted_context,
                    action=action,
                    reward=float(action_rewards[int(action)]),
                    done=predicted_context.done,
                    action_rewards=action_rewards,
                )
            )
            run_records += 1
            processed += 1

            if action == ThreeAction.RECORD:
                memory_size += 1
                last_record_step = current_step
            elif action == ThreeAction.ROLLBACK:
                last_rollback_step = current_step

            if progress_every > 0 and processed % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "run": run_dir.name,
                            "run_index": run_index,
                            "runs": len(run_dirs),
                            "records": len(records),
                        }
                    ),
                    flush=True,
                )

        per_run_counts[run_dir.name] = run_records

    return records, {
        "trajectory_root": str(trajectory_root),
        "runs": len(run_dirs),
        "skipped_runs": skipped_runs,
        "bad_samples": bad_samples,
        "per_run_records": per_run_counts,
        "history_length": history_length,
        "tau": tau,
        "current_tolerance": current_tolerance,
        "run_shard_index": run_shard_index,
        "run_shard_count": run_shard_count,
        "min_record_interval": min_record_interval,
        "rollback_cooldown": rollback_cooldown,
        "prediction_cache": str(cache.path) if cache.path is not None else None,
    }


def _sample_to_qwen_frame(sample: dict) -> LiveQwenFrame:
    robot_state = sample.get("robot_state") or {}
    return LiveQwenFrame(
        images=(
            _decode_jpeg_base64(sample.get("camera_image")),
            _decode_jpeg_base64(sample.get("wrist_image")),
        ),
        joint_pos=np.asarray(robot_state.get("robot0_joint_pos") or [], dtype=np.float32).reshape(-1),
        joint_vel=np.asarray(robot_state.get("robot0_joint_vel") or [], dtype=np.float32).reshape(-1),
        joint_torques=np.asarray(robot_state.get("robot0_joint_torques") or [], dtype=np.float32).reshape(-1),
    )


def _decode_jpeg_base64(value: str | None) -> Image.Image:
    if not value:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


class _PredictionCache:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._items: dict[str, dict[str, float]] = {}
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self._items[str(row["key"])] = dict(row["prediction"])

    def get(self, key: str) -> dict[str, float] | None:
        return self._items.get(key)

    def put(self, key: str, prediction: dict[str, float]) -> None:
        self._items[key] = prediction
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "prediction": prediction}) + "\n")


if __name__ == "__main__":
    main()
