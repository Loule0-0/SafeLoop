from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QwenTrainingConfig:
    mode: str
    upstream_root: Path
    model_dir: Path
    data: Path
    out: Path
    epochs: int = 3
    batch_size: int = 2
    lr: float = 1e-4
    fp16: bool = False
    validation_ratio: float = 0.1
    eval_batch_size: int = 4
    test_data: Path | None = None
    seed: int = 42
    w1: float = 1.0
    w2: float = 1.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1


def build_qwen_training_command(
    config: QwenTrainingConfig,
    python_executable: str | None = None,
) -> list[str]:
    python_executable = python_executable or sys.executable
    mode = config.mode.lower()
    if mode == "head":
        script_path = config.upstream_root / "onlyhead_ft" / "train.py"
        command = [
            python_executable,
            _path_text(script_path),
            "--model_dir",
            _path_text(config.model_dir),
            "--data",
            _path_text(config.data),
            "--out",
            _path_text(config.out),
            "--epochs",
            str(config.epochs),
            "--batch_size",
            str(config.batch_size),
            "--lr",
            str(config.lr),
            "--val_ratio",
            str(config.validation_ratio),
            "--eval_batch_size",
            str(config.eval_batch_size),
            "--seed",
            str(config.seed),
            "--w1",
            str(config.w1),
            "--w2",
            str(config.w2),
        ]
        if config.test_data is not None:
            command.extend(["--test_data", _path_text(config.test_data)])
    elif mode == "lora":
        script_path = config.upstream_root / "lora_ft" / "train.py"
        command = [
            python_executable,
            _path_text(script_path),
            "--model_dir",
            _path_text(config.model_dir),
            "--data",
            _path_text(config.data),
            "--out",
            _path_text(config.out),
            "--epochs",
            str(config.epochs),
            "--batch_size",
            str(config.batch_size),
            "--lr",
            str(config.lr),
            "--validation_split",
            str(config.validation_ratio),
            "--max_grad_norm",
            str(config.max_grad_norm),
            "--warmup_ratio",
            str(config.warmup_ratio),
        ]
    else:
        raise ValueError("mode must be either 'head' or 'lora'")

    if config.fp16:
        command.append("--fp16")
    return command


def run_qwen_training(
    config: QwenTrainingConfig,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | list[str]:
    command = build_qwen_training_command(config, python_executable=python_executable)
    if dry_run:
        return command

    script_dir = Path(command[1]).parent
    if not script_dir.is_dir():
        raise FileNotFoundError(script_dir)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [_path_text(script_dir), _path_text(config.upstream_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return subprocess.run(
        command,
        cwd=_path_text(script_dir),
        env=env,
        check=True,
        text=True,
    )


def _path_text(path: Path) -> str:
    return Path(path).as_posix()
