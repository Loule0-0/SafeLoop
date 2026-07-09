#!/usr/bin/env python
"""Run the release 24-task LIBERO evaluation matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "release" / "v56_24task_eval.json"


def _replace_placeholders(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in context.items():
            value = value.replace("${" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [_replace_placeholders(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, context) for key, item in value.items()}
    return value


def _append_arg(command: list[str], key: str, value: Any) -> None:
    flag = "--" + key
    if value is None or value is False:
        return
    if value is True:
        command.append(flag)
        return
    if isinstance(value, list):
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def _entry_name(entry: dict[str, Any]) -> str:
    tasks = "_".join(str(task_id) for task_id in entry["task_ids"])
    return f"{entry['benchmark']}_tasks_{tasks}_seed{entry['seed']}_{entry['mode']}"


def _build_command(
    config: dict[str, Any],
    entry: dict[str, Any],
    context: dict[str, str],
    output_root: Path,
) -> list[str]:
    args: dict[str, Any] = {}
    args.update(config["common_args"])
    args.update(config.get("gates", {}).get(entry.get("gate", "standard"), {}))
    args.update({
        "benchmark": entry["benchmark"],
        "task-ids": entry["task_ids"],
        "seed": entry["seed"],
        "mode": entry["mode"],
        "output-dir": str(output_root / _entry_name(entry)),
    })
    if entry["mode"] == "rl":
        checkpoints = config["checkpoints"]
        args["qwen-head-path"] = str(Path(context["weights_dir"]) / checkpoints[entry["head"]])
        args["decision-checkpoint"] = str(Path(context["weights_dir"]) / checkpoints["decision"])
    else:
        args.pop("predictor", None)
        args.pop("qwen-model-path", None)
        args["disable-safeloop"] = True

    args = _replace_placeholders(args, context)
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_pi0_safeguard_closed_loop.py")]
    for key, value in args.items():
        _append_arg(command, key, value)
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="paper_24task_release")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    profiles = config["profiles"]
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; choices: {', '.join(sorted(profiles))}")

    context = {
        "model_dir": str(args.model_dir),
        "weights_dir": str(args.weights_dir),
        "policy_host": args.policy_host,
        "policy_port": str(args.policy_port),
    }
    env = os.environ.copy()
    if args.libero_root:
        env["LIBERO_ROOT"] = str(args.libero_root)
    if args.openpi_root:
        env["OPENPI_ROOT"] = str(args.openpi_root)
    if args.checkpoint_dir:
        env["PI0_CHECKPOINT_DIR"] = str(args.checkpoint_dir)

    for entry in profiles[args.profile]["entries"]:
        command = _build_command(config, entry, context, args.output_root)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.check_call(command, cwd=PROJECT_ROOT, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
