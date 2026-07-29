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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_config(parent_path.resolve()), config)


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


def _entry_name(entry: dict[str, Any], seed: int) -> str:
    tasks = "_".join(str(task_id) for task_id in entry["task_ids"])
    return f"{entry['benchmark']}_tasks_{tasks}_seed{seed}_{entry['mode']}"


def _entry_seeds(config: dict[str, Any], profile: dict[str, Any], entry: dict[str, Any]) -> list[int]:
    if "seeds" in entry:
        values = entry["seeds"]
    elif "seeds" in profile:
        values = profile["seeds"]
    elif "seed_list" in config:
        values = config["seed_list"]
    else:
        values = [entry.get("seed", 0)]
    return [int(seed) for seed in values]


def _parse_task(value: str) -> tuple[str, int]:
    benchmark, separator, task_id = value.partition(":")
    if not separator or not benchmark or not task_id:
        raise argparse.ArgumentTypeError("tasks must use BENCHMARK:TASK_ID")
    try:
        parsed_task_id = int(task_id)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid task id in {value!r}") from exc
    if parsed_task_id < 0:
        raise argparse.ArgumentTypeError("task id must be non-negative")
    return benchmark, parsed_task_id


def _select_entries(
    entries: list[dict[str, Any]],
    selected_tasks: list[tuple[str, int]] | None,
) -> list[dict[str, Any]]:
    if not selected_tasks:
        return entries
    requested: dict[str, set[int]] = {}
    for benchmark, task_id in selected_tasks:
        requested.setdefault(benchmark, set()).add(task_id)

    selected: list[dict[str, Any]] = []
    found: set[tuple[str, int]] = set()
    for entry in entries:
        benchmark = entry["benchmark"]
        task_ids = [
            int(task_id)
            for task_id in entry["task_ids"]
            if int(task_id) in requested.get(benchmark, set())
        ]
        if task_ids:
            selected.append({**entry, "task_ids": task_ids})
            found.update((benchmark, task_id) for task_id in task_ids)

    missing = sorted(set(selected_tasks) - found)
    if missing:
        values = ", ".join(f"{benchmark}:{task_id}" for benchmark, task_id in missing)
        raise ValueError(f"selected task(s) are not present in the profile: {values}")
    return selected


def _build_command(
    config: dict[str, Any],
    entry: dict[str, Any],
    seed: int,
    context: dict[str, str],
    output_root: Path,
) -> list[str]:
    args: dict[str, Any] = {}
    args.update(config["common_args"])
    args.update(config.get("suite_args", {}).get(entry["benchmark"], {}))
    args.update(config.get("gates", {}).get(entry.get("gate", "standard"), {}))
    args.update({
        "benchmark": entry["benchmark"],
        "task-ids": entry["task_ids"],
        "seed": seed,
        "mode": entry["mode"],
        "output-dir": str(output_root / _entry_name(entry, seed)),
    })
    if config.get("seed_selects_init_state", False):
        args["init-state-start"] = seed
    if entry["mode"] == "rl":
        checkpoints = config["checkpoints"]
        args["qwen-head-path"] = str(Path(context["weights_dir"]) / checkpoints[entry["head"]])
        args["decision-checkpoint"] = context.get("decision_checkpoint") or str(
            Path(context["weights_dir"]) / checkpoints["decision"]
        )
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
    parser.add_argument("--profile", default="safeloop_all")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--decision-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--task",
        action="append",
        type=_parse_task,
        help="Run only BENCHMARK:TASK_ID; repeat to select multiple tasks",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = _load_config(args.config.resolve())
    profiles = config["profiles"]
    if args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; choices: {', '.join(sorted(profiles))}")

    context = {
        "model_dir": str(args.model_dir),
        "weights_dir": str(args.weights_dir),
        "policy_host": args.policy_host,
        "policy_port": str(args.policy_port),
        "decision_checkpoint": str(args.decision_checkpoint) if args.decision_checkpoint else "",
    }
    env = os.environ.copy()
    if args.libero_root:
        env["LIBERO_ROOT"] = str(args.libero_root)
    if args.openpi_root:
        env["OPENPI_ROOT"] = str(args.openpi_root)
    if args.checkpoint_dir:
        env["PI0_CHECKPOINT_DIR"] = str(args.checkpoint_dir)

    profile = profiles[args.profile]
    try:
        entries = _select_entries(profile["entries"], args.task)
    except ValueError as exc:
        parser.error(str(exc))
    for entry in entries:
        seeds = args.seeds if args.seeds is not None else _entry_seeds(config, profile, entry)
        for seed in seeds:
            command = _build_command(config, entry, seed, context, args.output_root)
            print(" ".join(command))
            if not args.dry_run:
                subprocess.check_call(command, cwd=PROJECT_ROOT, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
