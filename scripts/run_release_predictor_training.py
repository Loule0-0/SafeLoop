#!/usr/bin/env python
"""Run the release predictor-head training recipes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "release" / "v56_predictor_training.json"


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _build_command(recipe: dict[str, Any], context: dict[str, str]) -> list[str]:
    script = PROJECT_ROOT / "scripts" / "train_qwen_multitask_safety.py"
    args = _replace_placeholders(recipe["args"], context)
    command = [sys.executable, str(script)]
    for key, value in args.items():
        _append_arg(command, key, value)
    return command


def _select_recipes(config: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    recipes = config["recipes"]
    by_name = {recipe["name"]: recipe for recipe in recipes}
    if not names:
        return recipes
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"unknown recipe(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--recipe", action="append", help="Recipe name. Repeat to run multiple recipes.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Downloaded HF dataset directory.")
    parser.add_argument("--data-root", type=Path, help="Materialized image root. Defaults to DATASET_DIR/materialized.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Qwen2.5-VL backbone directory.")
    parser.add_argument("--weights-dir", type=Path, required=True, help="Downloaded SafeLoop weights directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data_root = args.data_root or (args.dataset_dir / "materialized")
    context = {
        "dataset_dir": str(args.dataset_dir),
        "data_root": str(data_root),
        "model_dir": str(args.model_dir),
        "weights_dir": str(args.weights_dir),
        "output_root": str(args.output_root),
    }
    config = _load_config(args.config)
    for recipe in _select_recipes(config, args.recipe or []):
        command = _build_command(recipe, context)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.check_call(command, cwd=PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
