#!/usr/bin/env python
"""Validate a SafeLoop release installation before training or evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 10)
PINNED_SUBMODULES = {
    "third_party/LIBERO": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "third_party/openpi": "b14bcf2989a46de9cc379f837b5a96a46a3948f4",
    "third_party/openvla-oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
}
REQUIRED_MODULES = {
    "accelerate": "accelerate",
    "huggingface-hub": "huggingface_hub",
    "imageio": "imageio",
    "numpy": "numpy",
    "peft": "peft",
    "pillow": "PIL",
    "qwen-vl-utils": "qwen_vl_utils",
    "torch": "torch",
    "transformers": "transformers",
    "libero": "libero",
    "openpi-client": "openpi_client",
    "robosuite": "robosuite",
}


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _asset_path(name: str, fallback: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def _check_install() -> tuple[list[str], list[str], dict[str, str]]:
    errors: list[str] = []
    warnings: list[str] = []
    versions: dict[str, str] = {}

    if sys.version_info[:2] != EXPECTED_PYTHON:
        errors.append(f"Python 3.10 is required; found {sys.version.split()[0]}")

    for distribution, module in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python module: {module}")
            continue
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "editable/source"

    for relative_path, expected_head in PINNED_SUBMODULES.items():
        path = PROJECT_ROOT / relative_path
        actual_head = _git_head(path)
        if actual_head is None:
            errors.append(f"submodule is not initialized: {relative_path}")
        elif actual_head != expected_head:
            errors.append(
                f"submodule revision mismatch for {relative_path}: "
                f"expected {expected_head}, found {actual_head}"
            )

    if importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            warnings.append("CUDA is not available; Qwen inference and training will be slow")

    return errors, warnings, versions


def _check_eval_assets() -> list[str]:
    errors: list[str] = []
    paths = {
        "LIBERO_ROOT": _asset_path("LIBERO_ROOT", PROJECT_ROOT / "third_party" / "LIBERO"),
        "OPENPI_ROOT": _asset_path("OPENPI_ROOT", PROJECT_ROOT / "third_party" / "openpi"),
        "QWEN_MODEL": _asset_path("QWEN_MODEL"),
        "SAFELOOP_WEIGHTS": _asset_path("SAFELOOP_WEIGHTS"),
    }
    for name, path in paths.items():
        if path is None:
            errors.append(f"environment variable is not set: {name}")
        elif not path.exists():
            errors.append(f"{name} does not exist: {path}")

    weights_dir = paths["SAFELOOP_WEIGHTS"]
    if weights_dir is not None and weights_dir.exists():
        config = json.loads(
            (PROJECT_ROOT / "configs" / "release" / "v56_24task_eval.json").read_text(encoding="utf-8")
        )
        for relative_path in config["checkpoints"].values():
            if not (weights_dir / relative_path).is_file():
                errors.append(f"missing release checkpoint: {weights_dir / relative_path}")
    return errors


def _check_policy_server(host: str, port: int, backend: str) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return None
    except OSError as exc:
        return f"cannot connect to {backend} policy server at {host}:{port}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["install", "eval"], default="install")
    parser.add_argument("--check-policy-server", action="store_true")
    parser.add_argument("--policy-backend", choices=["pi0", "openvla-oft"], default="pi0")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    args = parser.parse_args(argv)

    errors, warnings, versions = _check_install()
    if args.scope == "eval":
        errors.extend(_check_eval_assets())
    if args.check_policy_server:
        policy_error = _check_policy_server(args.policy_host, args.policy_port, args.policy_backend)
        if policy_error:
            errors.append(policy_error)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(json.dumps({"scope": args.scope, "versions": versions}, indent=2, sort_keys=True))
    if errors:
        print(f"Environment check failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("SafeLoop environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
