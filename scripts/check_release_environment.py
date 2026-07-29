#!/usr/bin/env python
"""Validate a SafeLoop release installation before training or evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 10)
ARTIFACT_MANIFEST = PROJECT_ROOT / "configs" / "release" / "artifacts_pi0_v1.json"
PINNED_SUBMODULES = {
    "third_party/LIBERO": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "third_party/openpi": "b14bcf2989a46de9cc379f837b5a96a46a3948f4",
}
PINNED_DISTRIBUTIONS = {
    "accelerate": "1.8.1",
    "future": "1.0.0",
    "huggingface-hub": "0.33.4",
    "imageio": "2.37.0",
    "imageio-ffmpeg": "0.6.0",
    "matplotlib": "3.10.3",
    "numpy": "1.26.4",
    "peft": "0.15.2",
    "pillow": "11.2.1",
    "qwen-vl-utils": "0.0.14",
    "termcolor": "3.1.0",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.53.2",
    "uv": "0.12.0",
    "websockets": "15.0.1",
}
REQUIRED_MODULES = {
    "accelerate": "accelerate",
    "future": "future",
    "huggingface-hub": "huggingface_hub",
    "imageio": "imageio",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "peft": "peft",
    "pillow": "PIL",
    "qwen-vl-utils": "qwen_vl_utils",
    "termcolor": "termcolor",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "websockets": "websockets",
    "libero": "libero.libero",
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

    libero_source = PROJECT_ROOT / "third_party" / "LIBERO"
    if libero_source.is_dir() and str(libero_source) not in sys.path:
        sys.path.insert(0, str(libero_source))

    for distribution, module in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python module: {module}")
            continue
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "editable/source"
        expected_version = PINNED_DISTRIBUTIONS.get(distribution)
        actual_version = versions[distribution].split("+", 1)[0]
        if expected_version is not None and actual_version != expected_version:
            errors.append(
                f"package version mismatch for {distribution}: "
                f"expected {expected_version}, found {versions[distribution]}"
            )

    try:
        versions["uv"] = importlib.metadata.version("uv")
    except importlib.metadata.PackageNotFoundError:
        errors.append("missing Python distribution: uv")
    else:
        if versions["uv"].split("+", 1)[0] != PINNED_DISTRIBUTIONS["uv"]:
            errors.append(
                "package version mismatch for uv: "
                f"expected {PINNED_DISTRIBUTIONS['uv']}, found {versions['uv']}"
            )
    if shutil.which("uv") is None:
        errors.append("missing executable: uv")

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_eval_assets() -> list[str]:
    errors: list[str] = []
    paths = {
        "LIBERO_ROOT": _asset_path("LIBERO_ROOT", PROJECT_ROOT / "third_party" / "LIBERO"),
        "OPENPI_ROOT": _asset_path("OPENPI_ROOT", PROJECT_ROOT / "third_party" / "openpi"),
        "QWEN_MODEL": _asset_path("QWEN_MODEL"),
        "PI0_CHECKPOINT_DIR": _asset_path("PI0_CHECKPOINT_DIR"),
        "SAFELOOP_WEIGHTS": _asset_path("SAFELOOP_WEIGHTS"),
    }
    for name, path in paths.items():
        if path is None:
            errors.append(f"environment variable is not set: {name}")
        elif not path.exists():
            errors.append(f"{name} does not exist: {path}")

    weights_dir = paths["SAFELOOP_WEIGHTS"]
    if weights_dir is not None and weights_dir.exists():
        manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
        for entry in manifest["weights"]["files"]:
            checkpoint = weights_dir / entry["path"]
            if not checkpoint.is_file():
                errors.append(f"missing release checkpoint: {checkpoint}")
                continue
            if checkpoint.stat().st_size != int(entry["size"]):
                errors.append(f"release checkpoint size mismatch: {checkpoint}")
                continue
            if _sha256(checkpoint) != entry["sha256"]:
                errors.append(f"release checkpoint SHA256 mismatch: {checkpoint}")
    return errors


def _check_policy_server(host: str, port: int, backend: str) -> str | None:
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    try:
        with connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=2.0,
            close_timeout=1.0,
        ) as connection:
            connection.recv(timeout=2.0)
    except (OSError, TimeoutError, WebSocketException) as exc:
        return f"cannot connect to {backend} policy server at {host}:{port}: {exc}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["install", "eval"], default="install")
    parser.add_argument("--check-policy-server", action="store_true")
    parser.add_argument("--policy-backend", choices=["pi0"], default="pi0")
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
