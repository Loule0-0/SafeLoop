#!/usr/bin/env python
"""Start the pinned OpenPI policy server without training-only LeRobot imports."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TrainingDatasetUnavailable:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "LeRobot datasets are unavailable in the inference-only OpenPI environment"
        )


def _install_lerobot_dataset_stub() -> None:
    packages: dict[str, types.ModuleType] = {}
    for name in ("lerobot", "lerobot.common", "lerobot.common.datasets"):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        packages[name] = module
        sys.modules[name] = module

    dataset_module = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = _TrainingDatasetUnavailable
    dataset_module.LeRobotDatasetMetadata = _TrainingDatasetUnavailable
    sys.modules[dataset_module.__name__] = dataset_module

    packages["lerobot"].common = packages["lerobot.common"]
    packages["lerobot.common"].datasets = packages["lerobot.common.datasets"]
    packages["lerobot.common.datasets"].lerobot_dataset = dataset_module


def main() -> None:
    openpi_root = Path(
        os.environ.get("OPENPI_ROOT", PROJECT_ROOT / "third_party" / "openpi")
    ).expanduser().resolve()
    server_script = openpi_root / "scripts" / "serve_policy.py"
    if not server_script.is_file():
        raise FileNotFoundError(f"OpenPI policy server not found: {server_script}")

    _install_lerobot_dataset_stub()
    sys.argv[0] = str(server_script)
    runpy.run_path(str(server_script), run_name="__main__")


if __name__ == "__main__":
    main()
