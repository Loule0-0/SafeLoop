#!/usr/bin/env python
"""Download and verify the pinned SafeLoop Pi0 release weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "release" / "artifacts_pi0_v1.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights", {})
    if not weights.get("repo_id") or not weights.get("revision") or not weights.get("files"):
        raise ValueError(f"invalid weights manifest: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(root: Path, entry: dict[str, Any]) -> str | None:
    path = root / entry["path"]
    if not path.is_file():
        return f"missing: {path}"
    expected_size = int(entry["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return f"size mismatch: {path} (expected {expected_size}, found {actual_size})"
    actual_hash = _sha256(path)
    if actual_hash != entry["sha256"]:
        return f"SHA256 mismatch: {path} (expected {entry['sha256']}, found {actual_hash})"
    return None


def download_weights(manifest: dict[str, Any], output_dir: Path, check_only: bool) -> None:
    weights = manifest["weights"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not check_only:
        from huggingface_hub import hf_hub_download

        for entry in weights["files"]:
            destination = output_dir / entry["path"]
            if verify_file(output_dir, entry) is None:
                print(f"verified {entry['path']}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=weights["repo_id"],
                filename=entry["path"],
                revision=weights["revision"],
                local_dir=output_dir,
            )

    errors = [
        error
        for entry in weights["files"]
        if (error := verify_file(output_dir, entry)) is not None
    ]
    if errors:
        raise RuntimeError("\n".join(errors))
    print(
        f"SafeLoop weights verified: {len(weights['files'])} files, "
        f"{weights['repo_id']}@{weights['revision']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ["SAFELOOP_WEIGHTS"]) if os.environ.get("SAFELOOP_WEIGHTS") else None,
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("--output-dir or SAFELOOP_WEIGHTS is required")

    manifest = _load_manifest(args.manifest.resolve())
    download_weights(manifest, args.output_dir.expanduser().resolve(), args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
