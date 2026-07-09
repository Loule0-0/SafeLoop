#!/usr/bin/env python
"""Extract the SafeLoop Hugging Face training-data shards."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def _safe_members(tar: tarfile.TarFile, out: Path):
    root = out.resolve()
    for member in tar.getmembers():
        target = (out / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"refusing to extract outside output root: {member.name}")
        yield member


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Downloaded HF dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Directory that will receive image paths.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    for shard in manifest["shards"]:
        tar_path = args.dataset_dir / shard["file"]
        done_path = args.out / (Path(shard["file"]).name + ".done")
        if args.skip_existing and done_path.exists():
            print(f"skipping {tar_path}")
            continue
        print(f"extracting {tar_path}")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(args.out, members=_safe_members(tar, args.out))
        done_path.write_text("ok\n", encoding="utf-8")
    print(f"materialized {manifest['image_refs']['unique']} image references under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
