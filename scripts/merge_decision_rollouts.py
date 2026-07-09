from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SafeLoop decision rollout npz shards.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    summaries = []
    action_counts: Counter[int] = Counter()

    for path in args.inputs:
        data = np.load(path)
        for key in data.files:
            arrays_by_key.setdefault(key, []).append(data[key])
        actions = data["actions"].astype(np.int64)
        action_counts.update(int(action) for action in actions)
        summary_path = path.with_suffix(".json")
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    merged = {
        key: np.concatenate(value, axis=0)
        for key, value in arrays_by_key.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged)

    summary = {
        "out": str(args.out),
        "inputs": [str(path) for path in args.inputs],
        "input_summaries": summaries,
        "records": int(merged["observations"].shape[0]),
        "input_dim": int(merged["observations"].shape[1]),
        "action_counts": {
            "noop": int(action_counts[0]),
            "record": int(action_counts[1]),
            "rollback": int(action_counts[2]),
        },
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
