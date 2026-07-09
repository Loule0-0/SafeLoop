from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Sample = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge SafeLoop Qwen rollout JSONL files and remove image-collision hard-case label noise.",
    )
    parser.add_argument("--base-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--extra-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--keep-colliding-rollback-groups", action="store_true")
    parser.add_argument("--drop-any-label-conflict", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            metadata = dict(sample.get("metadata") or {})
            metadata.setdefault("source_jsonl", str(path))
            metadata.setdefault("source_line", line_number)
            sample["metadata"] = metadata
            samples.append(sample)
    return samples


def merge_and_filter_samples(
    samples: list[Sample],
    *,
    data_root: Path = Path("."),
    require_images: bool = False,
    drop_colliding_rollback_groups: bool = True,
    dedupe: bool = True,
    drop_any_label_conflict: bool = False,
) -> tuple[list[Sample], dict[str, Any]]:
    drop_indices: set[int] = set()
    missing_image_count = 0
    if require_images:
        for index, sample in enumerate(samples):
            if not _sample_images_exist(sample, data_root):
                drop_indices.add(index)
                missing_image_count += 1

    collision_groups = []
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[_step_group_key(sample)].append(index)

    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        by_images: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index in indices:
            by_images[_image_signature(samples[index])].append(index)
        duplicate_image_group = any(len(image_indices) > 1 for image_indices in by_images.values())
        label_conflict = any(
            len({_label_signature(samples[index]) for index in image_indices}) > 1
            for image_indices in by_images.values()
        )
        rollback_label_conflict = any(
            len({_label_signature(samples[index]) for index in image_indices}) > 1
            and any(_is_rollback_source(samples[index]) for index in image_indices)
            for image_indices in by_images.values()
        )
        rollback_collision = (
            drop_colliding_rollback_groups
            and duplicate_image_group
            and any(_is_rollback_source(samples[index]) for index in indices)
        )
        should_drop = rollback_collision or rollback_label_conflict or (drop_any_label_conflict and label_conflict)
        if should_drop:
            drop_indices.update(indices)
            collision_groups.append(
                {
                    "key": list(key),
                    "samples": len(indices),
                    "label_conflict": bool(label_conflict),
                    "rollback_collision": bool(rollback_collision),
                    "sources": sorted({_sample_source(samples[index]) for index in indices}),
                }
            )

    kept: list[Sample] = []
    seen: set[str] = set()
    exact_duplicates = 0
    for index, sample in enumerate(samples):
        if index in drop_indices:
            continue
        if dedupe:
            signature = _dedupe_signature(sample)
            if signature in seen:
                exact_duplicates += 1
                continue
            seen.add(signature)
        kept.append(sample)

    report = {
        "input_samples": len(samples),
        "written_samples": len(kept),
        "dropped_samples": len(drop_indices),
        "missing_image_samples": missing_image_count,
        "exact_duplicates_removed": exact_duplicates,
        "collision_groups_removed": len(collision_groups),
        "collision_group_examples": collision_groups[:20],
        "source_counts_in": _source_counts(samples),
        "source_counts_out": _source_counts(kept),
        "label_counts_out": _label_counts(kept),
    }
    return kept, report


def write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_paths = [*args.base_jsonl, *args.extra_jsonl]
    if not input_paths:
        raise SystemExit("at least one --base-jsonl or --extra-jsonl is required")
    samples: list[Sample] = []
    for path in input_paths:
        samples.extend(read_jsonl(path))
    kept, report = merge_and_filter_samples(
        samples,
        data_root=args.data_root,
        require_images=args.require_images,
        drop_colliding_rollback_groups=not args.keep_colliding_rollback_groups,
        dedupe=not args.no_dedupe,
        drop_any_label_conflict=bool(args.drop_any_label_conflict),
    )
    write_jsonl(args.out, kept)
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _sample_source(sample: Sample) -> str:
    return str((sample.get("metadata") or {}).get("source") or "none")


def _is_rollback_source(sample: Sample) -> bool:
    return _sample_source(sample).startswith("rollback")


def _step_group_key(sample: Sample) -> tuple[Any, ...]:
    metadata = sample.get("metadata") or {}
    return (
        metadata.get("benchmark"),
        metadata.get("task_id"),
        metadata.get("episode"),
        _run_id_from_images(sample),
        metadata.get("sample_index", sample.get("sample_index")),
    )


def _run_id_from_images(sample: Sample) -> str:
    images = sample.get("images") or []
    if not images:
        return "unknown"
    parts = str(images[0]).split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else str(images[0])


def _image_signature(sample: Sample) -> tuple[str, ...]:
    return tuple(str(path) for path in (sample.get("images") or []))


def _label_signature(sample: Sample) -> str:
    labels = sample.get("labels") or {}
    fields = {
        key: float(labels.get(key, 0.0) or 0.0)
        for key in (
            "future_body",
            "future_body_tth",
            "future_object",
            "future_object_tth",
            "current_body",
            "current_object",
        )
    }
    return json.dumps(fields, sort_keys=True)


def _dedupe_signature(sample: Sample) -> str:
    user_text = ""
    messages = sample.get("messages") or []
    if messages:
        user_text = str(messages[0].get("content", ""))
    payload = {
        "images": _image_signature(sample),
        "labels": json.loads(_label_signature(sample)),
        "user": user_text,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _sample_images_exist(sample: Sample, data_root: Path) -> bool:
    for image in sample.get("images") or []:
        path = Path(str(image))
        if path.is_absolute():
            if not path.exists():
                return False
        elif not (data_root / path).exists():
            return False
    return True


def _source_counts(samples: list[Sample]) -> dict[str, int]:
    return dict(Counter(_sample_source(sample) for sample in samples))


def _label_counts(samples: list[Sample]) -> dict[str, int]:
    counts = {"future_body": 0, "future_object": 0, "current_body": 0, "current_object": 0}
    for sample in samples:
        labels = sample.get("labels") or {}
        for key in counts:
            counts[key] += int(float(labels.get(key, 0.0) or 0.0) > 0.0)
    return counts


if __name__ == "__main__":
    main()
