#!/usr/bin/env python
"""Aggregate SafeLoop closed-loop evaluation summaries."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _metric(summary: dict[str, Any], key: str, subkey: str = "any", default: float = 0.0) -> float:
    value = summary.get(key, default)
    if isinstance(value, dict):
        return float(value.get(subkey, default) or default)
    return float(value or default)


def _parse_task_seed(path: Path, payload: dict[str, Any]) -> tuple[int | None, int | None]:
    task = payload.get("task_id")
    seed = payload.get("seed")
    match = re.search(r"task(\d+)_seed(\d+)", path.parent.name)
    if match:
        task = int(match.group(1)) if task is None else task
        seed = int(match.group(2)) if seed is None else seed
    return task, seed


def _hazard_metric(summary: dict[str, Any], key: str, subkey: str) -> int | None:
    if summary.get("manual_hazard_review_required", False):
        return None
    return int(_metric(summary, key, subkey))


def _row(
    *,
    label: str,
    name: str,
    benchmark: str,
    task: int | None,
    seed: int | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "name": name,
        "benchmark": benchmark,
        "task": task,
        "seed": seed,
        "success": float(summary.get("success_rate", 0.0) or 0.0),
        "mean_steps": float(summary.get("mean_steps", 0.0) or 0.0),
        "mean_effective_control_steps": float(
            summary.get("mean_effective_control_steps", 0.0) or 0.0
        ),
        "hazard_metrics_available": not bool(
            summary.get("manual_hazard_review_required", False)
        ),
        "hazard_label_source": summary.get("hazard_label_source"),
        "hazard_events_any": _hazard_metric(summary, "hazard_events", "any"),
        "hazard_events_body": _hazard_metric(summary, "hazard_events", "body"),
        "hazard_events_object": _hazard_metric(summary, "hazard_events", "object"),
        "hazard_events_stuck": _hazard_metric(summary, "hazard_events", "stuck"),
        "hazard_steps_any": _hazard_metric(summary, "hazard_steps", "any"),
        "hazard_steps_body": _hazard_metric(summary, "hazard_steps", "body"),
        "hazard_steps_object": _hazard_metric(summary, "hazard_steps", "object"),
        "hazard_steps_stuck": _hazard_metric(summary, "hazard_steps", "stuck"),
        "events_per_1k": (
            None
            if summary.get("manual_hazard_review_required", False)
            else float(
                summary.get("events_per_1k", summary.get("hazard_events_per_1k", 0.0))
                or 0.0
            )
        ),
        "rollback_planned": int(summary.get("rollback_planned", 0) or 0),
        "rollback_failed": int(summary.get("rollback_failed", 0) or 0),
        "rollback_rendered_frames": int(summary.get("rollback_rendered_frames", 0) or 0),
    }


def load_rows(root: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        benchmark = str(payload.get("benchmark", "unknown"))
        task_payloads = payload.get("tasks")
        if isinstance(task_payloads, list) and task_payloads:
            for task_payload in task_payloads:
                task = task_payload.get("task_id")
                rows.append(
                    _row(
                        label=label,
                        name=f"{summary_path.parent.name}:{benchmark}:task{task}",
                        benchmark=benchmark,
                        task=int(task) if task is not None else None,
                        seed=int(payload["seed"]) if payload.get("seed") is not None else None,
                        summary=task_payload.get("summary", task_payload),
                    )
                )
            continue

        summary = payload.get("summary", payload)
        task, seed = _parse_task_seed(summary_path, payload)
        rows.append(
            _row(
                label=label,
                name=summary_path.parent.name,
                benchmark=benchmark,
                task=int(task) if task is not None else None,
                seed=int(seed) if seed is not None else None,
                summary=summary,
            )
        )
    return rows


def _sum_if_complete(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row[key] for row in rows]
    if any(value is None for value in values):
        return None
    return int(sum(values))


def _mean_if_complete(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows]
    if any(value is None for value in values):
        return None
    return float(sum(values) / len(values))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"n": 0}
    return {
        "n": count,
        "success_rate": sum(row["success"] for row in rows) / count,
        "mean_steps": sum(row["mean_steps"] for row in rows) / count,
        "mean_effective_control_steps": (
            sum(row["mean_effective_control_steps"] for row in rows) / count
        ),
        "hazard_metrics_available": all(row["hazard_metrics_available"] for row in rows),
        "hazard_events_any_sum": _sum_if_complete(rows, "hazard_events_any"),
        "hazard_events_body_sum": _sum_if_complete(rows, "hazard_events_body"),
        "hazard_events_object_sum": _sum_if_complete(rows, "hazard_events_object"),
        "hazard_events_stuck_sum": _sum_if_complete(rows, "hazard_events_stuck"),
        "hazard_steps_any_sum": _sum_if_complete(rows, "hazard_steps_any"),
        "hazard_steps_body_sum": _sum_if_complete(rows, "hazard_steps_body"),
        "hazard_steps_object_sum": _sum_if_complete(rows, "hazard_steps_object"),
        "hazard_steps_stuck_sum": _sum_if_complete(rows, "hazard_steps_stuck"),
        "events_per_1k_mean": _mean_if_complete(rows, "events_per_1k"),
        "rollback_planned_sum": sum(row["rollback_planned"] for row in rows),
        "rollback_failed_sum": sum(row["rollback_failed"] for row in rows),
        "rollback_rendered_frames_sum": sum(
            row["rollback_rendered_frames"] for row in rows
        ),
    }


def summarize_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['benchmark']}:{row['task']}"
        grouped.setdefault(key, []).append(row)
    return {key: summarize(grouped[key]) for key in sorted(grouped)}


def _delta(left: float | int | None, right: float | int | None) -> float | int | None:
    if left is None or right is None:
        return None
    return right - left


def paired_deltas(
    base_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = {
        (row["benchmark"], row["task"], row["seed"]): row
        for row in base_rows
    }
    candidate = {
        (row["benchmark"], row["task"], row["seed"]): row
        for row in candidate_rows
    }
    pairs = []
    for key in sorted(set(base) & set(candidate)):
        left = base[key]
        right = candidate[key]
        pairs.append(
            {
                "benchmark": key[0],
                "task": key[1],
                "seed": key[2],
                "base_success": left["success"],
                "candidate_success": right["success"],
                "success_delta": right["success"] - left["success"],
                "base_hazard_events_any": left["hazard_events_any"],
                "candidate_hazard_events_any": right["hazard_events_any"],
                "hazard_events_delta": _delta(
                    left["hazard_events_any"], right["hazard_events_any"]
                ),
                "base_hazard_steps_any": left["hazard_steps_any"],
                "candidate_hazard_steps_any": right["hazard_steps_any"],
                "hazard_steps_delta": _delta(
                    left["hazard_steps_any"], right["hazard_steps_any"]
                ),
                "base_steps": left["mean_steps"],
                "candidate_steps": right["mean_steps"],
                "steps_delta": right["mean_steps"] - left["mean_steps"],
                "rollback_planned": right["rollback_planned"],
                "rollback_failed": right["rollback_failed"],
            }
        )
    return pairs


def build_report(
    roots: list[tuple[str, Path]],
    base_label: str | None = None,
) -> dict[str, Any]:
    rows_by_label = {label: load_rows(root, label) for label, root in roots}
    report: dict[str, Any] = {
        "roots": {label: str(root) for label, root in roots},
        "summaries": {label: summarize(rows) for label, rows in rows_by_label.items()},
        "by_task": {
            label: summarize_by_task(rows) for label, rows in rows_by_label.items()
        },
        "rows": rows_by_label,
    }
    if base_label is not None:
        if base_label not in rows_by_label:
            raise ValueError(f"base label {base_label!r} is missing")
        base_rows = rows_by_label[base_label]
        base_summary = report["summaries"][base_label]
        comparisons = {}
        for label, rows in rows_by_label.items():
            if label == base_label:
                continue
            candidate_summary = report["summaries"][label]
            base_events = base_summary.get("hazard_events_any_sum")
            candidate_events = candidate_summary.get("hazard_events_any_sum")
            base_steps = base_summary.get("hazard_steps_any_sum")
            candidate_steps = candidate_summary.get("hazard_steps_any_sum")
            comparisons[label] = {
                "deltas": {
                    "success_rate_delta": (
                        candidate_summary["success_rate"] - base_summary["success_rate"]
                    ),
                    "hazard_events_any_delta": _delta(base_events, candidate_events),
                    "hazard_events_any_ratio_vs_base": (
                        candidate_events / base_events
                        if base_events not in (None, 0) and candidate_events is not None
                        else None
                    ),
                    "hazard_steps_any_delta": _delta(base_steps, candidate_steps),
                    "hazard_steps_any_ratio_vs_base": (
                        candidate_steps / base_steps
                        if base_steps not in (None, 0) and candidate_steps is not None
                        else None
                    ),
                    "mean_steps_delta": (
                        candidate_summary["mean_steps"] - base_summary["mean_steps"]
                    ),
                },
                "pairs": paired_deltas(base_rows, rows),
            }
        report["comparisons"] = comparisons
    return report


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("roots must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("root label must be non-empty")
    return label, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=parse_root,
        required=True,
        help="LABEL=closed_loop_root",
    )
    parser.add_argument("--base-label", help="Label used as paired-comparison baseline")
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)
    report = build_report(args.root, base_label=args.base_label)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
