from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any


def _has_any_hazard(signals: object) -> bool:
    return bool(
        getattr(signals, "any_hazard", False)
        or getattr(signals, "body_hazard", False)
        or getattr(signals, "object_hazard", False)
    )


def select_prehazard_sample_offsets(
    hazard_timeline: Sequence[object],
    *,
    tau: int,
    stride: int,
    existing_indices: set[int] | None = None,
) -> dict[int, int]:
    if int(stride) <= 0 or int(tau) <= 0:
        return {}
    existing = {int(index) for index in (existing_indices or set())}
    selected: dict[int, int] = {}
    horizon = int(tau)
    interval = max(1, int(stride))

    for hazard_index, signals in enumerate(hazard_timeline):
        if not _has_any_hazard(signals):
            continue
        start_index = max(0, int(hazard_index) - horizon)
        for sample_index in range(start_index, int(hazard_index)):
            if sample_index in existing or _has_any_hazard(hazard_timeline[sample_index]):
                continue
            offset = int(hazard_index) - int(sample_index)
            if offset != 1 and offset % interval != 0:
                continue
            previous = selected.get(sample_index)
            if previous is None or offset < previous:
                selected[sample_index] = offset

    return dict(sorted(selected.items()))


def append_prehazard_rollout_samples(
    samples: MutableSequence[dict[str, Any]],
    candidates_by_step: Mapping[int, dict[str, Any]],
    hazard_timeline: Sequence[object],
    *,
    tau: int,
    stride: int,
) -> int:
    existing_indices = {int(sample["sample_index"]) for sample in samples if "sample_index" in sample}
    selected = select_prehazard_sample_offsets(
        hazard_timeline,
        tau=tau,
        stride=stride,
        existing_indices=existing_indices,
    )
    added = 0
    for sample_index, offset in selected.items():
        candidate = candidates_by_step.get(int(sample_index))
        if candidate is None:
            continue
        sample = dict(candidate)
        sample["images"] = [
            image.copy() if hasattr(image, "copy") else image
            for image in candidate.get("images", [])
        ]
        metadata = dict(candidate.get("metadata") or {})
        metadata["source"] = "pre_hazard_dense"
        metadata["nearest_hazard_offset"] = int(offset)
        sample["metadata"] = metadata
        samples.append(sample)
        added += 1
    return added
