from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

import numpy as np

from .risk import RiskVector


@dataclass
class Waypoint:
    step_index: int
    state: Any
    risk: RiskVector | None = None
    metadata: dict = field(default_factory=dict)


class WaypointMemory:
    def __init__(self, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: Deque[Waypoint] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def record(
        self,
        step_index: int,
        state: Any,
        risk: RiskVector | None = None,
        metadata: dict | None = None,
    ) -> Waypoint:
        stored_state = state if not _is_array_like_state(state) else np.asarray(state, dtype=np.float32).copy()
        waypoint = Waypoint(
            step_index=step_index,
            state=stored_state,
            risk=risk,
            metadata=dict(metadata or {}),
        )
        self._items.append(waypoint)
        return waypoint

    def latest(self) -> Waypoint:
        if not self._items:
            raise IndexError("waypoint memory is empty")
        return self._items[-1]

    def select_rollback(
        self,
        current_step_index: int | None = None,
        safe_score_threshold: float = 0.6,
        max_safe_age: int = 120,
        min_safe_age: int = 30,
        require_safe: bool = False,
    ) -> Waypoint:
        if not self._items:
            raise IndexError("waypoint memory is empty")
        if current_step_index is not None:
            age_eligible: list[Waypoint] = []
            candidates: list[tuple[float, int, Waypoint]] = []
            for waypoint in reversed(self._items):
                age = int(current_step_index) - int(waypoint.step_index)
                if age < int(min_safe_age) or age > int(max_safe_age):
                    continue
                age_eligible.append(waypoint)
                risk_score = _waypoint_risk_score(waypoint)
                if risk_score <= float(safe_score_threshold):
                    candidates.append((risk_score, -int(waypoint.step_index), waypoint))
            if candidates:
                return min(candidates, key=lambda item: (item[0], item[1]))[2]
            if age_eligible and not require_safe:
                return max(age_eligible, key=lambda waypoint: int(waypoint.step_index))
            if age_eligible:
                raise IndexError("no mature low-risk waypoint is available")
            raise IndexError("no waypoint satisfies the rollback age constraints")
        return self.latest()

    def clear(self) -> None:
        self._items.clear()


def _is_array_like_state(state: Any) -> bool:
    return isinstance(state, (np.ndarray, list, tuple))


def _waypoint_risk_score(waypoint: Waypoint) -> float:
    scores: list[float] = []
    if waypoint.risk is not None:
        scores.append(float(waypoint.risk.max_probability))
        scores.append(float(max(0.0, 1.0 - waypoint.risk.min_tth)) * 0.25)
    for key in ("current_body_probability", "current_object_probability"):
        if key in waypoint.metadata:
            scores.append(float(waypoint.metadata[key]))
    if not scores:
        return 0.0
    return float(max(scores))
