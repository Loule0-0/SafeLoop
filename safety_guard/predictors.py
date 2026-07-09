from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .risk import RiskVector


@dataclass
class ConstantRiskPredictor:
    risk: RiskVector | Iterable[float] = RiskVector(0.0, 1.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.risk, RiskVector):
            self.risk = RiskVector.from_iterable(self.risk)

    def predict(self, observation, proposed_action, instruction: str | None = None) -> RiskVector:
        return self.risk


@dataclass
class ActionNormRiskPredictor:
    safe_norm: float = 0.5
    critical_norm: float = 2.0
    object_scale: float = 0.75

    def predict(self, observation, proposed_action, instruction: str | None = None) -> RiskVector:
        action = np.asarray(proposed_action, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(action))
        span = max(self.critical_norm - self.safe_norm, 1e-6)
        normalized = np.clip((norm - self.safe_norm) / span, 0.0, 1.0)
        probability = float(normalized)
        tth = float(1.0 - 0.9 * normalized)
        return RiskVector(
            body_probability=probability,
            body_tth=tth,
            object_probability=float(np.clip(probability * self.object_scale, 0.0, 1.0)),
            object_tth=tth,
        )
