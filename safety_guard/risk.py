from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RiskVector:
    body_probability: float
    body_tth: float
    object_probability: float
    object_tth: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "RiskVector":
        array = np.asarray(list(values), dtype=np.float32)
        if array.shape != (4,):
            raise ValueError(f"RiskVector needs four values, got shape {array.shape}")
        return cls(
            body_probability=float(array[0]),
            body_tth=float(array[1]),
            object_probability=float(array[2]),
            object_tth=float(array[3]),
        )

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.body_probability,
                self.body_tth,
                self.object_probability,
                self.object_tth,
            ],
            dtype=np.float32,
        )

    @property
    def max_probability(self) -> float:
        return max(self.body_probability, self.object_probability)

    @property
    def min_tth(self) -> float:
        return min(self.body_tth, self.object_tth)
