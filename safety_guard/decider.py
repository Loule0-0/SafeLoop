from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .memory import WaypointMemory
from .risk import RiskVector


class Intervention(str, Enum):
    NOOP = "noop"
    RECORD = "record"
    ROLLBACK = "rollback"


@dataclass
class RuleBasedDecider:
    record_probability: float = 0.05
    rollback_probability: float = 0.8
    rollback_tth: float = 0.25
    min_record_interval: int = 1
    rollback_cooldown: int = 0

    def decide(
        self,
        risk: RiskVector,
        memory: WaypointMemory,
        step_index: int,
        last_record_step: int | None = None,
        last_rollback_step: int | None = None,
    ) -> Intervention:
        can_rollback = len(memory) > 0 and self._cooldown_elapsed(
            step_index, last_rollback_step, self.rollback_cooldown
        )
        if can_rollback and (
            risk.max_probability >= self.rollback_probability or risk.min_tth <= self.rollback_tth
        ):
            return Intervention.ROLLBACK

        can_record = self._cooldown_elapsed(step_index, last_record_step, self.min_record_interval)
        if can_record and risk.max_probability <= self.record_probability:
            return Intervention.RECORD

        return Intervention.NOOP

    @staticmethod
    def _cooldown_elapsed(step_index: int, previous_step: int | None, interval: int) -> bool:
        if previous_step is None:
            return True
        return (step_index - previous_step) >= interval
