from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .controller import extract_robot_joint_vector
from .decider import Intervention
from .memory import WaypointMemory
from .risk import RiskVector
from .rl_decider import ThreeAction
from .rl_decision_data import build_action_mask, build_decision_features
from .rl_training import load_policy_checkpoint


THREE_ACTION_TO_INTERVENTION = {
    ThreeAction.NOOP: Intervention.NOOP,
    ThreeAction.RECORD: Intervention.RECORD,
    ThreeAction.ROLLBACK: Intervention.ROLLBACK,
}


@dataclass
class RLPolicyDecider:
    checkpoint_path: Path | str
    device: str = "cpu"
    history_length: int = 3
    min_record_interval: int = 1
    rollback_cooldown: int = 0
    max_steps: int | None = None
    memory_normalizer: float = 10.0
    _policy: Any = field(init=False, repr=False)
    input_dim: int = field(init=False)
    hidden_dim: int = field(init=False)
    joint_dim: int = field(init=False)
    last_decision_info: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        policy, metadata = load_policy_checkpoint(Path(self.checkpoint_path))
        self._policy = policy.to(self.device)
        self._policy.eval()
        self.input_dim = int(metadata["input_dim"])
        self.hidden_dim = int(metadata["hidden_dim"])
        self.joint_dim = infer_joint_dim(self.input_dim, self.history_length, context_dim=4)

    def decide(
        self,
        risk: RiskVector,
        memory: WaypointMemory,
        step_index: int,
        last_record_step: int | None = None,
        last_rollback_step: int | None = None,
        observation: Any | None = None,
        proposed_action: Any | None = None,
        joint_history: Sequence[Sequence[float]] | None = None,
        risk_history: Sequence[RiskVector] | None = None,
        current_hazard_history: Sequence[Sequence[float]] | None = None,
        current_body_probability: float = 0.0,
        current_object_probability: float = 0.0,
        max_steps: int | None = None,
    ) -> Intervention:
        can_record = _cooldown_elapsed(step_index, last_record_step, self.min_record_interval)
        can_rollback = len(memory) > 0 and _cooldown_elapsed(
            step_index,
            last_rollback_step,
            self.rollback_cooldown,
        )
        action_mask = build_action_mask(can_record=can_record, can_rollback=can_rollback)

        joints = self._coerce_joint_history(joint_history, observation, proposed_action)
        risks = list(risk_history or [])
        currents = list(current_hazard_history or [])
        if not risks:
            risks = [risk]
        if not currents:
            currents = [(current_body_probability, current_object_probability)]

        progress_denominator = max_steps or self.max_steps
        context_values = [
            float(can_record),
            float(can_rollback),
            float(np.clip(len(memory) / max(self.memory_normalizer, 1e-6), 0.0, 1.0)),
            _safe_ratio(step_index, progress_denominator),
        ]
        features = build_decision_features(
            joint_history=joints,
            risk_history=risks,
            current_hazard_history=currents,
            context_values=context_values,
            history_length=self.history_length,
        )
        features = fit_feature_dim(features, self.input_dim)
        tensor = torch.as_tensor(features, dtype=torch.float32, device=next(self._policy.parameters()).device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._policy(tensor, action_mask=action_mask)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            action = ThreeAction(int(torch.argmax(logits, dim=-1).item()))
        self.last_decision_info = {
            "decider": "rl_policy",
            "step_index": int(step_index),
            "memory_size": int(len(memory)),
            "can_record": bool(can_record),
            "can_rollback": bool(can_rollback),
            "action_mask": [bool(item) for item in action_mask],
            "action_probabilities": [float(item) for item in probabilities.tolist()],
            "selected_action": THREE_ACTION_TO_INTERVENTION[action].value,
        }
        return THREE_ACTION_TO_INTERVENTION[ThreeAction(action)]

    def _coerce_joint_history(
        self,
        joint_history: Sequence[Sequence[float]] | None,
        observation: Any | None,
        proposed_action: Any | None,
    ) -> list[np.ndarray]:
        if joint_history:
            raw = joint_history
        else:
            raw = [extract_robot_joint_vector(observation, proposed_action)]
        return [fit_feature_dim(np.asarray(item, dtype=np.float32).reshape(-1), self.joint_dim) for item in raw]


@dataclass
class RollbackGateDecider:
    inner: Any
    max_rollbacks_per_episode: int | None = None
    current_hazard_threshold: float = 1.01
    current_body_threshold: float | None = None
    current_object_threshold: float | None = None
    max_current_object_probability: float | None = None
    future_probability_threshold: float = 1.01
    future_body_probability_threshold: float | None = None
    future_object_probability_threshold: float | None = None
    future_tth_threshold: float = 0.0
    future_object_tth_threshold: float | None = None
    allow_risk_override: bool = False
    min_rollback_step: int = 0
    rollback_count: int = 0
    last_decision_info: dict[str, Any] = field(default_factory=dict, init=False)

    def decide(
        self,
        risk: RiskVector,
        current_body_probability: float = 0.0,
        current_object_probability: float = 0.0,
        **kwargs: Any,
    ) -> Intervention:
        action = self._decide_inner(
            risk=risk,
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            **kwargs,
        )
        inner_info = getattr(self.inner, "last_decision_info", None)
        self.last_decision_info = self._decision_info(
            inner_action=action,
            final_action=action,
            allowed=action == Intervention.ROLLBACK,
            reason="inner_not_rollback" if action != Intervention.ROLLBACK else "allowed",
            risk=risk,
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            inner_info=inner_info,
        )
        step_index = int(kwargs.get("step_index", 0))
        if action != Intervention.ROLLBACK:
            if self._risk_override_allowed(
                risk=risk,
                current_body_probability=current_body_probability,
                current_object_probability=current_object_probability,
                inner_info=inner_info,
                step_index=step_index,
                kwargs=kwargs,
            ):
                self.rollback_count += 1
                self.last_decision_info = self._decision_info(
                    inner_action=action,
                    final_action=Intervention.ROLLBACK,
                    allowed=True,
                    reason="risk_override",
                    risk=risk,
                    current_body_probability=current_body_probability,
                    current_object_probability=current_object_probability,
                    inner_info=inner_info,
                    step_index=step_index,
                )
                return Intervention.ROLLBACK
            return action
        if step_index < int(self.min_rollback_step):
            self.last_decision_info = self._decision_info(
                inner_action=action,
                final_action=Intervention.NOOP,
                allowed=False,
                reason="before_min_rollback_step",
                risk=risk,
                current_body_probability=current_body_probability,
                current_object_probability=current_object_probability,
                inner_info=inner_info,
                step_index=step_index,
            )
            return Intervention.NOOP
        if self.max_rollbacks_per_episode is not None and self.rollback_count >= self.max_rollbacks_per_episode:
            self.last_decision_info = self._decision_info(
                inner_action=action,
                final_action=Intervention.NOOP,
                allowed=False,
                reason="rollback_budget_exhausted",
                risk=risk,
                current_body_probability=current_body_probability,
                current_object_probability=current_object_probability,
                inner_info=inner_info,
                step_index=step_index,
            )
            return Intervention.NOOP
        if not is_high_confidence_rollback(
            risk,
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            current_hazard_threshold=self.current_hazard_threshold,
            current_body_threshold=self.current_body_threshold,
            current_object_threshold=self.current_object_threshold,
            max_current_object_probability=self.max_current_object_probability,
            future_probability_threshold=self.future_probability_threshold,
            future_body_probability_threshold=self.future_body_probability_threshold,
            future_object_probability_threshold=self.future_object_probability_threshold,
            future_tth_threshold=self.future_tth_threshold,
            future_object_tth_threshold=self.future_object_tth_threshold,
        ):
            self.last_decision_info = self._decision_info(
                inner_action=action,
                final_action=Intervention.NOOP,
                allowed=False,
                reason="below_gate_threshold",
                risk=risk,
                current_body_probability=current_body_probability,
                current_object_probability=current_object_probability,
                inner_info=inner_info,
                step_index=step_index,
            )
            return Intervention.NOOP
        self.rollback_count += 1
        self.last_decision_info = self._decision_info(
            inner_action=action,
            final_action=Intervention.ROLLBACK,
            allowed=True,
            reason="allowed",
            risk=risk,
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            inner_info=inner_info,
            step_index=step_index,
        )
        return Intervention.ROLLBACK

    def _risk_override_allowed(
        self,
        risk: RiskVector,
        current_body_probability: float,
        current_object_probability: float,
        inner_info: Any | None,
        step_index: int,
        kwargs: dict[str, Any],
    ) -> bool:
        if not self.allow_risk_override:
            return False
        if step_index < int(self.min_rollback_step):
            return False
        if self.max_rollbacks_per_episode is not None and self.rollback_count >= self.max_rollbacks_per_episode:
            return False
        if not self._has_available_rollback_target(kwargs=kwargs, inner_info=inner_info):
            return False
        return is_high_confidence_rollback(
            risk,
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            current_hazard_threshold=self.current_hazard_threshold,
            current_body_threshold=self.current_body_threshold,
            current_object_threshold=self.current_object_threshold,
            max_current_object_probability=self.max_current_object_probability,
            future_probability_threshold=self.future_probability_threshold,
            future_body_probability_threshold=self.future_body_probability_threshold,
            future_object_probability_threshold=self.future_object_probability_threshold,
            future_tth_threshold=self.future_tth_threshold,
            future_object_tth_threshold=self.future_object_tth_threshold,
        )

    @staticmethod
    def _has_available_rollback_target(kwargs: dict[str, Any], inner_info: Any | None) -> bool:
        if isinstance(inner_info, dict) and "can_rollback" in inner_info:
            return bool(inner_info["can_rollback"])
        memory = kwargs.get("memory")
        return memory is not None and len(memory) > 0

    def _decide_inner(self, **kwargs: Any) -> Intervention:
        decide = self.inner.decide
        signature = inspect.signature(decide)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return decide(**kwargs)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return decide(**accepted)

    def _decision_info(
        self,
        inner_action: Intervention,
        final_action: Intervention,
        allowed: bool,
        reason: str,
        risk: RiskVector,
        current_body_probability: float,
        current_object_probability: float,
        inner_info: Any | None,
        step_index: int | None = None,
    ) -> dict[str, Any]:
        return {
            "decider": "rollback_gate",
            "step_index": None if step_index is None else int(step_index),
            "inner_action": inner_action.value,
            "final_action": final_action.value,
            "gate_allowed": bool(allowed),
            "gate_reason": reason,
            "rollback_count": int(self.rollback_count),
            "min_rollback_step": int(self.min_rollback_step),
            "max_rollbacks_per_episode": (
                None if self.max_rollbacks_per_episode is None else int(self.max_rollbacks_per_episode)
            ),
            "current_hazard_threshold": float(self.current_hazard_threshold),
            "current_body_threshold": (
                None if self.current_body_threshold is None else float(self.current_body_threshold)
            ),
            "current_object_threshold": (
                None if self.current_object_threshold is None else float(self.current_object_threshold)
            ),
            "max_current_object_probability": (
                None
                if self.max_current_object_probability is None
                else float(self.max_current_object_probability)
            ),
            "future_probability_threshold": float(self.future_probability_threshold),
            "future_body_probability_threshold": (
                None
                if self.future_body_probability_threshold is None
                else float(self.future_body_probability_threshold)
            ),
            "future_object_probability_threshold": (
                None
                if self.future_object_probability_threshold is None
                else float(self.future_object_probability_threshold)
            ),
            "future_tth_threshold": float(self.future_tth_threshold),
            "future_object_tth_threshold": (
                None
                if self.future_object_tth_threshold is None
                else float(self.future_object_tth_threshold)
            ),
            "allow_risk_override": bool(self.allow_risk_override),
            "risk": {
                "body_probability": float(risk.body_probability),
                "body_tth": float(risk.body_tth),
                "object_probability": float(risk.object_probability),
                "object_tth": float(risk.object_tth),
                "current_body_probability": float(current_body_probability),
                "current_object_probability": float(current_object_probability),
            },
            "inner": inner_info,
        }


def is_high_confidence_rollback(
    risk: RiskVector,
    current_body_probability: float,
    current_object_probability: float,
    current_hazard_threshold: float = 1.01,
    current_body_threshold: float | None = None,
    current_object_threshold: float | None = None,
    max_current_object_probability: float | None = None,
    future_probability_threshold: float = 1.01,
    future_body_probability_threshold: float | None = None,
    future_object_probability_threshold: float | None = None,
    future_tth_threshold: float = 0.0,
    future_object_tth_threshold: float | None = None,
) -> bool:
    if (
        max_current_object_probability is not None
        and float(current_object_probability) > float(max_current_object_probability)
    ):
        return False
    body_threshold = (
        float(current_body_threshold)
        if current_body_threshold is not None
        else float(current_hazard_threshold)
    )
    object_threshold = (
        float(current_object_threshold)
        if current_object_threshold is not None
        else float(current_hazard_threshold)
    )
    if float(current_body_probability) >= body_threshold:
        return True
    if float(current_object_probability) >= object_threshold:
        return True
    future_body_threshold = (
        float(future_body_probability_threshold)
        if future_body_probability_threshold is not None
        else float(future_probability_threshold)
    )
    future_object_threshold = (
        float(future_object_probability_threshold)
        if future_object_probability_threshold is not None
        else float(future_probability_threshold)
    )
    object_tth_threshold = (
        float(future_object_tth_threshold)
        if future_object_tth_threshold is not None
        else float(future_tth_threshold)
    )
    if float(risk.body_probability) >= future_body_threshold and float(risk.body_tth) <= float(future_tth_threshold):
        return True
    if (
        float(risk.object_probability) >= future_object_threshold
        and float(risk.object_tth) <= object_tth_threshold
    ):
        return True
    return False


def infer_joint_dim(input_dim: int, history_length: int, context_dim: int = 4) -> int:
    numerator = int(input_dim) - int(context_dim)
    if numerator <= 0 or numerator % int(history_length) != 0:
        return 9
    row_dim = numerator // int(history_length)
    joint_dim = row_dim - 6
    return int(joint_dim) if joint_dim > 0 else 9


def fit_feature_dim(values: np.ndarray, target_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    target_dim = int(target_dim)
    if array.shape[0] == target_dim:
        return array.astype(np.float32)
    if array.shape[0] > target_dim:
        return array[:target_dim].astype(np.float32)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: array.shape[0]] = array
    return padded


def _cooldown_elapsed(step_index: int, previous_step: int | None, interval: int) -> bool:
    if previous_step is None:
        return True
    return (int(step_index) - int(previous_step)) >= int(interval)


def _safe_ratio(value: int, denominator: int | None) -> float:
    if denominator is None or denominator <= 0:
        return 0.0
    return float(np.clip(float(value) / float(denominator), 0.0, 1.0))
