from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .risk import RiskVector
from .rl_decider import ThreeAction
from .rl_training import RolloutBatch


@dataclass(frozen=True)
class DecisionContext:
    features: np.ndarray
    risk: RiskVector
    current_body_probability: float
    current_object_probability: float
    can_record: bool
    can_rollback: bool
    step_index: int = 0
    memory_size: int = 0
    done: bool = False

    @property
    def current_hazard_probability(self) -> float:
        return max(float(self.current_body_probability), float(self.current_object_probability))


@dataclass(frozen=True)
class DecisionRewardConfig:
    safe_probability: float = 0.12
    safe_tth: float = 0.75
    current_safe_probability: float = 0.20
    rollback_probability: float = 0.55
    rollback_tth: float = 0.35
    current_rollback_probability: float = 0.45
    invalid_action_penalty: float = -4.0
    noop_safe_reward: float = 0.15
    noop_danger_penalty: float = -2.0
    record_safe_reward: float = 0.75
    record_risky_penalty: float = -1.0
    record_cost: float = 0.05
    rollback_reward: float = 2.0
    rollback_false_positive_penalty: float = -0.6
    rollback_cost: float = 0.20


@dataclass(frozen=True)
class DecisionRecord:
    context: DecisionContext
    action: ThreeAction | int
    reward: float | None = None
    old_logprob: float = 0.0
    done: bool | None = None
    action_rewards: Sequence[float] | None = None


def build_decision_features(
    joint_history: Sequence[Sequence[float]],
    risk_history: Sequence[RiskVector],
    current_hazard_history: Sequence[Sequence[float]],
    context_values: Sequence[float] = (),
    history_length: int = 3,
) -> np.ndarray:
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if not (len(joint_history) == len(risk_history) == len(current_hazard_history)):
        raise ValueError("joint_history, risk_history, and current_hazard_history must have the same length")

    joint_arrays = [np.asarray(joints, dtype=np.float32).reshape(-1) for joints in joint_history]
    joint_dim = int(joint_arrays[-1].shape[0]) if joint_arrays else 7
    row_dim = joint_dim + 6
    rows: list[np.ndarray] = []
    start = max(0, len(joint_arrays) - history_length)

    for _ in range(history_length - len(joint_arrays[start:])):
        rows.append(np.zeros(row_dim, dtype=np.float32))

    for joints, risk, current in zip(
        joint_arrays[start:],
        list(risk_history)[start:],
        list(current_hazard_history)[start:],
    ):
        if joints.shape[0] != joint_dim:
            raise ValueError("all joint vectors must have the same dimension")
        current_array = np.asarray(current, dtype=np.float32).reshape(-1)
        if current_array.shape != (2,):
            raise ValueError(f"current hazard history entries must have shape (2,), got {current_array.shape}")
        rows.append(np.concatenate([joints, risk.as_array(), current_array]).astype(np.float32))

    context = np.asarray(list(context_values), dtype=np.float32).reshape(-1)
    return np.concatenate([*rows, context], axis=0).astype(np.float32)


def build_action_mask(can_record: bool, can_rollback: bool) -> list[bool]:
    return [True, bool(can_record), bool(can_rollback)]


class DecisionRewardModel:
    def __init__(self, config: DecisionRewardConfig | None = None) -> None:
        self.config = config or DecisionRewardConfig()

    def action_rewards(self, context: DecisionContext) -> np.ndarray:
        mask = build_action_mask(context.can_record, context.can_rollback)
        urgency = self._danger_urgency(context)
        is_dangerous = self._is_dangerous(context)
        is_safe_record = self._is_safe_record(context)

        rewards = np.zeros(len(ThreeAction), dtype=np.float32)
        rewards[ThreeAction.NOOP] = (
            self.config.noop_safe_reward
            if not is_dangerous
            else self.config.noop_danger_penalty * max(urgency, 0.1)
        )
        rewards[ThreeAction.RECORD] = (
            self.config.record_safe_reward - self.config.record_cost
            if is_safe_record
            else self.config.record_risky_penalty * max(urgency, 0.1) - self.config.record_cost
        )
        rewards[ThreeAction.ROLLBACK] = (
            self.config.rollback_reward * max(urgency, 0.1) - self.config.rollback_cost
            if is_dangerous
            else self.config.rollback_false_positive_penalty - self.config.rollback_cost
        )
        for action, is_valid in enumerate(mask):
            if not is_valid:
                rewards[action] = self.config.invalid_action_penalty
        return rewards

    def best_action(self, context: DecisionContext) -> ThreeAction:
        return ThreeAction(int(np.argmax(self.action_rewards(context))))

    def _is_safe_record(self, context: DecisionContext) -> bool:
        return (
            context.can_record
            and context.risk.max_probability <= self.config.safe_probability
            and context.risk.min_tth >= self.config.safe_tth
            and context.current_hazard_probability <= self.config.current_safe_probability
        )

    def _is_dangerous(self, context: DecisionContext) -> bool:
        return (
            context.risk.max_probability >= self.config.rollback_probability
            or context.risk.min_tth <= self.config.rollback_tth
            or context.current_hazard_probability >= self.config.current_rollback_probability
        )

    def _danger_urgency(self, context: DecisionContext) -> float:
        return float(
            max(
                context.risk.max_probability,
                1.0 - context.risk.min_tth,
                context.current_hazard_probability,
            )
        )


def decision_records_to_rollout_batch(
    records: Sequence[DecisionRecord],
    reward_model: DecisionRewardModel | None = None,
) -> RolloutBatch:
    if not records:
        raise ValueError("records must not be empty")
    reward_model = reward_model or DecisionRewardModel()
    observations = np.stack([np.asarray(record.context.features, dtype=np.float32).reshape(-1) for record in records])
    actions = np.asarray([int(record.action) for record in records], dtype=np.int64)
    old_logprobs = np.asarray([float(record.old_logprob) for record in records], dtype=np.float32)
    action_masks = np.asarray(
        [build_action_mask(record.context.can_record, record.context.can_rollback) for record in records],
        dtype=bool,
    )
    reward_matrix = np.stack(
        [
            np.asarray(record.action_rewards, dtype=np.float32)
            if record.action_rewards is not None
            else reward_model.action_rewards(record.context)
            for record in records
        ]
    )
    rewards = np.asarray(
        [
            float(record.reward)
            if record.reward is not None
            else float(reward_matrix[index, actions[index]])
            for index, record in enumerate(records)
        ],
        dtype=np.float32,
    )
    dones = np.asarray(
        [
            bool(record.done) if record.done is not None else bool(record.context.done)
            for record in records
        ],
        dtype=bool,
    )
    return RolloutBatch(
        observations=torch.from_numpy(observations),
        actions=torch.from_numpy(actions).long(),
        old_logprobs=torch.from_numpy(old_logprobs),
        rewards=torch.from_numpy(rewards),
        dones=torch.from_numpy(dones),
        action_masks=torch.from_numpy(action_masks),
        reward_matrix=torch.from_numpy(reward_matrix.astype(np.float32)),
    )


def save_decision_rollout_npz(batch: RolloutBatch, path) -> None:
    path = str(path)
    arrays = {
        "observations": batch.observations.detach().cpu().numpy().astype(np.float32),
        "actions": batch.actions.detach().cpu().numpy().astype(np.int64),
        "old_logprobs": batch.old_logprobs.detach().cpu().numpy().astype(np.float32),
        "rewards": batch.rewards.detach().cpu().numpy().astype(np.float32),
        "dones": batch.dones.detach().cpu().numpy().astype(bool),
    }
    if batch.action_masks is not None:
        arrays["action_masks"] = batch.action_masks.detach().cpu().numpy().astype(bool)
    if batch.reward_matrix is not None:
        arrays["reward_matrix"] = batch.reward_matrix.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)
