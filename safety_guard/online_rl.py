from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .decider import Intervention
from .rl_decider import ActorCriticPolicy, ThreeAction


@dataclass(frozen=True)
class OnlineRewardConfig:
    hazard_penalty: float = -1.0
    hazard_onset_penalty: float = 0.0
    body_hazard_penalty: float = 0.0
    object_hazard_penalty: float = 0.0
    stuck_hazard_penalty: float = 0.0
    rollback_penalty: float = -0.005
    rollback_failure_penalty: float = -0.25
    rollback_rendered_frame_penalty: float = 0.0
    rollback_resolved_bonus: float = 0.0
    rollback_preemptive_bonus: float = 0.0
    rollback_unresolved_penalty: float = 0.0
    rollback_episode_success_bonus: float = 0.0
    rollback_episode_failure_penalty: float = 0.0
    noop_safe_bonus: float = 0.0
    noop_hazard_penalty: float = 0.0
    rollback_count_scale: float = 0.25
    record_penalty: float = -0.0005
    step_penalty: float = -1e-5
    completion_reward: float = 1.0
    task_reward_scale: float = 0.0


@dataclass(frozen=True)
class OnlineStepSignals:
    body_hazard: bool = False
    object_hazard: bool = False
    stuck_hazard: bool = False
    hazard_event: bool = False
    success: bool = False
    task_reward: float = 0.0
    eef_speed: float = 0.0
    arm_contact_count: int = 0

    @property
    def any_hazard(self) -> bool:
        return bool(self.body_hazard or self.object_hazard or self.stuck_hazard)

    def privileged_features(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.body_hazard),
                float(self.object_hazard),
                float(self.stuck_hazard),
                float(self.hazard_event),
                float(self.success),
                float(np.clip(self.eef_speed, 0.0, 5.0) / 5.0),
                float(np.clip(self.arm_contact_count, 0, 10) / 10.0),
            ],
            dtype=np.float32,
        )


def privileged_feature_dim() -> int:
    return int(OnlineStepSignals().privileged_features().shape[0])


def online_hazard_score(signals: OnlineStepSignals, config: OnlineRewardConfig | None = None) -> float:
    config = config or OnlineRewardConfig()
    channel_weights = (
        abs(float(config.body_hazard_penalty)),
        abs(float(config.object_hazard_penalty)),
        abs(float(config.stuck_hazard_penalty)),
    )
    if any(weight > 0.0 for weight in channel_weights):
        score = 0.0
        if signals.body_hazard:
            score += channel_weights[0]
        if signals.object_hazard:
            score += channel_weights[1]
        if signals.stuck_hazard:
            score += channel_weights[2]
        return float(score)
    return 1.0 if signals.any_hazard else 0.0


def online_safeloop_reward(
    signals: OnlineStepSignals,
    intervention: Intervention | ThreeAction | int | str,
    rollback_count: int = 0,
    rollback_failed: bool = False,
    rollback_rendered_frames: int = 0,
    config: OnlineRewardConfig | None = None,
) -> float:
    config = config or OnlineRewardConfig()
    name = _intervention_name(intervention)
    reward = float(config.step_penalty)
    if signals.any_hazard:
        reward += float(config.hazard_penalty)
    if signals.hazard_event:
        reward += float(config.hazard_onset_penalty)
    if signals.body_hazard:
        reward += float(config.body_hazard_penalty)
    if signals.object_hazard:
        reward += float(config.object_hazard_penalty)
    if signals.stuck_hazard:
        reward += float(config.stuck_hazard_penalty)
    if name == "rollback":
        reward += float(config.rollback_penalty) * (1.0 + float(rollback_count) * float(config.rollback_count_scale))
        reward += float(config.rollback_rendered_frame_penalty) * max(int(rollback_rendered_frames), 0)
        if rollback_failed:
            reward += float(config.rollback_failure_penalty)
    elif name == "record":
        reward += float(config.record_penalty)
    if signals.success:
        reward += float(config.completion_reward)
    reward += float(config.task_reward_scale) * float(signals.task_reward)
    return float(reward)


def rollback_outcome_credit(
    post_rollback_hazard_steps: int,
    observed_steps: int,
    pre_rollback_hazard_steps: int | None = None,
    pre_observed_steps: int | None = None,
    preemptive_warning: bool = False,
    config: OnlineRewardConfig | None = None,
) -> float:
    config = config or OnlineRewardConfig()
    if int(observed_steps) <= 0:
        return 0.0
    if pre_rollback_hazard_steps is not None and pre_observed_steps is not None and int(pre_observed_steps) > 0:
        pre_rate = float(pre_rollback_hazard_steps) / float(max(int(pre_observed_steps), 1))
        post_rate = float(post_rollback_hazard_steps) / float(max(int(observed_steps), 1))
        if pre_rate <= 0.0 and post_rate <= 0.0:
            return float(config.rollback_preemptive_bonus) if preemptive_warning else 0.0
        if post_rate <= 0.0 and pre_rate > 0.0:
            return float(config.rollback_resolved_bonus)
        improvement = pre_rate - post_rate
        if improvement > 0.0:
            scale = float(np.clip(improvement / max(pre_rate, 1e-6), 0.0, 1.0))
            return float(config.rollback_resolved_bonus) * scale
        if post_rate > 0.0:
            return float(config.rollback_unresolved_penalty)
        return 0.0
    if int(post_rollback_hazard_steps) > 0:
        return float(config.rollback_unresolved_penalty)
    return float(config.rollback_resolved_bonus)


def rollback_terminal_credit(
    success: bool,
    rollback_count: int,
    config: OnlineRewardConfig | None = None,
) -> float:
    count = int(rollback_count)
    if count <= 0:
        return 0.0
    config = config or OnlineRewardConfig()
    total = (
        float(config.rollback_episode_success_bonus)
        if bool(success)
        else float(config.rollback_episode_failure_penalty)
    )
    return float(total / count)


def noop_outcome_credit(
    hazard_steps: float,
    observed_steps: int,
    config: OnlineRewardConfig | None = None,
) -> float:
    config = config or OnlineRewardConfig()
    if int(observed_steps) <= 0:
        return 0.0
    if float(hazard_steps) > 0.0:
        return float(config.noop_hazard_penalty)
    return float(config.noop_safe_bonus)


class AsymmetricActorCriticPolicy(nn.Module):
    def __init__(self, actor_dim: int, critic_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.critic_dim = int(critic_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.actor_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.critic_encoder = nn.Sequential(
            nn.Linear(self.critic_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(self.hidden_dim, len(ThreeAction))
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward_actor(
        self,
        actor_features: torch.Tensor,
        action_mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None = None,
    ) -> torch.Tensor:
        encoded = self.actor_encoder(actor_features.float())
        logits = self.actor(encoded)
        return _apply_action_mask(logits, action_mask)

    def forward_critic(self, critic_features: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(critic_features.float())
        return self.critic(encoded).squeeze(-1)

    def forward(
        self,
        actor_features: torch.Tensor,
        critic_features: torch.Tensor | None = None,
        action_mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_input = actor_features if critic_features is None else critic_features
        return self.forward_actor(actor_features, action_mask=action_mask), self.forward_critic(critic_input)

    def sample_action(
        self,
        actor_features: np.ndarray | Sequence[float],
        action_mask: Sequence[bool] | None = None,
    ) -> tuple[ThreeAction, torch.Tensor]:
        tensor = torch.as_tensor(actor_features, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        logits = self.forward_actor(tensor, action_mask=action_mask)
        distribution = Categorical(logits=logits)
        action_tensor = distribution.sample()
        logprob = distribution.log_prob(action_tensor).squeeze(0)
        return ThreeAction(int(action_tensor.item())), logprob

    def greedy_action(
        self,
        actor_features: np.ndarray | Sequence[float],
        action_mask: Sequence[bool] | None = None,
    ) -> ThreeAction:
        tensor = torch.as_tensor(actor_features, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        logits = self.forward_actor(tensor, action_mask=action_mask)
        return ThreeAction(int(torch.argmax(logits, dim=-1).item()))


def sample_action_with_rollback_exploration(
    policy: AsymmetricActorCriticPolicy,
    actor_features: np.ndarray | Sequence[float],
    action_mask: Sequence[bool] | None = None,
    rollback_exploration_probability: float = 0.0,
) -> tuple[ThreeAction, torch.Tensor, bool]:
    action, logprob, exploration = sample_action_with_intervention_exploration(
        policy,
        actor_features,
        action_mask=action_mask,
        rollback_exploration_probability=rollback_exploration_probability,
    )
    return action, logprob, exploration == "rollback"


def sample_action_with_intervention_exploration(
    policy: AsymmetricActorCriticPolicy,
    actor_features: np.ndarray | Sequence[float],
    action_mask: Sequence[bool] | None = None,
    rollback_exploration_probability: float = 0.0,
    noop_exploration_probability: float = 0.0,
) -> tuple[ThreeAction, torch.Tensor, str | None]:
    action, logprob = policy.sample_action(actor_features, action_mask=action_mask)
    rollback_available = action_mask is None or bool(action_mask[int(ThreeAction.ROLLBACK)])
    noop_available = action_mask is None or bool(action_mask[int(ThreeAction.NOOP)])
    rollback_probability = float(np.clip(rollback_exploration_probability, 0.0, 1.0))
    noop_probability = float(np.clip(noop_exploration_probability, 0.0, 1.0))
    if not rollback_available:
        return action, logprob, None
    if rollback_probability <= 0.0 and (noop_probability <= 0.0 or not noop_available):
        return action, logprob, None

    draw = float(np.random.random())
    forced_action = None
    exploration = None
    if noop_available and draw < noop_probability:
        forced_action = ThreeAction.NOOP
        exploration = "noop"
    elif draw < min(noop_probability + rollback_probability, 1.0):
        forced_action = ThreeAction.ROLLBACK
        exploration = "rollback"
    if forced_action is None:
        return action, logprob, None

    tensor = torch.as_tensor(
        actor_features,
        dtype=torch.float32,
        device=next(policy.parameters()).device,
    ).unsqueeze(0)
    logits = policy.forward_actor(tensor, action_mask=action_mask)
    distribution = Categorical(logits=logits)
    forced_tensor = torch.as_tensor([int(forced_action)], dtype=torch.long, device=logits.device)
    forced_logprob = distribution.log_prob(forced_tensor).squeeze(0)
    return forced_action, forced_logprob, exploration


@dataclass(frozen=True)
class OnlineRolloutBatch:
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_masks: torch.Tensor
    teacher_actions: torch.Tensor | None = None
    policy_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class AsymmetricPPOConfig:
    gamma: float = 0.99
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    bc_coef: float = 1.0
    learning_rate: float = 3e-4
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    class_balanced_bc: bool = False
    max_grad_norm: float = 1.0


class AsymmetricPPOTrainer:
    def __init__(self, policy: AsymmetricActorCriticPolicy, config: AsymmetricPPOConfig | None = None):
        self.policy = policy
        self.config = config or AsymmetricPPOConfig()
        self.actor_parameters = list(policy.actor_encoder.parameters()) + list(policy.actor.parameters())
        self.critic_parameters = list(policy.critic_encoder.parameters()) + list(policy.critic.parameters())
        actor_lr = (
            self.config.learning_rate
            if self.config.actor_learning_rate is None
            else self.config.actor_learning_rate
        )
        critic_lr = (
            self.config.learning_rate
            if self.config.critic_learning_rate is None
            else self.config.critic_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=critic_lr)

    def compute_returns(self, rewards: Sequence[float], dones: Sequence[bool], last_value: float = 0.0) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = float(last_value)
        for index in reversed(range(len(rewards))):
            if bool(dones[index]):
                running_return = 0.0
            running_return = float(rewards[index]) + self.config.gamma * running_return
            returns[index] = running_return
        return returns

    def update(self, batch: OnlineRolloutBatch, epochs: int = 1, bc_coef: float | None = None) -> list[dict[str, float]]:
        if batch.actor_observations.numel() == 0:
            return []
        device = next(self.policy.parameters()).device
        actor_obs = batch.actor_observations.to(device)
        critic_obs = batch.critic_observations.to(device)
        actions = batch.actions.to(device)
        old_logprobs = batch.old_logprobs.to(device)
        rewards = batch.rewards.detach().cpu().numpy().tolist()
        dones = batch.dones.detach().cpu().numpy().astype(bool).tolist()
        returns = torch.as_tensor(self.compute_returns(rewards, dones), dtype=torch.float32, device=device)
        masks = batch.action_masks.to(device)
        teacher_actions = batch.teacher_actions.to(device) if batch.teacher_actions is not None else None
        policy_weights = (
            batch.policy_weights.to(device).float()
            if batch.policy_weights is not None
            else torch.ones_like(actions, dtype=torch.float32, device=device)
        )
        with torch.no_grad():
            values = self.policy.forward_critic(critic_obs)
            advantages = returns - values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        coef = self.config.bc_coef if bc_coef is None else float(bc_coef)
        with torch.no_grad():
            initial_actions = torch.argmax(self.policy.forward_actor(actor_obs, action_mask=masks), dim=-1)
        history: list[dict[str, float]] = []
        for _ in range(int(epochs)):
            logits, values = self.policy(actor_obs, critic_obs, action_mask=masks)
            distribution = Categorical(logits=logits)
            logprobs = distribution.log_prob(actions)
            ratio = torch.exp(logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
            surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
            policy_weight_sum = policy_weights.sum()
            if float(policy_weight_sum.detach().cpu()) > 0.0:
                policy_loss = -(surrogate * policy_weights).sum() / policy_weight_sum
            else:
                policy_loss = logits.sum() * 0.0
            value_loss = F.mse_loss(values, returns)
            entropy = distribution.entropy().mean()
            bc_loss = torch.zeros((), dtype=torch.float32, device=device)
            bc_valid_count = 0
            bc_ignored_count = 0
            bc_action_counts = torch.zeros(masks.shape[-1], dtype=torch.long, device=device)
            bc_accuracy = 0.0
            if teacher_actions is not None and coef > 0.0:
                row_indices = torch.arange(teacher_actions.shape[0], device=device)
                in_range = (teacher_actions >= 0) & (teacher_actions < masks.shape[-1])
                valid_teacher = torch.zeros_like(teacher_actions, dtype=torch.bool, device=device)
                if torch.any(in_range):
                    valid_rows = row_indices[in_range]
                    valid_actions = teacher_actions[in_range]
                    valid_teacher[in_range] = masks[valid_rows, valid_actions]
                bc_valid_count = int(valid_teacher.sum().detach().cpu())
                bc_ignored_count = int((~valid_teacher).sum().detach().cpu())
                if bc_valid_count > 0:
                    valid_actions = teacher_actions[valid_teacher]
                    bc_action_counts = torch.bincount(valid_actions, minlength=masks.shape[-1])
                    class_weights = None
                    if self.config.class_balanced_bc:
                        class_weights = torch.zeros(masks.shape[-1], dtype=torch.float32, device=device)
                        present = bc_action_counts > 0
                        class_weights[present] = (
                            float(bc_valid_count)
                            / (float(present.sum()) * bc_action_counts[present].float())
                        )
                    bc_loss = F.cross_entropy(
                        logits[valid_teacher],
                        valid_actions,
                        weight=class_weights,
                    )
                    bc_accuracy = float(
                        (torch.argmax(logits[valid_teacher], dim=-1) == valid_actions).float().mean().detach().cpu()
                    )
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
                + coef * bc_loss
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.config.max_grad_norm)
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.config.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            with torch.no_grad():
                current_actions = torch.argmax(self.policy.forward_actor(actor_obs, action_mask=masks), dim=-1)
                action_change_fraction = float((current_actions != initial_actions).float().mean().cpu())
            history.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "ppo_valid_count": float((policy_weights > 0.0).sum().detach().cpu()),
                    "ppo_ignored_count": float((policy_weights <= 0.0).sum().detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "bc_coef": float(coef),
                    "bc_valid_count": float(bc_valid_count),
                    "bc_ignored_count": float(bc_ignored_count),
                    "bc_noop_count": float(bc_action_counts[int(ThreeAction.NOOP)].detach().cpu()),
                    "bc_record_count": float(bc_action_counts[int(ThreeAction.RECORD)].detach().cpu()),
                    "bc_rollback_count": float(bc_action_counts[int(ThreeAction.ROLLBACK)].detach().cpu()),
                    "bc_accuracy": float(bc_accuracy),
                    "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
                    "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
                    "action_change_fraction": action_change_fraction,
                }
            )
        return history


def mark_online_episode_terminal(records: list[dict]) -> None:
    if records:
        records[-1]["done"] = True


def export_actor_checkpoint(
    asymmetric_policy: AsymmetricActorCriticPolicy,
    path,
) -> None:
    actor_policy = ActorCriticPolicy(
        input_dim=asymmetric_policy.actor_dim,
        hidden_dim=asymmetric_policy.hidden_dim,
    )
    actor_policy.encoder.load_state_dict(asymmetric_policy.actor_encoder.state_dict())
    actor_policy.actor.load_state_dict(asymmetric_policy.actor.state_dict())
    torch.save(
        {
            "input_dim": asymmetric_policy.actor_dim,
            "hidden_dim": asymmetric_policy.hidden_dim,
            "state_dict": actor_policy.state_dict(),
            "source": "asymmetric_online_actor",
            "critic_dim": asymmetric_policy.critic_dim,
        },
        path,
    )


def actor_checkpoint_to_asymmetric(
    checkpoint_path,
    critic_dim: int,
    device: str = "cpu",
) -> AsymmetricActorCriticPolicy:
    from .rl_training import load_policy_checkpoint

    actor_policy, metadata = load_policy_checkpoint(checkpoint_path)
    policy = AsymmetricActorCriticPolicy(
        actor_dim=int(metadata["input_dim"]),
        critic_dim=int(critic_dim),
        hidden_dim=int(metadata["hidden_dim"]),
    )
    policy.actor_encoder.load_state_dict(actor_policy.encoder.state_dict())
    policy.actor.load_state_dict(actor_policy.actor.state_dict())
    return policy.to(device)


def batch_from_online_records(records: Sequence[dict]) -> OnlineRolloutBatch:
    if not records:
        raise ValueError("records must not be empty")
    actor = np.stack([np.asarray(item["actor_features"], dtype=np.float32) for item in records])
    critic = np.stack([np.asarray(item["critic_features"], dtype=np.float32) for item in records])
    actions = np.asarray([int(item["action"]) for item in records], dtype=np.int64)
    old_logprobs = np.asarray([float(item.get("old_logprob", 0.0)) for item in records], dtype=np.float32)
    rewards = np.asarray([float(item["reward"]) for item in records], dtype=np.float32)
    dones = np.asarray([bool(item.get("done", False)) for item in records], dtype=bool)
    masks = np.asarray([item["action_mask"] for item in records], dtype=bool)
    policy_weights = np.asarray([float(item.get("ppo_weight", 1.0)) for item in records], dtype=np.float32)
    teacher = None
    if all("teacher_action" in item for item in records):
        teacher = torch.from_numpy(np.asarray([int(item["teacher_action"]) for item in records], dtype=np.int64))
    return OnlineRolloutBatch(
        actor_observations=torch.from_numpy(actor),
        critic_observations=torch.from_numpy(critic),
        actions=torch.from_numpy(actions).long(),
        old_logprobs=torch.from_numpy(old_logprobs),
        rewards=torch.from_numpy(rewards),
        dones=torch.from_numpy(dones),
        action_masks=torch.from_numpy(masks),
        teacher_actions=teacher,
        policy_weights=torch.from_numpy(policy_weights),
    )


def _intervention_name(value: Intervention | ThreeAction | int | str) -> str:
    if isinstance(value, Intervention):
        return value.value
    if isinstance(value, ThreeAction):
        return value.name.lower()
    if isinstance(value, int):
        return ThreeAction(value).name.lower()
    return str(value).lower()


def _apply_action_mask(
    logits: torch.Tensor,
    action_mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None,
) -> torch.Tensor:
    if action_mask is None:
        return logits
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand_as(logits)
    if mask.shape != logits.shape:
        raise ValueError(f"action_mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}")
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("each action mask row must allow at least one action")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
