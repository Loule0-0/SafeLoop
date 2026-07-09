from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ThreeAction(IntEnum):
    NOOP = 0
    RECORD = 1
    ROLLBACK = 2


class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, len(ThreeAction))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features.float())
        logits = self.actor(encoded)
        logits = self._apply_action_mask(logits, action_mask)
        value = self.critic(encoded).squeeze(-1)
        return logits, value

    def sample_action(
        self,
        features: np.ndarray | Sequence[float],
        action_mask: Sequence[bool] | None = None,
    ) -> tuple[ThreeAction, torch.Tensor, torch.Tensor]:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        logits, value = self(tensor, action_mask=action_mask)
        distribution = Categorical(logits=logits)
        action_tensor = distribution.sample()
        logprob = distribution.log_prob(action_tensor).squeeze(0)
        return ThreeAction(int(action_tensor.item())), logprob, value.squeeze(0)

    def greedy_action(
        self,
        features: np.ndarray | Sequence[float],
        action_mask: Sequence[bool] | None = None,
    ) -> ThreeAction:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        logits, _ = self(tensor, action_mask=action_mask)
        return ThreeAction(int(torch.argmax(logits, dim=-1).item()))

    def _apply_action_mask(
        self,
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


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0


class PPOTrainer:
    def __init__(self, policy: ActorCriticPolicy, config: PPOConfig | None = None):
        self.policy = policy
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    def compute_returns(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        last_value: float = 0.0,
    ) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = float(last_value)
        for index in reversed(range(len(rewards))):
            if dones[index]:
                running_return = 0.0
            running_return = float(rewards[index]) + self.config.gamma * running_return
            returns[index] = running_return
        return returns

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> dict[str, float]:
        logits, values = self.policy(observations, action_mask=action_masks)
        distribution = Categorical(logits=logits)
        logprobs = distribution.log_prob(actions)
        ratio = torch.exp(logprobs - old_logprobs)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }
