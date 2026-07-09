from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .rl_decider import ActorCriticPolicy, PPOConfig, PPOTrainer


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_masks: torch.Tensor | None = None
    reward_matrix: torch.Tensor | None = None


def make_synthetic_rollout(input_dim: int, steps: int, seed: int = 0) -> RolloutBatch:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(steps, input_dim)).astype(np.float32)
    record_score = observations[:, 1] if input_dim > 1 else np.zeros(steps, dtype=np.float32)
    rollback_score = observations[:, 0]
    actions = np.where(rollback_score > 0.7, 2, np.where(record_score > 0.4, 1, 0)).astype(np.int64)
    rewards = np.where(actions == 2, 1.0, np.where(actions == 1, 0.4, 0.1)).astype(np.float32)
    dones = np.zeros(steps, dtype=bool)
    dones[31::32] = True
    dones[-1] = True
    old_logprobs = np.zeros(steps, dtype=np.float32)

    return RolloutBatch(
        observations=torch.from_numpy(observations),
        actions=torch.from_numpy(actions).long(),
        old_logprobs=torch.from_numpy(old_logprobs),
        rewards=torch.from_numpy(rewards),
        dones=torch.from_numpy(dones),
        action_masks=torch.ones(steps, 3, dtype=torch.bool),
    )


def load_rollout_npz(path: Path) -> RolloutBatch:
    data = np.load(path)
    required = {"observations", "actions", "old_logprobs", "rewards", "dones"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"rollout npz is missing fields: {sorted(missing)}")
    action_masks = None
    if "action_masks" in data.files:
        action_masks = torch.as_tensor(data["action_masks"], dtype=torch.bool)
    reward_matrix = None
    if "reward_matrix" in data.files:
        reward_matrix = torch.as_tensor(data["reward_matrix"], dtype=torch.float32)
    return RolloutBatch(
        observations=torch.as_tensor(data["observations"], dtype=torch.float32),
        actions=torch.as_tensor(data["actions"], dtype=torch.long),
        old_logprobs=torch.as_tensor(data["old_logprobs"], dtype=torch.float32),
        rewards=torch.as_tensor(data["rewards"], dtype=torch.float32),
        dones=torch.as_tensor(data["dones"], dtype=torch.bool),
        action_masks=action_masks,
        reward_matrix=reward_matrix,
    )


def train_ppo_on_rollout(
    policy: ActorCriticPolicy,
    batch: RolloutBatch,
    epochs: int,
    config: PPOConfig | None = None,
) -> list[dict[str, float]]:
    trainer = PPOTrainer(policy, config=config)
    returns_np = trainer.compute_returns(
        rewards=batch.rewards.cpu().numpy().tolist(),
        dones=batch.dones.cpu().numpy().astype(bool).tolist(),
    )
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=batch.observations.device)
    with torch.no_grad():
        _, values = policy(batch.observations)
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    history = []
    for _ in range(epochs):
        history.append(
            trainer.update(
                observations=batch.observations,
                actions=batch.actions,
                old_logprobs=batch.old_logprobs,
                returns=returns,
                advantages=advantages,
                action_masks=batch.action_masks,
            )
        )
    return history


def evaluate_decision_policy(policy: ActorCriticPolicy, batch: RolloutBatch) -> dict[str, float]:
    if batch.reward_matrix is None:
        raise ValueError("batch.reward_matrix is required for decision-policy evaluation")
    device = next(policy.parameters()).device
    observations = batch.observations.to(device)
    reward_matrix = batch.reward_matrix.to(device)
    if batch.action_masks is None:
        action_masks = torch.ones_like(reward_matrix, dtype=torch.bool, device=device)
    else:
        action_masks = batch.action_masks.to(device)

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        logits, _ = policy(observations, action_mask=action_masks)
        actions = torch.argmax(logits, dim=-1)
        row_indices = torch.arange(actions.numel(), device=device)
        valid_actions = action_masks[row_indices, actions]
        chosen_rewards = reward_matrix[row_indices, actions]

        masked_rewards = reward_matrix.masked_fill(~action_masks, torch.finfo(reward_matrix.dtype).min)
        oracle_actions = torch.argmax(masked_rewards, dim=-1)
        oracle_rewards = reward_matrix[row_indices, oracle_actions]
        random_valid_rewards = (reward_matrix * action_masks.float()).sum(dim=-1) / action_masks.float().sum(dim=-1)
        oracle_match = (actions == oracle_actions).float().mean()
        action_counts = torch.bincount(actions.detach().cpu(), minlength=3).float() / max(actions.numel(), 1)
    if was_training:
        policy.train()

    return {
        "mean_reward": float(chosen_rewards.mean().detach().cpu()),
        "oracle_mean_reward": float(oracle_rewards.mean().detach().cpu()),
        "random_valid_mean_reward": float(random_valid_rewards.mean().detach().cpu()),
        "oracle_action_match": float(oracle_match.detach().cpu()),
        "valid_action_rate": float(valid_actions.float().mean().detach().cpu()),
        "noop_rate": float(action_counts[0]),
        "record_rate": float(action_counts[1]),
        "rollback_rate": float(action_counts[2]),
    }


def train_oracle_policy_on_rollout(
    policy: ActorCriticPolicy,
    batch: RolloutBatch,
    epochs: int,
    learning_rate: float = 1e-3,
    value_coef: float = 0.1,
    entropy_coef: float = 0.0,
    action_weights: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    if batch.reward_matrix is None:
        raise ValueError("batch.reward_matrix is required for oracle policy training")
    if epochs <= 0:
        return []

    device = next(policy.parameters()).device
    observations = batch.observations.to(device)
    reward_matrix = batch.reward_matrix.to(device)
    if batch.action_masks is None:
        action_masks = torch.ones_like(reward_matrix, dtype=torch.bool, device=device)
    else:
        action_masks = batch.action_masks.to(device)

    masked_rewards = reward_matrix.masked_fill(~action_masks, torch.finfo(reward_matrix.dtype).min)
    target_actions = torch.argmax(masked_rewards, dim=-1)
    row_indices = torch.arange(target_actions.numel(), device=device)
    target_values = reward_matrix[row_indices, target_actions].detach()

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    loss_weights = (
        torch.as_tensor(action_weights, dtype=torch.float32, device=device)
        if action_weights is not None
        else None
    )
    history: list[dict[str, float]] = []
    policy.train()
    for _ in range(epochs):
        logits, values = policy(observations, action_mask=action_masks)
        distribution = Categorical(logits=logits)
        policy_loss = F.cross_entropy(logits, target_actions, weight=loss_weights)
        value_loss = F.mse_loss(values, target_values)
        entropy = distribution.entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        history.append(
            {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
        )
    return history


def save_policy_checkpoint(
    policy: ActorCriticPolicy,
    path: Path,
    input_dim: int,
    hidden_dim: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "state_dict": policy.state_dict(),
        },
        path,
    )


def load_policy_checkpoint(path: Path) -> tuple[ActorCriticPolicy, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    policy = ActorCriticPolicy(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    return policy, {"input_dim": int(checkpoint["input_dim"]), "hidden_dim": int(checkpoint["hidden_dim"])}
