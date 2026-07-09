from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from safety_guard.rl_decider import ActorCriticPolicy, PPOConfig
from safety_guard.rl_training import (
    RolloutBatch,
    evaluate_decision_policy,
    load_rollout_npz,
    make_synthetic_rollout,
    save_policy_checkpoint,
    train_oracle_policy_on_rollout,
    train_ppo_on_rollout,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-action SafeLoop policy: noop, record, rollback.")
    parser.add_argument("--rollout-npz", type=Path, help="Optional rollout npz with observations/actions/old_logprobs/rewards/dones.")
    parser.add_argument("--input-dim", type=int, help="Feature size. Inferred from --rollout-npz when omitted.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--synthetic-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--eval-before-after", action="store_true", help="Evaluate with reward_matrix before and after PPO.")
    parser.add_argument(
        "--oracle-pretrain-epochs",
        type=int,
        help="Supervised warm-up epochs against the reward_matrix oracle. Defaults to 50 when reward_matrix exists.",
    )
    parser.add_argument("--oracle-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--oracle-pretrain-value-coef", type=float, default=0.1)
    parser.add_argument(
        "--oracle-action-weights",
        type=str,
        help="Optional comma-separated CE weights for noop,record,rollback during oracle pretrain.",
    )
    parser.add_argument(
        "--min-reward-improvement",
        type=float,
        help="Optional gate: fail if eval_after mean_reward - eval_before mean_reward is below this value.",
    )
    return parser.parse_args(argv)


def move_batch_to_device(batch: RolloutBatch, device: torch.device) -> RolloutBatch:
    return RolloutBatch(
        observations=batch.observations.to(device),
        actions=batch.actions.to(device),
        old_logprobs=batch.old_logprobs.to(device),
        rewards=batch.rewards.to(device),
        dones=batch.dones.to(device),
        action_masks=batch.action_masks.to(device) if batch.action_masks is not None else None,
        reward_matrix=batch.reward_matrix.to(device) if batch.reward_matrix is not None else None,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.rollout_npz:
        batch = load_rollout_npz(args.rollout_npz)
        input_dim = int(args.input_dim or batch.observations.shape[1])
        if int(batch.observations.shape[1]) != input_dim:
            raise ValueError(
                f"--input-dim={input_dim} does not match rollout observations {tuple(batch.observations.shape)}"
            )
    else:
        if args.input_dim is None:
            raise ValueError("--input-dim is required when --rollout-npz is not provided")
        input_dim = int(args.input_dim)
        batch = make_synthetic_rollout(
            input_dim=input_dim,
            steps=args.synthetic_steps,
            seed=args.seed,
        )

    policy = ActorCriticPolicy(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    batch = move_batch_to_device(batch, device)

    should_eval = bool(args.eval_before_after or args.min_reward_improvement is not None or batch.reward_matrix is not None)
    if should_eval and batch.reward_matrix is None:
        raise ValueError("--eval-before-after requires rollout npz with reward_matrix")
    eval_before = evaluate_decision_policy(policy, batch) if should_eval else None
    best_state = _clone_policy_state(policy) if eval_before is not None else None
    best_eval_name = "before" if eval_before is not None else None
    best_eval = eval_before

    oracle_pretrain_epochs = (
        int(args.oracle_pretrain_epochs)
        if args.oracle_pretrain_epochs is not None
        else (50 if batch.reward_matrix is not None else 0)
    )
    oracle_history = train_oracle_policy_on_rollout(
        policy,
        batch,
        epochs=oracle_pretrain_epochs,
        learning_rate=args.oracle_pretrain_lr,
        value_coef=args.oracle_pretrain_value_coef,
        action_weights=_parse_action_weights(args.oracle_action_weights),
    )
    eval_after_oracle = evaluate_decision_policy(policy, batch) if should_eval and oracle_history else None
    if eval_after_oracle is not None and _is_better_eval(eval_after_oracle, best_eval):
        best_state = _clone_policy_state(policy)
        best_eval_name = "after_oracle_pretrain"
        best_eval = eval_after_oracle

    history = train_ppo_on_rollout(
        policy,
        batch,
        epochs=args.epochs,
        config=PPOConfig(learning_rate=args.lr, gamma=args.gamma, clip_ratio=args.clip_ratio),
    )
    eval_after = evaluate_decision_policy(policy, batch) if should_eval else None
    if eval_after is not None and _is_better_eval(eval_after, best_eval):
        best_state = _clone_policy_state(policy)
        best_eval_name = "after_ppo"
        best_eval = eval_after
    if best_state is not None:
        policy.load_state_dict(best_state)
    save_policy_checkpoint(policy.cpu(), args.out, input_dim=input_dim, hidden_dim=args.hidden_dim)

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics = {
        "updates": history,
        "oracle_pretrain_updates": oracle_history,
        "checkpoint": str(args.out),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "oracle_pretrain_epochs": oracle_pretrain_epochs,
    }
    if eval_before is not None:
        metrics.update(
            {
                "eval_before": eval_before,
                "saved_eval_name": best_eval_name,
                "saved_eval": best_eval,
            }
        )
    if eval_after_oracle is not None:
        metrics["eval_after_oracle_pretrain"] = eval_after_oracle
    if eval_after is not None:
        metrics["eval_after"] = eval_after
    if eval_before is not None and best_eval is not None:
        metrics["mean_reward_improvement"] = float(best_eval["mean_reward"] - eval_before["mean_reward"])
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if (
        args.min_reward_improvement is not None
        and metrics.get("mean_reward_improvement", 0.0) < args.min_reward_improvement
    ):
        raise RuntimeError(
            "mean reward improvement "
            f"{metrics['mean_reward_improvement']:.6f} is below gate {args.min_reward_improvement:.6f}; "
            f"metrics were written to {metrics_path}"
        )
    print(json.dumps({"checkpoint": str(args.out), "metrics": str(metrics_path), "last": history[-1] if history else {}}, indent=2))


def _clone_policy_state(policy: ActorCriticPolicy) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in policy.state_dict().items()}


def _is_better_eval(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    return incumbent is None or float(candidate["mean_reward"]) > float(incumbent["mean_reward"])


def _parse_action_weights(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    values = [float(part.strip()) for part in raw.split(",")]
    if len(values) != 3:
        raise ValueError("--oracle-action-weights must contain exactly three values")
    if any(value <= 0.0 for value in values):
        raise ValueError("--oracle-action-weights values must be positive")
    return values


if __name__ == "__main__":
    main()
