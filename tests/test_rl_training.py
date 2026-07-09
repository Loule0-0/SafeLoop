import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class RLTrainingTests(unittest.TestCase):
    def test_synthetic_rollout_contains_three_action_ids_and_rewards(self):
        from safety_guard.rl_training import make_synthetic_rollout

        batch = make_synthetic_rollout(input_dim=5, steps=24, seed=7)

        self.assertEqual(batch.observations.shape, (24, 5))
        self.assertEqual(batch.actions.dtype, torch.long)
        self.assertTrue(set(batch.actions.tolist()).issubset({0, 1, 2}))
        self.assertEqual(batch.rewards.shape, (24,))
        self.assertEqual(batch.dones.shape, (24,))

    def test_policy_checkpoint_round_trips_metadata(self):
        from safety_guard.rl_decider import ActorCriticPolicy
        from safety_guard.rl_training import load_policy_checkpoint, save_policy_checkpoint

        policy = ActorCriticPolicy(input_dim=6, hidden_dim=12)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.pt"
            save_policy_checkpoint(policy, path, input_dim=6, hidden_dim=12)
            loaded_policy, metadata = load_policy_checkpoint(path)

        self.assertEqual(metadata["input_dim"], 6)
        self.assertEqual(metadata["hidden_dim"], 12)
        self.assertEqual(loaded_policy.greedy_action([0, 0, 0, 0, 0, 0]).value in {0, 1, 2}, True)

    def test_oracle_reward_matrix_training_improves_policy_reward(self):
        from safety_guard.rl_decider import ActorCriticPolicy
        from safety_guard.rl_training import RolloutBatch, evaluate_decision_policy, train_oracle_policy_on_rollout

        torch.manual_seed(5)
        policy = ActorCriticPolicy(input_dim=2, hidden_dim=16)
        batch = RolloutBatch(
            observations=torch.tensor(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.0, 0.0]],
                dtype=torch.float32,
            ),
            actions=torch.tensor([1, 1, 2, 2, 0]),
            old_logprobs=torch.zeros(5),
            rewards=torch.tensor([1.0, 1.0, 1.5, 1.5, 0.2]),
            dones=torch.tensor([False, False, False, False, True]),
            action_masks=torch.tensor(
                [
                    [True, True, False],
                    [True, True, False],
                    [True, False, True],
                    [True, False, True],
                    [True, False, False],
                ]
            ),
            reward_matrix=torch.tensor(
                [
                    [0.0, 1.0, -4.0],
                    [0.0, 1.0, -4.0],
                    [0.0, -4.0, 1.5],
                    [0.0, -4.0, 1.5],
                    [0.2, -4.0, -4.0],
                ],
                dtype=torch.float32,
            ),
        )

        before = evaluate_decision_policy(policy, batch)
        train_oracle_policy_on_rollout(policy, batch, epochs=120, learning_rate=5e-2)
        after = evaluate_decision_policy(policy, batch)

        self.assertGreater(after["mean_reward"], before["mean_reward"])
        self.assertGreaterEqual(after["oracle_action_match"], 0.8)

    def test_train_script_preserves_optional_rollout_fields_on_device(self):
        module = _load_train_script()
        from safety_guard.rl_training import RolloutBatch

        batch = RolloutBatch(
            observations=torch.zeros(2, 3),
            actions=torch.zeros(2, dtype=torch.long),
            old_logprobs=torch.zeros(2),
            rewards=torch.ones(2),
            dones=torch.tensor([False, True]),
            action_masks=torch.tensor([[True, True, False], [True, False, True]]),
            reward_matrix=torch.tensor([[0.1, 0.5, -4.0], [0.2, -4.0, 1.0]]),
        )

        moved = module.move_batch_to_device(batch, torch.device("cpu"))

        self.assertIsNotNone(moved.action_masks)
        self.assertIsNotNone(moved.reward_matrix)
        self.assertEqual(moved.action_masks.tolist(), [[True, True, False], [True, False, True]])
        self.assertEqual(tuple(moved.reward_matrix.shape), (2, 3))

    def test_train_script_writes_before_after_eval_metrics(self):
        module = _load_train_script()

        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_path = Path(tmpdir) / "rollout.npz"
            checkpoint_path = Path(tmpdir) / "policy.pt"
            np.savez_compressed(
                rollout_path,
                observations=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
                actions=np.asarray([1, 2, 2, 0], dtype=np.int64),
                old_logprobs=np.zeros(4, dtype=np.float32),
                rewards=np.asarray([1.0, 1.5, 1.5, 0.2], dtype=np.float32),
                dones=np.asarray([False, False, False, True], dtype=bool),
                action_masks=np.asarray(
                    [[True, True, False], [True, False, True], [True, True, True], [True, False, False]],
                    dtype=bool,
                ),
                reward_matrix=np.asarray(
                    [[0.0, 1.0, -4.0], [0.0, -4.0, 1.5], [0.0, 0.5, 1.5], [0.2, -4.0, -4.0]],
                    dtype=np.float32,
                ),
            )

            module.main(
                [
                    "--rollout-npz",
                    str(rollout_path),
                    "--input-dim",
                    "2",
                    "--hidden-dim",
                    "8",
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "2",
                    "--device",
                    "cpu",
                    "--eval-before-after",
                ]
            )

            metrics = __import__("json").loads(checkpoint_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
            checkpoint_exists = checkpoint_path.exists()

        self.assertIn("eval_before", metrics)
        self.assertIn("eval_after", metrics)
        self.assertIn("saved_eval", metrics)
        self.assertIn("oracle_mean_reward", metrics["eval_after"])
        self.assertIn("random_valid_mean_reward", metrics["eval_after"])
        self.assertTrue(checkpoint_exists)

    def test_train_script_parser_accepts_effect_gate(self):
        module = _load_train_script()

        args = module.parse_args(
            [
                "--input-dim",
                "4",
                "--out",
                "/tmp/policy.pt",
                "--eval-before-after",
                "--oracle-pretrain-epochs",
                "3",
                "--min-reward-improvement",
                "0.05",
            ]
        )

        self.assertTrue(args.eval_before_after)
        self.assertEqual(args.oracle_pretrain_epochs, 3)
        self.assertAlmostEqual(args.min_reward_improvement, 0.05)


def _load_train_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_three_action_decider.py"
    spec = importlib.util.spec_from_file_location("train_three_action_decider", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
