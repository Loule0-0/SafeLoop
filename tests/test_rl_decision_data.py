import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from safety_guard.risk import RiskVector
from safety_guard.rl_decider import ThreeAction


class RLDecisionRewardTests(unittest.TestCase):
    def test_build_decision_features_includes_current_hazard_history_and_context(self):
        from safety_guard.rl_decision_data import build_decision_features

        features = build_decision_features(
            joint_history=[[1.0, 2.0]],
            risk_history=[RiskVector(0.1, 0.9, 0.2, 0.8)],
            current_hazard_history=[(0.3, 0.4)],
            context_values=[1.0, 0.0],
            history_length=2,
        )

        self.assertEqual(features.shape, (18,))
        np.testing.assert_allclose(features[:8], np.zeros(8, dtype=np.float32))
        np.testing.assert_allclose(features[8:16], [1.0, 2.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
        np.testing.assert_allclose(features[16:], [1.0, 0.0])

    def test_action_mask_blocks_unavailable_rollback(self):
        from safety_guard.rl_decision_data import build_action_mask

        self.assertEqual(build_action_mask(can_record=True, can_rollback=False), [True, True, False])
        self.assertEqual(build_action_mask(can_record=False, can_rollback=True), [True, False, True])

    def test_reward_model_prefers_rollback_when_current_hazard_is_high(self):
        from safety_guard.rl_decision_data import DecisionContext, DecisionRewardModel

        context = DecisionContext(
            features=np.zeros(6, dtype=np.float32),
            risk=RiskVector(body_probability=0.8, body_tth=0.1, object_probability=0.2, object_tth=0.8),
            current_body_probability=0.9,
            current_object_probability=0.1,
            can_record=True,
            can_rollback=True,
        )

        rewards = DecisionRewardModel().action_rewards(context)

        self.assertGreater(rewards[ThreeAction.ROLLBACK], rewards[ThreeAction.NOOP])
        self.assertGreater(rewards[ThreeAction.ROLLBACK], rewards[ThreeAction.RECORD])

    def test_reward_model_penalizes_invalid_rollback(self):
        from safety_guard.rl_decision_data import DecisionContext, DecisionRewardModel

        context = DecisionContext(
            features=np.zeros(6, dtype=np.float32),
            risk=RiskVector(body_probability=0.95, body_tth=0.05, object_probability=0.1, object_tth=1.0),
            current_body_probability=0.8,
            current_object_probability=0.0,
            can_record=True,
            can_rollback=False,
        )

        rewards = DecisionRewardModel().action_rewards(context)

        self.assertLess(rewards[ThreeAction.ROLLBACK], -1.0)
        self.assertGreater(rewards[ThreeAction.NOOP], rewards[ThreeAction.ROLLBACK])


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class RLDecisionBatchTests(unittest.TestCase):
    def test_decision_records_to_rollout_batch_preserves_masks_and_rewards(self):
        from safety_guard.rl_decision_data import (
            DecisionContext,
            DecisionRecord,
            DecisionRewardModel,
            decision_records_to_rollout_batch,
        )

        contexts = [
            DecisionContext(
                features=np.asarray([1.0, 0.0], dtype=np.float32),
                risk=RiskVector(0.05, 1.0, 0.05, 1.0),
                current_body_probability=0.0,
                current_object_probability=0.0,
                can_record=True,
                can_rollback=False,
            ),
            DecisionContext(
                features=np.asarray([0.0, 1.0], dtype=np.float32),
                risk=RiskVector(0.9, 0.1, 0.1, 1.0),
                current_body_probability=0.7,
                current_object_probability=0.0,
                can_record=False,
                can_rollback=True,
                done=True,
            ),
        ]
        records = [
            DecisionRecord(context=contexts[0], action=ThreeAction.RECORD),
            DecisionRecord(context=contexts[1], action=ThreeAction.ROLLBACK),
        ]

        batch = decision_records_to_rollout_batch(records, DecisionRewardModel())

        self.assertEqual(tuple(batch.observations.shape), (2, 2))
        self.assertEqual(batch.actions.tolist(), [ThreeAction.RECORD, ThreeAction.ROLLBACK])
        self.assertEqual(batch.action_masks.tolist(), [[True, True, False], [True, False, True]])
        self.assertEqual(tuple(batch.reward_matrix.shape), (2, 3))
        self.assertTrue(batch.dones.tolist()[-1])
        self.assertAlmostEqual(float(batch.rewards[0]), float(batch.reward_matrix[0, ThreeAction.RECORD]))

    def test_evaluate_decision_policy_reports_oracle_and_random_valid_baselines(self):
        from safety_guard.rl_decider import ActorCriticPolicy
        from safety_guard.rl_training import RolloutBatch, evaluate_decision_policy

        policy = ActorCriticPolicy(input_dim=2, hidden_dim=4)
        with torch.no_grad():
            policy.actor.weight.zero_()
            policy.actor.bias[:] = torch.tensor([0.0, 0.0, 5.0])
        batch = RolloutBatch(
            observations=torch.zeros(2, 2),
            actions=torch.tensor([ThreeAction.ROLLBACK, ThreeAction.NOOP]),
            old_logprobs=torch.zeros(2),
            rewards=torch.zeros(2),
            dones=torch.tensor([False, True]),
            action_masks=torch.tensor([[True, True, True], [True, False, True]]),
            reward_matrix=torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, -4.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )

        metrics = evaluate_decision_policy(policy, batch)

        self.assertAlmostEqual(metrics["mean_reward"], 1.0)
        self.assertAlmostEqual(metrics["oracle_mean_reward"], 1.5)
        self.assertAlmostEqual(metrics["random_valid_mean_reward"], 0.75)
        self.assertAlmostEqual(metrics["oracle_action_match"], 0.5)
        self.assertAlmostEqual(metrics["valid_action_rate"], 1.0)


class RLDecisionScriptTests(unittest.TestCase):
    def test_build_decision_rollout_parser_accepts_pi0_inputs(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_decision_rollout_from_pi0.py"
        spec = importlib.util.spec_from_file_location("build_decision_rollout_from_pi0", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.parse_args(
            [
                "--trajectory-root",
                "/data/full_trajectory_pi0",
                "--out",
                "/tmp/decision_rollout.npz",
                "--history-length",
                "3",
                "--tau",
                "50",
                "--max-runs",
                "2",
            ]
        )

        self.assertEqual(args.history_length, 3)
        self.assertEqual(args.tau, 50)
        self.assertEqual(args.max_runs, 2)

    def test_build_decision_rollout_skips_corrupt_sample_files(self):
        module = _load_rollout_script()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "libero_10_00_run000"
            run_dir.mkdir()
            (run_dir / "trajectory_metadata.json").write_text(
                json.dumps({"marked_steps": [[10, 0]], "max_steps": 20}),
                encoding="utf-8",
            )
            (run_dir / "sample_00000.json").write_text(
                json.dumps(
                    {
                        "global_step": 0,
                        "robot_state": {
                            "robot0_joint_pos": [0.0, 0.1],
                            "robot0_gripper_qpos": [0.2],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "sample_00001.json").write_text("", encoding="utf-8")

            records, summary = module.build_records_from_pi0_trajectories(Path(tmpdir), history_length=2)

        self.assertEqual(len(records), 1)
        self.assertEqual(len(summary["bad_samples"]), 1)

def _load_rollout_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_decision_rollout_from_pi0.py"
    spec = importlib.util.spec_from_file_location("build_decision_rollout_from_pi0", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
