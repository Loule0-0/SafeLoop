import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path

import torch


class OnlineRLTests(unittest.TestCase):
    def test_prehazard_dense_rollout_sampling_prefers_safe_future_positive_steps(self):
        from safety_guard.rollout_sampling import select_prehazard_sample_offsets

        timeline = [
            SimpleNamespace(any_hazard=False, body_hazard=False, object_hazard=False)
            for _ in range(36)
        ]
        timeline[35] = SimpleNamespace(any_hazard=True, body_hazard=True, object_hazard=False)

        selected = select_prehazard_sample_offsets(
            timeline,
            tau=30,
            stride=10,
            existing_indices={15},
        )

        self.assertEqual(selected, {5: 30, 25: 10, 34: 1})

    def test_rollback_gate_can_use_separate_future_body_and_object_thresholds(self):
        from safety_guard.risk import RiskVector
        from safety_guard.rl_policy_decider import is_high_confidence_rollback

        body_warning = RiskVector(
            body_probability=0.61,
            body_tth=0.9,
            object_probability=0.69,
            object_tth=0.3,
        )
        object_warning = RiskVector(
            body_probability=0.2,
            body_tth=0.9,
            object_probability=0.69,
            object_tth=0.3,
        )

        self.assertTrue(
            is_high_confidence_rollback(
                body_warning,
                current_body_probability=0.0,
                current_object_probability=0.0,
                future_probability_threshold=1.01,
                future_body_probability_threshold=0.60,
                future_object_probability_threshold=0.72,
                future_tth_threshold=0.95,
            )
        )
        self.assertFalse(
            is_high_confidence_rollback(
                object_warning,
                current_body_probability=0.0,
                current_object_probability=0.0,
                future_probability_threshold=1.01,
                future_body_probability_threshold=0.60,
                future_object_probability_threshold=0.72,
                future_tth_threshold=0.95,
            )
        )

    def test_rollback_gate_can_use_tighter_future_object_tth_than_body(self):
        from safety_guard.risk import RiskVector
        from safety_guard.rl_policy_decider import is_high_confidence_rollback

        body_warning = RiskVector(
            body_probability=0.81,
            body_tth=0.55,
            object_probability=0.2,
            object_tth=0.9,
        )
        late_object_warning = RiskVector(
            body_probability=0.2,
            body_tth=0.9,
            object_probability=0.81,
            object_tth=0.55,
        )
        near_object_warning = RiskVector(
            body_probability=0.2,
            body_tth=0.9,
            object_probability=0.81,
            object_tth=0.35,
        )

        shared = dict(
            current_body_probability=0.0,
            current_object_probability=0.0,
            future_probability_threshold=1.01,
            future_body_probability_threshold=0.75,
            future_object_probability_threshold=0.75,
            future_tth_threshold=0.95,
            future_object_tth_threshold=0.40,
        )

        self.assertTrue(is_high_confidence_rollback(body_warning, **shared))
        self.assertFalse(is_high_confidence_rollback(late_object_warning, **shared))
        self.assertTrue(is_high_confidence_rollback(near_object_warning, **shared))

    def test_rollback_gate_can_veto_high_current_object_probability(self):
        from safety_guard.risk import RiskVector
        from safety_guard.rl_policy_decider import is_high_confidence_rollback

        future_object_warning = RiskVector(
            body_probability=0.1,
            body_tth=0.9,
            object_probability=0.82,
            object_tth=0.3,
        )

        self.assertFalse(
            is_high_confidence_rollback(
                future_object_warning,
                current_body_probability=0.0,
                current_object_probability=0.81,
                future_probability_threshold=1.01,
                future_object_probability_threshold=0.75,
                future_tth_threshold=0.45,
                max_current_object_probability=0.75,
            )
        )
        self.assertTrue(
            is_high_confidence_rollback(
                future_object_warning,
                current_body_probability=0.0,
                current_object_probability=0.70,
                future_probability_threshold=1.01,
                future_object_probability_threshold=0.75,
                future_tth_threshold=0.45,
                max_current_object_probability=0.75,
            )
        )

    def test_record_gate_blocks_unsafe_memory_candidates(self):
        from scripts.train_online_safeguard_decider import record_allowed_by_gate
        from safety_guard.risk import RiskVector

        args = Namespace(
            record_max_risk_score=0.40,
            record_max_current_body_probability=0.25,
            record_max_current_object_probability=0.30,
        )
        safe_candidate = RiskVector(
            body_probability=0.10,
            body_tth=0.90,
            object_probability=0.20,
            object_tth=0.80,
        )
        high_future_risk = RiskVector(
            body_probability=0.10,
            body_tth=0.90,
            object_probability=0.55,
            object_tth=0.80,
        )

        self.assertTrue(record_allowed_by_gate(args, safe_candidate, current_body=0.05, current_object=0.10))
        self.assertFalse(record_allowed_by_gate(args, high_future_risk, current_body=0.05, current_object=0.10))
        self.assertFalse(record_allowed_by_gate(args, safe_candidate, current_body=0.05, current_object=0.35))

    def test_online_reward_penalizes_hazard_and_intervention(self):
        from safety_guard.decider import Intervention
        from safety_guard.online_rl import OnlineStepSignals, online_safeloop_reward

        safe = online_safeloop_reward(OnlineStepSignals(), Intervention.NOOP)
        unsafe = online_safeloop_reward(OnlineStepSignals(body_hazard=True), Intervention.NOOP)
        rollback = online_safeloop_reward(OnlineStepSignals(body_hazard=True), Intervention.ROLLBACK, rollback_count=3)
        self.assertLess(unsafe, safe)
        self.assertLess(rollback, unsafe)

    def test_online_reward_supports_channel_specific_hazard_penalties(self):
        from safety_guard.decider import Intervention
        from safety_guard.online_rl import OnlineRewardConfig, OnlineStepSignals, online_hazard_score
        from safety_guard.online_rl import online_safeloop_reward

        config = OnlineRewardConfig(
            hazard_penalty=-0.25,
            body_hazard_penalty=-1.0,
            object_hazard_penalty=-0.5,
            stuck_hazard_penalty=-1.5,
        )

        object_reward = online_safeloop_reward(
            OnlineStepSignals(object_hazard=True),
            Intervention.NOOP,
            config=config,
        )
        stuck_reward = online_safeloop_reward(
            OnlineStepSignals(stuck_hazard=True),
            Intervention.NOOP,
            config=config,
        )

        self.assertLess(stuck_reward, object_reward)
        self.assertEqual(online_hazard_score(OnlineStepSignals(stuck_hazard=True), config), 1.5)

    def test_privileged_features_include_stuck_hazard(self):
        from safety_guard.online_rl import OnlineStepSignals, privileged_feature_dim

        signals = OnlineStepSignals(stuck_hazard=True)

        self.assertTrue(signals.any_hazard)
        self.assertEqual(privileged_feature_dim(), 7)
        self.assertEqual(float(signals.privileged_features()[2]), 1.0)

    def test_online_reward_penalizes_failed_rollback(self):
        from safety_guard.decider import Intervention
        from safety_guard.online_rl import OnlineRewardConfig, OnlineStepSignals, online_safeloop_reward

        config = OnlineRewardConfig(rollback_failure_penalty=-0.5)
        successful = online_safeloop_reward(
            OnlineStepSignals(),
            Intervention.ROLLBACK,
            rollback_count=1,
            rollback_failed=False,
            config=config,
        )
        failed = online_safeloop_reward(
            OnlineStepSignals(),
            Intervention.ROLLBACK,
            rollback_count=1,
            rollback_failed=True,
            config=config,
        )

        self.assertLess(failed, successful)

    def test_online_reward_penalizes_long_rendered_rollback(self):
        from safety_guard.decider import Intervention
        from safety_guard.online_rl import OnlineRewardConfig, OnlineStepSignals, online_safeloop_reward

        config = OnlineRewardConfig(rollback_rendered_frame_penalty=-0.01)
        short = online_safeloop_reward(
            OnlineStepSignals(),
            Intervention.ROLLBACK,
            rollback_rendered_frames=2,
            config=config,
        )
        long = online_safeloop_reward(
            OnlineStepSignals(),
            Intervention.ROLLBACK,
            rollback_rendered_frames=20,
            config=config,
        )

        self.assertLess(long, short)

    def test_rollback_outcome_credit_rewards_resolution_and_penalizes_persistent_hazard(self):
        from safety_guard.online_rl import OnlineRewardConfig, rollback_outcome_credit

        config = OnlineRewardConfig(
            rollback_resolved_bonus=0.75,
            rollback_unresolved_penalty=-1.25,
        )

        resolved = rollback_outcome_credit(post_rollback_hazard_steps=0, observed_steps=5, config=config)
        unresolved = rollback_outcome_credit(post_rollback_hazard_steps=2, observed_steps=5, config=config)
        incomplete = rollback_outcome_credit(post_rollback_hazard_steps=0, observed_steps=0, config=config)

        self.assertEqual(resolved, 0.75)
        self.assertEqual(unresolved, -1.25)
        self.assertEqual(incomplete, 0.0)

    def test_rollback_outcome_credit_uses_pre_post_hazard_improvement(self):
        from safety_guard.online_rl import OnlineRewardConfig, rollback_outcome_credit

        config = OnlineRewardConfig(
            rollback_resolved_bonus=1.0,
            rollback_unresolved_penalty=-2.0,
        )

        worsened = rollback_outcome_credit(
            post_rollback_hazard_steps=20,
            observed_steps=20,
            pre_rollback_hazard_steps=0,
            pre_observed_steps=20,
            config=config,
        )
        improved = rollback_outcome_credit(
            post_rollback_hazard_steps=2,
            observed_steps=20,
            pre_rollback_hazard_steps=20,
            pre_observed_steps=20,
            config=config,
        )
        unchanged_safe = rollback_outcome_credit(
            post_rollback_hazard_steps=0,
            observed_steps=20,
            pre_rollback_hazard_steps=0,
            pre_observed_steps=20,
            config=config,
        )

        self.assertEqual(worsened, -2.0)
        self.assertAlmostEqual(improved, 0.9)
        self.assertEqual(unchanged_safe, 0.0)

    def test_rollback_terminal_credit_distributes_episode_success_or_failure(self):
        from safety_guard.online_rl import OnlineRewardConfig, rollback_terminal_credit

        config = OnlineRewardConfig(
            rollback_episode_success_bonus=2.0,
            rollback_episode_failure_penalty=-1.0,
        )

        self.assertEqual(rollback_terminal_credit(True, rollback_count=2, config=config), 1.0)
        self.assertEqual(rollback_terminal_credit(False, rollback_count=2, config=config), -0.5)
        self.assertEqual(rollback_terminal_credit(True, rollback_count=0, config=config), 0.0)

    def test_rollback_exploration_forces_allowed_rollback_and_keeps_logprob(self):
        from safety_guard.online_rl import AsymmetricActorCriticPolicy, sample_action_with_rollback_exploration
        from safety_guard.rl_decider import ThreeAction

        policy = AsymmetricActorCriticPolicy(actor_dim=4, critic_dim=10, hidden_dim=8)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
            policy.actor.bias[:] = torch.tensor([8.0, 0.0, -8.0])

        action, logprob, explored = sample_action_with_rollback_exploration(
            policy,
            actor_features=[0.0, 0.0, 0.0, 0.0],
            action_mask=[True, True, True],
            rollback_exploration_probability=1.0,
        )

        self.assertEqual(action, ThreeAction.ROLLBACK)
        self.assertTrue(explored)
        self.assertTrue(torch.isfinite(logprob))

    def test_asymmetric_ppo_skips_masked_teacher_actions_for_bc(self):
        from safety_guard.online_rl import (
            AsymmetricActorCriticPolicy,
            AsymmetricPPOConfig,
            AsymmetricPPOTrainer,
            OnlineRolloutBatch,
        )

        policy = AsymmetricActorCriticPolicy(actor_dim=4, critic_dim=6, hidden_dim=8)
        trainer = AsymmetricPPOTrainer(policy, AsymmetricPPOConfig(bc_coef=1.0))
        batch = OnlineRolloutBatch(
            actor_observations=torch.zeros(3, 4),
            critic_observations=torch.zeros(3, 6),
            actions=torch.tensor([0, 0, 1], dtype=torch.long),
            old_logprobs=torch.zeros(3),
            rewards=torch.tensor([0.0, 0.0, 1.0]),
            dones=torch.tensor([False, False, True]),
            action_masks=torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                    [True, True, False],
                ],
                dtype=torch.bool,
            ),
            teacher_actions=torch.tensor([2, 1, 1], dtype=torch.long),
        )

        history = trainer.update(batch, epochs=1, bc_coef=1.0)

        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["bc_loss"])))
        self.assertEqual(history[-1]["bc_valid_count"], 1.0)
        self.assertEqual(history[-1]["bc_ignored_count"], 2.0)

    def test_min_step_and_max_count_do_not_enable_risk_gate_by_themselves(self):
        from scripts.train_online_safeguard_decider import rollback_gate_enabled

        args = Namespace(
            max_rollbacks_per_episode=1,
            min_rollback_step=80,
            rollback_gate_current_threshold=1.01,
            rollback_gate_current_body_threshold=None,
            rollback_gate_current_object_threshold=None,
            rollback_gate_future_probability=1.01,
            rollback_gate_future_body_probability=None,
            rollback_gate_future_object_probability=None,
        )

        self.assertFalse(rollback_gate_enabled(args))

    def test_asymmetric_policy_exports_actor_checkpoint(self):
        from safety_guard.online_rl import (
            AsymmetricActorCriticPolicy,
            actor_checkpoint_to_asymmetric,
            export_actor_checkpoint,
        )
        from safety_guard.rl_training import load_policy_checkpoint

        policy = AsymmetricActorCriticPolicy(actor_dim=8, critic_dim=14, hidden_dim=16)
        features = torch.randn(4, 8)
        critic_features = torch.randn(4, 14)
        logits, values = policy(features, critic_features)
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(values.shape), (4,))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actor.pt"
            export_actor_checkpoint(policy, path)
            loaded, metadata = load_policy_checkpoint(path)
            self.assertEqual(metadata["input_dim"], 8)
            self.assertEqual(metadata["hidden_dim"], 16)
            loaded_logits, _ = loaded(features)
            self.assertEqual(tuple(loaded_logits.shape), (4, 3))
            asymmetric = actor_checkpoint_to_asymmetric(path, critic_dim=14)
            self.assertEqual(asymmetric.actor_dim, 8)


if __name__ == "__main__":
    unittest.main()
