import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class RLDeciderTests(unittest.TestCase):
    def test_actor_critic_outputs_three_actions_and_value(self):
        from safety_guard.rl_decider import ActorCriticPolicy

        policy = ActorCriticPolicy(input_dim=33, hidden_dim=16)
        logits, value = policy(torch.zeros(2, 33))

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(value.shape, (2,))

    def test_sample_action_maps_to_three_action_enum(self):
        from safety_guard.rl_decider import ActorCriticPolicy, ThreeAction

        policy = ActorCriticPolicy(input_dim=4, hidden_dim=8)
        action, logprob, value = policy.sample_action(np.zeros(4, dtype=np.float32))

        self.assertIsInstance(action, ThreeAction)
        self.assertEqual(logprob.ndim, 0)
        self.assertEqual(value.ndim, 0)

    def test_greedy_action_respects_action_mask(self):
        from safety_guard.rl_decider import ActorCriticPolicy, ThreeAction

        policy = ActorCriticPolicy(input_dim=4, hidden_dim=8)
        with torch.no_grad():
            policy.actor.bias[:] = torch.tensor([0.0, 1.0, 9.0])

        action = policy.greedy_action(
            np.zeros(4, dtype=np.float32),
            action_mask=[True, True, False],
        )

        self.assertEqual(action, ThreeAction.RECORD)

    def test_sample_action_respects_single_valid_action_mask(self):
        from safety_guard.rl_decider import ActorCriticPolicy, ThreeAction

        policy = ActorCriticPolicy(input_dim=4, hidden_dim=8)

        for _ in range(5):
            action, _, _ = policy.sample_action(
                np.zeros(4, dtype=np.float32),
                action_mask=[False, True, False],
            )
            self.assertEqual(action, ThreeAction.RECORD)

    def test_ppo_update_accepts_action_masks(self):
        from safety_guard.rl_decider import ActorCriticPolicy, PPOTrainer

        policy = ActorCriticPolicy(input_dim=4, hidden_dim=8)
        trainer = PPOTrainer(policy)
        stats = trainer.update(
            observations=torch.zeros(3, 4),
            actions=torch.tensor([0, 1, 0]),
            old_logprobs=torch.zeros(3),
            returns=torch.ones(3),
            advantages=torch.ones(3),
            action_masks=torch.tensor(
                [
                    [True, True, False],
                    [False, True, False],
                    [True, False, False],
                ]
            ),
        )

        self.assertIn("loss", stats)

    def test_discounted_returns_respect_episode_boundaries(self):
        from safety_guard.rl_decider import ActorCriticPolicy, PPOConfig, PPOTrainer

        trainer = PPOTrainer(ActorCriticPolicy(4, 8), PPOConfig(gamma=0.9))

        returns = trainer.compute_returns([1.0, 1.0, 1.0], [False, True, False], last_value=10.0)

        np.testing.assert_allclose(returns, [1.9, 1.0, 10.0], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
