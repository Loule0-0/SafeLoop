from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from safety_guard.controller import SafeLoopController
from safety_guard.decider import Intervention
from safety_guard.memory import Waypoint
from safety_guard.predictors import ConstantRiskPredictor
from safety_guard.risk import RiskVector
from safety_guard.rl_decider import ActorCriticPolicy
from safety_guard.rl_policy_decider import RLPolicyDecider
from safety_guard.rl_training import save_policy_checkpoint


class AlwaysRollbackAfterRecord:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs):
        self.calls += 1
        return Intervention.RECORD if self.calls == 1 else Intervention.ROLLBACK


class FakeRollbackExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.last_waypoint = None

    def rollback(self, env, waypoint: Waypoint):
        self.calls += 1
        self.last_waypoint = waypoint
        return {"rolled_back": True}, {"rollback_mode": "motion-plan", "reached": True}


class FakeEnv:
    def __init__(self) -> None:
        self.set_init_state_calls = 0
        self.state = np.zeros(3, dtype=np.float32)

    def get_sim_state(self):
        return self.state.copy()

    def set_init_state(self, state):
        self.set_init_state_calls += 1
        self.state = np.asarray(state, dtype=np.float32)
        return {"restored": True}

    def step(self, action):
        return {"stepped": True}, 0.25, False, {}


class SafeGuardIntegrationTests(unittest.TestCase):
    def test_controller_uses_rollback_executor(self) -> None:
        env = FakeEnv()
        executor = FakeRollbackExecutor()
        controller = SafeLoopController(
            predictor=ConstantRiskPredictor(RiskVector(1.0, 0.0, 0.0, 1.0)),
            decider=AlwaysRollbackAfterRecord(),
            rollback_executor=executor,
            rollback_target_min_age=0,
        )

        first = controller.step(env, {"robot0_joint_pos": np.zeros(7)}, np.zeros(7))
        second = controller.step(env, {"robot0_joint_pos": np.ones(7)}, np.zeros(7))

        self.assertEqual(first.intervention, Intervention.RECORD)
        self.assertEqual(second.intervention, Intervention.ROLLBACK)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(env.set_init_state_calls, 0)
        self.assertEqual(second.info["safeloop"]["rollback_mode"], "motion-plan")

    def test_rl_policy_decider_loads_checkpoint_and_respects_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "policy.pt"
            policy = ActorCriticPolicy(input_dim=49, hidden_dim=16)
            with torch.no_grad():
                policy.actor.weight.zero_()
                policy.actor.bias[:] = torch.tensor([-1.0, 2.0, 0.0])
            save_policy_checkpoint(policy, checkpoint, input_dim=49, hidden_dim=16)

            decider = RLPolicyDecider(checkpoint, device="cpu")
            memory = type("Memory", (), {"__len__": lambda self: 0})()
            action = decider.decide(
                risk=RiskVector(0.0, 1.0, 0.0, 1.0),
                memory=memory,
                step_index=0,
                joint_history=[np.zeros(9, dtype=np.float32)],
                risk_history=[RiskVector(0.0, 1.0, 0.0, 1.0)],
                current_hazard_history=[(0.0, 0.0)],
            )
            self.assertEqual(action, Intervention.RECORD)

            action = decider.decide(
                risk=RiskVector(0.0, 1.0, 0.0, 1.0),
                memory=memory,
                step_index=0,
                last_record_step=0,
                joint_history=[np.zeros(9, dtype=np.float32)],
                risk_history=[RiskVector(0.0, 1.0, 0.0, 1.0)],
                current_hazard_history=[(0.0, 0.0)],
            )
            self.assertEqual(action, Intervention.NOOP)


if __name__ == "__main__":
    unittest.main()
