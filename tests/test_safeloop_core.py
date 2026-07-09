import unittest

import numpy as np

from safety_guard import (
    Intervention,
    RiskVector,
    RuleBasedDecider,
    SafeLoopController,
    WaypointMemory,
    build_actor_features,
)
from safety_guard.libero_motion import LiberoSimState
from safety_guard.rl_policy_decider import RollbackGateDecider


class FakeLiberoEnv:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float32)
        self.executed_actions = []

    def get_sim_state(self):
        return self.state.copy()

    def set_init_state(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()
        return {"state": self.state.copy()}

    def step(self, action):
        self.executed_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.state = self.state + np.asarray(action, dtype=np.float32)
        return {"state": self.state.copy()}, 0.0, False, {}


class FakePreciseData:
    def __init__(self):
        self.time = 0.0
        self.qpos = np.asarray([0.0, 0.0], dtype=np.float32)
        self.qvel = np.asarray([0.0, 0.0], dtype=np.float32)
        self.ctrl = np.asarray([0.0, 0.0], dtype=np.float32)


class FakePreciseSim:
    def __init__(self):
        self.data = FakePreciseData()
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class FakePreciseLiberoEnv:
    def __init__(self):
        self.sim = FakePreciseSim()
        self.executed_actions = []
        self.set_init_state_calls = 0
        self.regenerate_calls = 0

    def get_sim_state(self):
        return self.sim.data.qpos.copy()

    def set_init_state(self, state):
        self.set_init_state_calls += 1
        self.sim.data.qpos[:] = np.asarray(state, dtype=np.float32)
        return {"state": self.sim.data.qpos.copy()}

    def regenerate_obs_from_state(self, state):
        self.regenerate_calls += 1
        return {"qpos": self.sim.data.qpos.copy(), "time": self.sim.data.time}

    def step(self, action):
        self.executed_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.sim.data.qpos[:] = self.sim.data.qpos + np.asarray(action, dtype=np.float32)
        self.sim.data.qvel[:] = np.asarray(action, dtype=np.float32)
        self.sim.data.time += 0.05
        return {"qpos": self.sim.data.qpos.copy()}, 0.0, False, {}


class QueuePredictor:
    def __init__(self, risks):
        self.risks = list(risks)

    def predict(self, observation, proposed_action, instruction=None):
        if not self.risks:
            raise AssertionError("predict called more times than expected")
        return self.risks.pop(0)


class RejectingRollbackExecutor:
    def rollback(self, env, waypoint):
        raise RuntimeError("rollback motion plan rejected: collision_on_linear_path")


class ConstantInterventionDecider:
    def __init__(self, intervention):
        self.intervention = intervention

    def decide(self, **kwargs):
        return self.intervention


class TraceNoopDecider:
    def __init__(self):
        self.last_decision_info = {"decider": "trace_noop", "selected_action": "noop"}

    def decide(self, **kwargs):
        return Intervention.NOOP


class SafeLoopCoreTests(unittest.TestCase):
    def test_low_risk_records_waypoint_and_executes_nominal_action(self):
        env = FakeLiberoEnv([3.0, 4.0])
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.01, body_tth=1.0, object_probability=0.02, object_tth=1.0)]
        )
        decider = RuleBasedDecider(record_probability=0.05, rollback_probability=0.8)
        controller = SafeLoopController(predictor=predictor, decider=decider)

        result = controller.step(env, observation={"joint_pos": [0.1, 0.2]}, proposed_action=[1.0, -1.0])

        self.assertEqual(result.intervention, Intervention.RECORD)
        self.assertTrue(result.executed_nominal_action)
        self.assertEqual(len(controller.memory), 1)
        np.testing.assert_allclose(controller.memory.latest().state, [3.0, 4.0])
        np.testing.assert_allclose(env.executed_actions[0], [1.0, -1.0])
        np.testing.assert_allclose(result.observation["state"], [4.0, 3.0])

    def test_controller_auto_records_safe_anchor_during_noop(self):
        env = FakeLiberoEnv([3.0, 4.0])
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.01, body_tth=1.0, object_probability=0.02, object_tth=1.0)]
        )
        controller = SafeLoopController(
            predictor=predictor,
            decider=TraceNoopDecider(),
            auto_record_safe_anchors=True,
            auto_record_min_interval=20,
            auto_record_max_risk_score=0.4,
        )

        result = controller.step(env, observation={"joint_pos": [0.1, 0.2]}, proposed_action=[1.0, -1.0])

        self.assertEqual(result.intervention, Intervention.NOOP)
        self.assertTrue(result.executed_nominal_action)
        self.assertEqual(len(controller.memory), 1)
        self.assertTrue(result.info["safeloop"]["recorded_waypoint"])
        self.assertTrue(result.info["safeloop"]["auto_recorded"])
        np.testing.assert_allclose(controller.memory.latest().state, [3.0, 4.0])

    def test_controller_record_gate_blocks_unsafe_record_waypoint(self):
        env = FakeLiberoEnv([3.0, 4.0])
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.95, body_tth=0.1, object_probability=0.02, object_tth=1.0)]
        )
        controller = SafeLoopController(
            predictor=predictor,
            decider=ConstantInterventionDecider(Intervention.RECORD),
            record_max_risk_score=0.4,
            record_max_current_body_probability=0.3,
            record_max_current_object_probability=0.45,
        )

        result = controller.step(env, observation={"joint_pos": [0.1, 0.2]}, proposed_action=[1.0, -1.0])

        self.assertEqual(result.intervention, Intervention.RECORD)
        self.assertTrue(result.executed_nominal_action)
        self.assertEqual(len(controller.memory), 0)
        self.assertFalse(result.info["safeloop"]["recorded_waypoint"])
        self.assertTrue(result.info["safeloop"]["record_blocked"])

    def test_high_risk_rolls_back_to_latest_waypoint_without_nominal_action(self):
        env = FakeLiberoEnv([9.0, 9.0])
        memory = WaypointMemory()
        memory.record(step_index=0, state=np.asarray([1.0, 2.0], dtype=np.float32))
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.95, body_tth=0.1, object_probability=0.05, object_tth=1.0)]
        )
        decider = RuleBasedDecider(record_probability=0.05, rollback_probability=0.8)
        controller = SafeLoopController(predictor=predictor, decider=decider, memory=memory)

        result = controller.step(env, observation={}, proposed_action=[5.0, 5.0])

        self.assertEqual(result.intervention, Intervention.ROLLBACK)
        self.assertFalse(result.executed_nominal_action)
        self.assertEqual(len(env.executed_actions), 0)
        np.testing.assert_allclose(env.state, [1.0, 2.0])
        np.testing.assert_allclose(result.observation["state"], [1.0, 2.0])

    def test_controller_uses_strict_low_risk_rollback_target(self):
        env = FakeLiberoEnv([9.0, 9.0])
        memory = WaypointMemory()
        safe = memory.record(
            step_index=30,
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            risk=RiskVector(0.05, 1.0, 0.05, 1.0),
            metadata={"current_body_probability": 0.02, "current_object_probability": 0.03},
        )
        memory.record(
            step_index=60,
            state=np.asarray([8.0, 8.0], dtype=np.float32),
            risk=RiskVector(0.9, 0.1, 0.75, 0.1),
            metadata={"current_body_probability": 0.80, "current_object_probability": 0.70},
        )
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.95, body_tth=0.1, object_probability=0.05, object_tth=1.0)]
        )
        controller = SafeLoopController(
            predictor=predictor,
            decider=ConstantInterventionDecider(Intervention.ROLLBACK),
            memory=memory,
            rollback_target_safe_score_threshold=0.4,
            rollback_target_min_age=30,
            rollback_target_max_age=120,
            rollback_target_require_safe=True,
        )
        controller.step_index = 100

        result = controller.step(env, observation={}, proposed_action=[5.0, 5.0])

        self.assertEqual(result.intervention, Intervention.ROLLBACK)
        self.assertFalse(result.executed_nominal_action)
        self.assertIs(safe, memory.select_rollback(current_step_index=100, require_safe=True))
        self.assertEqual(result.info["safeloop"]["rollback_step"], 30)
        np.testing.assert_allclose(env.state, [1.0, 2.0])

    def test_controller_reports_missing_strict_rollback_target_without_crashing(self):
        env = FakeLiberoEnv([9.0, 9.0])
        memory = WaypointMemory()
        memory.record(
            step_index=60,
            state=np.asarray([8.0, 8.0], dtype=np.float32),
            risk=RiskVector(0.9, 0.1, 0.75, 0.1),
            metadata={"current_body_probability": 0.80, "current_object_probability": 0.70},
        )
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.95, body_tth=0.1, object_probability=0.05, object_tth=1.0)]
        )
        controller = SafeLoopController(
            predictor=predictor,
            decider=ConstantInterventionDecider(Intervention.ROLLBACK),
            memory=memory,
            rollback_target_safe_score_threshold=0.4,
            rollback_target_require_safe=True,
        )
        controller.step_index = 100

        result = controller.step(env, observation={"state": env.state.copy()}, proposed_action=[5.0, 5.0])

        self.assertEqual(result.intervention, Intervention.ROLLBACK)
        self.assertFalse(result.executed_nominal_action)
        self.assertEqual(len(env.executed_actions), 0)
        np.testing.assert_allclose(env.state, [9.0, 9.0])
        self.assertEqual(result.info["safeloop"]["reason"], "no_rollback_target")
        self.assertFalse(result.info["safeloop"]["planned"])

    def test_rejected_motion_plan_is_reported_without_teleport_or_nominal_action(self):
        env = FakeLiberoEnv([9.0, 9.0])
        memory = WaypointMemory()
        memory.record(step_index=0, state=np.asarray([1.0, 2.0], dtype=np.float32))
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.95, body_tth=0.1, object_probability=0.05, object_tth=1.0)]
        )
        decider = RuleBasedDecider(record_probability=0.05, rollback_probability=0.8)
        controller = SafeLoopController(
            predictor=predictor,
            decider=decider,
            memory=memory,
            rollback_executor=RejectingRollbackExecutor(),
        )

        result = controller.step(env, observation={"state": env.state.copy()}, proposed_action=[5.0, 5.0])

        self.assertEqual(result.intervention, Intervention.ROLLBACK)
        self.assertFalse(result.executed_nominal_action)
        self.assertEqual(len(env.executed_actions), 0)
        np.testing.assert_allclose(env.state, [9.0, 9.0])
        np.testing.assert_allclose(result.observation["state"], [9.0, 9.0])
        self.assertFalse(result.info["safeloop"]["planned"])
        self.assertFalse(result.info["safeloop"]["safe"])
        self.assertFalse(result.info["safeloop"]["reached"])
        self.assertEqual(result.info["safeloop"]["reason"], "collision_on_linear_path")

    def test_controller_uses_precise_sim_state_when_env_exposes_sim(self):
        env = FakePreciseLiberoEnv()
        env.sim.data.qpos[:] = [1.0, 2.0]
        env.sim.data.qvel[:] = [0.1, 0.2]
        env.sim.data.time = 3.0
        predictor = QueuePredictor(
            [
                RiskVector(body_probability=0.01, body_tth=1.0, object_probability=0.01, object_tth=1.0),
                RiskVector(body_probability=0.95, body_tth=0.1, object_probability=0.01, object_tth=1.0),
            ]
        )
        controller = SafeLoopController(
            predictor=predictor,
            decider=RuleBasedDecider(record_probability=0.05, rollback_probability=0.8),
        )

        controller.step(env, observation={}, proposed_action=[5.0, 5.0])
        env.sim.data.qpos[:] = [9.0, 9.0]
        env.sim.data.qvel[:] = [8.0, 8.0]
        env.sim.data.time = 9.0
        result = controller.step(env, observation={}, proposed_action=[7.0, 7.0])

        self.assertIsInstance(controller.memory.latest().state, LiberoSimState)
        self.assertEqual(result.intervention, Intervention.ROLLBACK)
        self.assertEqual(env.set_init_state_calls, 0)
        self.assertEqual(env.regenerate_calls, 1)
        np.testing.assert_allclose(env.sim.data.qpos, [1.0, 2.0])
        np.testing.assert_allclose(env.sim.data.qvel, [0.1, 0.2])
        self.assertEqual(env.sim.data.time, 3.0)

    def test_actor_features_use_three_step_joint_and_risk_history(self):
        joint_history = [[1, 2], [3, 4]]
        risk_history = [
            RiskVector(0.1, 0.9, 0.2, 0.8),
            RiskVector(0.3, 0.7, 0.4, 0.6),
        ]

        features = build_actor_features(joint_history, risk_history, history_length=3)

        self.assertEqual(features.shape, (18,))
        np.testing.assert_allclose(features[:6], [0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(features[6:12], [1, 2, 0.1, 0.9, 0.2, 0.8])
        np.testing.assert_allclose(features[12:], [3, 4, 0.3, 0.7, 0.4, 0.6])

    def test_rollback_gate_vetoes_low_confidence_rollbacks(self):
        decider = RollbackGateDecider(
            ConstantInterventionDecider(Intervention.ROLLBACK),
            current_hazard_threshold=0.8,
            future_probability_threshold=0.95,
            future_tth_threshold=0.2,
        )

        action = decider.decide(
            risk=RiskVector(0.9, 0.5, 0.2, 1.0),
            current_body_probability=0.1,
            current_object_probability=0.2,
        )

        self.assertEqual(action, Intervention.NOOP)
        self.assertEqual(decider.rollback_count, 0)
        self.assertEqual(decider.last_decision_info["inner_action"], "rollback")
        self.assertEqual(decider.last_decision_info["final_action"], "noop")
        self.assertFalse(decider.last_decision_info["gate_allowed"])
        self.assertEqual(decider.last_decision_info["gate_reason"], "below_gate_threshold")

    def test_rollback_gate_allows_high_confidence_rollbacks_up_to_budget(self):
        decider = RollbackGateDecider(
            ConstantInterventionDecider(Intervention.ROLLBACK),
            max_rollbacks_per_episode=1,
            current_hazard_threshold=0.8,
            future_probability_threshold=0.95,
            future_tth_threshold=0.2,
        )
        risk = RiskVector(0.96, 0.1, 0.2, 1.0)

        first = decider.decide(risk=risk, current_body_probability=0.1, current_object_probability=0.2)
        second = decider.decide(risk=risk, current_body_probability=0.1, current_object_probability=0.2)

        self.assertEqual(first, Intervention.ROLLBACK)
        self.assertEqual(second, Intervention.NOOP)
        self.assertEqual(decider.rollback_count, 1)

    def test_rollback_gate_blocks_rollbacks_before_minimum_step(self):
        decider = RollbackGateDecider(
            ConstantInterventionDecider(Intervention.ROLLBACK),
            min_rollback_step=120,
            current_hazard_threshold=0.8,
        )

        action = decider.decide(
            risk=RiskVector(0.99, 0.1, 0.2, 1.0),
            current_body_probability=0.99,
            current_object_probability=0.1,
            step_index=80,
        )

        self.assertEqual(action, Intervention.NOOP)
        self.assertEqual(decider.last_decision_info["gate_reason"], "before_min_rollback_step")
        self.assertEqual(decider.last_decision_info["step_index"], 80)

    def test_rollback_gate_supports_separate_current_body_and_object_thresholds(self):
        decider = RollbackGateDecider(
            ConstantInterventionDecider(Intervention.ROLLBACK),
            current_body_threshold=0.97,
            current_object_threshold=0.65,
        )
        risk = RiskVector(0.2, 1.0, 0.2, 1.0)

        body_false_positive = decider.decide(
            risk=risk,
            current_body_probability=0.9,
            current_object_probability=0.1,
        )
        object_alarm = decider.decide(
            risk=risk,
            current_body_probability=0.2,
            current_object_probability=0.7,
        )

        self.assertEqual(body_false_positive, Intervention.NOOP)
        self.assertEqual(object_alarm, Intervention.ROLLBACK)

    def test_rollback_gate_can_override_noop_on_high_confidence_risk(self):
        memory = WaypointMemory()
        memory.record(
            step_index=10,
            state=np.asarray([1.0], dtype=np.float32),
            risk=RiskVector(0.05, 1.0, 0.05, 1.0),
        )
        decider = RollbackGateDecider(
            ConstantInterventionDecider(Intervention.NOOP),
            allow_risk_override=True,
            future_body_probability_threshold=0.8,
            future_tth_threshold=0.6,
            min_rollback_step=20,
            max_rollbacks_per_episode=1,
        )

        action = decider.decide(
            risk=RiskVector(0.9, 0.4, 0.1, 1.0),
            current_body_probability=0.1,
            current_object_probability=0.1,
            memory=memory,
            step_index=40,
        )
        blocked_by_budget = decider.decide(
            risk=RiskVector(0.9, 0.4, 0.1, 1.0),
            current_body_probability=0.1,
            current_object_probability=0.1,
            memory=memory,
            step_index=60,
        )

        self.assertEqual(action, Intervention.ROLLBACK)
        self.assertEqual(blocked_by_budget, Intervention.NOOP)
        self.assertEqual(decider.last_decision_info["gate_reason"], "inner_not_rollback")
        self.assertEqual(decider.rollback_count, 1)

    def test_rollback_gate_does_not_override_without_memory(self):
        decider = RollbackGateDecider(
            ConstantInterventionDecider(Intervention.NOOP),
            allow_risk_override=True,
            future_body_probability_threshold=0.8,
            future_tth_threshold=0.6,
        )

        action = decider.decide(
            risk=RiskVector(0.9, 0.4, 0.1, 1.0),
            current_body_probability=0.1,
            current_object_probability=0.1,
            step_index=40,
        )

        self.assertEqual(action, Intervention.NOOP)
        self.assertEqual(decider.rollback_count, 0)

    def test_waypoint_memory_prefers_recent_low_risk_point_when_current_step_is_known(self):
        memory = WaypointMemory()
        safe = memory.record(
            step_index=20,
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            risk=RiskVector(0.05, 1.0, 0.05, 1.0),
            metadata={"current_body_probability": 0.02, "current_object_probability": 0.03},
        )
        memory.record(
            step_index=80,
            state=np.asarray([9.0, 9.0], dtype=np.float32),
            risk=RiskVector(0.75, 0.1, 0.2, 1.0),
            metadata={"current_body_probability": 0.8, "current_object_probability": 0.4},
        )

        self.assertIs(memory.select_rollback(), memory.latest())
        self.assertIs(memory.select_rollback(current_step_index=100), safe)

    def test_waypoint_memory_skips_too_recent_rollback_targets(self):
        memory = WaypointMemory()
        mature_safe = memory.record(
            step_index=60,
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            risk=RiskVector(0.05, 1.0, 0.05, 1.0),
        )
        recent_safe = memory.record(
            step_index=92,
            state=np.asarray([3.0, 4.0], dtype=np.float32),
            risk=RiskVector(0.05, 1.0, 0.05, 1.0),
        )

        self.assertIs(memory.select_rollback(current_step_index=100, min_safe_age=30), mature_safe)
        self.assertIs(memory.select_rollback(current_step_index=100, min_safe_age=5), recent_safe)

    def test_waypoint_memory_prefers_lowest_risk_mature_target(self):
        memory = WaypointMemory()
        low_risk = memory.record(
            step_index=60,
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            risk=RiskVector(0.05, 1.0, 0.05, 1.0),
            metadata={"current_body_probability": 0.02, "current_object_probability": 0.03},
        )
        memory.record(
            step_index=80,
            state=np.asarray([3.0, 4.0], dtype=np.float32),
            risk=RiskVector(0.45, 1.0, 0.20, 1.0),
            metadata={"current_body_probability": 0.10, "current_object_probability": 0.20},
        )

        self.assertIs(memory.select_rollback(current_step_index=120, min_safe_age=30), low_risk)

    def test_waypoint_memory_strict_mode_rejects_missing_safe_target(self):
        memory = WaypointMemory()
        latest = memory.record(
            step_index=90,
            state=np.asarray([1.0, 2.0], dtype=np.float32),
            risk=RiskVector(0.9, 0.1, 0.8, 0.1),
            metadata={"current_body_probability": 0.85, "current_object_probability": 0.82},
        )

        self.assertIs(memory.select_rollback(current_step_index=120), latest)
        with self.assertRaises(IndexError):
            memory.select_rollback(current_step_index=120, require_safe=True)

    def test_controller_includes_decision_trace_in_safeloop_info(self):
        env = FakeLiberoEnv([3.0, 4.0])
        predictor = QueuePredictor(
            [RiskVector(body_probability=0.1, body_tth=1.0, object_probability=0.1, object_tth=1.0)]
        )
        controller = SafeLoopController(predictor=predictor, decider=TraceNoopDecider())

        result = controller.step(env, observation={"joint_pos": [0.1, 0.2]}, proposed_action=[1.0, -1.0])

        self.assertEqual(result.info["safeloop"]["decision"]["decider"], "trace_noop")
        self.assertEqual(result.info["safeloop"]["decision"]["selected_action"], "noop")


if __name__ == "__main__":
    unittest.main()
