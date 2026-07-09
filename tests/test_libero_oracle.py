import unittest
import importlib.util
import sys
import types
from dataclasses import dataclass

import numpy as np

if importlib.util.find_spec("safety_guard.libero_motion") is None:
    libero_motion = types.ModuleType("safety_guard.libero_motion")

    def build_robot_geom_sets(sim, robot_body_prefix="robot0"):
        return set(), set(), set()

    libero_motion.build_robot_geom_sets = build_robot_geom_sets
    sys.modules["safety_guard.libero_motion"] = libero_motion

if importlib.util.find_spec("safety_guard.online_rl") is None:
    online_rl = types.ModuleType("safety_guard.online_rl")

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

    online_rl.OnlineStepSignals = OnlineStepSignals
    sys.modules["safety_guard.online_rl"] = online_rl


class FakeContact:
    def __init__(self, geom1=0, geom2=0):
        self.geom1 = geom1
        self.geom2 = geom2


class FakeModel:
    nbody = 3
    ngeom = 0
    body_mass = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    geom_bodyid = np.asarray([], dtype=np.int32)
    joint_names = []
    actuator_names = []

    def body_id2name(self, body_id):
        return {0: "world", 1: "red_mug", 2: "robot0_eef"}.get(body_id, "")

    def body_name2id(self, name):
        if name == "robot0_eef":
            return 2
        raise KeyError(name)

    def geom_id2name(self, geom_id):
        return ""


class FakeData:
    def __init__(self):
        self.time = 1.0
        self.ncon = 0
        self.contact = []
        self.body_xpos = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.1, 0.5],
                [0.0, 0.0, 0.8],
            ],
            dtype=np.float32,
        )


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()


class LiberoHazardOracleTests(unittest.TestCase):
    def test_kinematic_position_jitter_without_time_delta_is_not_unstable_motion(self):
        from safety_guard.libero_oracle import LiberoHazardOracle

        sim = FakeSim()
        oracle = LiberoHazardOracle(sim)

        sim.data.body_xpos[1] += np.asarray([2e-4, 0.0, 0.0], dtype=np.float32)
        signals = oracle.read()

        self.assertFalse(signals.object_hazard)

    def test_sustained_arm_contact_with_low_eef_speed_is_stuck_hazard(self):
        from safety_guard.libero_oracle import LiberoHazardOracle

        sim = FakeSim()
        sim.data.ncon = 1
        sim.data.contact = [FakeContact(10, 20)]
        oracle = LiberoHazardOracle(sim, stuck_contact_steps=2)
        oracle._arm_geoms = {10}
        oracle._gripper_geoms = set()

        first = oracle.read()
        sim.data.time += 0.1
        second = oracle.read()

        self.assertTrue(first.body_hazard)
        self.assertFalse(first.stuck_hazard)
        self.assertTrue(second.stuck_hazard)
        self.assertTrue(second.any_hazard)

    def test_intermittent_contact_with_sustained_low_eef_speed_is_stuck_hazard(self):
        from safety_guard.libero_oracle import LiberoHazardOracle

        sim = FakeSim()
        oracle = LiberoHazardOracle(
            sim,
            stuck_contact_steps=10,
            stuck_window_steps=8,
            stuck_window_eef_speed_threshold=0.05,
            stuck_window_min_low_speed_steps=6,
            stuck_window_min_contact_steps=3,
        )
        oracle._arm_geoms = {10}
        oracle._gripper_geoms = set()
        signals = []
        for step in range(8):
            sim.data.time += 0.1
            sim.data.body_xpos[2, 0] += 0.001
            if step in {1, 3, 5}:
                sim.data.ncon = 1
                sim.data.contact = [FakeContact(10, 20)]
            else:
                sim.data.ncon = 0
                sim.data.contact = []
            signals.append(oracle.read())

        self.assertFalse(any(item.stuck_hazard for item in signals[:5]))
        self.assertTrue(signals[-1].stuck_hazard)

    def test_low_eef_speed_without_contact_is_not_stuck_hazard(self):
        from safety_guard.libero_oracle import LiberoHazardOracle

        sim = FakeSim()
        oracle = LiberoHazardOracle(sim, stuck_window_steps=8, stuck_window_min_low_speed_steps=6)
        signals = []
        for _ in range(8):
            sim.data.time += 0.1
            sim.data.body_xpos[2, 0] += 0.001
            signals.append(oracle.read())

        self.assertFalse(any(item.stuck_hazard for item in signals))


if __name__ == "__main__":
    unittest.main()
