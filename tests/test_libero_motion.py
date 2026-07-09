import unittest

import numpy as np

from safety_guard.libero_motion import (
    JointPathExecutor,
    arm_joint_linf_error,
    arm_joint_positions_from_state,
    LinearJointPlanner,
    MotionPlanningRollbackExecutor,
    build_robot_geom_sets,
    capture_libero_sim_state,
    check_collision_filtered,
    find_arm_indices,
    interpolate_joint_path,
    render_joint_path_kinematic,
    restore_libero_sim_state,
)
from safety_guard.memory import Waypoint


class FakeModel:
    joint_names = ["robot0_joint1", "robot0_joint2", "robot0_gripper_joint"]
    actuator_names = ["robot0_torq_j1", "robot0_torq_j2", "robot0_gripper_actuator"]
    ngeom = 3
    geom_bodyid = [0, 1, 2]

    def geom_id2name(self, geom_id):
        return ["arm_geom", "finger_geom", "table_geom"][geom_id]

    def body_id2name(self, body_id):
        return ["robot0_link1", "robot0_gripper", "table"][body_id]


class FakeData:
    def __init__(self):
        self.time = 1.5
        self.qpos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        self.qvel = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        self.act = np.asarray([0.5], dtype=np.float32)
        self.ctrl = np.zeros(3, dtype=np.float32)
        self.mocap_pos = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.mocap_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.contact = []

    @property
    def ncon(self):
        return len(self.contact)


class FakeContact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()
        self.rendered = 0
        self.forward_calls = 0

    def step(self):
        self.data.qvel[:2] = 0.25 * self.data.ctrl[:2]
        self.data.qpos[:2] += 0.05 * self.data.ctrl[:2]

    def forward(self):
        self.forward_calls += 1

    def render(self, height, width, camera_name):
        self.rendered += 1
        return np.zeros((height, width, 3), dtype=np.uint8)


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()


class LiberoMotionTests(unittest.TestCase):
    def test_find_arm_indices_ignores_gripper(self):
        env = FakeEnv()

        qpos_indices, actuator_indices = find_arm_indices(env)

        self.assertEqual(qpos_indices, [0, 1])
        self.assertEqual(actuator_indices, [0, 1])

    def test_collision_filter_ignores_gripper_object_contact_but_not_arm_contact(self):
        sim = FakeSim()
        _, arm_geoms, gripper_geoms = build_robot_geom_sets(sim, "robot0")

        sim.data.contact = [FakeContact(1, 2)]
        self.assertTrue(check_collision_filtered(sim, arm_geoms, gripper_geoms))

        sim.data.contact = [FakeContact(0, 2)]
        self.assertFalse(check_collision_filtered(sim, arm_geoms, gripper_geoms))

    def test_interpolate_joint_path_limits_per_step_motion(self):
        path = interpolate_joint_path([0.0, 0.0], [1.0, 0.0], max_joint_step=0.25)

        self.assertEqual(len(path), 4)
        np.testing.assert_allclose(path[-1], [1.0, 0.0])
        for before, after in zip([[0.0, 0.0]] + path[:-1], path):
            self.assertLessEqual(np.linalg.norm(np.asarray(after) - np.asarray(before)), 0.250001)

    def test_linear_planner_rejects_colliding_direct_path(self):
        planner = LinearJointPlanner(max_joint_step=0.5)

        plan = planner.plan([0.0], [1.0], is_state_valid=lambda q: q[0] < 0.75)

        self.assertFalse(plan.safe)
        self.assertEqual(plan.path, [])

    def test_joint_path_executor_renders_rollback_frames(self):
        env = FakeEnv()
        executor = JointPathExecutor(kp=2.0, kd=0.0, tolerance=0.02, max_steps_per_waypoint=50)
        frames = []

        reached = executor.execute(env, [np.asarray([0.2, 0.0])], [0, 1], [0, 1], frames)

        self.assertTrue(reached)
        self.assertGreater(len(frames), 0)
        self.assertGreater(env.sim.rendered, 0)

    def test_motion_planning_executor_defaults_to_smooth_kinematic_execution(self):
        executor = MotionPlanningRollbackExecutor()

        self.assertEqual(executor.motion_execution_mode, "kinematic")
        self.assertGreater(executor.kinematic_substeps_per_waypoint, 1)

    def test_kinematic_renderer_interpolates_substeps_instead_of_jumping_waypoints(self):
        env = FakeEnv()
        frames = []

        reached = render_joint_path_kinematic(
            env,
            [np.asarray([0.1, 0.0]), np.asarray([0.2, -0.1])],
            [0, 1],
            frames=frames,
            render_height=16,
            render_width=16,
            substeps_per_waypoint=4,
        )

        self.assertTrue(reached)
        np.testing.assert_allclose(env.sim.data.qpos[:2], [0.2, -0.1])
        self.assertEqual(len(frames), 8)
        self.assertGreater(env.sim.forward_calls, 2)

    def test_motion_planning_executor_counts_kinematic_substeps_without_video_frames(self):
        env = FakeEnv()
        state = capture_libero_sim_state(env.sim)
        env.sim.data.qpos[:2] = [0.4, 0.0]
        executor = MotionPlanningRollbackExecutor(
            max_joint_step=0.2,
            motion_execution_mode="kinematic",
            kinematic_substeps_per_waypoint=3,
            frames=None,
        )

        _, info = executor.rollback(env, Waypoint(step_index=0, state=state))

        self.assertTrue(info["reached"])
        self.assertGreater(info["rendered_frames"], 0)
        self.assertEqual(info["rendered_frames"], info["waypoints"] * 3)

    def test_capture_and_restore_libero_sim_state_deep_copies_mutable_arrays(self):
        sim = FakeSim()
        sim.data.qpos[:] = [0.1, 0.2, 0.3]
        sim.data.qvel[:] = [0.4, 0.5, 0.6]
        sim.data.ctrl[:] = [0.7, 0.8, 0.9]
        sim.data.time = 2.5

        state = capture_libero_sim_state(sim)
        state.qpos[:] = -1
        sim.data.qpos[:] = [9.0, 9.0, 9.0]
        sim.data.qvel[:] = [8.0, 8.0, 8.0]
        sim.data.ctrl[:] = [7.0, 7.0, 7.0]
        sim.data.mocap_pos[:] = [[6.0, 6.0, 6.0]]
        sim.data.time = 9.5

        restore_libero_sim_state(sim, state)

        self.assertEqual(sim.data.time, 2.5)
        np.testing.assert_allclose(sim.data.qpos, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(sim.data.qvel, [0.4, 0.5, 0.6])
        np.testing.assert_allclose(sim.data.ctrl, [0.7, 0.8, 0.9])
        np.testing.assert_allclose(sim.data.mocap_pos, [[1.0, 2.0, 3.0]])
        self.assertGreater(sim.forward_calls, 0)

    def test_extracts_target_arm_positions_and_motion_error_from_saved_state(self):
        sim = FakeSim()
        sim.data.qpos[:] = [0.1, 0.2, 0.3]
        state = capture_libero_sim_state(sim)
        sim.data.qpos[:] = [0.12, 0.18, 0.3]

        target = arm_joint_positions_from_state(state, [0, 1])
        error = arm_joint_linf_error(sim, state, [0, 1])

        np.testing.assert_allclose(target, [0.1, 0.2])
        self.assertAlmostEqual(error, 0.02, places=6)


if __name__ == "__main__":
    unittest.main()
