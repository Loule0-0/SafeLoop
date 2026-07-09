from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LiberoSimState:
    _qpos: np.ndarray
    _qvel: np.ndarray
    time: float | None = None
    _ctrl: np.ndarray | None = None
    _act: np.ndarray | None = None
    _mocap_pos: np.ndarray | None = None
    _mocap_quat: np.ndarray | None = None

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos.copy()

    @property
    def qvel(self) -> np.ndarray:
        return self._qvel.copy()

    @property
    def ctrl(self) -> np.ndarray | None:
        return None if self._ctrl is None else self._ctrl.copy()

    @property
    def act(self) -> np.ndarray | None:
        return None if self._act is None else self._act.copy()

    @property
    def mocap_pos(self) -> np.ndarray | None:
        return None if self._mocap_pos is None else self._mocap_pos.copy()

    @property
    def mocap_quat(self) -> np.ndarray | None:
        return None if self._mocap_quat is None else self._mocap_quat.copy()


def _copy_optional_array(data, name: str) -> np.ndarray | None:
    value = getattr(data, name, None)
    if value is None:
        return None
    return np.asarray(value).copy()


def capture_libero_sim_state(sim) -> LiberoSimState:
    return LiberoSimState(
        _qpos=np.asarray(sim.data.qpos).copy(),
        _qvel=np.asarray(sim.data.qvel).copy(),
        time=float(sim.data.time) if hasattr(sim.data, "time") else None,
        _ctrl=_copy_optional_array(sim.data, "ctrl"),
        _act=_copy_optional_array(sim.data, "act"),
        _mocap_pos=_copy_optional_array(sim.data, "mocap_pos"),
        _mocap_quat=_copy_optional_array(sim.data, "mocap_quat"),
    )


def _restore_optional_array(data, name: str, value: np.ndarray | None) -> None:
    if value is not None and hasattr(data, name):
        getattr(data, name)[:] = value


def restore_libero_sim_state(sim, state: LiberoSimState) -> None:
    if state.time is not None and hasattr(sim.data, "time"):
        sim.data.time = state.time
    sim.data.qpos[:] = state._qpos
    sim.data.qvel[:] = state._qvel
    _restore_optional_array(sim.data, "ctrl", state._ctrl)
    _restore_optional_array(sim.data, "act", state._act)
    _restore_optional_array(sim.data, "mocap_pos", state._mocap_pos)
    _restore_optional_array(sim.data, "mocap_quat", state._mocap_quat)
    sim.forward()


def arm_joint_positions_from_state(state: LiberoSimState, arm_joint_indices: Sequence[int]) -> np.ndarray:
    return state.qpos[list(arm_joint_indices)].astype(np.float32)


def arm_joint_linf_error(sim, state: LiberoSimState, arm_joint_indices: Sequence[int]) -> float:
    target = arm_joint_positions_from_state(state, arm_joint_indices)
    current = np.asarray(sim.data.qpos[list(arm_joint_indices)], dtype=np.float32)
    return float(np.max(np.abs(current - target))) if target.size else 0.0


def build_robot_geom_sets(sim, robot_body_prefix: str = "robot0") -> tuple[set[int], set[int], set[int]]:
    gripper_keywords = {"gripper", "finger", "pad", "tip", "hand", "palm", "eef"}
    robot_geom_ids: set[int] = set()
    robot_arm_geom_ids: set[int] = set()
    robot_gripper_geom_ids: set[int] = set()

    for geom_id in range(sim.model.ngeom):
        geom_name = sim.model.geom_id2name(geom_id) or ""
        body_id = sim.model.geom_bodyid[geom_id]
        body_name = sim.model.body_id2name(body_id) or ""
        if robot_body_prefix not in body_name:
            continue
        robot_geom_ids.add(geom_id)
        lower_name = (body_name + geom_name).lower()
        if any(keyword in lower_name for keyword in gripper_keywords):
            robot_gripper_geom_ids.add(geom_id)
        else:
            robot_arm_geom_ids.add(geom_id)

    return robot_geom_ids, robot_arm_geom_ids, robot_gripper_geom_ids


def check_collision_filtered(
    sim,
    robot_arm_geom_ids: set[int],
    robot_gripper_geom_ids: set[int],
    ignore_gripper_object_contact: bool = True,
) -> bool:
    all_robot_geom_ids = robot_arm_geom_ids | robot_gripper_geom_ids
    for contact_index in range(sim.data.ncon):
        contact = sim.data.contact[contact_index]
        geom1, geom2 = contact.geom1, contact.geom2
        involves_robot1 = geom1 in all_robot_geom_ids
        involves_robot2 = geom2 in all_robot_geom_ids
        if not (involves_robot1 or involves_robot2):
            continue

        if ignore_gripper_object_contact:
            gripper1 = geom1 in robot_gripper_geom_ids
            gripper2 = geom2 in robot_gripper_geom_ids
            if (gripper1 and not involves_robot2) or (gripper2 and not involves_robot1):
                continue

        return False
    return True


def find_arm_indices(env) -> tuple[list[int], list[int]]:
    joint_names = list(env.sim.model.joint_names)
    actuator_names = list(env.sim.model.actuator_names)
    qpos_indices = [
        index
        for index, name in enumerate(joint_names)
        if "robot0_joint" in name and "gripper" not in name.lower()
    ]
    actuator_indices = [
        index
        for index, name in enumerate(actuator_names)
        if "robot0_torq" in name and "gripper" not in name.lower()
    ]
    if not qpos_indices or not actuator_indices:
        raise ValueError("Could not find LIBERO robot arm joint or actuator indices")
    return qpos_indices, actuator_indices


def interpolate_joint_path(
    start: Sequence[float],
    goal: Sequence[float],
    max_joint_step: float = 0.08,
) -> list[np.ndarray]:
    start_array = np.asarray(start, dtype=np.float32)
    goal_array = np.asarray(goal, dtype=np.float32)
    delta = goal_array - start_array
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        return [goal_array.copy()]
    steps = max(1, int(np.ceil(distance / max_joint_step)))
    return [(start_array + delta * (i / steps)).astype(np.float32) for i in range(1, steps + 1)]


@dataclass
class JointPlan:
    path: list[np.ndarray]
    safe: bool
    reason: str = ""


@dataclass
class LinearJointPlanner:
    max_joint_step: float = 0.08

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        is_state_valid: Callable[[np.ndarray], bool],
    ) -> JointPlan:
        path = interpolate_joint_path(start, goal, self.max_joint_step)
        for waypoint in path:
            if not is_state_valid(waypoint):
                return JointPlan(path=[], safe=False, reason="collision_on_linear_path")
        return JointPlan(path=path, safe=True)


def make_libero_state_validator(
    sim,
    arm_joint_indices: Sequence[int],
    collision_check: Callable[[object], bool],
) -> Callable[[np.ndarray], bool]:
    arm_joint_indices = list(arm_joint_indices)

    def _is_valid(joint_angles: np.ndarray) -> bool:
        original_qpos = sim.data.qpos.copy()
        original_qvel = sim.data.qvel.copy()
        try:
            sim.data.qpos[arm_joint_indices] = joint_angles
            sim.data.qvel[arm_joint_indices] = 0
            sim.forward()
            return bool(collision_check(sim))
        finally:
            sim.data.qpos[:] = original_qpos
            sim.data.qvel[:] = original_qvel
            sim.forward()

    return _is_valid


@dataclass
class JointPathExecutor:
    kp: float = 50.0
    kd: float = 5.0
    tolerance: float = 0.05
    velocity_tolerance: float = 0.08
    max_steps_per_waypoint: int = 200
    render_height: int = 224
    render_width: int = 224
    render_camera: str = "agentview"

    def execute(
        self,
        env,
        path: Iterable[np.ndarray],
        arm_joint_indices: Sequence[int],
        actuator_indices: Sequence[int],
        frames: list[np.ndarray] | None = None,
    ) -> bool:
        arm_joint_indices = list(arm_joint_indices)
        actuator_indices = list(actuator_indices)
        all_reached = True

        for waypoint in path:
            target = np.asarray(waypoint, dtype=np.float32)
            reached = False
            for _ in range(self.max_steps_per_waypoint):
                current_pos = env.sim.data.qpos[arm_joint_indices]
                current_vel = env.sim.data.qvel[arm_joint_indices]
                error = target - current_pos
                if np.linalg.norm(error) < self.tolerance and np.linalg.norm(current_vel) < self.velocity_tolerance:
                    reached = True
                    break

                torque = self.kp * error - self.kd * current_vel
                env.sim.data.ctrl[actuator_indices] = torque
                env.sim.step()

                if frames is not None:
                    frame = env.sim.render(
                        height=self.render_height,
                        width=self.render_width,
                        camera_name=self.render_camera,
                    )
                    frames.append(np.ascontiguousarray(frame[::-1, ::-1]))
            all_reached = all_reached and reached

        return all_reached


def render_joint_path_kinematic(
    env,
    path: Iterable[np.ndarray],
    arm_joint_indices: Sequence[int],
    frames: list[np.ndarray] | None = None,
    render_height: int = 224,
    render_width: int = 224,
    render_camera: str = "agentview",
    substeps_per_waypoint: int = 4,
) -> bool:
    arm_joint_indices = list(arm_joint_indices)
    substeps = max(1, int(substeps_per_waypoint))
    for waypoint in path:
        start = np.asarray(env.sim.data.qpos[arm_joint_indices], dtype=np.float32).copy()
        target = np.asarray(waypoint, dtype=np.float32)
        for index in range(1, substeps + 1):
            alpha = float(index) / float(substeps)
            env.sim.data.qpos[arm_joint_indices] = (start + alpha * (target - start)).astype(np.float32)
            env.sim.data.qvel[arm_joint_indices] = 0
            env.sim.forward()
            if frames is not None:
                frame = env.sim.render(
                    height=render_height,
                    width=render_width,
                    camera_name=render_camera,
                )
                frames.append(np.ascontiguousarray(frame[::-1, ::-1]))
    return True


@dataclass
class MotionPlanningRollbackExecutor:
    max_joint_step: float = 0.08
    motion_execution_mode: str = "kinematic"
    kp: float = 50.0
    kd: float = 5.0
    tolerance: float = 0.05
    velocity_tolerance: float = 0.08
    max_steps_per_waypoint: int = 120
    kinematic_substeps_per_waypoint: int = 2
    render_height: int = 224
    render_width: int = 224
    render_camera: str = "agentview"
    allow_restore_fallback: bool = False
    frames: list[np.ndarray] | None = None

    def rollback(self, env, waypoint) -> tuple[object, dict]:
        target_state = waypoint.state
        if not isinstance(target_state, LiberoSimState):
            if not self.allow_restore_fallback:
                raise TypeError("MotionPlanningRollbackExecutor requires a LiberoSimState waypoint")
            if hasattr(env, "set_init_state"):
                observation = env.set_init_state(target_state)
            else:
                observation = current_libero_observation(env)
            return observation, {"rollback_mode": "restore_fallback_non_libero_state"}

        arm_joint_indices, actuator_indices = find_arm_indices(env)
        start_qpos = np.asarray(env.sim.data.qpos[arm_joint_indices], dtype=np.float32).copy()
        target_qpos = arm_joint_positions_from_state(target_state, arm_joint_indices)
        _, arm_geoms, gripper_geoms = build_robot_geom_sets(env.sim, "robot0")
        validator = make_libero_state_validator(
            env.sim,
            arm_joint_indices,
            lambda sim: check_collision_filtered(sim, arm_geoms, gripper_geoms),
        )
        plan = LinearJointPlanner(max_joint_step=self.max_joint_step).plan(start_qpos, target_qpos, validator)
        if not plan.safe:
            if not self.allow_restore_fallback:
                raise RuntimeError(f"rollback motion plan rejected: {plan.reason}")
            restore_libero_sim_state(env.sim, target_state)
            return current_libero_observation(env), {
                "rollback_mode": "restore_fallback",
                "planned": False,
                "safe": False,
                "reason": plan.reason,
                "waypoints": 0,
                "reached": True,
                "arm_qpos_linf": arm_joint_linf_error(env.sim, target_state, arm_joint_indices),
            }

        frames_before = len(self.frames) if self.frames is not None else 0
        if self.motion_execution_mode == "kinematic":
            reached = render_joint_path_kinematic(
                env,
                plan.path,
                arm_joint_indices,
                frames=self.frames,
                render_height=self.render_height,
                render_width=self.render_width,
                render_camera=self.render_camera,
                substeps_per_waypoint=self.kinematic_substeps_per_waypoint,
            )
        elif self.motion_execution_mode == "pd":
            executor = JointPathExecutor(
                kp=self.kp,
                kd=self.kd,
                tolerance=self.tolerance,
                velocity_tolerance=self.velocity_tolerance,
                max_steps_per_waypoint=self.max_steps_per_waypoint,
                render_height=self.render_height,
                render_width=self.render_width,
                render_camera=self.render_camera,
            )
            reached = executor.execute(env, plan.path, arm_joint_indices, actuator_indices, self.frames)
        else:
            raise ValueError(f"unsupported motion_execution_mode: {self.motion_execution_mode}")

        if self.frames is not None:
            frames_after = len(self.frames)
            rendered_frames = int(frames_after - frames_before)
        elif self.motion_execution_mode == "kinematic":
            rendered_frames = int(len(plan.path) * max(1, int(self.kinematic_substeps_per_waypoint)))
        else:
            rendered_frames = 0
        return current_libero_observation(env), {
            "rollback_mode": "motion-plan",
            "execution_mode": self.motion_execution_mode,
            "planned": True,
            "safe": True,
            "waypoints": len(plan.path),
            "reached": bool(reached),
            "rendered_frames": rendered_frames,
            "arm_qpos_linf": arm_joint_linf_error(env.sim, target_state, arm_joint_indices),
        }


def current_libero_observation(env):
    if hasattr(env, "regenerate_obs_from_state") and hasattr(env, "get_sim_state"):
        return env.regenerate_obs_from_state(env.get_sim_state())
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
        if hasattr(env, "_post_process"):
            env._post_process()
        if hasattr(env, "_update_observables"):
            env._update_observables(force=True)
        return env.env._get_observations()
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    return None
