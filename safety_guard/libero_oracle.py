from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .libero_motion import build_robot_geom_sets
from .online_rl import OnlineStepSignals


STATIC_BODY_KEYWORDS = (
    "world",
    "table",
    "arena",
    "floor",
    "wall",
    "bin",
    "cabinet",
    "drawer",
    "robot",
    "gripper",
    "camera",
    "mount",
    "base",
)


@dataclass
class LiberoHazardOracle:
    sim: object
    robot_body_prefix: str = "robot0"
    object_drop_threshold: float = 0.08
    object_lift_threshold: float = 0.06
    object_speed_threshold: float = 1.5
    object_motion_warmup_steps: int = 20
    object_motion_min_displacement: float = 0.03
    object_unstable_sustain_steps: int = 3
    stuck_eef_speed_threshold: float = 0.03
    stuck_contact_steps: int = 4
    stuck_window_steps: int = 60
    stuck_window_eef_speed_threshold: float = 0.05
    stuck_window_min_low_speed_steps: int = 35
    stuck_window_min_contact_steps: int = 5
    stuck_progress_window_steps: int = 70
    stuck_progress_min_low_speed_steps: int = 50
    stuck_progress_max_displacement: float = 0.025
    _robot_geoms: set[int] = field(init=False, repr=False)
    _arm_geoms: set[int] = field(init=False, repr=False)
    _gripper_geoms: set[int] = field(init=False, repr=False)
    _object_body_ids: list[int] = field(init=False, repr=False)
    _initial_pos: dict[int, np.ndarray] = field(init=False, repr=False)
    _initial_z: dict[int, float] = field(init=False, repr=False)
    _max_z: dict[int, float] = field(init=False, repr=False)
    _previous_pos: dict[int, np.ndarray] = field(init=False, repr=False)
    _unstable_motion_counts: dict[int, int] = field(init=False, repr=False)
    _object_step_count: int = field(default=0, init=False, repr=False)
    _last_object_hazard_reasons: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _previous_time: float = field(init=False, repr=False)
    _previous_eef_time: float = field(init=False, repr=False)
    _previous_any_hazard: bool = field(default=False, init=False, repr=False)
    _previous_eef_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _stuck_contact_count: int = field(default=0, init=False, repr=False)
    _stuck_window: deque = field(default_factory=deque, init=False, repr=False)
    _stuck_progress_window: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        self._robot_geoms, self._arm_geoms, self._gripper_geoms = build_robot_geom_sets(
            self.sim,
            self.robot_body_prefix,
        )
        self._object_body_ids = self._discover_object_bodies()
        self._initial_pos = {body_id: self._body_pos(body_id) for body_id in self._object_body_ids}
        self._initial_z = {body_id: float(pos[2]) for body_id, pos in self._initial_pos.items()}
        self._max_z = dict(self._initial_z)
        self._previous_pos = {body_id: pos.copy() for body_id, pos in self._initial_pos.items()}
        self._unstable_motion_counts = {body_id: 0 for body_id in self._object_body_ids}
        self._previous_time = self._sim_time()
        self._previous_eef_time = self._previous_time
        self._previous_eef_pos = self._eef_pos()

    def read(self, success: bool = False, task_reward: float = 0.0) -> OnlineStepSignals:
        body_hazard, arm_contact_count = self._body_hazard()
        object_hazard = self._object_hazard()
        eef_speed = self._eef_speed()
        stuck_hazard = self._stuck_hazard(eef_speed=eef_speed, arm_contact_count=arm_contact_count)
        any_hazard = bool(body_hazard or object_hazard or stuck_hazard)
        event = any_hazard and not self._previous_any_hazard
        self._previous_any_hazard = any_hazard
        return OnlineStepSignals(
            body_hazard=body_hazard,
            object_hazard=object_hazard,
            stuck_hazard=stuck_hazard,
            hazard_event=event,
            success=bool(success),
            task_reward=float(task_reward),
            eef_speed=float(eef_speed),
            arm_contact_count=int(arm_contact_count),
        )

    def _discover_object_bodies(self) -> list[int]:
        body_ids: list[int] = []
        for body_id in range(1, int(self.sim.model.nbody)):
            name = (self.sim.model.body_id2name(body_id) or "").lower()
            if not name:
                continue
            if any(keyword in name for keyword in STATIC_BODY_KEYWORDS):
                continue
            mass = float(np.asarray(self.sim.model.body_mass)[body_id]) if hasattr(self.sim.model, "body_mass") else 1.0
            if mass <= 0.0:
                continue
            body_ids.append(body_id)
        return body_ids

    def _body_hazard(self) -> tuple[bool, int]:
        arm_contact_count = 0
        all_robot_geoms = self._arm_geoms | self._gripper_geoms
        for contact_index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            involves_arm = geom1 in self._arm_geoms or geom2 in self._arm_geoms
            involves_robot = geom1 in all_robot_geoms or geom2 in all_robot_geoms
            if not involves_robot:
                continue
            if involves_arm:
                arm_contact_count += 1
            gripper1 = geom1 in self._gripper_geoms
            gripper2 = geom2 in self._gripper_geoms
            other_robot1 = geom1 in all_robot_geoms
            other_robot2 = geom2 in all_robot_geoms
            if (gripper1 and not other_robot2) or (gripper2 and not other_robot1):
                continue
            if involves_arm:
                return True, arm_contact_count
        return False, arm_contact_count

    def _object_hazard(self) -> bool:
        now = self._sim_time()
        dt = now - self._previous_time
        time_advanced = dt > 1e-9
        warmup_done = self._object_step_count >= max(0, int(self.object_motion_warmup_steps))
        hazard = False
        reasons: list[dict[str, object]] = []
        for body_id in self._object_body_ids:
            pos = self._body_pos(body_id)
            z = float(pos[2])
            if not warmup_done:
                self._initial_pos[body_id] = pos.copy()
                self._initial_z[body_id] = z
                self._max_z[body_id] = z
                self._previous_pos[body_id] = pos
                self._unstable_motion_counts[body_id] = 0
                continue
            initial_z = self._initial_z[body_id]
            self._max_z[body_id] = max(self._max_z[body_id], z)
            previous = self._previous_pos[body_id]
            speed = float(np.linalg.norm(pos - previous) / dt) if time_advanced else 0.0
            displacement = float(np.linalg.norm(pos - self._initial_pos[body_id]))
            was_lifted = self._max_z[body_id] >= initial_z + self.object_lift_threshold
            dropped_after_lift = was_lifted and z <= self._max_z[body_id] - self.object_drop_threshold
            fell_below_start = z <= initial_z - self.object_drop_threshold
            raw_unstable_motion = (
                warmup_done
                and time_advanced
                and speed >= self.object_speed_threshold
                and displacement >= self.object_motion_min_displacement
                and z <= initial_z + self.object_lift_threshold
            )
            if raw_unstable_motion:
                self._unstable_motion_counts[body_id] += 1
            else:
                self._unstable_motion_counts[body_id] = 0
            unstable_motion = self._unstable_motion_counts[body_id] >= max(1, int(self.object_unstable_sustain_steps))
            causes = []
            if dropped_after_lift:
                causes.append("dropped_after_lift")
            if fell_below_start:
                causes.append("fell_below_start")
            if unstable_motion:
                causes.append("unstable_motion")
            if causes:
                reasons.append(
                    {
                        "body_id": int(body_id),
                        "body_name": self.sim.model.body_id2name(body_id) or "",
                        "causes": causes,
                        "z": float(z),
                        "initial_z": float(initial_z),
                        "max_z": float(self._max_z[body_id]),
                        "speed": float(speed),
                        "displacement": float(displacement),
                        "step_count": int(self._object_step_count),
                    }
                )
            hazard = hazard or bool(causes)
            self._previous_pos[body_id] = pos
        self._previous_time = now
        self._object_step_count += 1
        self._last_object_hazard_reasons = reasons
        return bool(hazard)

    def last_object_hazard_reasons(self) -> list[dict[str, object]]:
        return list(self._last_object_hazard_reasons)

    def _stuck_hazard(self, eef_speed: float, arm_contact_count: int) -> bool:
        has_arm_contact = int(arm_contact_count) > 0
        low_speed_contact = has_arm_contact and float(eef_speed) <= float(self.stuck_eef_speed_threshold)
        if low_speed_contact:
            self._stuck_contact_count += 1
        else:
            self._stuck_contact_count = 0
        hard_contact_stuck = self._stuck_contact_count >= max(1, int(self.stuck_contact_steps))

        low_speed = float(eef_speed) <= float(self.stuck_window_eef_speed_threshold)
        self._stuck_window.append((bool(low_speed), bool(has_arm_contact)))
        max_window = max(1, int(self.stuck_window_steps))
        while len(self._stuck_window) > max_window:
            self._stuck_window.popleft()
        low_speed_steps = sum(1 for item in self._stuck_window if item[0])
        contact_steps = sum(1 for item in self._stuck_window if item[1])
        stalled_contact_stuck = (
            low_speed
            and low_speed_steps >= max(1, int(self.stuck_window_min_low_speed_steps))
            and contact_steps >= max(1, int(self.stuck_window_min_contact_steps))
        )

        current_pos = None if self._previous_eef_pos is None else self._previous_eef_pos.copy()
        self._stuck_progress_window.append((bool(low_speed), current_pos))
        progress_window = max(1, int(self.stuck_progress_window_steps))
        while len(self._stuck_progress_window) > progress_window:
            self._stuck_progress_window.popleft()
        progress_low_speed_steps = sum(1 for item in self._stuck_progress_window if item[0])
        progress_stuck = False
        if len(self._stuck_progress_window) >= progress_window:
            valid_positions = [item[1] for item in self._stuck_progress_window if item[1] is not None]
            if len(valid_positions) >= 2:
                displacement = float(np.linalg.norm(valid_positions[-1] - valid_positions[0]))
                progress_stuck = (
                    low_speed
                    and progress_low_speed_steps >= max(1, int(self.stuck_progress_min_low_speed_steps))
                    and displacement <= float(self.stuck_progress_max_displacement)
                )

        return bool(hard_contact_stuck or stalled_contact_stuck or progress_stuck)

    def _eef_speed(self) -> float:
        now = self._sim_time()
        pos = self._eef_pos()
        if pos is None:
            self._previous_eef_time = now
            return 0.0
        previous = self._previous_eef_pos
        previous_time = self._previous_eef_time
        self._previous_eef_pos = pos
        self._previous_eef_time = now
        if previous is None:
            return 0.0
        dt = now - previous_time
        if dt <= 1e-9:
            return 0.0
        return float(np.linalg.norm(pos - previous) / dt)

    def _body_pos(self, body_id: int) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[body_id], dtype=np.float32).copy()

    def _sim_time(self) -> float:
        return float(getattr(self.sim.data, "time", 0.0))

    def _eef_pos(self) -> np.ndarray | None:
        for name in ("robot0_eef", "robot0_right_hand", "gripper0_eef"):
            try:
                body_id = self.sim.model.body_name2id(name)
            except Exception:
                continue
            return self._body_pos(body_id)
        return None
