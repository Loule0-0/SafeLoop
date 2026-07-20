from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Protocol

import numpy as np

from .decider import Intervention, RuleBasedDecider
from .libero_motion import LiberoSimState, capture_libero_sim_state, restore_libero_sim_state
from .memory import Waypoint, WaypointMemory, initial_anchor_rollback_allowed
from .risk import RiskVector


class RiskPredictor(Protocol):
    def predict(
        self,
        observation: Any,
        proposed_action: Any,
        instruction: str | None = None,
    ) -> RiskVector:
        ...


class RollbackExecutor(Protocol):
    def rollback(self, env: Any, waypoint: Waypoint) -> Any:
        ...


@dataclass
class ProprioceptiveStuckMonitor:
    window_steps: int = 70
    min_low_motion_steps: int = 50
    max_step_displacement: float = 0.0025
    max_window_displacement: float = 0.025
    _positions: Deque[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_steps < 2:
            raise ValueError("window_steps must be at least two")
        self._positions = deque(maxlen=int(self.window_steps))

    def reset(self) -> None:
        self._positions.clear()

    def update(self, observation: Any) -> bool:
        position = extract_robot_eef_position(observation)
        if position is None:
            self.reset()
            return False
        self._positions.append(position)
        if len(self._positions) < int(self.window_steps):
            return False
        positions = np.stack(list(self._positions))
        step_displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        low_motion_steps = int(np.count_nonzero(step_displacements <= float(self.max_step_displacement)))
        window_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        return bool(
            low_motion_steps >= int(self.min_low_motion_steps)
            and window_displacement <= float(self.max_window_displacement)
        )


def apply_stuck_fallback(
    risk: RiskVector,
    current_body_probability: float,
    stuck_detected: bool,
) -> tuple[RiskVector, float]:
    if not stuck_detected:
        return risk, float(current_body_probability)
    return (
        RiskVector(
            body_probability=max(float(risk.body_probability), 1.0),
            body_tth=0.0,
            object_probability=float(risk.object_probability),
            object_tth=float(risk.object_tth),
        ),
        max(float(current_body_probability), 1.0),
    )


@dataclass
class SafeLoopStepResult:
    observation: Any
    reward: float
    done: bool
    info: dict[str, Any]
    risk: RiskVector
    intervention: Intervention
    executed_nominal_action: bool


@dataclass
class SafeLoopController:
    predictor: RiskPredictor
    decider: Any
    memory: WaypointMemory = field(default_factory=WaypointMemory)
    rollback_executor: RollbackExecutor | None = None
    max_history: int = 3
    max_steps: int | None = None
    prediction_period: int = 1
    rollback_target_safe_score_threshold: float = 0.6
    rollback_target_max_age: int = 120
    rollback_target_min_age: int = 30
    rollback_target_require_safe: bool = False
    rollback_target_prefer_recent_safe: bool = False
    stuck_rollback_target_max_age: int | None = None
    record_max_risk_score: float | None = None
    record_max_current_body_probability: float | None = None
    record_max_current_object_probability: float | None = None
    record_initial_safe_anchor: bool = False
    initial_anchor_rollback_min_current_body_probability: float | None = None
    initial_anchor_rollback_min_current_object_probability: float | None = None
    auto_record_safe_anchors: bool = False
    auto_record_min_interval: int = 20
    auto_record_max_risk_score: float = 0.4
    auto_record_max_current_body_probability: float = 0.3
    auto_record_max_current_object_probability: float = 0.45
    stuck_fallback: bool = False
    stuck_window_steps: int = 70
    stuck_min_low_motion_steps: int = 50
    stuck_max_step_displacement: float = 0.0025
    stuck_max_window_displacement: float = 0.025
    step_index: int = 0
    last_record_step: int | None = None
    last_rollback_step: int | None = None
    _joint_history: Deque[np.ndarray] = field(init=False, repr=False)
    _risk_history: Deque[RiskVector] = field(init=False, repr=False)
    _current_hazard_history: Deque[tuple[float, float]] = field(init=False, repr=False)
    _stuck_monitor: ProprioceptiveStuckMonitor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        if self.prediction_period <= 0:
            raise ValueError("prediction_period must be positive")
        self._joint_history = deque(maxlen=self.max_history)
        self._risk_history = deque(maxlen=self.max_history)
        self._current_hazard_history = deque(maxlen=self.max_history)
        self._stuck_monitor = ProprioceptiveStuckMonitor(
            window_steps=self.stuck_window_steps,
            min_low_motion_steps=self.stuck_min_low_motion_steps,
            max_step_displacement=self.stuck_max_step_displacement,
            max_window_displacement=self.stuck_max_window_displacement,
        )

    def reset(self) -> None:
        self.memory.clear()
        self.step_index = 0
        self.last_record_step = None
        self.last_rollback_step = None
        self._joint_history.clear()
        self._risk_history.clear()
        self._current_hazard_history.clear()
        self._stuck_monitor.reset()

    def step(
        self,
        env: Any,
        observation: Any,
        proposed_action: Any,
        instruction: str | None = None,
    ) -> SafeLoopStepResult:
        initial_safe_anchor_recorded = False
        if self.record_initial_safe_anchor and self.step_index == 0 and len(self.memory) == 0:
            self.memory.record(
                step_index=0,
                state=capture_env_state(env),
                risk=None,
                metadata={"source": "initial_safe_anchor"},
            )
            self.last_record_step = 0
            initial_safe_anchor_recorded = True

        stuck_detected = bool(self.stuck_fallback and self._stuck_monitor.update(observation))
        if self.step_index % int(self.prediction_period) != 0:
            risk = self._risk_history[-1] if self._risk_history else RiskVector(0.0, 1.0, 0.0, 1.0)
            current_body_probability, current_object_probability = (
                self._current_hazard_history[-1] if self._current_hazard_history else (0.0, 0.0)
            )
            next_observation, reward, done, info = env.step(proposed_action)
            info = dict(info or {})
            info["safeloop"] = {
                "intervention": Intervention.NOOP.value,
                "risk": risk_info_dict(risk, current_body_probability, current_object_probability),
                "recorded_waypoint": False,
                "initial_safe_anchor_recorded": initial_safe_anchor_recorded,
                "stuck_fallback": stuck_detected,
                "decision": {
                    "decider": "periodic",
                    "period": int(self.prediction_period),
                    "due": False,
                    "selected_action": Intervention.NOOP.value,
                },
            }
            self.step_index += 1
            return SafeLoopStepResult(
                observation=next_observation,
                reward=float(reward),
                done=bool(done),
                info=info,
                risk=risk,
                intervention=Intervention.NOOP,
                executed_nominal_action=True,
            )

        prediction = self.predictor.predict(observation, proposed_action, instruction=instruction)
        risk, current_body_probability, current_object_probability = normalize_risk_prediction(prediction)
        risk, current_body_probability = apply_stuck_fallback(
            risk,
            current_body_probability,
            stuck_detected,
        )
        self._joint_history.append(extract_robot_joint_vector(observation, proposed_action))
        self._risk_history.append(risk)
        self._current_hazard_history.append((current_body_probability, current_object_probability))

        rollback_waypoint = None
        initial_anchor_rollback_blocked = False
        rollback_max_age = (
            self.stuck_rollback_target_max_age
            if stuck_detected and self.stuck_rollback_target_max_age is not None
            else self.rollback_target_max_age
        )
        if len(self.memory) > 0:
            try:
                rollback_waypoint = self.memory.select_rollback(
                    current_step_index=self.step_index,
                    safe_score_threshold=self.rollback_target_safe_score_threshold,
                    max_safe_age=rollback_max_age,
                    min_safe_age=self.rollback_target_min_age,
                    require_safe=self.rollback_target_require_safe,
                    prefer_recent_safe=self.rollback_target_prefer_recent_safe,
                )
            except IndexError:
                rollback_waypoint = None
        if rollback_waypoint is not None and not initial_anchor_rollback_allowed(
            rollback_waypoint,
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            stuck_detected=stuck_detected,
            min_current_body_probability=self.initial_anchor_rollback_min_current_body_probability,
            min_current_object_probability=self.initial_anchor_rollback_min_current_object_probability,
        ):
            rollback_waypoint = None
            initial_anchor_rollback_blocked = True

        intervention = call_decider(
            self.decider,
            risk=risk,
            memory=self.memory,
            step_index=self.step_index,
            last_record_step=self.last_record_step,
            last_rollback_step=self.last_rollback_step,
            observation=observation,
            proposed_action=proposed_action,
            joint_history=list(self._joint_history),
            risk_history=list(self._risk_history),
            current_hazard_history=list(self._current_hazard_history),
            current_body_probability=current_body_probability,
            current_object_probability=current_object_probability,
            stuck_detected=stuck_detected,
            rollback_target_available=rollback_waypoint is not None,
            max_steps=self.max_steps,
        )
        decision_info = getattr(self.decider, "last_decision_info", None)

        if intervention == Intervention.ROLLBACK:
            try:
                waypoint = rollback_waypoint
                if waypoint is None:
                    raise IndexError("no rollback target satisfies the configured safety and age constraints")
            except IndexError as exc:
                self.last_rollback_step = self.step_index
                self.step_index += 1
                safeloop_info = {
                    "intervention": intervention.value,
                    "rollback_mode": "none",
                    "planned": False,
                    "safe": False,
                    "reached": False,
                    "reason": "no_rollback_target",
                    "error_type": type(exc).__name__,
                    "risk": risk_info_dict(risk, current_body_probability, current_object_probability),
                    "initial_safe_anchor_recorded": initial_safe_anchor_recorded,
                }
                if decision_info is not None:
                    safeloop_info["decision"] = decision_info
                return SafeLoopStepResult(
                    observation=current_env_observation(env),
                    reward=0.0,
                    done=False,
                    info={"safeloop": safeloop_info},
                    risk=risk,
                    intervention=intervention,
                    executed_nominal_action=False,
                )
            rollback_observation, rollback_info = execute_rollback(env, waypoint, self.rollback_executor)
            self._stuck_monitor.reset()
            refreshed_rollback_anchor = False
            if rollback_info.get("safe", True) is not False and rollback_info.get("reached", True) is not False:
                anchor_metadata = dict(waypoint.metadata)
                anchor_metadata["source"] = "rollback_anchor"
                self.memory.record(
                    step_index=self.step_index,
                    state=capture_env_state(env),
                    risk=waypoint.risk,
                    metadata=anchor_metadata,
                )
                self.last_record_step = self.step_index
                refreshed_rollback_anchor = True
            self.last_rollback_step = self.step_index
            self.step_index += 1
            safeloop_info = {
                "intervention": intervention.value,
                "rollback_step": waypoint.step_index,
                "risk": risk_info_dict(risk, current_body_probability, current_object_probability),
                "stuck_fallback": stuck_detected,
                "refreshed_rollback_anchor": refreshed_rollback_anchor,
                "initial_safe_anchor_recorded": initial_safe_anchor_recorded,
            }
            if decision_info is not None:
                safeloop_info["decision"] = decision_info
            safeloop_info.update(rollback_info)
            return SafeLoopStepResult(
                observation=rollback_observation,
                reward=0.0,
                done=False,
                info={"safeloop": safeloop_info},
                risk=risk,
                intervention=intervention,
                executed_nominal_action=False,
            )

        recorded_waypoint = initial_safe_anchor_recorded
        auto_recorded = False
        record_blocked = False
        if intervention == Intervention.RECORD and not initial_safe_anchor_recorded:
            if self._record_allowed(risk, current_body_probability, current_object_probability):
                self._record_waypoint(env, risk, current_body_probability, current_object_probability)
                recorded_waypoint = True
            else:
                record_blocked = True
        elif not recorded_waypoint and self._should_auto_record(
            risk,
            current_body_probability,
            current_object_probability,
        ):
            self._record_waypoint(env, risk, current_body_probability, current_object_probability)
            recorded_waypoint = True
            auto_recorded = True

        next_observation, reward, done, info = env.step(proposed_action)
        info = dict(info or {})
        info["safeloop"] = {
            "intervention": intervention.value,
            "risk": risk_info_dict(risk, current_body_probability, current_object_probability),
            "recorded_waypoint": recorded_waypoint,
            "initial_safe_anchor_recorded": initial_safe_anchor_recorded,
            "stuck_fallback": stuck_detected,
        }
        if auto_recorded:
            info["safeloop"]["auto_recorded"] = True
        if record_blocked:
            info["safeloop"]["record_blocked"] = True
        if initial_anchor_rollback_blocked:
            info["safeloop"]["initial_anchor_rollback_blocked"] = True
        if decision_info is not None:
            info["safeloop"]["decision"] = decision_info
        self.step_index += 1
        return SafeLoopStepResult(
            observation=next_observation,
            reward=float(reward),
            done=bool(done),
            info=info,
            risk=risk,
            intervention=intervention,
            executed_nominal_action=True,
        )

    def _record_waypoint(
        self,
        env: Any,
        risk: RiskVector,
        current_body_probability: float,
        current_object_probability: float,
    ) -> None:
        self.memory.record(
            step_index=self.step_index,
            state=capture_env_state(env),
            risk=risk,
            metadata={
                "current_body_probability": current_body_probability,
                "current_object_probability": current_object_probability,
            },
        )
        self.last_record_step = self.step_index

    def _record_allowed(
        self,
        risk: RiskVector,
        current_body_probability: float,
        current_object_probability: float,
    ) -> bool:
        if self.record_max_risk_score is not None and (
            candidate_risk_score(risk, current_body_probability, current_object_probability)
            > float(self.record_max_risk_score)
        ):
            return False
        if self.record_max_current_body_probability is not None and (
            float(current_body_probability) > float(self.record_max_current_body_probability)
        ):
            return False
        if self.record_max_current_object_probability is not None and (
            float(current_object_probability) > float(self.record_max_current_object_probability)
        ):
            return False
        return True

    def _should_auto_record(
        self,
        risk: RiskVector,
        current_body_probability: float,
        current_object_probability: float,
    ) -> bool:
        if not self.auto_record_safe_anchors:
            return False
        if self.last_record_step is not None:
            interval = int(self.step_index) - int(self.last_record_step)
            if interval < int(self.auto_record_min_interval):
                return False
        if (
            candidate_risk_score(risk, current_body_probability, current_object_probability)
            > float(self.auto_record_max_risk_score)
        ):
            return False
        if float(current_body_probability) > float(self.auto_record_max_current_body_probability):
            return False
        if float(current_object_probability) > float(self.auto_record_max_current_object_probability):
            return False
        return True


def normalize_risk_prediction(prediction: Any) -> tuple[RiskVector, float, float]:
    risk = getattr(prediction, "risk", prediction)
    if not isinstance(risk, RiskVector):
        risk = RiskVector.from_iterable(risk)
    return (
        risk,
        float(getattr(prediction, "current_body_probability", 0.0)),
        float(getattr(prediction, "current_object_probability", 0.0)),
    )


def risk_info_dict(
    risk: RiskVector,
    current_body_probability: float,
    current_object_probability: float,
) -> dict[str, float]:
    return {
        "body_probability": float(risk.body_probability),
        "body_tth": float(risk.body_tth),
        "object_probability": float(risk.object_probability),
        "object_tth": float(risk.object_tth),
        "current_body_probability": float(current_body_probability),
        "current_object_probability": float(current_object_probability),
    }


def candidate_risk_score(
    risk: RiskVector,
    current_body_probability: float,
    current_object_probability: float,
) -> float:
    tth_risk = max(0.0, 1.0 - float(risk.min_tth)) * 0.25
    return float(
        max(
            float(risk.max_probability),
            tth_risk,
            float(current_body_probability),
            float(current_object_probability),
        )
    )


def call_decider(decider: Any, **kwargs: Any) -> Intervention:
    decide = decider.decide
    signature = inspect.signature(decide)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return decide(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return decide(**accepted)


def execute_rollback(
    env: Any,
    waypoint: Waypoint,
    rollback_executor: RollbackExecutor | None,
) -> tuple[Any, dict[str, Any]]:
    if rollback_executor is None:
        return restore_env_state(env, waypoint.state), {"rollback_mode": "restore"}

    try:
        result = rollback_executor.rollback(env, waypoint)
    except RuntimeError as exc:
        return current_env_observation(env), rollback_failure_info(exc)
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, dict(info or {})
    if hasattr(result, "observation"):
        return result.observation, dict(getattr(result, "info", {}) or {})
    return result, {}


def rollback_failure_info(exc: RuntimeError) -> dict[str, Any]:
    reason = str(exc)
    prefix = "rollback motion plan rejected: "
    if reason.startswith(prefix):
        reason = reason[len(prefix) :]
    return {
        "rollback_mode": "motion-plan",
        "planned": False,
        "safe": False,
        "reached": False,
        "reason": reason,
        "error_type": type(exc).__name__,
    }


def capture_env_state(env: Any) -> Any:
    if hasattr(env, "sim"):
        return capture_libero_sim_state(env.sim)
    return np.asarray(env.get_sim_state(), dtype=np.float32)


def restore_env_state(env: Any, state: Any) -> Any:
    if isinstance(state, LiberoSimState) and hasattr(env, "sim"):
        restore_libero_sim_state(env.sim, state)
        return current_env_observation(env)
    return env.set_init_state(state)


def current_env_observation(env: Any) -> Any:
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
    if hasattr(env, "get_sim_state"):
        return {"state": np.asarray(env.get_sim_state(), dtype=np.float32)}
    return None


def extract_robot_joint_vector(observation: Any, proposed_action: Any | None = None) -> np.ndarray:
    if isinstance(observation, dict):
        joint_pos = _array_from_first_key(
            observation,
            ("robot0_joint_pos", "joint_pos", "observation/joint_pos"),
        )
        gripper = _array_from_first_key(
            observation,
            ("robot0_gripper_qpos", "gripper_qpos", "observation/gripper_qpos"),
        )
        if joint_pos is not None:
            parts = [joint_pos.reshape(-1).astype(np.float32)]
            if gripper is not None:
                parts.append(gripper.reshape(-1).astype(np.float32))
            return np.concatenate(parts).astype(np.float32)
        state = _array_from_first_key(observation, ("observation/state", "state"))
        if state is not None:
            return state.reshape(-1).astype(np.float32)

    if proposed_action is not None:
        action = np.asarray(proposed_action, dtype=np.float32).reshape(-1)
        if action.size:
            return action
    return np.zeros(9, dtype=np.float32)


def extract_robot_eef_position(observation: Any) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    position = _array_from_first_key(
        observation,
        ("robot0_eef_pos", "eef_pos", "observation/eef_pos"),
    )
    if position is None or position.size < 3:
        return None
    return position.reshape(-1)[:3].astype(np.float32)


def _array_from_first_key(observation: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in observation and observation[key] is not None:
            return np.asarray(observation[key], dtype=np.float32)
    return None
