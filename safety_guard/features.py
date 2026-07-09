from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .risk import RiskVector


def build_actor_features(
    joint_history: Sequence[Sequence[float]],
    risk_history: Sequence[RiskVector],
    history_length: int = 3,
) -> np.ndarray:
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if len(joint_history) != len(risk_history):
        raise ValueError("joint_history and risk_history must have the same length")

    joint_arrays = [np.asarray(joints, dtype=np.float32).reshape(-1) for joints in joint_history]
    if joint_arrays:
        joint_dim = joint_arrays[-1].shape[0]
    else:
        joint_dim = 7

    rows = []
    start = max(0, len(joint_arrays) - history_length)
    selected_joints = joint_arrays[start:]
    selected_risks = list(risk_history[start:])

    pad_count = history_length - len(selected_joints)
    for _ in range(pad_count):
        rows.append(np.zeros(joint_dim + 4, dtype=np.float32))

    for joints, risk in zip(selected_joints, selected_risks):
        if joints.shape[0] != joint_dim:
            raise ValueError("all joint vectors must have the same dimension")
        rows.append(np.concatenate([joints, risk.as_array()]).astype(np.float32))

    return np.concatenate(rows, axis=0)
