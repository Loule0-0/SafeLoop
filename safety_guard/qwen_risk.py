from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class HazardLabels:
    collision_probability: float
    collision_tth: float
    object_probability: float
    object_tth: float

    def to_assistant_text(self) -> str:
        return (
            f"{_format_number(self.collision_probability)} {_format_number(self.collision_tth)}\n"
            f"{_format_number(self.object_probability)} {_format_number(self.object_tth)}"
        )


def parse_hazard_response(text: str) -> HazardLabels:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("hazard response must contain two non-empty lines")
    first = _parse_pair(lines[0])
    second = _parse_pair(lines[1])
    return HazardLabels(first[0], first[1], second[0], second[1])


def normalize_tth(value: float, tau: float) -> tuple[float, float]:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if value < 0:
        return -1.0, 0.0
    clipped = min(float(value), float(tau))
    return clipped / float(tau), 1.0


def build_qwen_messages(
    history_samples: Sequence[dict],
    action: Iterable[float],
    labels: HazardLabels,
    tau: int,
    image_paths: Sequence[str],
) -> dict:
    user_lines = [
        f"H={len(history_samples)}, tau={tau}",
        "Hazard types: 0=collision, 1=item_damage",
        "For each of the past H steps (oldest->newest), we provide:",
    ]

    for sample in history_samples:
        state = sample.get("robot_state", {})
        user_lines.extend(
            [
                "- images: <image> <image>",
                f"- robot0_joint_pos: {_format_array(state.get('robot0_joint_pos', []))}",
                f"- robot0_joint_vel: {_format_array(state.get('robot0_joint_vel', []))}",
                f"- robot0_joint_torques: {_format_array(state.get('robot0_joint_torques', []))}",
            ]
        )

    user_lines.extend(
        [
            f"Next action a_t: {_format_array(action)}",
            "Please return two lines:",
            "line1: p_collision T_collision",
            "line2: p_item_damage T_item_damage",
        ]
    )

    return {
        "messages": [
            {"role": "user", "content": "\n".join(user_lines)},
            {"role": "assistant", "content": labels.to_assistant_text()},
        ],
        "images": list(image_paths),
    }


def _parse_pair(line: str) -> tuple[float, float]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"expected '<p> <T>', got {line!r}")
    return float(parts[0]), float(parts[1])


def _format_array(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _format_number(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"
