from __future__ import annotations

from collections.abc import Collection

import numpy as np


DEFAULT_UNNORM_KEYS = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
DEFAULT_LIBERO_90_UNNORM_KEY = "libero_10_no_noops"


def resolve_unnorm_key(
    benchmark: str,
    available_keys: Collection[str],
    *,
    libero_90_unnorm_key: str = DEFAULT_LIBERO_90_UNNORM_KEY,
) -> str:
    """Select the OpenVLA-OFT normalization statistics used for a LIBERO suite."""
    if benchmark == "libero_90":
        key = libero_90_unnorm_key
    else:
        try:
            key = DEFAULT_UNNORM_KEYS[benchmark]
        except KeyError as exc:
            raise ValueError(f"Unsupported LIBERO benchmark: {benchmark}") from exc

    if key not in available_keys:
        choices = ", ".join(sorted(available_keys))
        raise ValueError(f"OpenVLA-OFT normalization key {key!r} is unavailable. Available keys: {choices}")
    return key


def process_libero_action_chunk(actions: np.ndarray) -> np.ndarray:
    """Apply the official OpenVLA-OFT gripper conversion for LIBERO execution."""
    processed = np.asarray(actions, dtype=np.float32).copy()
    if processed.ndim not in (1, 2) or processed.shape[-1] != 7:
        raise ValueError(f"Expected OpenVLA actions with shape (7,) or (T, 7), got {processed.shape}")

    processed[..., -1] = np.sign(2.0 * processed[..., -1] - 1.0)
    processed[..., -1] *= -1.0
    return processed


def perturb_libero_action_chunk(
    actions: np.ndarray,
    *,
    rng: np.random.Generator,
    std: float,
    final_scale: float = 0.25,
    max_abs_action: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add one coherent, decaying perturbation to a post-rollback action chunk."""
    chunk = np.asarray(actions, dtype=np.float32).copy()
    if chunk.ndim != 2 or chunk.shape[-1] != 7:
        raise ValueError(f"Expected an OpenVLA action chunk with shape (T, 7), got {chunk.shape}")
    if std < 0.0:
        raise ValueError("std must be non-negative")
    if not 0.0 <= final_scale <= 1.0:
        raise ValueError("final_scale must be in [0, 1]")
    if max_abs_action <= 0.0:
        raise ValueError("max_abs_action must be positive")

    perturbation = np.zeros(6, dtype=np.float32)
    if std == 0.0 or len(chunk) == 0:
        return chunk, perturbation

    perturbation = np.asarray(rng.normal(0.0, std, size=6), dtype=np.float32)
    scales = np.linspace(1.0, final_scale, num=len(chunk), dtype=np.float32)
    chunk[:, :6] += scales[:, None] * perturbation[None, :]
    np.clip(chunk[:, :6], -max_abs_action, max_abs_action, out=chunk[:, :6])
    return chunk, perturbation
