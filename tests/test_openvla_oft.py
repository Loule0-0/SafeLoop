from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from safety_guard.openvla_oft import (
    perturb_libero_action_chunk,
    process_libero_action_chunk,
    resolve_unnorm_key,
)
from scripts.libero_policy_utils import policy_observation
from scripts.serve_openvla_oft import disable_tensorflow_gpu


def test_resolve_unnorm_key_uses_suite_stats_and_explicit_libero_90_fallback() -> None:
    keys = {"libero_object_no_noops", "libero_10_no_noops"}
    assert resolve_unnorm_key("libero_object", keys) == "libero_object_no_noops"
    assert resolve_unnorm_key("libero_90", keys) == "libero_10_no_noops"


def test_resolve_unnorm_key_rejects_missing_stats() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        resolve_unnorm_key("libero_goal", {"libero_10_no_noops"})


def test_process_libero_action_chunk_matches_official_gripper_conversion() -> None:
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, -1] = [0.0, 0.5, 1.0]
    processed = process_libero_action_chunk(actions)
    np.testing.assert_array_equal(processed[:, -1], [1.0, 0.0, -1.0])


def test_post_rollback_perturbation_is_reproducible_decaying_and_preserves_gripper() -> None:
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, -1] = [1.0, -1.0, 1.0, -1.0]
    first, first_noise = perturb_libero_action_chunk(
        actions,
        rng=np.random.default_rng(17),
        std=0.02,
        final_scale=0.25,
    )
    second, second_noise = perturb_libero_action_chunk(
        actions,
        rng=np.random.default_rng(17),
        std=0.02,
        final_scale=0.25,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_noise, second_noise)
    np.testing.assert_allclose(first[0, :6], first_noise)
    np.testing.assert_allclose(first[-1, :6], first_noise * 0.25)
    np.testing.assert_array_equal(first[:, -1], actions[:, -1])


def test_zero_post_rollback_perturbation_is_an_exact_noop() -> None:
    actions = np.linspace(-1.0, 1.0, num=21, dtype=np.float32).reshape(3, 7)
    perturbed, noise = perturb_libero_action_chunk(
        actions,
        rng=np.random.default_rng(1),
        std=0.0,
    )
    np.testing.assert_array_equal(perturbed, actions)
    np.testing.assert_array_equal(noise, np.zeros(6, dtype=np.float32))


def test_openvla_observation_keeps_raw_images_and_adds_suite_metadata() -> None:
    obs = {
        "agentview_image": np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3),
        "robot0_eye_in_hand_image": np.zeros((4, 5, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    prepared = policy_observation(
        obs,
        "pick up the object",
        224,
        policy_backend="openvla-oft",
        benchmark="libero_object",
    )
    assert prepared["observation/image"].shape == (4, 5, 3)
    np.testing.assert_array_equal(prepared["observation/image"], obs["agentview_image"][::-1, ::-1])
    assert prepared["policy/benchmark"] == "libero_object"
    assert prepared["observation/state"].shape == (8,)


def test_disable_tensorflow_gpu_hides_cuda_during_device_discovery(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def set_visible_devices(devices, device_type) -> None:
        observed["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        observed["devices"] = devices
        observed["device_type"] = device_type

    fake_tensorflow = SimpleNamespace(config=SimpleNamespace(set_visible_devices=set_visible_devices))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")

    disable_tensorflow_gpu()

    assert observed == {"cuda_visible_devices": "", "devices": [], "device_type": "GPU"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
