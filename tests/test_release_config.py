from __future__ import annotations

import json
from pathlib import Path

from scripts.run_release_24task_eval import _build_command, _load_config
from scripts.run_release_decider_training import _parse_override


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_openvla_release_config_inherits_full_task_and_seed_matrix() -> None:
    config = _load_config(PROJECT_ROOT / "configs" / "release" / "openvla_oft_24task_eval.json")
    profile = config["profiles"]["safeloop_all"]

    assert sum(len(entry["task_ids"]) for entry in profile["entries"]) == 24
    assert len(profile["seeds"]) == 16
    assert config["seed_selects_init_state"] is True
    assert all(entry["mode"] == "rl" for entry in profile["entries"])
    assert config["common_args"]["policy-backend"] == "openvla-oft"
    assert config["common_args"]["max-rollout-steps"] == 520
    assert config["common_args"]["replan-steps"] == 8
    assert config["common_args"]["post-rollback-action-noise-std"] == 0.04
    assert config["common_args"]["post-rollback-action-noise-final-scale"] == 0.25
    assert config["common_args"]["post-rollback-action-noise-replans"] == 1
    assert config["common_args"]["rollback-target-require-safe"] is True
    assert config["common_args"]["stuck-fallback"] is True
    assert config["common_args"]["stuck-window-steps"] == 70
    assert config["common_args"]["rollback-gate-allow-stuck-object-override"] is True
    assert config["common_args"]["stuck-rollback-target-max-age"] == 320
    assert config["common_args"]["rollback-target-max-age"] == 200
    assert config["common_args"]["rollback-gate-max-current-object-probability"] == 1.01
    assert config["checkpoints"]["decision"] == (
        "decision_heads/openvla_oft_libero90/online_decider_update001.pt"
    )

    entry = profile["entries"][0]
    command = _build_command(
        config,
        entry,
        seed=7,
        context={
            "model_dir": "/models/qwen",
            "weights_dir": "/models/safeloop",
            "policy_host": "127.0.0.1",
            "policy_port": "8001",
            "decision_checkpoint": "/models/decider.pt",
        },
        output_root=Path("/tmp/eval"),
    )
    assert command[command.index("--init-state-start") + 1] == "7"
    assert command[command.index("--decision-checkpoint") + 1] == "/models/decider.pt"


def test_openvla_decider_recipe_uses_full_libero90_episodes() -> None:
    path = PROJECT_ROOT / "configs" / "release" / "openvla_oft_decider_training.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))["recipe"]["args"]

    assert recipe["max-rollout-steps"] == 400
    assert recipe["rollback-target-max-age"] == 200
    assert recipe["rollback-gate-max-current-object-probability"] == 1.01
    assert recipe["object-hazard-penalty"] == -0.25


def test_decider_recipe_override_accepts_json_and_strings() -> None:
    assert _parse_override("object-hazard-penalty=-0.18") == ("object-hazard-penalty", -0.18)
    assert _parse_override("task-ids=[0,3]") == ("task-ids", [0, 3])
    assert _parse_override("qwen-dtype=bfloat16") == ("qwen-dtype", "bfloat16")
