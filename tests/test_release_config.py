from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.download_release_weights import verify_file
from scripts.run_release_24task_eval import (
    _build_command,
    _load_config,
    _parse_task,
    _select_entries,
)
from scripts.run_release_decider_training import _parse_override


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pi0_release_config_has_full_matrix_and_suite_horizons() -> None:
    config = _load_config(PROJECT_ROOT / "configs" / "release" / "v56_24task_eval.json")
    profile = config["profiles"]["safeloop_all"]
    assert sum(len(entry["task_ids"]) for entry in profile["entries"]) == 24
    assert len(profile["seeds"]) == 16
    assert config["seed_selects_init_state"] is True
    assert all(entry["mode"] == "rl" for entry in profile["entries"])
    assert config["suite_args"] == {
        "libero_spatial": {"max-rollout-steps": 220},
        "libero_object": {"max-rollout-steps": 280},
        "libero_goal": {"max-rollout-steps": 300},
        "libero_10": {"max-rollout-steps": 520},
        "libero_90": {"max-rollout-steps": 400},
    }


def test_pi0_release_can_select_one_quickstart_task() -> None:
    config = _load_config(PROJECT_ROOT / "configs" / "release" / "v56_24task_eval.json")
    entries = _select_entries(
        config["profiles"]["safeloop_all"]["entries"],
        [_parse_task("libero_10:9")],
    )
    assert len(entries) == 1
    assert entries[0]["task_ids"] == [9]
    command = _build_command(
        config,
        entries[0],
        seed=214,
        context={
            "model_dir": "/models/qwen",
            "weights_dir": "/models/safeloop",
            "policy_host": "127.0.0.1",
            "policy_port": "8000",
            "decision_checkpoint": "",
        },
        output_root=Path("/tmp/eval"),
    )
    assert command[command.index("--max-rollout-steps") + 1] == "520"
    assert command[command.index("--task-ids") + 1] == "9"
    assert command[command.index("--init-state-start") + 1] == "214"


def test_task_selector_rejects_tasks_outside_profile() -> None:
    config = _load_config(PROJECT_ROOT / "configs" / "release" / "v56_24task_eval.json")
    with pytest.raises(ValueError, match="not present"):
        _select_entries(
            config["profiles"]["safeloop_all"]["entries"],
            [_parse_task("libero_10:99")],
        )


def test_pi0_artifact_manifest_matches_release_files(tmp_path: Path) -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "configs" / "release" / "artifacts_pi0_v1.json").read_text(
            encoding="utf-8"
        )
    )
    files = manifest["weights"]["files"]
    assert manifest["weights"]["revision"] == "pi0-v1"
    assert len(files) == 8
    assert {entry["path"] for entry in files} == {
        "decision_heads/g6_safeanchor_update004.pt",
        "decision_heads/init_g4_success_bodygate_update000.pt",
        "prediction_heads/init_v13_ultra_lowdrift_success_step0100_multitask_head.pt",
        "prediction_heads/init_v8_balanced_step1500_multitask_head.pt",
        "prediction_heads/v23_success_balanced_step1000_multitask_head.pt",
        "prediction_heads/v30_lowdrift_success_step300_multitask_head.pt",
        "prediction_heads/v30_lowdrift_success_step600_multitask_head.pt",
        "prediction_heads/v9_object_recall_step3500_multitask_head.pt",
    }

    entry = files[0]
    target = tmp_path / entry["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not-a-checkpoint")
    assert "size mismatch" in verify_file(tmp_path, entry)


def test_decider_recipe_override_accepts_json_and_strings() -> None:
    assert _parse_override("object-hazard-penalty=-0.18") == ("object-hazard-penalty", -0.18)
    assert _parse_override("task-ids=[0,3]") == ("task-ids", [0, 3])
    assert _parse_override("qwen-dtype=bfloat16") == ("qwen-dtype", "bfloat16")
