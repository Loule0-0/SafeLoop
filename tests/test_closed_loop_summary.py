import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.evaluate_pi0_safeguard_closed_loop import summarize, write_qwen_rollout_samples
from safety_guard.online_rl import OnlineStepSignals


class ClosedLoopSummaryTests(unittest.TestCase):
    def test_summary_counts_rollback_rendered_frames_as_effective_control_steps(self):
        reports = [
            {
                "success": True,
                "steps": 10,
                "hazard_events": {"body": 0, "object": 0, "stuck": 1, "any": 1},
                "hazard_steps": {"body": 0, "object": 0, "stuck": 3, "any": 3},
                "interventions": {"noop": 8, "record": 1, "rollback": 1},
                "rollback_planned": 1,
                "rollback_failed": 0,
                "rollback_rendered_frames": 12,
                "nominal_after_rollback": 4,
            },
            {
                "success": False,
                "steps": 20,
                "hazard_events": {"body": 1, "object": 0, "any": 1},
                "hazard_steps": {"body": 2, "object": 0, "any": 2},
                "interventions": {"noop": 17, "record": 2, "rollback": 1},
                "rollback_planned": 1,
                "rollback_failed": 1,
                "rollback_rendered_frames": 8,
                "nominal_after_rollback": 3,
            },
        ]

        summary = summarize(reports)

        self.assertEqual(summary["rollback_rendered_frames"], 20)
        self.assertEqual(summary["effective_control_steps"], 50)
        self.assertEqual(summary["mean_effective_control_steps"], 25.0)
        self.assertEqual(summary["hazard_events"]["stuck"], 1)
        self.assertEqual(summary["hazard_steps"]["stuck"], 3)

    def test_qwen_rollout_export_respects_configured_tau(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                qwen_rollout_jsonl_out=root / "samples.jsonl",
                qwen_rollout_image_root=root / "images",
                qwen_tau=100,
                benchmark="libero_10",
                task_id=9,
                seed=7,
                qwen_rollout_run_tag=None,
            )
            task = SimpleNamespace(name="task9", language="dummy instruction")
            timeline = [OnlineStepSignals() for _ in range(101)]
            timeline[80] = OnlineStepSignals(body_hazard=True)
            samples = [
                {
                    "sample_index": 0,
                    "text": "H=3, tau=100",
                    "images": [Image.new("RGB", (4, 4), color="white")],
                    "current_body": 0.0,
                    "current_object": 0.0,
                }
            ]

            write_qwen_rollout_samples(args, task, 0, samples, timeline)

            exported = args.qwen_rollout_jsonl_out.read_text(encoding="utf-8")
            self.assertIn('"future_body": 1.0', exported)
            self.assertIn('"future_body_tth": 0.8', exported)
            self.assertIn('"content": "1 80\\n0 -1"', exported)

    def test_qwen_rollout_export_treats_stuck_as_future_body_hazard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                qwen_rollout_jsonl_out=root / "samples.jsonl",
                qwen_rollout_image_root=root / "images",
                qwen_tau=50,
                benchmark="libero_10",
                task_id=6,
                seed=9,
                qwen_rollout_run_tag=None,
            )
            task = SimpleNamespace(name="task6", language="dummy instruction")
            timeline = [OnlineStepSignals() for _ in range(30)]
            timeline[12] = OnlineStepSignals(stuck_hazard=True)
            samples = [
                {
                    "sample_index": 2,
                    "text": "H=3, tau=50",
                    "images": [Image.new("RGB", (4, 4), color="white")],
                    "current_body": 0.0,
                    "current_object": 0.0,
                }
            ]

            write_qwen_rollout_samples(args, task, 0, samples, timeline)

            exported = args.qwen_rollout_jsonl_out.read_text(encoding="utf-8")
            self.assertIn('"future_body": 1.0', exported)
            self.assertIn('"future_body_tth": 0.2', exported)
            self.assertIn('"content": "1 10\\n0 -1"', exported)

    def test_qwen_rollout_export_keeps_same_step_sources_in_separate_images(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                qwen_rollout_jsonl_out=root / "samples.jsonl",
                qwen_rollout_image_root=root / "images",
                qwen_tau=50,
                benchmark="libero_10",
                task_id=8,
                seed=11,
                qwen_rollout_run_tag="unit_tag",
            )
            task = SimpleNamespace(name="task8", language="dummy instruction")
            timeline = [OnlineStepSignals() for _ in range(20)]
            samples = [
                {
                    "sample_index": 10,
                    "text": "H=3, tau=50",
                    "images": [Image.new("RGB", (4, 4), color="white")],
                    "current_body": 0.0,
                    "current_object": 1.0,
                    "metadata": {"source": "rollback_pre"},
                },
                {
                    "sample_index": 10,
                    "text": "H=3, tau=50",
                    "images": [Image.new("RGB", (4, 4), color="black")],
                    "current_body": 0.0,
                    "current_object": 0.0,
                    "metadata": {"source": "rollback_post"},
                },
            ]

            write_qwen_rollout_samples(args, task, 0, samples, timeline)

            rows = [json.loads(line) for line in args.qwen_rollout_jsonl_out.read_text(encoding="utf-8").splitlines()]
            self.assertNotEqual(rows[0]["images"][0], rows[1]["images"][0])
            self.assertIn("rollback_pre", rows[0]["images"][0])
            self.assertIn("unit_tag", rows[0]["images"][0])
            self.assertIn("rollback_post", rows[1]["images"][0])
            self.assertTrue((args.qwen_rollout_image_root / rows[0]["images"][0]).exists())
            self.assertTrue((args.qwen_rollout_image_root / rows[1]["images"][0]).exists())


if __name__ == "__main__":
    unittest.main()
