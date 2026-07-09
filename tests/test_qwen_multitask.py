import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


class QwenMultitaskDataTests(unittest.TestCase):
    def test_qwenize_image_placeholders_replaces_each_plain_image_marker(self):
        from safety_guard.qwen_multitask import qwenize_image_placeholders

        text = "first <image> second <image>"

        converted = qwenize_image_placeholders(text)

        self.assertNotIn("<image>", converted)
        self.assertEqual(converted.count("<|image_pad|>"), 2)
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", converted)

    def test_parse_future_labels_uses_paper_tth_target_for_negatives(self):
        from safety_guard.qwen_multitask import parse_future_labels

        sample = {
            "messages": [
                {"role": "user", "content": "H=3, tau=50, current_global_step=30"},
                {"role": "assistant", "content": "0 328\n1 25"},
            ],
            "images": ["full_trajectory_pi0/libero_10_00_run002/images/00000_camera.jpg"],
        }

        labels = parse_future_labels(sample)

        self.assertEqual(labels["future_body"], 0.0)
        self.assertEqual(labels["future_object"], 1.0)
        self.assertEqual(labels["future_body_tth"], 1.0)
        self.assertEqual(labels["future_object_tth"], 0.5)

    def test_current_labels_from_marked_steps_uses_tolerance(self):
        from safety_guard.qwen_multitask import current_labels_from_marked_steps

        marked_steps = [(95, 0), (111, 1), (140, 0)]

        labels = current_labels_from_marked_steps(100, marked_steps, tolerance=5)
        far_labels = current_labels_from_marked_steps(100, marked_steps, tolerance=4)

        self.assertEqual(labels, (1.0, 0.0))
        self.assertEqual(far_labels, (0.0, 0.0))

    def test_load_marked_steps_reads_trajectory_metadata_by_run(self):
        from safety_guard.qwen_multitask import load_marked_steps

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "full_trajectory_pi0" / "libero_10_00_run000"
            run_dir.mkdir(parents=True)
            (run_dir / "trajectory_metadata.json").write_text(
                json.dumps({"marked_steps": [[90, 0], [110, 1]]}),
                encoding="utf-8",
            )

            marked = load_marked_steps(root)

        self.assertEqual(
            marked["full_trajectory_pi0/libero_10_00_run000"],
            [(90, 0), (110, 1)],
        )

    def test_build_group_split_keeps_runs_disjoint(self):
        from safety_guard.qwen_multitask import build_group_split

        samples = [
            {"run_id": "run_a"},
            {"run_id": "run_a"},
            {"run_id": "run_b"},
            {"run_id": "run_c"},
        ]

        train_indices, val_indices = build_group_split(samples, val_ratio=0.34, seed=7)
        train_runs = {samples[index]["run_id"] for index in train_indices}
        val_runs = {samples[index]["run_id"] for index in val_indices}

        self.assertTrue(train_indices)
        self.assertTrue(val_indices)
        self.assertTrue(train_runs.isdisjoint(val_runs))

    def test_build_group_split_accepts_safety_sample_dataclasses(self):
        from safety_guard.qwen_multitask import SafetySample, build_group_split

        samples = [
            SafetySample(
                text="x",
                image_paths=(),
                run_id="run_a",
                current_step=10,
                future_body=0.0,
                future_body_tth=1.0,
                future_object=0.0,
                future_object_tth=1.0,
                current_body=0.0,
                current_object=0.0,
            ),
            SafetySample(
                text="x",
                image_paths=(),
                run_id="run_b",
                current_step=20,
                future_body=1.0,
                future_body_tth=0.2,
                future_object=0.0,
                future_object_tth=1.0,
                current_body=1.0,
                current_object=0.0,
            ),
        ]

        train_indices, val_indices = build_group_split(samples, val_ratio=0.5, seed=1)

        self.assertEqual(len(train_indices), 1)
        self.assertEqual(len(val_indices), 1)

    def test_current_positive_sample_weights_boosts_only_current_hazards(self):
        from safety_guard.qwen_multitask import SafetySample, current_positive_sample_weights

        samples = [
            SafetySample(
                text="x",
                image_paths=(),
                run_id="run_a",
                current_step=10,
                future_body=1.0,
                future_body_tth=0.2,
                future_object=0.0,
                future_object_tth=1.0,
                current_body=0.0,
                current_object=0.0,
            ),
            SafetySample(
                text="x",
                image_paths=(),
                run_id="run_b",
                current_step=20,
                future_body=0.0,
                future_body_tth=1.0,
                future_object=0.0,
                future_object_tth=1.0,
                current_body=1.0,
                current_object=0.0,
            ),
            SafetySample(
                text="x",
                image_paths=(),
                run_id="run_c",
                current_step=30,
                future_body=0.0,
                future_body_tth=1.0,
                future_object=0.0,
                future_object_tth=1.0,
                current_body=0.0,
                current_object=1.0,
            ),
        ]

        self.assertEqual(current_positive_sample_weights(samples, positive_weight=6.0), [1.0, 6.0, 6.0])
        self.assertEqual(
            current_positive_sample_weights(
                samples,
                positive_weight=6.0,
                body_positive_weight=4.0,
                object_positive_weight=12.0,
            ),
            [1.0, 4.0, 12.0],
        )


class QwenMultitaskModelTests(unittest.TestCase):
    def test_multitask_head_returns_risk_and_current_logits(self):
        import torch

        from safety_guard.qwen_multitask import MultitaskSafetyHead

        head = MultitaskSafetyHead(hidden_size=8, neck_size=4, dropout=0.0)
        outputs = head(torch.ones(3, 8))

        self.assertEqual(set(outputs), {"risk_logits", "risk_tth", "current_logits"})
        self.assertEqual(tuple(outputs["risk_logits"].shape), (3, 2))
        self.assertEqual(tuple(outputs["risk_tth"].shape), (3, 2))
        self.assertEqual(tuple(outputs["current_logits"].shape), (3, 2))
        self.assertTrue(torch.all(outputs["risk_tth"] >= 0.0))
        self.assertTrue(torch.all(outputs["risk_tth"] <= 1.0))

    def test_safety_multitask_loss_combines_future_and_current_terms(self):
        import torch

        from safety_guard.qwen_multitask import SafetyMultitaskLoss

        predictions = {
            "risk_logits": torch.tensor([[0.0, 1.0], [1.0, -1.0]], dtype=torch.float32),
            "risk_tth": torch.tensor([[1.0, 0.2], [0.1, 1.0]], dtype=torch.float32),
            "current_logits": torch.tensor([[0.0, -2.0], [2.0, 0.0]], dtype=torch.float32),
        }
        labels = {
            "future_labels": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            "future_tth": torch.tensor([[1.0, 0.25], [0.2, 1.0]], dtype=torch.float32),
            "current_labels": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        }

        loss, parts = SafetyMultitaskLoss()(predictions, labels)

        self.assertGreater(float(loss), 0.0)
        self.assertIn("future_bce", parts)
        self.assertIn("future_tth", parts)
        self.assertIn("current_bce", parts)

    def test_compute_multitask_metrics_reports_current_recall(self):
        import torch

        from safety_guard.qwen_multitask import compute_multitask_metrics

        predictions = {
            "risk_logits": torch.tensor([[-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]]),
            "risk_tth": torch.tensor([[1.0, 0.2], [0.3, 1.0], [1.0, 1.0]]),
            "current_logits": torch.tensor([[-2.0, -2.0], [3.0, -2.0], [-2.0, 3.0]]),
        }
        labels = {
            "future_labels": torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
            "future_tth": torch.tensor([[1.0, 0.25], [0.2, 1.0], [1.0, 1.0]]),
            "current_labels": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        }

        metrics = compute_multitask_metrics(predictions, labels, threshold=0.5)

        self.assertAlmostEqual(metrics["current_body_recall"], 1.0)
        self.assertAlmostEqual(metrics["current_object_recall"], 1.0)
        self.assertIn("future_body_acc", metrics)
        self.assertAlmostEqual(metrics["current_body_ap"], 1.0)
        self.assertGreater(metrics["current_body_prob_pos_mean"], metrics["current_body_prob_neg_mean"])

    def test_compute_multitask_metrics_at_thresholds_reports_lower_threshold_recall(self):
        import torch

        from safety_guard.qwen_multitask import compute_multitask_metrics_at_thresholds

        predictions = {
            "risk_logits": torch.tensor([[-2.0, -2.0], [-2.0, -2.0]]),
            "risk_tth": torch.ones(2, 2),
            "current_logits": torch.tensor([[-2.0, -2.0], [-0.4, -2.0]]),
        }
        labels = {
            "future_labels": torch.zeros(2, 2),
            "future_tth": torch.ones(2, 2),
            "current_labels": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        }

        metrics = compute_multitask_metrics_at_thresholds(predictions, labels, thresholds=(0.5, 0.3))

        self.assertEqual(metrics["current_body_recall"], 0.0)
        self.assertEqual(metrics["current_body_recall_t030"], 1.0)

    def test_multitask_values_to_output_exposes_risk_and_current_state(self):
        import math

        from safety_guard.qwen_multitask import multitask_values_to_output

        output = multitask_values_to_output(
            future_body_logit=0.0,
            future_body_tth=0.2,
            future_object_logit=math.log(3.0),
            future_object_tth=0.8,
            current_body_logit=math.log(9.0),
            current_object_logit=-10.0,
        )

        self.assertAlmostEqual(output.risk.body_probability, 0.5)
        self.assertAlmostEqual(output.risk.object_probability, 0.75)
        self.assertEqual(output.risk.body_tth, 0.2)
        self.assertTrue(output.is_current_hazard(threshold=0.5))
        self.assertAlmostEqual(output.current_body_probability, 0.9)
        self.assertLess(output.current_object_probability, 0.01)

    def test_sibling_lora_adapter_path_detects_saved_adapter(self):
        from safety_guard.qwen_multitask import _sibling_lora_adapter_path

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best" / "multitask_head.pt"
            adapter = checkpoint.parent / "lora_adapter"
            adapter.mkdir(parents=True)
            checkpoint.write_bytes(b"placeholder")

            self.assertEqual(_sibling_lora_adapter_path(checkpoint), adapter)

    def test_sibling_lora_adapter_path_returns_none_for_frozen_head(self):
        from safety_guard.qwen_multitask import _sibling_lora_adapter_path

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best" / "multitask_head.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"placeholder")

            self.assertIsNone(_sibling_lora_adapter_path(checkpoint))


class QwenMultitaskCliTests(unittest.TestCase):
    def test_training_cli_parser_has_expected_defaults(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_qwen_multitask_safety.py"
        spec = importlib.util.spec_from_file_location("train_qwen_multitask_safety", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.parse_args(
            [
                "--model-dir",
                "/models/qwen",
                "--data",
                "/data/train.jsonl",
                "--data-root",
                "/data",
                "--out",
                "/out",
            ]
        )

        self.assertEqual(args.train_mode, "frozen")
        self.assertEqual(args.current_tolerance, 5)
        self.assertEqual(args.val_ratio, 0.1)
        self.assertEqual(args.neck_size, 512)
        self.assertEqual(args.current_positive_sample_weight, 1.0)
        self.assertIsNone(args.current_body_positive_sample_weight)
        self.assertIsNone(args.current_object_positive_sample_weight)
        self.assertFalse(args.gradient_checkpointing)

    def test_training_cli_resolves_hidden_size_from_text_config(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_qwen_multitask_safety.py"
        spec = importlib.util.spec_from_file_location("train_qwen_multitask_safety_hidden", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class TextConfig:
            hidden_size = 2048

        class Config:
            hidden_size = None
            text_config = TextConfig()

        self.assertEqual(module._model_hidden_size(Config()), 2048)

    def test_evaluation_cli_parser_accepts_checkpoint_and_thresholds(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_qwen_multitask_safety.py"
        spec = importlib.util.spec_from_file_location("evaluate_qwen_multitask_safety", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.parse_args(
            [
                "--model-dir",
                "/models/qwen",
                "--data",
                "/data/train.jsonl",
                "--data-root",
                "/data",
                "--checkpoint",
                "/out/best/multitask_head.pt",
                "--thresholds",
                "0.5",
                "0.25",
            ]
        )

        self.assertEqual(args.checkpoint, Path("/out/best/multitask_head.pt"))
        self.assertEqual(args.thresholds, [0.5, 0.25])


if __name__ == "__main__":
    unittest.main()
