import unittest
from pathlib import Path

from safety_guard.qwen_multitask import parse_future_labels
from scripts.train_qwen_multitask_safety import _jsonable_args, _selection_score, parse_args


class QwenMultitaskTrainingTests(unittest.TestCase):
    def test_parser_accepts_initial_lora_adapter_for_continued_training(self):
        args = parse_args(
            [
                "--model-dir",
                "/models/qwen",
                "--data",
                "/data/train.jsonl",
                "--data-root",
                "/data/root",
                "--out",
                "/out",
                "--train-mode",
                "lora",
                "--init-lora-adapter",
                "/runs/ap_best/lora_adapter",
            ]
        )

        self.assertEqual(args.init_lora_adapter, Path("/runs/ap_best/lora_adapter"))
        self.assertEqual(_jsonable_args(args)["init_lora_adapter"], "/runs/ap_best/lora_adapter")

    def test_parser_accepts_reproduction_guide_aliases(self):
        args = parse_args(
            [
                "--model-path",
                "/models/qwen",
                "--data-jsonl",
                "/data/train.jsonl",
                "--data-root",
                "/data/root",
                "--output-dir",
                "/out",
                "--freeze-backbone",
                "--learning-rate",
                "2e-4",
                "--grad-accum-steps",
                "8",
                "--save-steps",
                "1000",
                "--future-window",
                "100",
                "--body-weight",
                "1.2",
                "--object-weight",
                "1.0",
                "--thresholds",
                "0.5",
                "0.9",
                "0.95",
            ]
        )

        self.assertEqual(args.model_dir, Path("/models/qwen"))
        self.assertEqual(args.data, Path("/data/train.jsonl"))
        self.assertEqual(args.out, Path("/out"))
        self.assertEqual(args.train_mode, "frozen")
        self.assertEqual(args.lr, 2e-4)
        self.assertEqual(args.grad_accum_steps, 8)
        self.assertEqual(args.save_steps, 1000)
        self.assertEqual(args.future_window, 100)
        self.assertEqual(args.body_weight, 1.2)
        self.assertEqual(args.thresholds, [0.5, 0.9, 0.95])

    def test_parse_future_labels_prefers_explicit_rollout_labels(self):
        sample = {
            "messages": [
                {"role": "user", "content": "H=3, tau=50, current_global_step=120"},
                {"role": "assistant", "content": "0 -1\n0 -1"},
            ],
            "labels": {
                "future_body": 1.0,
                "future_body_tth": 0.2,
                "future_object": 0.0,
                "future_object_tth": 1.0,
            },
        }

        labels = parse_future_labels(sample)

        self.assertEqual(labels["future_body"], 1.0)
        self.assertEqual(labels["future_body_tth"], 0.2)
        self.assertEqual(labels["future_object"], 0.0)

    def test_selection_score_prefers_balanced_current_object_head(self):
        recall_only = {
            "future_body_acc": 0.90,
            "future_object_acc": 0.90,
            "current_body_precision": 0.0,
            "current_body_recall": 0.0,
            "current_object_ap": 0.45,
            "current_object_precision": 0.20,
            "current_object_recall": 0.90,
            "current_object_precision_t090": 0.25,
        }
        balanced = {
            "future_body_acc": 0.88,
            "future_object_acc": 0.88,
            "current_body_precision": 0.0,
            "current_body_recall": 0.0,
            "current_object_ap": 0.58,
            "current_object_precision": 0.60,
            "current_object_recall": 0.60,
            "current_object_precision_t090": 0.70,
        }

        self.assertGreater(_selection_score(balanced), _selection_score(recall_only))


if __name__ == "__main__":
    unittest.main()
