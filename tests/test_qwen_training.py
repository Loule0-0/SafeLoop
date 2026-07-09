import sys
import unittest
from pathlib import Path

from safety_guard.qwen_training import QwenTrainingConfig, build_qwen_training_command


class QwenTrainingTests(unittest.TestCase):
    def test_head_mode_uses_onlyhead_train_arguments(self):
        config = QwenTrainingConfig(
            mode="head",
            upstream_root=Path("/repo"),
            model_dir=Path("/models/qwen"),
            data=Path("/data/train.jsonl"),
            out=Path("/out/head"),
            epochs=2,
            batch_size=4,
            lr=1e-5,
            fp16=True,
            validation_ratio=0.2,
            test_data=Path("/data/test.jsonl"),
        )

        command = build_qwen_training_command(config, python_executable=sys.executable)

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "/repo/onlyhead_ft/train.py")
        self.assertIn("--val_ratio", command)
        self.assertIn("--test_data", command)
        self.assertIn("--fp16", command)
        self.assertNotIn("--validation_split", command)

    def test_lora_mode_uses_lora_train_arguments(self):
        config = QwenTrainingConfig(
            mode="lora",
            upstream_root=Path("/repo"),
            model_dir=Path("/models/qwen"),
            data=Path("/data/train.jsonl"),
            out=Path("/out/lora"),
            epochs=3,
            batch_size=1,
            lr=5e-5,
            validation_ratio=0.15,
            max_grad_norm=0.7,
        )

        command = build_qwen_training_command(config, python_executable="python")

        self.assertEqual(command[1], "/repo/lora_ft/train.py")
        self.assertIn("--validation_split", command)
        self.assertIn("--max_grad_norm", command)
        self.assertIn("0.7", command)
        self.assertNotIn("--val_ratio", command)

    def test_unknown_mode_is_rejected(self):
        config = QwenTrainingConfig(
            mode="bad",
            upstream_root=Path("/repo"),
            model_dir=Path("/models/qwen"),
            data=Path("/data/train.jsonl"),
            out=Path("/out"),
        )

        with self.assertRaises(ValueError):
            build_qwen_training_command(config)


if __name__ == "__main__":
    unittest.main()
