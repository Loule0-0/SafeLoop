import unittest

from safety_guard.qwen_risk import (
    HazardLabels,
    build_qwen_messages,
    parse_hazard_response,
    normalize_tth,
)


class QwenRiskTests(unittest.TestCase):
    def test_parse_hazard_response_accepts_two_line_format(self):
        labels = parse_hazard_response("1 12\n0 -1")

        self.assertEqual(labels.collision_probability, 1.0)
        self.assertEqual(labels.collision_tth, 12.0)
        self.assertEqual(labels.object_probability, 0.0)
        self.assertEqual(labels.object_tth, -1.0)

    def test_normalize_tth_masks_missing_or_negative_tth(self):
        self.assertEqual(normalize_tth(-1, tau=50), (-1.0, 0.0))
        self.assertEqual(normalize_tth(75, tau=50), (1.0, 1.0))
        self.assertEqual(normalize_tth(25, tau=50), (0.5, 1.0))

    def test_build_qwen_messages_preserves_pi0_state_and_action_context(self):
        labels = HazardLabels(1, 10, 0, -1)
        history = [
            {
                "robot_state": {
                    "robot0_joint_pos": [1, 2],
                    "robot0_joint_vel": [0.1, 0.2],
                    "robot0_joint_torques": [0.0, 0.0],
                }
            }
        ]

        sample = build_qwen_messages(history, [0.3, 0.4], labels, tau=50, image_paths=["a.jpg", "b.jpg"])

        self.assertEqual(sample["messages"][0]["role"], "user")
        self.assertIn("Next action a_t", sample["messages"][0]["content"])
        self.assertIn("robot0_joint_pos", sample["messages"][0]["content"])
        self.assertEqual(sample["messages"][1]["content"], "1 10\n0 -1")
        self.assertEqual(sample["images"], ["a.jpg", "b.jpg"])


if __name__ == "__main__":
    unittest.main()
