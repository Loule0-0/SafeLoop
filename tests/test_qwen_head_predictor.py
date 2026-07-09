import math
import unittest

from safety_guard.qwen_head_predictor import head_values_to_risk


class QwenHeadPredictorTests(unittest.TestCase):
    def test_head_values_to_risk_sigmoids_probabilities_and_clips_times(self):
        risk = head_values_to_risk(
            p0_logit=0.0,
            t0_norm=1.25,
            p1_logit=math.log(3.0),
            t1_norm=-0.5,
        )

        self.assertAlmostEqual(risk.body_probability, 0.5)
        self.assertAlmostEqual(risk.object_probability, 0.75)
        self.assertEqual(risk.body_tth, 1.0)
        self.assertEqual(risk.object_tth, 0.0)


if __name__ == "__main__":
    unittest.main()
