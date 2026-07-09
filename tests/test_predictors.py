import unittest

from safety_guard.predictors import ActionNormRiskPredictor, ConstantRiskPredictor


class PredictorTests(unittest.TestCase):
    def test_constant_predictor_returns_configured_risk(self):
        predictor = ConstantRiskPredictor([0.1, 0.9, 0.2, 0.8])

        risk = predictor.predict(observation={}, proposed_action=[100.0])

        self.assertAlmostEqual(risk.body_probability, 0.1)
        self.assertAlmostEqual(risk.body_tth, 0.9)
        self.assertAlmostEqual(risk.object_probability, 0.2)
        self.assertAlmostEqual(risk.object_tth, 0.8)

    def test_action_norm_predictor_increases_risk_for_large_actions(self):
        predictor = ActionNormRiskPredictor(safe_norm=0.5, critical_norm=2.0)

        low = predictor.predict(observation={}, proposed_action=[0.1, 0.1])
        high = predictor.predict(observation={}, proposed_action=[2.0, 2.0])

        self.assertLess(low.body_probability, high.body_probability)
        self.assertGreater(low.body_tth, high.body_tth)
        self.assertGreaterEqual(high.body_probability, 0.8)


if __name__ == "__main__":
    unittest.main()
