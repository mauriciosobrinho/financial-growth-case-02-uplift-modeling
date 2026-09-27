import unittest

import numpy as np
import pandas as pd

from src.uplift_utils import (
    difference_in_means,
    evaluate_policy,
    exact_top_fraction,
    qini_curve,
    qini_metrics,
    uplift_at_fraction,
)


class UpliftUtilityTests(unittest.TestCase):
    def test_difference_in_means_recovers_constructed_effect(self):
        treatment = np.array([0, 0, 0, 1, 1, 1])
        outcome = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = difference_in_means(outcome, treatment)
        self.assertEqual(result.estimate, 1.0)
        self.assertEqual(result.n_treated, 3)
        self.assertEqual(result.n_control, 3)

    def test_exact_top_fraction_has_requested_size(self):
        score = np.arange(10)
        selected = exact_top_fraction(score, 0.2)
        self.assertEqual(selected.sum(), 2)
        self.assertTrue(selected[-2:].all())

    def test_qini_metrics_reward_correct_ranking(self):
        rng = np.random.default_rng(7)
        n = 20_000
        feature = rng.normal(size=n)
        treatment = rng.binomial(1, 0.5, size=n)
        baseline = 0.15
        uplift = np.where(feature > 0, 0.25, -0.05)
        outcome = rng.binomial(1, np.clip(baseline + treatment * uplift, 0.001, 0.999))
        good = qini_metrics(qini_curve(outcome, treatment, feature))
        bad = qini_metrics(qini_curve(outcome, treatment, -feature))
        self.assertGreater(good["qini_coefficient"], bad["qini_coefficient"])
        self.assertGreater(
            uplift_at_fraction(outcome, treatment, feature, 0.2),
            uplift_at_fraction(outcome, treatment, -feature, 0.2),
        )

    def test_policy_economics_use_incremental_margin_not_gross_margin(self):
        frame = pd.DataFrame(
            {
                "treatment_assigned": [0, 0, 1, 1],
                "conversion_30d": [0, 0, 1, 1],
                "revenue_30d": [0.0, 0.0, 10.0, 10.0],
                "margin_30d": [0.0, 0.0, 4.0, 4.0],
                "campaign_cost": [0.0, 0.0, 1.0, 1.0],
            }
        )
        result = evaluate_policy(frame, np.ones(len(frame), dtype=bool), "all")
        self.assertEqual(result["incremental_margin"], 16.0)
        self.assertEqual(result["campaign_cost"], 4.0)
        self.assertEqual(result["net_incremental_value"], 12.0)


if __name__ == "__main__":
    unittest.main()
