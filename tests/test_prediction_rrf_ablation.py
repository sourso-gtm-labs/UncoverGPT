from __future__ import annotations

import unittest

import numpy as np

from scripts.run_prediction_experiment import COMPACT_FEATURES, RRF_K0
from scripts.run_prediction_rrf_ablation import (
    G_A_RRF_OVERLAPPING_FEATURES,
    OVERALL_RRF_FEATURES,
    descending_rrf_contribution,
)


class PredictionRrfAblationTest(unittest.TestCase):
    def test_feature_sets_exclude_base_and_direct_components(self) -> None:
        for granularity, features in OVERALL_RRF_FEATURES.items():
            self.assertNotIn("rrf_score", features)
            self.assertEqual(
                set(features),
                set(COMPACT_FEATURES[granularity])
                - set(G_A_RRF_OVERLAPPING_FEATURES[granularity]),
            )

    def test_descending_rank_uses_average_ties(self) -> None:
        scores, active = descending_rrf_contribution([3.0, 2.0, 2.0, 1.0])
        self.assertTrue(active)
        expected = np.asarray(
            [
                1.0 / (RRF_K0 + 1.0),
                1.0 / (RRF_K0 + 2.5),
                1.0 / (RRF_K0 + 2.5),
                1.0 / (RRF_K0 + 4.0),
            ]
        )
        np.testing.assert_allclose(scores, expected)

    def test_constant_feature_has_no_rrf_vote(self) -> None:
        scores, active = descending_rrf_contribution([1.0, 1.0, 1.0])
        self.assertFalse(active)
        np.testing.assert_array_equal(scores, np.zeros(3))

    def test_missing_candidate_receives_no_vote(self) -> None:
        scores, active = descending_rrf_contribution([2.0, float("nan"), 1.0])
        self.assertTrue(active)
        self.assertGreater(scores[0], scores[2])
        self.assertEqual(scores[1], 0.0)


if __name__ == "__main__":
    unittest.main()
