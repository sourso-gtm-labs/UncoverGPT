"""Regression tests for the Google/API candidate visibility union."""

import importlib.util
from collections import Counter
from pathlib import Path
import unittest


EVALUATOR_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_proxy_effectiveness.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("proxy_effectiveness", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CandidateUnionTest(unittest.TestCase):
    def test_top_k_visibility_uses_fixed_k_denominator(self) -> None:
        evaluator = load_evaluator()
        metrics = evaluator.top_k_visibility_metrics(
            {"a", "c"},
            ["a", "b", "c", "d"],
            3,
        )

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)

        short_ranking = evaluator.top_k_visibility_metrics({"a"}, ["a"], 5)
        self.assertIsNotNone(short_ranking)
        assert short_ranking is not None
        self.assertAlmostEqual(short_ranking["precision"], 0.2)
        self.assertAlmostEqual(short_ranking["recall"], 1.0)

    def test_union_visibility_has_no_implicit_ranking(self) -> None:
        evaluator = load_evaluator()
        proxies = evaluator._with_candidate_union(
            {
                "G": {"set": {"target-a", "google-only"}, "ranking": ["target-a", "google-only"]},
                "A_candidate": {
                    "set": {"target-b", "api-only"},
                    "ranking": ["target-b", "api-only"],
                },
                "A_citation": {"set": {"target-b"}, "ranking": ["target-b"]},
            }
        )

        scored = evaluator._score_granularity(
            Counter({"target-a": 2, "target-b": 1}),
            proxies,
        )["proxies"][evaluator.UNION_PROXY]

        self.assertEqual(
            proxies[evaluator.UNION_PROXY]["set"],
            {"target-a", "target-b", "google-only", "api-only"},
        )
        self.assertAlmostEqual(scored["visibility"]["precision"], 0.5)
        self.assertAlmostEqual(scored["visibility"]["recall"], 1.0)
        self.assertAlmostEqual(scored["visibility"]["f1"], 2 / 3)
        self.assertTrue(all(value is None for value in scored["ranking"].values()))


if __name__ == "__main__":
    unittest.main()
