from __future__ import annotations

import unittest
from collections import Counter

from scripts.run_prediction_experiment import (
    COMPACT_FEATURES,
    CandidateRow,
    QuestionBundle,
    _feature_matrix,
    classification_metrics,
    make_fold_map,
    query_prediction_metrics,
    rrf_score,
    top_k_metrics,
)


class PredictionExperimentTest(unittest.TestCase):
    def test_classification_metrics_report_auc_and_average_precision(self) -> None:
        rows = [
            CandidateRow("q", "a", 1, 3.0, {}, 0.0, 0.0),
            CandidateRow("q", "b", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "c", 1, 1.0, {}, 0.0, 0.0),
        ]

        metrics = classification_metrics(rows, [0.9, 0.2, 0.8])

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)

    def test_query_metrics_report_precision_recall_and_f1_at_k(self) -> None:
        rows = [
            CandidateRow("q", "a", 1, 3.0, {}, 0.0, 0.0),
            CandidateRow("q", "b", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "c", 1, 1.0, {}, 0.0, 0.0),
            CandidateRow("q", "d", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "e", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "f", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "g", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "h", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "i", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "j", 0, 0.0, {}, 0.0, 0.0),
        ]

        metrics = query_prediction_metrics(
            rows, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        )

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics["precision@1"], 1.0)
        self.assertAlmostEqual(metrics["recall@1"], 0.5)
        self.assertAlmostEqual(metrics["f1@1"], 2 / 3)
        self.assertAlmostEqual(metrics["precision@3"], 2 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["f1@3"], 0.8)
        self.assertAlmostEqual(metrics["precision@10"], 0.2)
        self.assertAlmostEqual(metrics["recall@10"], 1.0)

    def test_top_k_metrics_average_hits_at_a_score_tie(self) -> None:
        rows = [
            CandidateRow("q", "a", 1, 1.0, {}, 0.0, 0.0),
            CandidateRow("q", "b", 0, 0.0, {}, 0.0, 0.0),
            CandidateRow("q", "c", 1, 1.0, {}, 0.0, 0.0),
            CandidateRow("q", "d", 0, 0.0, {}, 0.0, 0.0),
        ]

        metrics = top_k_metrics(rows, [0.5, 0.5, 0.5, 0.5], 1)

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.25)
        self.assertAlmostEqual(metrics["f1"], 1 / 3)

    def test_feature_matrix_uses_only_the_compact_feature_set(self) -> None:
        features = {name: float(index) for index, name in enumerate(COMPACT_FEATURES["url"])}
        features["path_depth"] = 99.0
        rows = [CandidateRow("q", "a", 1, 1.0, features, 0.0, 0.0)]

        matrix, labels, feature_names = _feature_matrix(rows, "url")

        self.assertEqual(feature_names, list(COMPACT_FEATURES["url"]))
        self.assertEqual(list(matrix.columns), list(COMPACT_FEATURES["url"]))
        self.assertNotIn("path_depth", matrix.columns)
        self.assertEqual(labels.tolist(), [1])

    def test_fold_map_keeps_queries_grouped_and_balances_strata(self) -> None:
        questions = []
        for intent in ("analysis", "factual_lookup", "info_synthesis", "recommendation"):
            for language in ("english", "chinese"):
                for index in range(5):
                    questions.append(
                        QuestionBundle(
                            dataset="d",
                            subdataset=intent,
                            qid=f"q{language[0]}{index}",
                            query="q",
                            language=language,
                            intent=intent,
                            n_api_runs=20,
                            web_url_relevance=Counter(),
                            web_domain_relevance=Counter(),
                            google_urls={},
                            api_urls={},
                            entity_cache={"counts": {}},
                        )
                    )

        fold_map = make_fold_map(questions)

        self.assertEqual(len(fold_map), 40)
        self.assertEqual({fold_map[question.group] for question in questions}, set(range(5)))
        for fold in range(5):
            self.assertEqual(sum(value == fold for value in fold_map.values()), 8)

    def test_rrf_requires_an_explicit_rank_rule(self) -> None:
        self.assertAlmostEqual(rrf_score(1, 2), 1 / 61 + 1 / 62)
        self.assertAlmostEqual(rrf_score(1, None), 1 / 61)
        self.assertEqual(rrf_score(None, None), 0.0)


if __name__ == "__main__":
    unittest.main()
