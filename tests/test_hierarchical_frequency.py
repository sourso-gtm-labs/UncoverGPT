from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_hierarchical_frequency import (
    FREQUENCY_BINS,
    Observation,
    RunGroup,
    analyze_observations,
    build_leave_one_run_out_rows,
    build_per_appearance_rows,
    load_web_frequency_rows,
)


class HierarchicalFrequencyTest(unittest.TestCase):
    def test_domain_frequency_deduplicates_multiple_urls_within_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = root / "v9_analysis" / "search_detail" / "q001"
            question.mkdir(parents=True)
            runs = [
                {
                    "all_urls": [
                        {"url": "https://example.com/a", "cited": False},
                        {"url": "https://example.com/b", "cited": True},
                    ]
                },
                {"all_urls": [{"url": "https://example.com/c", "cited": False}]},
            ]
            for index, payload in enumerate(runs):
                (question / f"run_{index}_detail.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            rows = load_web_frequency_rows(root)

            domain = next(row for row in rows["domain"] if row.item == "example.com")
            self.assertEqual(domain.value, 1.0)
            self.assertEqual(domain.label, 1)
            self.assertEqual(len(rows["url"]), 3)

    def test_analysis_reports_increasing_frequency_signal(self) -> None:
        rows = [
            Observation("q1", "a", 0.05, 0),
            Observation("q1", "b", 0.10, 0),
            Observation("q2", "c", 0.50, 1),
            Observation("q2", "d", 1.00, 1),
        ]

        result = analyze_observations(rows, FREQUENCY_BINS)

        self.assertGreater(result["roc_auc"], 0.5)
        self.assertGreater(result["logistic"]["beta"], 0)
        self.assertGreater(result["ordered_bin_trend"]["z"], 0)

    def test_leave_one_run_out_frequency_excludes_the_heldout_run(self) -> None:
        run_group = RunGroup(
            group="v9_analysis/q001",
            appeared=(
                frozenset({"a", "one-shot"}),
                frozenset({"a"}),
                frozenset({"a"}),
            ),
            cited=(frozenset({"a"}), frozenset(), frozenset({"a"})),
        )

        rows = build_leave_one_run_out_rows(
            {"url": [run_group], "domain": []}
        )["url"]

        one_shot = next(row for row in rows if row.item.startswith("one-shot@@"))
        recurrent = [row for row in rows if row.item.startswith("a@@")]
        self.assertEqual(one_shot.value, 0.0)
        self.assertEqual(len(recurrent), 3)
        self.assertTrue(all(row.value == 1.0 for row in recurrent))
        self.assertEqual([row.label for row in recurrent], [1, 0, 1])

    def test_per_appearance_propensity_does_not_use_ever_cited_label(self) -> None:
        run_group = RunGroup(
            group="v9_analysis/q001",
            appeared=(
                frozenset({"a"}),
                frozenset({"a"}),
                frozenset({"a"}),
                frozenset({"a"}),
            ),
            cited=(frozenset({"a"}), frozenset(), frozenset(), frozenset()),
        )

        row = build_per_appearance_rows(
            {"url": [run_group], "domain": []}
        )["url"][0]

        self.assertEqual(row.value, 1.0)
        self.assertEqual(row.successes, 1)
        self.assertEqual(row.trials, 4)


if __name__ == "__main__":
    unittest.main()
