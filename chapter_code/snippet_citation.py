"""Analyze retrieval-snippet length for cited and uncited Web candidates."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "gpt-5.3": ROOT / "dataset-5.3-1600",
    "gpt-5.5": ROOT / "dataset-5.5-1600",
}
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20_260_731


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def probability_superiority(positive: list[int], negative: list[int]) -> float | None:
    """Return P(positive > negative), with half credit for ties."""
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if cited > uncited else 0.5 if cited == uncited else 0.0
        for cited in positive
        for uncited in negative
    )
    return wins / (len(positive) * len(negative))


def bootstrap_mean(values: list[float]) -> dict[str, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    sample_count = len(values)
    samples = sorted(
        mean(values[rng.randrange(sample_count)] for _ in range(sample_count))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "mean": mean(values),
        "ci_low": samples[int(0.025 * BOOTSTRAP_SAMPLES)],
        "ci_high": samples[int(0.975 * BOOTSTRAP_SAMPLES) - 1],
    }


def load_runs(dataset: Path, search_candidates_only: bool = False) -> dict[str, list[list[dict[str, Any]]]]:
    runs_by_query: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    pattern = "v9_*/search_detail/q*/*_detail.json"
    for path in sorted(dataset.glob(pattern)):
        relative = path.relative_to(dataset)
        query_key = f"{relative.parts[0]}/{relative.parts[2]}"
        data = json.loads(path.read_text())
        rows = []
        for item in data.get("all_urls", []):
            snippet = item.get("snippet") or ""
            if not snippet:
                continue
            if search_candidates_only and not item.get("is_search_candidate", True):
                continue
            rows.append(
                {
                    "length": len(snippet),
                    "cited": bool(item.get("cited")),
                }
            )
        runs_by_query[query_key].append(rows)
    return runs_by_query


def summarize(dataset: Path, search_candidates_only: bool = False) -> dict[str, Any]:
    runs_by_query = load_runs(dataset, search_candidates_only)
    pooled = [row for runs in runs_by_query.values() for run in runs for row in run]
    cited = [row["length"] for row in pooled if row["cited"]]
    uncited = [row["length"] for row in pooled if not row["cited"]]

    query_auc: dict[str, float] = {}
    for query_key, runs in runs_by_query.items():
        run_auc = []
        for rows in runs:
            positive = [row["length"] for row in rows if row["cited"]]
            negative = [row["length"] for row in rows if not row["cited"]]
            auc = probability_superiority(positive, negative)
            if auc is not None:
                run_auc.append(auc)
        if run_auc:
            query_auc[query_key] = mean(run_auc)

    return {
        "n_queries": len(runs_by_query),
        "n_evaluable_queries": len(query_auc),
        "n_nonempty_instances": len(pooled),
        "cited_n": len(cited),
        "uncited_n": len(uncited),
        "cited_mean": mean(cited),
        "uncited_mean": mean(uncited),
        "cited_median": median(cited),
        "uncited_median": median(uncited),
        "cited_p90": percentile(cited, 0.9),
        "uncited_p90": percentile(uncited, 0.9),
        "cited_ge_500_rate": sum(value >= 500 for value in cited) / len(cited),
        "uncited_ge_500_rate": sum(value >= 500 for value in uncited) / len(uncited),
        "query_macro_within_run_auc": bootstrap_mean(list(query_auc.values())),
        "query_auc": query_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "chapter_code/generated/snippet_citation.json",
    )
    args = parser.parse_args()

    main_results = {model: summarize(path) for model, path in DATASETS.items()}
    sensitivity = {
        model: summarize(path, search_candidates_only=True)
        for model, path in DATASETS.items()
    }

    shared_queries = sorted(
        set(main_results["gpt-5.3"]["query_auc"])
        & set(main_results["gpt-5.5"]["query_auc"])
    )
    paired_delta = [
        main_results["gpt-5.5"]["query_auc"][query]
        - main_results["gpt-5.3"]["query_auc"][query]
        for query in shared_queries
    ]

    output = {
        "protocol": {
            "candidate_pool": "all_urls, including supporting webpages",
            "snippet_filter": "non-empty snippets",
            "citation_label": "project-wide cited=True",
            "auc": "within-run probability superiority, averaged per query",
            "confidence_interval": "2,000 query-bootstrap samples",
        },
        "main": main_results,
        "search_candidate_only_sensitivity": sensitivity,
        "paired_auc_delta_gpt_5_5_minus_gpt_5_3": {
            "n_queries": len(shared_queries),
            **bootstrap_mean(paired_delta),
        },
    }
    for section in (main_results, sensitivity):
        for result in section.values():
            result.pop("query_auc")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
