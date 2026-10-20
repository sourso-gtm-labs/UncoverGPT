"""Fuse compact Prediction features into one overall reciprocal-rank score.

G_A_RRF remains the base score, preserving its separate Google and API rank
contributions.  Every remaining compact feature is converted into a ranked
list within each query and contributes ``1 / (k0 + rank)``.  Features used to
construct G_A_RRF are excluded to avoid counting the same evidence twice.

The experiment also reports leave-one-feature-out ablations.  It writes
separate artifacts and does not modify the canonical Prediction experiment or
the Geo Notion page.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import run_prediction_experiment as prediction
except ModuleNotFoundError:  # Imported as scripts.run_prediction_rrf_ablation.
    from scripts import run_prediction_experiment as prediction


BOOTSTRAP_REPEATS = prediction.BOOTSTRAP_REPEATS
COMPACT_FEATURES = prediction.COMPACT_FEATURES
DEFAULT_DATASETS = prediction.DEFAULT_DATASETS
GRANULARITIES = prediction.GRANULARITIES
ROOT = prediction.ROOT
RRF_K0 = prediction.RRF_K0
SEED = prediction.SEED
TOP_K_VALUES = prediction.TOP_K_VALUES
CandidateRow = prediction.CandidateRow
_data_summary = prediction._data_summary
_paired_delta = prediction._paired_delta
build_rows = prediction.build_rows
evaluate_method = prediction.evaluate_method
fit_oof_xgboost = prediction.fit_oof_xgboost
load_questions = prediction.load_questions
make_fold_map = prediction.make_fold_map
stable_seed = prediction.stable_seed


# These compact features directly encode inputs already used to construct the
# Google/API rank lists behind G_A_RRF.  Keeping them would double-count rank or
# API exposure evidence in the overall fusion.
G_A_RRF_OVERLAPPING_FEATURES = {
    "url": ("rrf_score", "api_best_rank_log", "api_appear_rate"),
    "domain": ("rrf_score", "api_best_rank_log", "api_appear_rate"),
    "entity_brand": (
        "rrf_score",
        "api_exposure_per_run",
        "source_consensus",
        "api_url_support_log",
    ),
}
OVERALL_RRF_FEATURES = {
    granularity: tuple(
        feature
        for feature in COMPACT_FEATURES[granularity]
        if feature not in G_A_RRF_OVERLAPPING_FEATURES[granularity]
    )
    for granularity in GRANULARITIES
}
PRIMARY_METHODS = ("G_A_RRF", "Overall_RRF", "Full_compact_XGBoost")
PAIRED_METRICS = ("roc_auc", "average_precision", "f1@5", "f1@10")


def descending_rrf_contribution(values: Sequence[float]) -> tuple[np.ndarray, bool]:
    """Convert one feature into a tie-aware descending RRF contribution.

    Missing values receive no contribution.  A feature with no within-query
    variation cannot rank candidates and therefore also contributes zero.
    """
    result = np.zeros(len(values), dtype=float)
    finite = [
        (index, float(value))
        for index, value in enumerate(values)
        if math.isfinite(float(value))
    ]
    distinct = {value for _, value in finite}
    if len(distinct) < 2:
        return result, False

    by_value: dict[float, list[int]] = defaultdict(list)
    for index, value in finite:
        by_value[value].append(index)

    first_rank = 1
    for value in sorted(by_value, reverse=True):
        indices = by_value[value]
        last_rank = first_rank + len(indices) - 1
        average_rank = (first_rank + last_rank) / 2.0
        contribution = 1.0 / (RRF_K0 + average_rank)
        for index in indices:
            result[index] = contribution
        first_rank = last_rank + 1
    return result, True


def feature_rrf_contributions(
    rows: list[CandidateRow], feature_names: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Return per-row RRF contributions and within-query activity counts."""
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[row.group].append(index)

    contributions = {
        feature: np.zeros(len(rows), dtype=float) for feature in feature_names
    }
    diagnostics = {
        feature: {"active_queries": 0, "active_rows": 0}
        for feature in feature_names
    }
    for feature in feature_names:
        for indices in grouped_indices.values():
            group_values = [rows[index].features.get(feature, math.nan) for index in indices]
            group_contribution, active = descending_rrf_contribution(group_values)
            if not active:
                continue
            diagnostics[feature]["active_queries"] += 1
            diagnostics[feature]["active_rows"] += int(
                np.count_nonzero(group_contribution)
            )
            contributions[feature][indices] = group_contribution
    return contributions, diagnostics


def overall_rrf_scores(
    rows: list[CandidateRow], feature_names: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Add feature-level RRF lists to the existing G_A_RRF base score."""
    contributions, diagnostics = feature_rrf_contributions(rows, feature_names)
    scores = np.asarray([row.rrf_score for row in rows], dtype=float)
    for contribution in contributions.values():
        scores += contribution
    return scores, contributions, diagnostics


def _comparison(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    dataset: str,
    granularity: str,
    name: str,
) -> dict[str, object]:
    return {
        metric: _paired_delta(
            left,
            right,
            metric,
            stable_seed(SEED, dataset, granularity, name, metric),
        )
        for metric in PAIRED_METRICS
    }


def _evaluate_granularity(
    dataset: str,
    questions: list[object],
    granularity: str,
    fold_map: dict[str, int],
) -> dict[str, object]:
    rows = build_rows(questions, granularity)
    feature_names = OVERALL_RRF_FEATURES[granularity]
    overall_scores, contributions, feature_diagnostics = overall_rrf_scores(
        rows, feature_names
    )
    xgb_scores, xgb_diagnostics = fit_oof_xgboost(rows, fold_map, granularity)

    score_sets = {
        "G_A_RRF": np.asarray([row.rrf_score for row in rows], dtype=float),
        "Overall_RRF": overall_scores,
        "Full_compact_XGBoost": xgb_scores,
    }
    for feature, contribution in contributions.items():
        score_sets[f"Overall_RRF_without_{feature}"] = overall_scores - contribution

    methods: dict[str, object] = {}
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for method, scores in score_sets.items():
        methods[method], per_query[method] = evaluate_method(rows, scores, method)

    paired_comparisons = {
        "Overall_RRF_minus_G_A_RRF": _comparison(
            per_query["Overall_RRF"],
            per_query["G_A_RRF"],
            dataset,
            granularity,
            "overall_minus_base",
        ),
        "Overall_RRF_minus_Full_compact_XGBoost": _comparison(
            per_query["Overall_RRF"],
            per_query["Full_compact_XGBoost"],
            dataset,
            granularity,
            "overall_minus_xgboost",
        ),
    }
    feature_ablation = {}
    for feature in feature_names:
        without = f"Overall_RRF_without_{feature}"
        feature_ablation[feature] = {
            "diagnostics": feature_diagnostics[feature],
            "overall_minus_without_feature": _comparison(
                per_query["Overall_RRF"],
                per_query[without],
                dataset,
                granularity,
                f"overall_minus_without_{feature}",
            ),
        }

    return {
        "data": _data_summary(questions, rows, granularity),
        "feature_sets": {
            "base": "G_A_RRF(k0=60)",
            "included_as_rank_lists": list(feature_names),
            "excluded_as_G_A_RRF_overlapping": list(
                G_A_RRF_OVERLAPPING_FEATURES[granularity]
            ),
            "rank_direction": "descending; average ranks for ties",
        },
        "methods": methods,
        "paired_comparisons": paired_comparisons,
        "feature_ablation": feature_ablation,
        "xgboost_diagnostics": xgb_diagnostics,
        "per_query_metrics": per_query,
    }


def evaluate_dataset(dataset_root: Path) -> dict[str, object]:
    questions = load_questions(dataset_root)
    fold_map = make_fold_map(questions)
    return {
        "dataset": dataset_root.name,
        "granularities": {
            granularity: _evaluate_granularity(
                dataset_root.name, questions, granularity, fold_map
            )
            for granularity in GRANULARITIES
        },
    }


def _interval(summary: dict[str, float]) -> str:
    return (
        f"{summary['mean']:.3f} "
        f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]"
    )


def _delta(summary: dict[str, float]) -> str:
    return (
        f"{summary['mean']:+.3f} "
        f"[{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]"
    )


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Prediction Overall-RRF Feature Ablation",
        "",
        "G_A_RRF is retained as the base sum of the Google and A_candidate rank "
        "lists. Each remaining compact feature is independently ranked within a "
        "query (higher values first, average ranks for ties) and contributes "
        "1/(60+rank) to Overall_RRF. Features used to construct G_A_RRF are "
        "excluded so rank/exposure evidence is not counted twice.",
        "",
        "Query-level context features are retained in the declared feature set, "
        "but naturally contribute zero because they have no within-query variation.",
        "",
        "AUC/AP and P/R/F1@K use the canonical query-macro evaluation with 2,000 "
        "query-bootstrap resamples. Full compact XGBoost uses the same query-grouped "
        "five-fold OOF split as the canonical Prediction experiment.",
        "",
    ]
    for model in result["models"]:
        lines.extend([f"## {model['dataset']}", ""])
        for granularity in GRANULARITIES:
            section = model["granularities"][granularity]
            features = section["feature_sets"]
            lines.extend(
                [
                    f"### {granularity}",
                    "",
                    "Included feature rank lists: "
                    f"{', '.join(features['included_as_rank_lists']) or 'none'}.",
                    "Excluded as G_A_RRF-overlapping: "
                    f"{', '.join(features['excluded_as_G_A_RRF_overlapping'])}.",
                    "",
                    "| Method | Query-macro AUC [95% CI] | Query-macro AP [95% CI] | F1@5 | F1@10 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method in PRIMARY_METHODS:
                metrics = section["methods"][method]
                lines.append(
                    f"| {method} | {_interval(metrics['roc_auc'])} "
                    f"| {_interval(metrics['average_precision'])} "
                    f"| {metrics['f1@5']['mean']:.3f} "
                    f"| {metrics['f1@10']['mean']:.3f} |"
                )
            lines.extend(
                [
                    "",
                    "| Paired comparison | Delta AUC [95% CI] | Delta AP [95% CI] | Delta F1@5 | Delta F1@10 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for name, comparison in section["paired_comparisons"].items():
                lines.append(
                    f"| {name} | {_delta(comparison['roc_auc'])} "
                    f"| {_delta(comparison['average_precision'])} "
                    f"| {_delta(comparison['f1@5'])} "
                    f"| {_delta(comparison['f1@10'])} |"
                )
            lines.extend(
                [
                    "",
                    "The leave-one-feature-out delta is Full Overall_RRF minus "
                    "Overall_RRF without that feature; a positive delta means "
                    "the feature improves Overall_RRF.",
                    "",
                    "| Leave-one-feature-out | Active queries | Delta AUC | Delta AP | Delta F1@5 | Delta F1@10 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for feature, ablation in section["feature_ablation"].items():
                comparison = ablation["overall_minus_without_feature"]
                lines.append(
                    f"| {feature} | {ablation['diagnostics']['active_queries']} "
                    f"| {_delta(comparison['roc_auc'])} "
                    f"| {_delta(comparison['average_precision'])} "
                    f"| {_delta(comparison['f1@5'])} "
                    f"| {_delta(comparison['f1@10'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--output-prefix",
        default="result/prediction_overall_rrf_ablation_1600run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = []
    for dataset in args.datasets:
        dataset_root = (ROOT / dataset).resolve()
        print(f"Running overall-RRF feature ablation for {dataset_root.name}...")
        models.append(evaluate_dataset(dataset_root))
    result = {
        "method": {
            "base": "G_A_RRF(k0=60)",
            "fusion": "base + sum(1/(60 + within-query descending feature rank))",
            "feature_sets": {
                name: list(features) for name, features in OVERALL_RRF_FEATURES.items()
            },
            "excluded_as_G_A_RRF_overlapping": {
                name: list(features)
                for name, features in G_A_RRF_OVERLAPPING_FEATURES.items()
            },
            "tie_handling": "average rank",
            "missing_handling": "zero contribution",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "seed": SEED,
        },
        "models": models,
    }
    prefix = (ROOT / args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {prefix.with_suffix('.json')}")
    print(f"Wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
