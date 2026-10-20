"""Analyze hierarchical candidate frequency and multi-URL support.

This script extends the existing exact-URL Web candidate-frequency analysis to
unique hostnames and audits the frequency/support features used by the
conditional Prediction experiment.  It uses only existing Web, Google, API,
and cached entity data; no model or search API is called.

Four quantities are kept separate:

* Ever-cited visibility: whether a URL or hostname was cited at least once in
  the same runs used to calculate its Web run frequency.  This is retained as
  a descriptive legacy result only.
* Held-out citation propensity: whether an item is cited in one Web run,
  predicted by its frequency in the other runs.
* API run frequency: fraction of API runs containing a URL or hostname,
  evaluated conditionally among A_candidate-supported union candidates.
* Multi-URL support: number of distinct candidate URLs supporting a hostname
  or entity/brand.

The cached entity pipeline stores summed URL exposure rather than per-run
entity membership.  Entity results are therefore labelled exposure intensity,
not strict run frequency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

try:
    from evaluate_proxy_effectiveness import deep_normalize, hostname
    from run_prediction_experiment import build_rows, load_questions
except ModuleNotFoundError:  # Imported as scripts.analyze_hierarchical_frequency.
    from scripts.evaluate_proxy_effectiveness import deep_normalize, hostname
    from scripts.run_prediction_experiment import build_rows, load_questions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("dataset-5.3-1600", "dataset-5.5-1600")
SUBDATASETS = (
    "v9_analysis",
    "v9_factual_lookup",
    "v9_info_synthesis",
    "v9_recommendation",
)

FREQUENCY_BINS = (
    ("1-9%", 1e-12, 0.10),
    ("10-29%", 0.10, 0.30),
    ("30-49%", 0.30, 0.50),
    ("50-69%", 0.50, 0.70),
    ("70-89%", 0.70, 0.90),
    ("90-100%", 0.90, math.inf),
)
HELDOUT_FREQUENCY_BINS = (
    ("0% prior", 0.0, 1e-12),
    *FREQUENCY_BINS,
)
SUPPORT_BINS = (
    ("1 URL", 1.0, 2.0),
    ("2 URLs", 2.0, 3.0),
    ("3-4 URLs", 3.0, 5.0),
    ("5-9 URLs", 5.0, 10.0),
    ("10+ URLs", 10.0, math.inf),
)
EXPOSURE_BINS = (
    ("0.01-0.09/run", 1e-12, 0.10),
    ("0.10-0.49/run", 0.10, 0.50),
    ("0.50-0.99/run", 0.50, 1.00),
    ("1.00-1.99/run", 1.00, 2.00),
    ("2.00-4.99/run", 2.00, 5.00),
    ("5.00+/run", 5.00, math.inf),
)


@dataclass(frozen=True)
class Observation:
    """One candidate x query observation for a univariate audit."""

    group: str
    item: str
    value: float
    label: int


@dataclass(frozen=True)
class TrialObservation:
    """One candidate x query row with citation successes and appearances."""

    group: str
    item: str
    value: float
    successes: int
    trials: int


@dataclass(frozen=True)
class RunGroup:
    """Per-run candidate and citation sets for one query."""

    group: str
    appeared: tuple[frozenset[str], ...]
    cited: tuple[frozenset[str], ...]


BOOTSTRAP_REPEATS = 2000


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_mean(values: Sequence[float], *seed_parts: object) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {
            "mean": math.nan,
            "median": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "n": 0,
        }
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    draws = rng.choice(
        array, size=(BOOTSTRAP_REPEATS, len(array)), replace=True
    ).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n": len(array),
    }


def _fit_logistic(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Fit intercept + one feature logistic regression with Newton IRLS."""
    matrix = np.column_stack([np.ones(len(values)), values]).astype(float)
    labels = labels.astype(float)
    beta = np.zeros(2, dtype=float)
    for _ in range(100):
        eta = np.clip(matrix @ beta, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        weights = probabilities * (1.0 - probabilities)
        hessian = (matrix.T * weights) @ matrix + 1e-8 * np.eye(2)
        score = matrix.T @ (labels - probabilities)
        delta = np.linalg.solve(hessian, score)
        beta += delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break

    eta = np.clip(matrix @ beta, -30.0, 30.0)
    probabilities = 1.0 / (1.0 + np.exp(-eta))
    weights = probabilities * (1.0 - probabilities)
    covariance = np.linalg.inv((matrix.T * weights) @ matrix + 1e-8 * np.eye(2))
    standard_errors = np.sqrt(np.diag(covariance))
    z_score = float(beta[1] / standard_errors[1])
    p_value = float(2.0 * stats.norm.sf(abs(z_score)))
    epsilon = 1e-12
    log_likelihood = float(
        np.sum(
            labels * np.log(probabilities + epsilon)
            + (1.0 - labels) * np.log(1.0 - probabilities + epsilon)
        )
    )
    null_probability = float(labels.mean())
    null_log_likelihood = float(
        np.sum(
            labels * math.log(null_probability + epsilon)
            + (1.0 - labels) * math.log(1.0 - null_probability + epsilon)
        )
    )
    return {
        "beta": float(beta[1]),
        "standard_error": float(standard_errors[1]),
        "z": z_score,
        "p_value": p_value,
        "odds_ratio_per_unit": float(math.exp(float(beta[1]))),
        "log_likelihood": log_likelihood,
        "mcfadden_pseudo_r2": float(1.0 - log_likelihood / null_log_likelihood),
    }


def _assign_bin(
    value: float, bins: Sequence[tuple[str, float, float]]
) -> str | None:
    for name, lower, upper in bins:
        if lower <= value < upper:
            return name
    return None


def _trend_test(bin_rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    nonempty = [row for row in bin_rows if int(row["n_total"]) > 0]
    if len(nonempty) < 2:
        return {"z": math.nan, "p_value": math.nan}
    cited = np.asarray([row["n_positive"] for row in nonempty], dtype=float)
    totals = np.asarray([row["n_total"] for row in nonempty], dtype=float)
    scores = np.arange(1, len(nonempty) + 1, dtype=float)
    total = float(totals.sum())
    positives = float(cited.sum())
    positive_rate = positives / total
    weighted_score = float((totals * scores).sum())
    variance = positive_rate * (1.0 - positive_rate) * (
        total * float((totals * scores**2).sum()) - weighted_score**2
    ) / total
    if variance <= 0:
        return {"z": math.nan, "p_value": math.nan}
    statistic = float((cited * scores).sum()) - positives * weighted_score / total
    z_score = statistic / math.sqrt(variance)
    return {
        "z": float(z_score),
        "p_value": float(2.0 * stats.norm.sf(abs(z_score))),
    }


def analyze_observations(
    observations: Sequence[Observation],
    bins: Sequence[tuple[str, float, float]],
    transform: Callable[[float], float] | None = None,
    transform_name: str = "identity",
) -> dict[str, object]:
    """Summarize bins and a one-feature logistic model."""
    if not observations:
        raise ValueError("No observations to analyze")
    labels = np.asarray([row.label for row in observations], dtype=int)
    raw_values = np.asarray([row.value for row in observations], dtype=float)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Observations must contain positive and negative labels")
    model_values = np.asarray(
        [(transform or (lambda value: value))(value) for value in raw_values],
        dtype=float,
    )

    bin_counts = {
        name: {"n_total": 0, "n_positive": 0}
        for name, _, _ in bins
    }
    for row in observations:
        bin_name = _assign_bin(row.value, bins)
        if bin_name is None:
            continue
        bin_counts[bin_name]["n_total"] += 1
        bin_counts[bin_name]["n_positive"] += row.label

    bin_table: list[dict[str, float | int | str]] = []
    for name, _, _ in bins:
        total = bin_counts[name]["n_total"]
        positive = bin_counts[name]["n_positive"]
        if total:
            interval = stats.binomtest(positive, total).proportion_ci(method="wilson")
            low, high = float(interval.low), float(interval.high)
        else:
            low = high = math.nan
        bin_table.append(
            {
                "bin": name,
                "n_total": total,
                "n_positive": positive,
                "positive_rate": positive / total if total else math.nan,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
            }
        )

    logistic = _fit_logistic(model_values, labels)
    logistic["feature_transform"] = transform_name
    spearman = stats.spearmanr(raw_values, labels)
    return {
        "n": len(observations),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "value_min": float(raw_values.min()),
        "value_max": float(raw_values.max()),
        "bin_table": bin_table,
        "ordered_bin_trend": _trend_test(bin_table),
        "logistic": logistic,
        "roc_auc": float(roc_auc_score(labels, model_values)),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }


def _per_query_auc(observations: Sequence[Observation]) -> dict[str, float]:
    grouped: defaultdict[str, list[Observation]] = defaultdict(list)
    for row in observations:
        grouped[row.group].append(row)
    output: dict[str, float] = {}
    for group, rows in grouped.items():
        labels = [row.label for row in rows]
        if len(set(labels)) < 2:
            continue
        output[group] = float(roc_auc_score(labels, [row.value for row in rows]))
    return output


def query_macro_auc(
    observations: Sequence[Observation], *seed_parts: object
) -> dict[str, object]:
    """Compute per-query AUC and a query-resampled confidence interval."""
    per_query = _per_query_auc(observations)
    summary = _bootstrap_mean(list(per_query.values()), *seed_parts, "query_auc")
    return {"summary": summary, "per_query": dict(sorted(per_query.items()))}


def paired_query_auc_delta(
    left: Sequence[Observation],
    right: Sequence[Observation],
    *seed_parts: object,
) -> dict[str, float | int]:
    """Return right-minus-left AUC with paired query bootstrap."""
    left_auc = _per_query_auc(left)
    right_auc = _per_query_auc(right)
    groups = sorted(set(left_auc) & set(right_auc))
    deltas = np.asarray([right_auc[group] - left_auc[group] for group in groups])
    summary = _bootstrap_mean(deltas.tolist(), *seed_parts, "paired_auc_delta")
    if not len(deltas):
        summary["probability_le_zero"] = math.nan
        return summary
    rng = np.random.default_rng(_stable_seed(*seed_parts, "paired_auc_probability"))
    draws = rng.choice(
        deltas, size=(BOOTSTRAP_REPEATS, len(deltas)), replace=True
    ).mean(axis=1)
    summary["probability_le_zero"] = float((draws <= 0).mean())
    return summary


def query_macro_bin_table(
    observations: Sequence[Observation],
    bins: Sequence[tuple[str, float, float]],
    *seed_parts: object,
) -> list[dict[str, float | int | str]]:
    """Compute citation rates within query, then macro-average by bin."""
    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for row in observations:
        bin_name = _assign_bin(row.value, bins)
        if bin_name is not None:
            grouped[(bin_name, row.group)].append(row.label)
    output: list[dict[str, float | int | str]] = []
    for bin_name, _, _ in bins:
        query_rates = [
            statistics.fmean(labels)
            for (name, _), labels in grouped.items()
            if name == bin_name
        ]
        total = sum(
            len(labels) for (name, _), labels in grouped.items() if name == bin_name
        )
        positives = sum(
            sum(labels) for (name, _), labels in grouped.items() if name == bin_name
        )
        summary = _bootstrap_mean(
            query_rates, *seed_parts, "bin", bin_name
        )
        output.append(
            {
                "bin": bin_name,
                "n_queries": summary["n"],
                "n_occurrences": total,
                "n_cited": positives,
                "micro_rate": positives / total if total else math.nan,
                "macro_rate": summary["mean"],
                "macro_ci_low": summary["ci_low"],
                "macro_ci_high": summary["ci_high"],
            }
        )
    return output


def trial_macro_bin_table(
    observations: Sequence[TrialObservation],
    bins: Sequence[tuple[str, float, float]],
    *seed_parts: object,
) -> list[dict[str, float | int | str]]:
    """Macro-average per-query cited/appearance rates within frequency bins."""
    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for row in observations:
        bin_name = _assign_bin(row.value, bins)
        if bin_name is None:
            continue
        grouped[(bin_name, row.group)][0] += row.successes
        grouped[(bin_name, row.group)][1] += row.trials
    output: list[dict[str, float | int | str]] = []
    for bin_name, _, _ in bins:
        pairs = [
            counts for (name, _), counts in grouped.items() if name == bin_name
        ]
        query_rates = [successes / trials for successes, trials in pairs if trials]
        successes = sum(pair[0] for pair in pairs)
        trials = sum(pair[1] for pair in pairs)
        summary = _bootstrap_mean(
            query_rates, *seed_parts, "trial_bin", bin_name
        )
        output.append(
            {
                "bin": bin_name,
                "n_queries": summary["n"],
                "n_candidates": sum(
                    1
                    for row in observations
                    if _assign_bin(row.value, bins) == bin_name
                ),
                "n_appearances": trials,
                "n_citations": successes,
                "micro_rate": successes / trials if trials else math.nan,
                "macro_rate": summary["mean"],
                "macro_ci_low": summary["ci_low"],
                "macro_ci_high": summary["ci_high"],
            }
        )
    return output


def load_web_run_groups(dataset_root: Path) -> dict[str, list[RunGroup]]:
    """Load per-run URL and unique-hostname sets for every query."""
    output: dict[str, list[RunGroup]] = {"url": [], "domain": []}
    for subdataset in SUBDATASETS:
        search_root = dataset_root / subdataset / "search_detail"
        for question_dir in sorted(search_root.glob("q*")):
            paths = sorted(question_dir.glob("*_detail.json"))
            if not paths:
                continue
            url_appeared_runs: list[frozenset[str]] = []
            url_cited_runs: list[frozenset[str]] = []
            domain_appeared_runs: list[frozenset[str]] = []
            domain_cited_runs: list[frozenset[str]] = []
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                run_urls: set[str] = set()
                run_cited_urls: set[str] = set()
                for item in payload.get("all_urls") or []:
                    normalized = deep_normalize(item.get("url") or "")
                    if not normalized:
                        continue
                    run_urls.add(normalized)
                    if item.get("cited"):
                        run_cited_urls.add(normalized)
                run_domains = {hostname(url) for url in run_urls if hostname(url)}
                run_cited_domains = {
                    hostname(url) for url in run_cited_urls if hostname(url)
                }
                url_appeared_runs.append(frozenset(run_urls))
                url_cited_runs.append(frozenset(run_cited_urls))
                domain_appeared_runs.append(frozenset(run_domains))
                domain_cited_runs.append(frozenset(run_cited_domains))

            group = f"{subdataset}/{question_dir.name}"
            output["url"].append(
                RunGroup(group, tuple(url_appeared_runs), tuple(url_cited_runs))
            )
            output["domain"].append(
                RunGroup(
                    group, tuple(domain_appeared_runs), tuple(domain_cited_runs)
                )
            )
    return output


def aggregate_web_frequency_rows(
    run_groups: dict[str, list[RunGroup]],
) -> dict[str, list[Observation]]:
    """Build legacy ever-cited observations from per-run sets."""
    output: dict[str, list[Observation]] = {"url": [], "domain": []}
    for granularity, groups in run_groups.items():
        for run_group in groups:
            appearances = Counter(
                item for run in run_group.appeared for item in run
            )
            citations = Counter(item for run in run_group.cited for item in run)
            n_runs = len(run_group.appeared)
            output[granularity].extend(
                Observation(
                    group=run_group.group,
                    item=item,
                    value=appear_count / n_runs,
                    label=int(citations[item] > 0),
                )
                for item, appear_count in appearances.items()
            )
    return output


def load_web_frequency_rows(dataset_root: Path) -> dict[str, list[Observation]]:
    """Build strict per-run URL and unique-hostname Web frequency rows."""
    return aggregate_web_frequency_rows(load_web_run_groups(dataset_root))


def build_per_appearance_rows(
    run_groups: dict[str, list[RunGroup]],
) -> dict[str, list[TrialObservation]]:
    """Build full-sample cited/appearance rows for conditional propensity."""
    output: dict[str, list[TrialObservation]] = {"url": [], "domain": []}
    for granularity, groups in run_groups.items():
        for run_group in groups:
            appearances = Counter(
                item for run in run_group.appeared for item in run
            )
            citations = Counter(item for run in run_group.cited for item in run)
            n_runs = len(run_group.appeared)
            output[granularity].extend(
                TrialObservation(
                    group=run_group.group,
                    item=item,
                    value=appear_count / n_runs,
                    successes=citations[item],
                    trials=appear_count,
                )
                for item, appear_count in appearances.items()
            )
    return output


def build_leave_one_run_out_rows(
    run_groups: dict[str, list[RunGroup]],
) -> dict[str, list[Observation]]:
    """Predict held-out per-appearance citation from the other runs' frequency."""
    output: dict[str, list[Observation]] = {"url": [], "domain": []}
    for granularity, groups in run_groups.items():
        for run_group in groups:
            n_runs = len(run_group.appeared)
            if n_runs < 2:
                continue
            total_appearances = Counter(
                item for run in run_group.appeared for item in run
            )
            for run_index, (appeared, cited) in enumerate(
                zip(run_group.appeared, run_group.cited)
            ):
                for item in appeared:
                    prior_count = total_appearances[item] - 1
                    output[granularity].append(
                        Observation(
                            group=run_group.group,
                            item=f"{item}@@run{run_index:02d}",
                            value=prior_count / (n_runs - 1),
                            label=int(item in cited),
                        )
                    )
    return output


def load_web_entity_rows(dataset_root: Path) -> dict[str, list[Observation]]:
    """Load cached entity exposure intensity and URL support observations."""
    exposure_rows: list[Observation] = []
    support_rows: list[Observation] = []
    cache_root = dataset_root / "v9_trend_analysis" / "results" / "entity_llm_cache"
    for path in sorted(cache_root.glob("v9_*__q*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = payload["counts"]
        group = f"{payload['subdataset']}/{payload['qid']}"
        w_exposure = counts.get("W_exp") or {}
        w_citation = counts.get("W_cite") or {}
        w_support = counts.get("W_n") or {}
        for entity, exposure in w_exposure.items():
            if float(exposure) <= 0:
                continue
            label = int(float(w_citation.get(entity, 0)) > 0)
            exposure_rows.append(
                Observation(group, entity, float(exposure) / 20.0, label)
            )
            support = float(w_support.get(entity, 0))
            if support > 0:
                support_rows.append(Observation(group, entity, support, label))
    return {"exposure": exposure_rows, "support": support_rows}


def load_prediction_feature_rows(dataset_root: Path) -> dict[str, dict[str, list[Observation]]]:
    """Expose the exact frequency/support variables used by Prediction."""
    questions = load_questions(dataset_root)
    output: dict[str, dict[str, list[Observation]]] = {}
    for granularity in ("url", "domain", "entity_brand"):
        rows = build_rows(questions, granularity)
        frequency: list[Observation] = []
        support_union: list[Observation] = []
        support_api: list[Observation] = []
        for row in rows:
            if granularity == "entity_brand":
                frequency_value = float(row.features["api_exposure_per_run"])
                api_support_value = round(
                    math.expm1(float(row.features["api_url_support_log"]))
                )
                union_support_value = 0
            elif granularity == "domain":
                frequency_value = float(row.features["api_appear_rate"])
                union_support_value = round(
                    math.expm1(float(row.features["candidate_url_count_log"]))
                )
                api_support_value = round(
                    math.expm1(float(row.features["api_url_count_log"]))
                )
            else:
                frequency_value = float(row.features["api_appear_rate"])
                union_support_value = api_support_value = 0
            # Frequency is conditional on appearing in A_candidate at least once.
            if frequency_value > 0:
                frequency.append(
                    Observation(row.group, row.item, frequency_value, row.label)
                )
            if union_support_value > 0:
                support_union.append(
                    Observation(
                        row.group, row.item, float(union_support_value), row.label
                    )
                )
            if api_support_value > 0:
                support_api.append(
                    Observation(
                        row.group, row.item, float(api_support_value), row.label
                    )
                )
        output[granularity] = {
            "frequency": frequency,
            "support_union": support_union,
            "support_api": support_api,
        }
    return output


def analyze_query_signal(
    observations: Sequence[Observation],
    bins: Sequence[tuple[str, float, float]],
    *seed_parts: object,
    transform: Callable[[float], float] | None = None,
    transform_name: str = "identity",
) -> dict[str, object]:
    """Report legacy pooled diagnostics plus query-macro inference."""
    return {
        "pooled_descriptive": analyze_observations(
            observations,
            bins,
            transform=transform,
            transform_name=transform_name,
        ),
        "query_macro_auc": query_macro_auc(observations, *seed_parts),
        "query_macro_bins": query_macro_bin_table(
            observations, bins, *seed_parts
        ),
    }


def analyze_dataset(dataset_root: Path) -> dict[str, object]:
    run_groups = load_web_run_groups(dataset_root)
    web_rows = aggregate_web_frequency_rows(run_groups)
    per_appearance_rows = build_per_appearance_rows(run_groups)
    heldout_rows = build_leave_one_run_out_rows(run_groups)
    web_entity = load_web_entity_rows(dataset_root)
    prediction_rows = load_prediction_feature_rows(dataset_root)
    web_heldout = {
        granularity: {
            "query_macro_auc": query_macro_auc(
                rows, dataset_root.name, "web_heldout", granularity
            ),
            "query_macro_bins": query_macro_bin_table(
                rows,
                HELDOUT_FREQUENCY_BINS,
                dataset_root.name,
                "web_heldout",
                granularity,
            ),
        }
        for granularity, rows in heldout_rows.items()
    }
    web_heldout["domain_minus_url_auc"] = paired_query_auc_delta(
        heldout_rows["url"],
        heldout_rows["domain"],
        dataset_root.name,
        "web_heldout",
        "domain_minus_url",
    )

    prediction_frequency = {
        granularity: analyze_query_signal(
            rows["frequency"],
            EXPOSURE_BINS if granularity == "entity_brand" else FREQUENCY_BINS,
            dataset_root.name,
            "prediction_frequency",
            granularity,
            transform=math.log1p if granularity == "entity_brand" else None,
            transform_name=(
                "log1p(A_exposures_per_run)"
                if granularity == "entity_brand"
                else "identity"
            ),
        )
        for granularity, rows in prediction_rows.items()
    }
    prediction_frequency["domain_minus_url_auc"] = paired_query_auc_delta(
        prediction_rows["url"]["frequency"],
        prediction_rows["domain"]["frequency"],
        dataset_root.name,
        "prediction_frequency",
        "domain_minus_url",
    )

    prediction_support = {
        "domain_api_only": analyze_query_signal(
            prediction_rows["domain"]["support_api"],
            SUPPORT_BINS,
            dataset_root.name,
            "prediction_support",
            "domain_api_only",
            transform=lambda value: math.log2(1.0 + value),
            transform_name="log2(1 + A_distinct_url_count)",
        ),
        "entity_api_only": analyze_query_signal(
            prediction_rows["entity_brand"]["support_api"],
            SUPPORT_BINS,
            dataset_root.name,
            "prediction_support",
            "entity_api_only",
            transform=lambda value: math.log2(1.0 + value),
            transform_name="log2(1 + A_distinct_url_count)",
        ),
        "domain_union_diagnostic": analyze_query_signal(
            prediction_rows["domain"]["support_union"],
            SUPPORT_BINS,
            dataset_root.name,
            "prediction_support",
            "domain_union_diagnostic",
            transform=lambda value: math.log2(1.0 + value),
            transform_name="log2(1 + G_union_A_distinct_url_count)",
        ),
    }
    return {
        "dataset": dataset_root.name,
        "web_internal_frequency": {
            granularity: analyze_observations(rows, FREQUENCY_BINS)
            for granularity, rows in web_rows.items()
        },
        "web_per_appearance_propensity": {
            granularity: {
                "query_macro_bins": trial_macro_bin_table(
                    rows,
                    FREQUENCY_BINS,
                    dataset_root.name,
                    "web_per_appearance",
                    granularity,
                )
            }
            for granularity, rows in per_appearance_rows.items()
        },
        "web_heldout_propensity": web_heldout,
        "web_entity_cache": {
            "exposure_intensity": analyze_observations(
                web_entity["exposure"],
                EXPOSURE_BINS,
                transform=math.log1p,
                transform_name="log1p(exposures_per_run)",
            ),
            "url_support": analyze_observations(
                web_entity["support"],
                SUPPORT_BINS,
                transform=lambda value: math.log2(1.0 + value),
                transform_name="log2(1 + distinct_url_count)",
            ),
        },
        "prediction_proxy": {
            "frequency_or_exposure": prediction_frequency,
            "url_support": prediction_support,
        },
    }


def _pct(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value * 100:.1f}%"


def _summary_row(label: str, result: dict[str, object]) -> str:
    logistic = result["logistic"]
    trend = result["ordered_bin_trend"]
    return (
        f"| {label} | {result['n']:,} | {_pct(result['positive_rate'])} "
        f"| {result['roc_auc']:.3f} | {logistic['mcfadden_pseudo_r2']:.3f} "
        f"| {logistic['p_value']:.3g} | {trend['z']:.2f} / {trend['p_value']:.3g} |"
    )


def _auc_row(label: str, result: dict[str, object]) -> str:
    summary = result["query_macro_auc"]["summary"]
    return (
        f"| {label} | {summary['n']} | {summary['mean']:.3f} "
        f"| [{summary['ci_low']:.3f}, {summary['ci_high']:.3f}] |"
    )


def _delta_text(result: dict[str, object]) -> str:
    return (
        f"{result['mean']:+.3f} [{result['ci_low']:+.3f}, "
        f"{result['ci_high']:+.3f}] (n={result['n']})"
    )


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Hierarchical Frequency and Multi-URL Support",
        "",
        "All analyses use existing dataset records. No API or model call is made.",
        "",
        "- Web URL/domain frequency is strict per-run presence with within-run deduplication.",
        "- Prediction proxy frequency is conditional on A_candidate support (>0 API runs).",
        "- Entity caches contain summed URL exposure, so entity frequency rows are labelled exposure intensity rather than strict run frequency.",
        "- Legacy ever-cited tables are retained as visibility diagnostics, not per-appearance propensity.",
        "- Corrected inference uses leave-one-run-out frequency, per-appearance citation labels, query-macro AUC, and query bootstrap.",
        "- A-only domain/entity URL support is the matched comparison and maps to the compact Prediction support features; G union A domain support is a standalone diagnostic, not a compact-model input.",
        "",
    ]
    for model in result["models"]:
        lines.extend(
            [
                f"## {model['dataset']}",
                "",
                "### Legacy ever-cited visibility (pooled descriptive)",
                "",
                "| Analysis | N | Ever-cited rate | AUC | McFadden R2 | Logistic p | Ordered-bin z / p |",
                "|---|---:|---:|---:|---:|---:|---:|",
                _summary_row("W exact URL frequency", model["web_internal_frequency"]["url"]),
                _summary_row("W hostname frequency", model["web_internal_frequency"]["domain"]),
                "",
                "### Corrected query-macro inference",
                "",
                "| Analysis | Evaluated queries | Macro AUC | Query-bootstrap 95% CI |",
                "|---|---:|---:|---:|",
                _auc_row("W held-out exact URL propensity", model["web_heldout_propensity"]["url"]),
                _auc_row("W held-out hostname propensity", model["web_heldout_propensity"]["domain"]),
                _auc_row("A exact URL frequency -> W", model["prediction_proxy"]["frequency_or_exposure"]["url"]),
                _auc_row("A hostname frequency -> W", model["prediction_proxy"]["frequency_or_exposure"]["domain"]),
                _auc_row("A entity exposure -> W", model["prediction_proxy"]["frequency_or_exposure"]["entity_brand"]),
                _auc_row("A-only hostname URL support -> W", model["prediction_proxy"]["url_support"]["domain_api_only"]),
                _auc_row("A-only entity URL support -> W", model["prediction_proxy"]["url_support"]["entity_api_only"]),
                _auc_row("G/A hostname URL support -> W", model["prediction_proxy"]["url_support"]["domain_union_diagnostic"]),
                "",
                "Paired hostname-minus-URL AUC:",
                "",
                f"- W held-out propensity: {_delta_text(model['web_heldout_propensity']['domain_minus_url_auc'])}",
                f"- A frequency -> W: {_delta_text(model['prediction_proxy']['frequency_or_exposure']['domain_minus_url_auc'])}",
                "",
                "### Full-sample per-appearance propensity bins",
                "",
                "| Granularity | Frequency | Queries | Appearances | Citations | Micro rate | Query-macro rate (95% CI) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for granularity in ("url", "domain"):
            section = model["web_per_appearance_propensity"][granularity]
            for row in section["query_macro_bins"]:
                lines.append(
                    f"| {granularity} | {row['bin']} | {row['n_queries']} "
                    f"| {row['n_appearances']} | {row['n_citations']} "
                    f"| {_pct(row['micro_rate'])} "
                    f"| {_pct(row['macro_rate'])} [{_pct(row['macro_ci_low'])}, {_pct(row['macro_ci_high'])}] |"
                )
        lines.extend(
            [
                "",
                "### Leave-one-run-out held-out propensity bins",
                "",
                "| Granularity | Prior frequency | Queries | Held-out appearances | Citations | Micro rate | Query-macro rate (95% CI) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for granularity in ("url", "domain"):
            section = model["web_heldout_propensity"][granularity]
            for row in section["query_macro_bins"]:
                lines.append(
                    f"| {granularity} | {row['bin']} | {row['n_queries']} "
                    f"| {row['n_occurrences']} | {row['n_cited']} "
                    f"| {_pct(row['micro_rate'])} "
                    f"| {_pct(row['macro_rate'])} [{_pct(row['macro_ci_low'])}, {_pct(row['macro_ci_high'])}] |"
                )
        lines.extend(
            [
                "",
                "### Matched A-only support-breadth bins",
                "",
                "| Granularity | Distinct A URLs | Queries | Candidates | W cited | Micro rate | Query-macro rate (95% CI) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, key in (("domain", "domain_api_only"), ("entity_brand", "entity_api_only")):
            section = model["prediction_proxy"]["url_support"][key]
            for row in section["query_macro_bins"]:
                lines.append(
                    f"| {label} | {row['bin']} | {row['n_queries']} "
                    f"| {row['n_occurrences']} | {row['n_cited']} "
                    f"| {_pct(row['micro_rate'])} "
                    f"| {_pct(row['macro_rate'])} [{_pct(row['macro_ci_low'])}, {_pct(row['macro_ci_high'])}] |"
                )
        lines.append("")
    return "\n".join(lines)


def _stable_numbers(value: object) -> object:
    """Remove platform-level floating point noise from serialized results."""
    if isinstance(value, float):
        return value if not math.isfinite(value) else float(f"{value:.10g}")
    if isinstance(value, dict):
        return {key: _stable_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_numbers(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--output-prefix",
        default="result/hierarchical_frequency_1600run",
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "method": {
            "legacy_web_visibility": (
                "per-query per-run presence, deduplicated within run; label is "
                "whether the item was ever cited in the same runs"
            ),
            "web_per_appearance_propensity": (
                "citation count divided by appearance count for each query-item"
            ),
            "web_heldout_propensity": (
                "leave one Web run out; predict citation on each held-out appearance "
                "from frequency in the other runs"
            ),
            "prediction_frequency": (
                "A_candidate-supported rows only; evaluated against W citation"
            ),
            "support_breadth_matched": (
                "distinct normalized A_candidate URLs for both hostname and "
                "entity/brand"
            ),
            "support_breadth_diagnostic": (
                "distinct normalized G union A_candidate URLs for hostname only"
            ),
            "entity_boundary": (
                "cached entity counts sum URL exposures and do not retain per-run entity membership"
            ),
            "inference": (
                "query-macro AUC with 2000 deterministic query-bootstrap resamples; "
                "hostname-minus-URL deltas use paired queries"
            ),
        },
        "models": [
            analyze_dataset((ROOT / dataset).resolve()) for dataset in args.datasets
        ],
    }
    result = _stable_numbers(result)
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
