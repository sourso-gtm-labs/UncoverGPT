"""Analyze secondary candidate-to-citation signals for the paper.

The analysis uses only the 1,600 Web runs in each Instant-model dataset. It
reports retrieval position, publication recency for dated news candidates,
reference type, and cross-run citation-set stability by query intent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import fmean, median
from urllib.parse import parse_qs, unquote, urlparse, urlunparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("dataset-5.3-1600", "dataset-5.5-1600")
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "srsltid",
    "ref",
    "fbclid",
    "gclid",
    "source",
    "from",
}
POSITION_BUCKETS = ("1-2", "3-5", "6-9", "10-14", "15+")
RECENCY_BUCKETS = ("0-7d", "8-30d", "31-365d", ">365d")
INTENT_LABELS = {
    "v9_analysis": "Analysis",
    "v9_factual_lookup": "Factual",
    "v9_info_synthesis": "Info.",
    "v9_recommendation": "Rec.",
}


def deep_normalize(url: str) -> str:
    """Normalize URLs using the project-wide comparison rule."""
    try:
        parsed = urlparse(unquote(url or ""))
        scheme = parsed.scheme.lower() or "https"
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        host = re.sub(r":443$", "", host)
        host = re.sub(r":80$", "", host)
        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        params = parse_qs(parsed.query, keep_blank_values=True)
        for key in TRACKING_PARAMS:
            params.pop(key, None)
        clean_query = "&".join(
            f"{key}={values[0]}" for key, values in sorted(params.items())
        )
        return urlunparse((scheme, host, path, "", clean_query, ""))
    except Exception:
        return url


def iter_questions(dataset_root: Path):
    """Yield question directories from the four v9 Web cohorts."""
    for subset in sorted(dataset_root.glob("v9_*")):
        detail_root = subset / "search_detail"
        if not detail_root.is_dir():
            continue
        for question_dir in sorted(detail_root.iterdir()):
            if question_dir.is_dir() and question_dir.name.startswith("q"):
                files = sorted(question_dir.glob("*_detail.json"))
                if files:
                    yield subset.name, question_dir.name, files


def run_datetime(path: Path) -> datetime:
    """Recover the 2026 collection timestamp encoded in a Web detail filename."""
    match = re.search(r"_(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})_detail\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot parse collection timestamp from {path.name}")
    month, day, hour, minute, second = map(int, match.groups())
    return datetime(2026, month, day, hour, minute, second, tzinfo=timezone.utc)


def position_bucket(ref_index: object) -> str | None:
    """Map a zero-based SSE reference index to a one-based display bucket."""
    if not isinstance(ref_index, int) or ref_index < 0:
        return None
    position = ref_index + 1
    if position <= 2:
        return "1-2"
    if position <= 5:
        return "3-5"
    if position <= 9:
        return "6-9"
    if position <= 14:
        return "10-14"
    return "15+"


def recency_bucket(age_days: float) -> str:
    """Map nonnegative publication age to a reporting bucket."""
    if age_days <= 7:
        return "0-7d"
    if age_days <= 30:
        return "8-30d"
    if age_days <= 365:
        return "31-365d"
    return ">365d"


def jaccard(left: set[str], right: set[str]) -> float:
    """Compute set Jaccard, treating two empty citation sets as identical."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def auc_probability(rows: list[tuple[float, bool]]) -> float | None:
    """Return probability that a positive item has a larger score than a negative."""
    positives = [score for score, label in rows if label]
    negatives = [score for score, label in rows if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def bootstrap_mean(
    values: list[float], *, seed: int = 20260731, samples: int = 2000
) -> dict[str, float]:
    """Return a mean and query-level percentile bootstrap interval."""
    if not values:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = random.Random(seed)
    estimates = sorted(
        fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return {
        "mean": fmean(values),
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
    }


def paired_difference(
    earlier: dict[str, float],
    later: dict[str, float],
) -> dict:
    """Bootstrap the later-minus-earlier difference over shared queries."""
    shared = sorted(set(earlier) & set(later))
    deltas = [later[key] - earlier[key] for key in shared]
    return {"n_queries": len(shared), **bootstrap_mean(deltas)}


def analyze_dataset(dataset: str) -> dict:
    """Compute all secondary-signal statistics for one Web dataset."""
    dataset_root = ROOT / dataset
    position_counts = {
        bucket: {"n": 0, "cited": 0} for bucket in POSITION_BUCKETS
    }
    recency_counts = {
        bucket: {"n": 0, "cited": 0} for bucket in RECENCY_BUCKETS
    }
    ref_type_urls: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"types": set(), "cited": False}
    )
    position_auc_by_query: dict[str, list[float]] = defaultdict(list)
    recency_auc_by_query: dict[str, list[float]] = defaultdict(list)
    citation_jaccard_by_intent: dict[str, list[float]] = defaultdict(list)
    recency_ages = {"cited": [], "uncited": []}
    news_instances = 0
    n_runs = 0
    n_questions = 0

    for subset, qid, files in iter_questions(dataset_root):
        n_questions += 1
        question_key = f"{subset}/{qid}"
        citation_sets: list[set[str]] = []

        for path in files:
            n_runs += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            collected_at = run_datetime(path)
            run_positions: list[tuple[float, bool]] = []
            run_recencies: list[tuple[float, bool]] = []
            cited_urls: set[str] = set()
            seen_ref_urls: set[str] = set()

            for item in payload.get("all_urls") or []:
                normalized = deep_normalize(item.get("url") or "")
                if not normalized:
                    continue
                cited = bool(item.get("cited"))
                if cited:
                    cited_urls.add(normalized)

                ref_type = str(item.get("ref_type") or "")
                if normalized not in seen_ref_urls:
                    seen_ref_urls.add(normalized)
                    key = (subset, qid, normalized)
                    if ref_type:
                        ref_type_urls[key]["types"].add(ref_type)
                    ref_type_urls[key]["cited"] |= cited

                if not item.get("is_search_candidate"):
                    continue

                bucket = position_bucket(item.get("ref_index"))
                if bucket is not None:
                    position_counts[bucket]["n"] += 1
                    position_counts[bucket]["cited"] += int(cited)
                    run_positions.append((-float(item["ref_index"]), cited))

                pub_date = item.get("pub_date")
                if ref_type != "news":
                    continue
                news_instances += 1
                if not isinstance(pub_date, (int, float)):
                    continue
                if pub_date <= 0:
                    continue
                published_at = datetime.fromtimestamp(pub_date, tz=timezone.utc)
                age_days = max(
                    0.0, (collected_at - published_at).total_seconds() / 86400.0
                )
                bucket = recency_bucket(age_days)
                recency_counts[bucket]["n"] += 1
                recency_counts[bucket]["cited"] += int(cited)
                recency_ages["cited" if cited else "uncited"].append(age_days)
                run_recencies.append((-math.log1p(age_days), cited))

            citation_sets.append(cited_urls)
            position_auc = auc_probability(run_positions)
            if position_auc is not None:
                position_auc_by_query[question_key].append(position_auc)
            recency_auc = auc_probability(run_recencies)
            if recency_auc is not None:
                recency_auc_by_query[question_key].append(recency_auc)

        pairwise = [
            jaccard(left, right) for left, right in combinations(citation_sets, 2)
        ]
        if pairwise:
            intent = INTENT_LABELS.get(subset, subset)
            citation_jaccard_by_intent[intent].append(fmean(pairwise))

    position_query_means = [
        fmean(values) for values in position_auc_by_query.values() if values
    ]
    recency_query_means = [
        fmean(values) for values in recency_auc_by_query.values() if values
    ]

    primary_order = ("news", "search", "academia", "reddit")
    ref_type_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "ever_cited": 0}
    )
    for item in ref_type_urls.values():
        if not item["types"]:
            continue
        primary = next(
            (ref_type for ref_type in primary_order if ref_type in item["types"]),
            sorted(item["types"])[0],
        )
        if primary == "supporting":
            continue
        ref_type_counts[primary]["n"] += 1
        ref_type_counts[primary]["ever_cited"] += int(item["cited"])

    cited_news_n = sum(
        values["cited"] for values in recency_counts.values()
    )
    uncited_news_n = sum(
        values["n"] - values["cited"] for values in recency_counts.values()
    )
    return {
        "dataset": dataset,
        "n_questions": n_questions,
        "n_runs": n_runs,
        "position": {
            "buckets": [
                {
                    "bucket": bucket,
                    **position_counts[bucket],
                    "cite_rate": (
                        position_counts[bucket]["cited"]
                        / position_counts[bucket]["n"]
                    ),
                }
                for bucket in POSITION_BUCKETS
            ],
            "within_run_auc_query_macro": {
                "n_queries": len(position_query_means),
                **bootstrap_mean(position_query_means),
            },
            "per_query_auc": {
                key: fmean(values)
                for key, values in sorted(position_auc_by_query.items())
                if values
            },
        },
        "publication_recency_news": {
            "all_news_instances": news_instances,
            "dated_instances": cited_news_n + uncited_news_n,
            "cited_instances": cited_news_n,
            "uncited_instances": uncited_news_n,
            "buckets": [
                {
                    "bucket": bucket,
                    **recency_counts[bucket],
                    "cite_rate": (
                        recency_counts[bucket]["cited"]
                        / recency_counts[bucket]["n"]
                        if recency_counts[bucket]["n"]
                        else math.nan
                    ),
                    "share_of_cited": (
                        recency_counts[bucket]["cited"] / cited_news_n
                        if cited_news_n
                        else math.nan
                    ),
                    "share_of_uncited": (
                        (
                            recency_counts[bucket]["n"]
                            - recency_counts[bucket]["cited"]
                        )
                        / uncited_news_n
                        if uncited_news_n
                        else math.nan
                    ),
                }
                for bucket in RECENCY_BUCKETS
            ],
            "median_age_days": {
                label: median(values) if values else math.nan
                for label, values in recency_ages.items()
            },
            "within_run_auc_query_macro": {
                "n_queries": len(recency_query_means),
                **bootstrap_mean(recency_query_means),
            },
            "per_query_auc": {
                key: fmean(values)
                for key, values in sorted(recency_auc_by_query.items())
                if values
            },
        },
        "reference_type": {
            "query_specific_unique_urls": sum(
                values["n"] for values in ref_type_counts.values()
            ),
            "types": [
                {
                    "ref_type": ref_type,
                    **values,
                    "ever_cited_rate": values["ever_cited"] / values["n"],
                }
                for ref_type, values in sorted(
                    ref_type_counts.items(),
                    key=lambda pair: (-pair[1]["n"], pair[0]),
                )
            ],
        },
        "citation_set_stability": {
            "level": "URL",
            "intent_query_macro_pairwise_jaccard": {
                intent: {
                    "n_queries": len(values),
                    **bootstrap_mean(values),
                }
                for intent, values in citation_jaccard_by_intent.items()
            },
        },
    }


def rounded(value):
    """Round floats recursively while preserving JSON structure."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "chapter_code/generated/other_candidate_signals.json",
    )
    args = parser.parse_args()

    results = {
        dataset: analyze_dataset(dataset)
        for dataset in args.datasets
    }
    if len(args.datasets) == 2:
        earlier = results[args.datasets[0]]
        later = results[args.datasets[1]]
        results["cross_model_later_minus_earlier"] = {
            "position_auc": paired_difference(
                earlier["position"]["per_query_auc"],
                later["position"]["per_query_auc"],
            ),
            "publication_recency_auc": paired_difference(
                earlier["publication_recency_news"]["per_query_auc"],
                later["publication_recency_news"]["per_query_auc"],
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rounded(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(rounded(results), ensure_ascii=False, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
