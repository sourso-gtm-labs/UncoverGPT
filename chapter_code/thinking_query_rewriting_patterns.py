#!/usr/bin/env python3
"""Analyze iterative query rewriting in GPT-5.4 Thinking traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from data_collection_stats import SPECS, is_complete_run, load_json
from query_rewriting_stats import normalize_query


MODEL_KEY = "gpt_5_4_thinking"
WORD_RE = re.compile(r"[a-z0-9]+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]+")
QUOTED_RE = re.compile(r"""["'][^"']+["']""")
TIME_RE = re.compile(
    r"\b(?:20\d{2}|today|latest|current|recent|now|this year)\b",
    flags=re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "at",
    "be",
    "best",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "latest",
    "me",
    "of",
    "on",
    "or",
    "site",
    "that",
    "the",
    "this",
    "to",
    "today",
    "vs",
    "what",
    "which",
    "with",
}


def query_text(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("query") or "").strip()
    return str(item or "").strip()


def query_domains(item: object) -> list[str]:
    if not isinstance(item, dict):
        return []
    domains = item.get("domains")
    if not isinstance(domains, list):
        return []
    return [str(domain).strip() for domain in domains if str(domain).strip()]


def query_tokens(value: str) -> set[str]:
    normalized = normalize_query(value)
    tokens = {
        token for token in WORD_RE.findall(normalized) if token not in STOPWORDS
    }
    for span in HAN_RE.findall(normalized):
        if len(span) == 1:
            tokens.add(span)
        else:
            tokens.update(span[index : index + 2] for index in range(len(span) - 1))
    return tokens


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def is_site_directed(value: str) -> bool:
    return "site:" in value.lower()


def has_quoted_phrase(value: str) -> bool:
    return QUOTED_RE.search(value) is not None


def has_time_constraint(value: str) -> bool:
    return TIME_RE.search(value) is not None


def round_items(search_round: dict[str, Any]) -> list[dict[str, Any]]:
    items = search_round.get("queries")
    if not isinstance(items, list) or not items:
        raise ValueError("Missing round queries")
    records: list[dict[str, Any]] = []
    for item in items:
        text = query_text(item)
        if not text:
            raise ValueError("Empty query text")
        records.append(
            {
                "query": text,
                "domains": query_domains(item),
                "site_directed": is_site_directed(text),
                "quoted_phrase": has_quoted_phrase(text),
                "time_constraint": has_time_constraint(text),
                "tokens": query_tokens(text),
            }
        )
    return records


def load_runs(repo_root: Path) -> list[dict[str, Any]]:
    spec = next(spec for spec in SPECS if spec.key == MODEL_KEY)
    dataset_root = repo_root / spec.dataset_dir
    runs: list[dict[str, Any]] = []

    for subset_name in spec.subsets:
        detail_root = dataset_root / subset_name / "search_detail"
        for path in sorted(detail_root.glob("q*/*_detail.json")):
            detail = load_json(path)
            slug = str((detail.get("server_meta") or {}).get("model_slug") or "")
            if slug not in spec.expected_model_slugs:
                raise ValueError(f"Unexpected model slug {slug!r}: {path}")
            if not is_complete_run(detail):
                raise ValueError(f"Incomplete run: {path}")

            raw_rounds = detail.get("search_rounds")
            if not isinstance(raw_rounds, list) or not raw_rounds:
                raise ValueError(f"Missing search rounds: {path}")
            rounds = [round_items(search_round) for search_round in raw_rounds]
            flat_queries = [item["query"] for search_round in rounds for item in search_round]
            if len(flat_queries) != detail.get("search_query_count"):
                raise ValueError(f"search_query_count mismatch: {path}")
            if len(rounds) != detail.get("search_round_count"):
                raise ValueError(f"search_round_count mismatch: {path}")

            original = str(detail.get("original_query") or "").strip()
            if not original:
                raise ValueError(f"Missing original query: {path}")
            runs.append(
                {
                    "subset": subset_name,
                    "query_id": path.parent.name,
                    "path": str(path.relative_to(repo_root)),
                    "original_query": original,
                    "rounds": rounds,
                }
            )
    return runs


def compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "subset": run["subset"],
        "query_id": run["query_id"],
        "source_path": run["path"],
        "original_query": run["original_query"],
        "round_count": len(run["rounds"]),
        "query_count": sum(len(search_round) for search_round in run["rounds"]),
        "rounds": [
            [item["query"] for item in search_round] for search_round in run["rounds"]
        ],
    }


def analyze_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    round_counts: list[int] = []
    query_counts: list[int] = []
    per_round_query_counts: list[int] = []
    first_queries: list[dict[str, Any]] = []
    later_queries: list[dict[str, Any]] = []
    later_overlaps: list[float] = []
    repeated_later_queries = 0
    exact_original_runs = 0
    round_position_query_counts: defaultdict[int, list[int]] = defaultdict(list)
    round_distribution: Counter[int] = Counter()
    query_distribution: Counter[int] = Counter()
    subset_round_counts: defaultdict[str, list[int]] = defaultdict(list)
    per_query_stats: defaultdict[
        tuple[str, str, str], list[tuple[int, int]]
    ] = defaultdict(list)
    decomposition_candidates: list[tuple[float, dict[str, Any]]] = []
    refinement_candidates: list[tuple[float, dict[str, Any]]] = []
    balanced_candidates: list[tuple[float, dict[str, Any]]] = []

    for run in runs:
        rounds = run["rounds"]
        original = normalize_query(run["original_query"])
        flat = [item for search_round in rounds for item in search_round]
        normalized_flat = [normalize_query(item["query"]) for item in flat]
        exact_original_runs += int(original in normalized_flat)

        round_count = len(rounds)
        query_count = len(flat)
        round_counts.append(round_count)
        query_counts.append(query_count)
        round_distribution[round_count] += 1
        query_distribution[query_count] += 1
        subset_round_counts[run["subset"]].append(round_count)
        per_query_stats[
            (run["subset"], run["query_id"], run["original_query"])
        ].append((round_count, query_count))
        first_queries.extend(rounds[0])

        prior_queries: list[dict[str, Any]] = []
        run_later_overlaps: list[float] = []
        run_low_overlap = 0
        run_high_overlap_constrained = 0
        run_new_constraints = 0
        for round_index, search_round in enumerate(rounds):
            per_round_query_counts.append(len(search_round))
            round_position_query_counts[round_index].append(len(search_round))
            if round_index > 0:
                later_queries.extend(search_round)
                prior_normalized = {
                    normalize_query(item["query"]) for item in prior_queries
                }
                prior_has_site = any(item["site_directed"] for item in prior_queries)
                prior_has_quote = any(item["quoted_phrase"] for item in prior_queries)
                prior_has_time = any(item["time_constraint"] for item in prior_queries)
                for item in search_round:
                    normalized = normalize_query(item["query"])
                    repeated_later_queries += int(normalized in prior_normalized)
                    similarities = [
                        jaccard(item["tokens"], prior["tokens"])
                        for prior in prior_queries
                    ]
                    max_overlap = max(similarities, default=0.0)
                    later_overlaps.append(max_overlap)
                    run_later_overlaps.append(max_overlap)
                    run_low_overlap += int(max_overlap <= 0.25)
                    added_constraint = (
                        (item["site_directed"] and not prior_has_site)
                        or (item["quoted_phrase"] and not prior_has_quote)
                        or (item["time_constraint"] and not prior_has_time)
                    )
                    run_new_constraints += int(added_constraint)
                    run_high_overlap_constrained += int(
                        max_overlap >= 0.40
                        and (
                            item["site_directed"]
                            or item["quoted_phrase"]
                            or item["time_constraint"]
                        )
                    )
            prior_queries.extend(search_round)

        if round_count > 1:
            later_count = sum(len(search_round) for search_round in rounds[1:])
            mean_overlap = fmean(run_later_overlaps) if run_later_overlaps else 0.0
            decomposition_score = (
                run_low_overlap / later_count
                + min(round_count, 8) / 8
                + min(later_count, 24) / 24
            )
            refinement_score = (
                run_high_overlap_constrained / later_count
                + run_new_constraints / later_count
                + mean_overlap
                + min(round_count, 8) / 16
            )
            decomposition_candidates.append((decomposition_score, run))
            refinement_candidates.append((refinement_score, run))
            if (
                3 <= round_count <= 5
                and HAN_RE.search(run["original_query"]) is None
            ):
                balanced_candidates.append(
                    (decomposition_score + refinement_score, run)
                )

    def query_feature_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        return {
            "queries": total,
            "site_directed_queries": sum(item["site_directed"] for item in items),
            "site_directed_rate": (
                sum(item["site_directed"] for item in items) / total if total else 0.0
            ),
            "queries_with_domain_filter": sum(bool(item["domains"]) for item in items),
            "domain_filter_rate": (
                sum(bool(item["domains"]) for item in items) / total if total else 0.0
            ),
            "quoted_phrase_queries": sum(item["quoted_phrase"] for item in items),
            "quoted_phrase_rate": (
                sum(item["quoted_phrase"] for item in items) / total if total else 0.0
            ),
            "time_constrained_queries": sum(item["time_constraint"] for item in items),
            "time_constraint_rate": (
                sum(item["time_constraint"] for item in items) / total if total else 0.0
            ),
        }

    decomposition_candidates.sort(key=lambda item: (-item[0], item[1]["path"]))
    refinement_candidates.sort(key=lambda item: (-item[0], item[1]["path"]))
    balanced_candidates.sort(key=lambda item: (-item[0], item[1]["path"]))

    return {
        "methodology": {
            "unit": "complete GPT-5.4 Thinking ChatGPT Web run",
            "round_definition": (
                "one structured search_rounds entry; multiple queries in the same "
                "entry belong to one retrieval round"
            ),
            "exact_original": (
                "at least one issued query equals the original user query after "
                "lowercasing and whitespace normalization"
            ),
            "later_query_overlap": (
                "maximum token-set Jaccard similarity between a query in round "
                "r>1 and any query in an earlier round; used only to rank runs for "
                "manual case selection"
            ),
            "site_directed": "query text contains site:",
            "candidate_scores": (
                "heuristics for case selection only; decomposition and refinement "
                "remain qualitative patterns verified from the selected traces"
            ),
        },
        "aggregate": {
            "runs": len(runs),
            "queries": sum(query_counts),
            "rounds": sum(round_counts),
            "mean_queries_per_run": fmean(query_counts),
            "median_queries_per_run": float(median(query_counts)),
            "mean_rounds_per_run": fmean(round_counts),
            "median_rounds_per_run": float(median(round_counts)),
            "multi_round_runs": sum(count > 1 for count in round_counts),
            "multi_round_rate": sum(count > 1 for count in round_counts) / len(runs),
            "exact_original_runs": exact_original_runs,
            "exact_original_rate": exact_original_runs / len(runs),
            "mean_queries_per_round": fmean(per_round_query_counts),
            "median_queries_per_round": float(median(per_round_query_counts)),
            "later_round_queries": len(later_queries),
            "repeated_later_queries": repeated_later_queries,
            "repeated_later_query_rate": (
                repeated_later_queries / len(later_queries) if later_queries else 0.0
            ),
            "mean_later_query_max_overlap": fmean(later_overlaps),
            "median_later_query_max_overlap": float(median(later_overlaps)),
        },
        "round_count_distribution": dict(sorted(round_distribution.items())),
        "query_count_distribution": dict(sorted(query_distribution.items())),
        "query_features": {
            "first_round": query_feature_summary(first_queries),
            "later_rounds": query_feature_summary(later_queries),
        },
        "round_position": {
            str(index + 1): {
                "runs_reaching_round": len(counts),
                "mean_queries_in_round": fmean(counts),
                "median_queries_in_round": float(median(counts)),
            }
            for index, counts in sorted(round_position_query_counts.items())
        },
        "by_subset": {
            subset: {
                "runs": len(counts),
                "mean_rounds_per_run": fmean(counts),
                "multi_round_rate": sum(count > 1 for count in counts) / len(counts),
            }
            for subset, counts in sorted(subset_round_counts.items())
        },
        "by_query": [
            {
                "subset": subset,
                "query_id": query_id,
                "original_query": original_query,
                "runs": len(values),
                "mean_rounds_per_run": fmean(value[0] for value in values),
                "multi_round_rate": (
                    sum(value[0] > 1 for value in values) / len(values)
                ),
                "mean_queries_per_run": fmean(value[1] for value in values),
                "max_rounds": max(value[0] for value in values),
            }
            for (subset, query_id, original_query), values in sorted(
                per_query_stats.items()
            )
        ],
        "case_candidates": {
            "decomposition": [
                {"selection_score": score, **compact_run(run)}
                for score, run in decomposition_candidates[:20]
            ],
            "refinement": [
                {"selection_score": score, **compact_run(run)}
                for score, run in refinement_candidates[:20]
            ],
            "balanced_english_3_to_5_rounds": [
                {"selection_score": score, **compact_run(run)}
                for score, run in balanced_candidates[:20]
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "generated" / "thinking_query_rewriting_patterns.json",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_runs(load_runs(args.repo_root.resolve()))
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"Generated file is stale: {args.output}")
        print("Thinking-mode query rewriting analysis is current.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
