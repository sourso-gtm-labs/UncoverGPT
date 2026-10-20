#!/usr/bin/env python3
"""Audit observable query rewriting and generate its overview table."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

from data_collection_stats import SPECS, DatasetSpec, is_complete_run, load_json


LATEX_MODELS = {
    "gpt_5_2_instant": r"\webtwo",
    "gpt_5_3_instant": r"\webthree",
    "gpt_5_4_thinking": r"\webfour",
    "gpt_5_5_instant": r"\webfive",
}


@dataclass(frozen=True)
class QueryRewritingStats:
    model: str
    dataset_dir: str
    total_runs: int
    observed_runs: int
    excluded_fallback_runs: int
    query_observation_rate: float
    single_query_runs: int
    multi_query_runs: int
    multi_query_rate: float
    mean_queries_per_run: float
    median_queries_per_run: float
    max_queries_per_run: int
    single_round_runs: int
    multi_round_runs: int
    multi_round_rate: float
    mean_search_rounds_per_run: float
    median_search_rounds_per_run: float
    max_search_rounds_per_run: int
    exact_original_runs: int
    exact_original_rate: float
    global_unique_queries: int


def normalize_query(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def audit_model(repo_root: Path, spec: DatasetSpec) -> QueryRewritingStats:
    root = repo_root / spec.dataset_dir
    query_counts: list[int] = []
    round_counts: list[int] = []
    exact_original = 0
    unique_queries: set[str] = set()
    total_runs = 0
    excluded_fallback_runs = 0

    for subset_name in spec.subsets:
        detail_root = root / subset_name / "search_detail"
        for path in sorted(detail_root.glob("q*/*_detail.json")):
            detail = load_json(path)
            slug = str((detail.get("server_meta") or {}).get("model_slug") or "")
            if slug not in spec.expected_model_slugs:
                raise ValueError(f"Unexpected model slug {slug!r}: {path}")
            if not is_complete_run(detail):
                raise ValueError(f"Incomplete run: {path}")

            total_runs += 1
            if spec.key == "gpt_5_4_thinking":
                queries = detail.get("search_queries")
            else:
                query_source = detail.get("search_query_source")
                if query_source == "search_code_fallback":
                    excluded_fallback_runs += 1
                    continue
                if query_source != "search_model_queries":
                    raise ValueError(f"Unexpected query source {query_source!r}: {path}")
                queries = detail.get("actual_search_queries")
                if queries != detail.get("search_queries"):
                    raise ValueError(f"Selected and actual queries differ: {path}")

            rounds = detail.get("search_rounds")
            if not isinstance(queries, list) or not queries:
                raise ValueError(f"Missing observable search queries: {path}")
            if not isinstance(rounds, list) or not rounds:
                raise ValueError(f"Missing search rounds: {path}")
            if detail.get("search_query_count") != len(queries):
                raise ValueError(f"search_query_count mismatch: {path}")
            if detail.get("search_round_count") != len(rounds):
                raise ValueError(f"search_round_count mismatch: {path}")

            original = normalize_query(detail.get("original_query"))
            normalized_queries = [normalize_query(query) for query in queries]
            if not original:
                raise ValueError(f"Missing original query: {path}")

            query_counts.append(len(queries))
            round_counts.append(len(rounds))
            exact_original += int(original in normalized_queries)
            unique_queries.update(
                str(query).strip() for query in queries if str(query).strip()
            )

    if not query_counts:
        raise ValueError(f"No valid runs found for {spec.label}")

    observed_runs = len(query_counts)
    return QueryRewritingStats(
        model=spec.label,
        dataset_dir=spec.dataset_dir,
        total_runs=total_runs,
        observed_runs=observed_runs,
        excluded_fallback_runs=excluded_fallback_runs,
        query_observation_rate=observed_runs / total_runs,
        single_query_runs=sum(count == 1 for count in query_counts),
        multi_query_runs=sum(count > 1 for count in query_counts),
        multi_query_rate=sum(count > 1 for count in query_counts) / observed_runs,
        mean_queries_per_run=fmean(query_counts),
        median_queries_per_run=float(median(query_counts)),
        max_queries_per_run=max(query_counts),
        single_round_runs=sum(count == 1 for count in round_counts),
        multi_round_runs=sum(count > 1 for count in round_counts),
        multi_round_rate=sum(count > 1 for count in round_counts) / observed_runs,
        mean_search_rounds_per_run=fmean(round_counts),
        median_search_rounds_per_run=float(median(round_counts)),
        max_search_rounds_per_run=max(round_counts),
        exact_original_runs=exact_original,
        exact_original_rate=exact_original / observed_runs,
        global_unique_queries=len(unique_queries),
    )


def render_table(stats: dict[str, QueryRewritingStats]) -> str:
    rows = []
    for key in LATEX_MODELS:
        item = stats[key]
        rows.append(
            "        "
            f"{LATEX_MODELS[key]} & {item.observed_runs}/{item.total_runs} "
            f"& {item.mean_queries_per_run:.2f} "
            f"& {item.multi_query_rate:.1%} "
            f"& {item.mean_search_rounds_per_run:.2f} "
            f"& {item.multi_round_rate:.1%} "
            f"& {item.exact_original_rate:.1%} \\\\"
        )

    table_rows = "\n".join(rows).replace("%", r"\%")
    return rf"""% Generated by chapter_code/query_rewriting_stats.py. Do not edit manually.
\begin{{table*}}[t]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \caption{{Query rewriting statistics across the four ChatGPT versions. Observed/All reports runs with recoverable executed search queries over all valid runs. Multi-query and multi-round report the percentages of observed runs with more than one query and retrieval round, respectively.}}
    \label{{tab:query-rewriting-overview}}
    \begin{{tabular}}{{lrrrrrr}}
        \toprule
        Model & Observed/All & Queries/run & Multi-query & Rounds/run & Multi-round & Exact original \\
        \midrule
{table_rows}
        \bottomrule
    \end{{tabular}}
\end{{table*}}
"""


def serialize_outputs(repo_root: Path) -> tuple[str, str]:
    stats = {spec.key: audit_model(repo_root, spec) for spec in SPECS}
    payload: dict[str, Any] = {
        "methodology": {
            "unit": "valid ChatGPT Web run",
            "query_fields": {
                "gpt_5_2_instant": "actual_search_queries from search_model_queries",
                "gpt_5_3_instant": "actual_search_queries from search_model_queries",
                "gpt_5_4_thinking": (
                    "search_queries from search_model_queries metadata or explicit "
                    "structured web.run requests"
                ),
                "gpt_5_5_instant": "actual_search_queries from search_model_queries",
            },
            "round_field": "search_rounds",
            "excluded_runs": (
                "runs with only a top-level search(...) code argument and no "
                "recoverable executed search queries"
            ),
            "multi_query": "more than one executed query in an observed run",
            "multi_round": "more than one retrieval round in an observed run",
            "exact_original": (
                "at least one executed query equals the original user query after "
                "lowercasing and whitespace normalization"
            ),
        },
        "models": {key: asdict(value) for key, value in stats.items()},
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return json_text, render_table(stats)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[1])
    parser.add_argument("--output-dir", type=Path, default=script.parent / "generated")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_text, table_text = serialize_outputs(args.repo_root.resolve())
    outputs = {
        args.output_dir / "query_rewriting_stats.json": json_text,
        args.output_dir / "query_rewriting_table.tex": table_text,
    }

    if args.check:
        stale = [
            path
            for path, text in outputs.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            raise SystemExit(
                "Generated query rewriting files are stale: "
                + ", ".join(str(path) for path in stale)
            )
        print("Query rewriting statistics and table are current.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
