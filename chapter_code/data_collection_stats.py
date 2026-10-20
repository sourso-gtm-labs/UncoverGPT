#!/usr/bin/env python3
"""Audit dataset coverage and generate the Data Collection table."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RUNS_PER_QUERY = 20


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    mode: str
    dataset_dir: str
    subsets: tuple[str, ...]
    expected_model_slugs: frozenset[str]
    target_queries: int
    publish_completed_counts: bool
    api_dir: str | None = None
    google_dir: str | None = None


SPECS = (
    DatasetSpec(
        key="gpt_5_2_instant",
        label="GPT-5.2 Instant",
        mode="Instant",
        dataset_dir="dataset-5.2",
        subsets=(
            "v9_analysis",
            "v9_factual_lookup",
            "v9_info_synthesis",
            "v9_recommendation",
        ),
        expected_model_slugs=frozenset({"gpt-5-2-instant"}),
        target_queries=32,
        publish_completed_counts=True,
        api_dir="v9_api_pool/api_responses",
        google_dir="v9_google_pool/google_pools",
    ),
    DatasetSpec(
        key="gpt_5_3_instant",
        label="GPT-5.3 Instant",
        mode="Instant",
        dataset_dir="dataset-5.3-1600",
        subsets=(
            "v9_analysis",
            "v9_factual_lookup",
            "v9_info_synthesis",
            "v9_recommendation",
        ),
        # The production collector accepts both server-side Instant slugs.
        expected_model_slugs=frozenset({"gpt-5-3", "gpt-5-3-instant"}),
        target_queries=80,
        publish_completed_counts=True,
        api_dir="v9_api_pool/api_responses_parsed",
        google_dir="v9_google_pool/google_pools",
    ),
    DatasetSpec(
        key="gpt_5_4_thinking",
        label="GPT-5.4 Thinking",
        mode="Thinking",
        dataset_dir="dataset-5.4-thinking",
        subsets=(
            "v10_analysis_5.4",
            "v10_factual_lookup_5.4",
            "v10_info_synthesis_5.4",
            "v10_recommendation_5.4",
        ),
        expected_model_slugs=frozenset(
            {"gpt-5-4-thinking", "gpt-5-4-auto-thinking"}
        ),
        target_queries=30,
        publish_completed_counts=True,
    ),
    DatasetSpec(
        key="gpt_5_5_instant",
        label="GPT-5.5 Instant",
        mode="Instant",
        dataset_dir="dataset-5.5-1600",
        subsets=(
            "v9_analysis",
            "v9_factual_lookup",
            "v9_info_synthesis",
            "v9_recommendation",
        ),
        expected_model_slugs=frozenset({"gpt-5-5", "gpt-5-5-instant"}),
        target_queries=80,
        publish_completed_counts=True,
        api_dir="v9_api_pool/api_responses_parsed",
        google_dir="v9_google_pool/google_pools",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def normalized_query_id(value: object) -> str:
    text = str(value)
    return text[1:] if text.startswith("q") else text


def is_complete_run(detail: dict[str, Any]) -> bool:
    stream = detail.get("stream_meta") or {}
    if "capture_complete" in stream:
        return stream.get("capture_complete") is True
    return (
        stream.get("saw_done") is True
        and stream.get("saw_message_stream_complete") is True
        and detail.get("final_answer_is_partial") is not True
        and stream.get("search_query_status") != "incomplete"
    )


def count_json_files(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def count_query_scoped_json_files(
    path: Path | None, query_keys: set[tuple[str, str]]
) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(
        count_json_files(path / subset / f"q{query_id}") or 0
        for subset, query_id in query_keys
    )


def audit_api_proxy(
    path: Path | None, query_keys: set[tuple[str, str]]
) -> dict[str, int | float] | None:
    """Count source entries and URL-citation annotations per API response."""
    if path is None or not path.exists():
        return None

    responses = 0
    candidate_entries = 0
    citation_annotations = 0
    responses_with_candidates = 0
    responses_with_citations = 0

    for subset, query_id in sorted(query_keys):
        response_paths = (path / subset / f"q{query_id}").glob("run_*.json")
        for response_path in sorted(response_paths):
            response = load_json(response_path)
            sources = [
                source
                for source in response.get("sources") or []
                if isinstance(source, dict) and str(source.get("url") or "").strip()
            ]
            citations = [
                citation
                for citation in response.get("citations") or []
                if isinstance(citation, dict)
                and str(citation.get("url") or "").strip()
            ]
            responses += 1
            candidate_entries += len(sources)
            citation_annotations += len(citations)
            responses_with_candidates += bool(sources)
            responses_with_citations += bool(citations)

    if not responses:
        return None
    return {
        "responses": responses,
        "candidate_source_entries": candidate_entries,
        "mean_candidate_source_entries_per_response": round(
            candidate_entries / responses, 4
        ),
        "citation_annotations": citation_annotations,
        "mean_citation_annotations_per_response": round(
            citation_annotations / responses, 4
        ),
        "responses_with_candidates": responses_with_candidates,
        "responses_with_citations": responses_with_citations,
    }


def audit_google_proxy(
    path: Path | None, query_keys: set[tuple[str, str]]
) -> dict[str, int | float] | None:
    """Count the organic-result entries returned by each cached query snapshot."""
    if path is None or not path.exists():
        return None

    result_counts: list[int] = []
    fetched_page_counts: list[int] = []
    for subset, query_id in sorted(query_keys):
        pool_path = path / f"{subset}__q{query_id}.json"
        if not pool_path.exists():
            continue
        pool = load_json(pool_path)
        pages = pool.get("pages") or []
        result_counts.append(
            sum(
                len(page.get("results") or [])
                for page in pages
                if isinstance(page, dict)
            )
        )
        fetched_page_counts.append(int(pool.get("pages_fetched") or len(pages)))

    if not result_counts:
        return None
    return {
        "query_snapshots": len(result_counts),
        "organic_result_entries": sum(result_counts),
        "mean_organic_result_entries_per_query": round(
            sum(result_counts) / len(result_counts), 4
        ),
        "min_organic_result_entries_per_query": min(result_counts),
        "max_organic_result_entries_per_query": max(result_counts),
        "mean_pages_fetched_per_query": round(
            sum(fetched_page_counts) / len(fetched_page_counts), 4
        ),
        "min_pages_fetched_per_query": min(fetched_page_counts),
        "max_pages_fetched_per_query": max(fetched_page_counts),
    }


def audit_dataset(repo_root: Path, spec: DatasetSpec) -> dict[str, Any]:
    dataset_root = repo_root / spec.dataset_dir
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    query_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    detail_paths: list[tuple[str, str, Path]] = []
    raw_sse_files = 0

    for subset_name in spec.subsets:
        subset = dataset_root / subset_name
        payload = load_json(subset / "query_list.json")
        queries = payload.get("queries")
        if not isinstance(queries, list):
            raise ValueError(f"Missing queries[]: {subset / 'query_list.json'}")
        for query in queries:
            if not isinstance(query, dict) or "id" not in query:
                raise ValueError(f"Malformed query entry: {subset / 'query_list.json'}")
            key = (subset_name, normalized_query_id(query["id"]))
            if key in query_metadata:
                raise ValueError(f"Duplicate query key: {key}")
            query_metadata[key] = query

        detail_root = subset / "search_detail"
        if detail_root.exists():
            for path in sorted(detail_root.glob("q*/*_detail.json")):
                query_id = normalized_query_id(path.parent.name)
                detail_paths.append((subset_name, query_id, path))

        raw_root = subset / "geo-web-crawl"
        if raw_root.exists():
            raw_sse_files += sum(1 for path in raw_root.rglob("*.txt") if path.is_file())

    runs_by_query: defaultdict[tuple[str, str], int] = defaultdict(int)
    model_slugs: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    totals: Counter[str] = Counter()

    for subset_name, query_id, path in detail_paths:
        query_key = (subset_name, query_id)
        if query_key not in query_metadata:
            invalid_reasons["query_missing_from_query_list"] += 1
            continue

        detail = load_json(path)
        slug = str((detail.get("server_meta") or {}).get("model_slug") or "")
        model_slugs[slug or "<missing>"] += 1
        if slug not in spec.expected_model_slugs:
            invalid_reasons["unexpected_model_slug"] += 1
            continue
        if not is_complete_run(detail):
            invalid_reasons["incomplete_stream"] += 1
            continue

        candidates = detail.get("total_candidate_urls")
        citations = detail.get("total_cited")
        search_queries = detail.get("search_query_count")
        if not all(isinstance(value, int) for value in (candidates, citations)):
            invalid_reasons["missing_url_totals"] += 1
            continue

        runs_by_query[query_key] += 1
        totals["valid_runs"] += 1
        totals["candidate_urls"] += candidates
        totals["citations"] += citations
        if isinstance(search_queries, int):
            totals["search_queries"] += search_queries
            totals["runs_with_search_query_count"] += 1

    effective_queries = set(runs_by_query)
    effective_metadata = [query_metadata[key] for key in sorted(effective_queries)]
    intents = sorted(
        {str(item.get("primary_intent")) for item in effective_metadata if item.get("primary_intent")}
    )
    languages = sorted(
        {str(item.get("language")) for item in effective_metadata if item.get("language")}
    )
    run_counts = sorted(runs_by_query.values())
    valid_runs = totals["valid_runs"]

    api_path = dataset_root / spec.api_dir if spec.api_dir else None
    google_path = dataset_root / spec.google_dir if spec.google_dir else None
    completed = (
        len(effective_queries) == spec.target_queries
        and valid_runs == spec.target_queries * RUNS_PER_QUERY
        and all(count == RUNS_PER_QUERY for count in run_counts)
    )

    def per_run(total_key: str, denominator: int | None = None) -> float | None:
        count = denominator if denominator is not None else valid_runs
        return round(totals[total_key] / count, 4) if count else None

    return {
        "label": spec.label,
        "mode": spec.mode,
        "dataset_dir": spec.dataset_dir,
        "target_queries": spec.target_queries,
        "declared_queries_in_lists": len(query_metadata),
        "effective_queries": len(effective_queries),
        "intent_count": len(intents),
        "intents": intents,
        "languages": languages,
        "runs_per_query_target": RUNS_PER_QUERY,
        "runs_per_query_observed_min": min(run_counts) if run_counts else None,
        "runs_per_query_observed_max": max(run_counts) if run_counts else None,
        "raw_sse_files": raw_sse_files,
        "parsed_detail_files": len(detail_paths),
        "valid_web_runs": valid_runs,
        "invalid_run_reasons": dict(sorted(invalid_reasons.items())),
        "model_slugs": dict(sorted(model_slugs.items())),
        "api_parsed_runs": count_query_scoped_json_files(api_path, effective_queries),
        "google_query_pools": count_json_files(google_path),
        "api_proxy_stats": audit_api_proxy(api_path, effective_queries),
        "google_proxy_stats": audit_google_proxy(google_path, effective_queries),
        "total_candidate_urls": totals["candidate_urls"],
        "mean_candidate_urls_per_run": per_run("candidate_urls"),
        "total_citations": totals["citations"],
        "mean_citations_per_run": per_run("citations"),
        "mean_search_queries_per_run": per_run(
            "search_queries", totals["runs_with_search_query_count"]
        ),
        "collection_complete": completed,
        "publish_completed_counts": spec.publish_completed_counts,
    }


def render_table(audits: dict[str, dict[str, Any]]) -> str:
    five_two = audits["gpt_5_2_instant"]
    five_three = audits["gpt_5_3_instant"]
    five_four = audits["gpt_5_4_thinking"]
    five_five = audits["gpt_5_5_instant"]

    if five_two["effective_queries"] != five_two["target_queries"]:
        raise ValueError("GPT-5.2 query coverage does not match its finalized cohort")
    if not five_three["collection_complete"]:
        raise ValueError("GPT-5.3 coverage is not complete; refusing to publish its row")
    if five_four["effective_queries"] != five_four["target_queries"]:
        raise ValueError("GPT-5.4 query coverage does not match its finalized cohort")
    if not five_five["collection_complete"]:
        raise ValueError("GPT-5.5 coverage is not complete; refusing to publish its row")

    def observed_runs(audit: dict[str, Any]) -> str:
        return (
            str(audit["runs_per_query_observed_min"])
            if audit["runs_per_query_observed_min"]
            == audit["runs_per_query_observed_max"]
            else (
                f"{audit['runs_per_query_observed_min']}"
                f"--{audit['runs_per_query_observed_max']}"
            )
        )

    five_two_runs = observed_runs(five_two)
    five_four_runs = observed_runs(five_four)

    return rf"""% Generated by chapter_code/data_collection_stats.py. Do not edit manually.
\begin{{table*}}[t]
    \centering
    {{\color{{blue}}
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \caption{{\textcolor{{blue}}{{Dataset coverage across the four ChatGPT versions. Runs/query reports the observed count or range; a dash indicates that the corresponding API proxy is unavailable.}}}}
    \label{{tab:dataset}}
    \begin{{tabular}}{{llrrrrr}}
        \toprule
        Model & Mode & \#Intents & \#Queries & Runs/query & Web runs & API runs \\
        \midrule
        \webtwo   & Instant  & {five_two['intent_count']} & {five_two['effective_queries']} & {five_two_runs} & {five_two['valid_web_runs']} & -- \\
        \webthree & Instant  & {five_three['intent_count']} & {five_three['effective_queries']} & {RUNS_PER_QUERY} & {five_three['valid_web_runs']} & {five_three['api_parsed_runs']} \\
        \webfour  & Thinking & {five_four['intent_count']} & {five_four['effective_queries']} & {five_four_runs} & {five_four['valid_web_runs']} & -- \\
        \webfive  & Instant  & {five_five['intent_count']} & {five_five['effective_queries']} & {RUNS_PER_QUERY} & {five_five['valid_web_runs']} & {five_five['api_parsed_runs']} \\
        \bottomrule
    \end{{tabular}}
    }}
\end{{table*}}
"""


def serialize_outputs(repo_root: Path) -> tuple[str, str]:
    audits = {spec.key: audit_dataset(repo_root, spec) for spec in SPECS}
    payload = {
        "runs_per_query": RUNS_PER_QUERY,
        "notes": {
            "gpt_5_2": (
                "GPT-5.2 reports the finalized observed cohort after excluding "
                "queries with only one to three stored runs. API runs are counted "
                "only for the same effective query keys."
            ),
            "gpt_5_4": (
                "GPT-5.4 was deprecated before every query reached the 20-run target. "
                "The paper reports the finalized observed cohort of 30 queries and 491 runs."
            ),
        },
        "datasets": audits,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return json_text, render_table(audits)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_repo_root = script.parents[1]
    default_output_dir = script.parent / "generated"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that generated files are current without rewriting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_text, table_text = serialize_outputs(args.repo_root.resolve())
    outputs = {
        args.output_dir / "data_collection_stats.json": json_text,
        args.output_dir / "data_collection_table.tex": table_text,
    }

    if args.check:
        stale = [path for path, text in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != text]
        if stale:
            names = ", ".join(str(path) for path in stale)
            raise SystemExit(f"Generated files are stale: {names}")
        print("Data Collection statistics and table are current.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
