#!/usr/bin/env python3
"""Analyze language patterns and representative query rewrites."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from data_collection_stats import SPECS, DatasetSpec, is_complete_run, load_json


INSTANT_MODEL_KEYS = (
    "gpt_5_2_instant",
    "gpt_5_3_instant",
    "gpt_5_5_instant",
)
EXAMPLE_SPECS = {
    "consolidated_simple_lookup": {
        "subset": "v9_recommendation",
        "query_id": "005",
        "criteria": {
            "gpt_5_2_instant": (2, 1, "monolingual_same_language"),
            "gpt_5_3_instant": (1, 1, "monolingual_same_language"),
            "gpt_5_5_instant": (1, 1, "monolingual_same_language"),
        },
    },
    "selective_multi_query_rewrite": {
        "subset": "v9_info_synthesis",
        "query_id": "004",
        "criteria": {
            "gpt_5_2_instant": (2, 1, "monolingual_same_language"),
            "gpt_5_3_instant": (1, 1, "monolingual_same_language"),
            "gpt_5_5_instant": (2, 1, "monolingual_same_language"),
        },
    },
}
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class RunRecord:
    model_key: str
    subset: str
    query_id: str
    original_query: str
    original_language: str
    search_queries: tuple[str, ...]
    search_round_count: int
    language_pattern: str
    source_path: str


def normalize_query_id(value: object) -> str:
    text = str(value)
    return text[1:] if text.startswith("q") else text


def query_language(query: str) -> str:
    """Classify the query language without treating Latin brand names as English."""
    if HAN_RE.search(query):
        return "Chinese"
    if LATIN_RE.search(query):
        return "English"
    return "Other"


def language_pattern(original_language: str, queries: tuple[str, ...]) -> str:
    languages = {query_language(query) for query in queries}
    languages.discard("Other")
    if {"Chinese", "English"}.issubset(languages):
        return "bilingual"
    if languages == {original_language}:
        return "monolingual_same_language"
    if len(languages) == 1:
        return "monolingual_translated"
    return "other"


def load_records(repo_root: Path, spec: DatasetSpec) -> list[RunRecord]:
    root = repo_root / spec.dataset_dir
    records: list[RunRecord] = []

    for subset_name in spec.subsets:
        query_list = load_json(root / subset_name / "query_list.json")
        metadata = {
            normalize_query_id(item["id"]): item
            for item in query_list.get("queries", [])
            if isinstance(item, dict) and "id" in item
        }
        detail_root = root / subset_name / "search_detail"
        for path in sorted(detail_root.glob("q*/*_detail.json")):
            detail = load_json(path)
            slug = str((detail.get("server_meta") or {}).get("model_slug") or "")
            if slug not in spec.expected_model_slugs:
                raise ValueError(f"Unexpected model slug {slug!r}: {path}")
            if not is_complete_run(detail):
                raise ValueError(f"Incomplete run: {path}")
            if detail.get("search_query_source") == "search_code_fallback":
                continue
            if detail.get("search_query_source") != "search_model_queries":
                raise ValueError(f"Unexpected query source in {path}")

            queries_value = detail.get("actual_search_queries")
            rounds = detail.get("search_rounds")
            if not isinstance(queries_value, list) or not queries_value:
                raise ValueError(f"Missing observable queries: {path}")
            if not isinstance(rounds, list) or not rounds:
                raise ValueError(f"Missing search rounds: {path}")
            queries = tuple(str(query).strip() for query in queries_value)

            query_id = normalize_query_id(path.parent.name)
            item = metadata.get(query_id)
            if item is None:
                raise ValueError(f"Missing query metadata for {path}")
            original = str(detail.get("original_query") or item.get("query") or "").strip()
            declared_language = str(item.get("language") or "")
            detected_language = query_language(original)
            if declared_language not in {"Chinese", "English"}:
                declared_language = detected_language
            if detected_language in {"Chinese", "English"} and (
                declared_language != detected_language
            ):
                raise ValueError(
                    f"Original-language mismatch: {declared_language} vs "
                    f"{detected_language}: {path}"
                )

            records.append(
                RunRecord(
                    model_key=spec.key,
                    subset=subset_name,
                    query_id=query_id,
                    original_query=original,
                    original_language=declared_language,
                    search_queries=queries,
                    search_round_count=len(rounds),
                    language_pattern=language_pattern(declared_language, queries),
                    source_path=str(path.relative_to(repo_root)),
                )
            )

    if not records:
        raise ValueError(f"No observed records for {spec.label}")
    return records


def summarize_group(records: list[RunRecord]) -> dict[str, Any]:
    patterns = Counter(record.language_pattern for record in records)
    total = len(records)
    return {
        "runs": total,
        "mean_queries_per_run": fmean(len(record.search_queries) for record in records),
        "multi_query_rate": (
            sum(len(record.search_queries) > 1 for record in records) / total
        ),
        "mean_rounds_per_run": fmean(record.search_round_count for record in records),
        "multi_round_rate": (
            sum(record.search_round_count > 1 for record in records) / total
        ),
        "pattern_counts": dict(sorted(patterns.items())),
        "pattern_rates": {
            key: value / total for key, value in sorted(patterns.items())
        },
    }


def summarize_model(records: list[RunRecord]) -> dict[str, Any]:
    by_language: defaultdict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_language[record.original_language].append(record)
    return {
        "overall": summarize_group(records),
        "by_original_language": {
            language: summarize_group(group)
            for language, group in sorted(by_language.items())
        },
    }


def summarize_queries(
    records_by_model: dict[str, list[RunRecord]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for model_key, records in records_by_model.items():
        for record in records:
            grouped[(model_key, record.subset, record.query_id)].append(record)

    shared_keys = set.intersection(
        *[
            {(record.subset, record.query_id) for record in records_by_model[key]}
            for key in INSTANT_MODEL_KEYS
        ]
    )
    summaries: list[dict[str, Any]] = []
    for subset, query_id in sorted(shared_keys):
        model_summaries: dict[str, Any] = {}
        originals: set[str] = set()
        for model_key in INSTANT_MODEL_KEYS:
            group = grouped[(model_key, subset, query_id)]
            originals.update(record.original_query for record in group)
            query_sets = Counter(record.search_queries for record in group)
            pattern_counts = Counter(record.language_pattern for record in group)
            model_summaries[model_key] = {
                "runs": len(group),
                "mean_queries_per_run": fmean(
                    len(record.search_queries) for record in group
                ),
                "multi_query_rate": (
                    sum(len(record.search_queries) > 1 for record in group) / len(group)
                ),
                "mean_rounds_per_run": fmean(
                    record.search_round_count for record in group
                ),
                "multi_round_rate": (
                    sum(record.search_round_count > 1 for record in group) / len(group)
                ),
                "language_pattern_counts": dict(sorted(pattern_counts.items())),
                "modal_query_sets": [
                    {
                        "count": count,
                        "queries": list(query_set),
                    }
                    for query_set, count in query_sets.most_common(3)
                ],
            }
        summaries.append(
            {
                "subset": subset,
                "query_id": query_id,
                "original_queries": sorted(originals),
                "models": model_summaries,
            }
        )
    return summaries


def select_representative_examples(
    records_by_model: dict[str, list[RunRecord]],
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for name, example_spec in EXAMPLE_SPECS.items():
        subset = str(example_spec["subset"])
        query_id = str(example_spec["query_id"])
        criteria = example_spec["criteria"]
        model_examples: dict[str, Any] = {}
        originals: set[str] = set()
        for model_key in INSTANT_MODEL_KEYS:
            group = [
                record
                for record in records_by_model[model_key]
                if record.subset == subset and record.query_id == query_id
            ]
            query_count, round_count, pattern = criteria[model_key]
            matching = [
                record
                for record in group
                if len(record.search_queries) == query_count
                and record.search_round_count == round_count
                and record.language_pattern == pattern
            ]
            if not matching:
                raise ValueError(
                    f"No representative run for {name}/{model_key}: "
                    f"{query_count} queries, {round_count} rounds, {pattern}"
                )
            record = sorted(matching, key=lambda item: item.source_path)[0]
            originals.add(record.original_query)
            model_examples[model_key] = {
                "support_runs": len(matching),
                "total_runs_for_query": len(group),
                "query_count": len(record.search_queries),
                "round_count": record.search_round_count,
                "language_pattern": record.language_pattern,
                "search_queries": list(record.search_queries),
                "source_path": record.source_path,
            }
        examples[name] = {
            "subset": subset,
            "query_id": query_id,
            "original_queries": sorted(originals),
            "models": model_examples,
        }
    return examples


def serialize_outputs(repo_root: Path) -> str:
    specs = {spec.key: spec for spec in SPECS}
    records_by_model = {
        key: load_records(repo_root, specs[key]) for key in INSTANT_MODEL_KEYS
    }
    payload = {
        "methodology": {
            "unit": "run with recoverable executed search_model_queries",
            "query_language": (
                "A query containing a Han character is Chinese; otherwise a query "
                "containing a Latin letter is English. This treats English brand "
                "names embedded in Chinese queries as part of a Chinese query."
            ),
            "monolingual_same_language": (
                "all issued queries use the original question's language"
            ),
            "monolingual_translated": (
                "all issued queries use the other analyzed language"
            ),
            "bilingual": "the run issues both Chinese and English queries",
            "fallback_policy": (
                "exclude runs without recoverable backend search_model_queries"
            ),
        },
        "models": {
            key: summarize_model(records)
            for key, records in records_by_model.items()
        },
        "shared_query_summaries": summarize_queries(records_by_model),
        "representative_examples": select_representative_examples(records_by_model),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "generated" / "query_rewriting_patterns.json",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    text = serialize_outputs(args.repo_root.resolve())
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"Generated query rewriting patterns are stale: {output}")
        print("Query rewriting pattern statistics are current.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
