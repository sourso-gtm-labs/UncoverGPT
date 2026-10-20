#!/usr/bin/env python3
"""Normalize GPT-5.3 Responses API captures without changing their raw payloads.

The retained historical component stores both derived fields and
``response_raw``.  The newer 820-run Nuwa pool deliberately stores a raw-only
envelope.  This module treats ``response_raw`` as authoritative and produces
one shared output schema for both collection generations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
from typing import Any


SUBDATASETS = (
    "v9_analysis",
    "v9_factual_lookup",
    "v9_info_synthesis",
    "v9_recommendation",
)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def extract_response_raw(response_raw: dict[str, Any]) -> dict[str, Any]:
    """Extract answer, search, and citation fields from one raw response."""

    text = ""
    citations: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    search_results: list[dict[str, Any]] = []
    search_queries: list[str] = []
    action_types: Counter[str] = Counter()
    web_search_call_count = 0

    output = response_raw.get("output")
    if not isinstance(output, list):
        output = []

    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "web_search_call":
            web_search_call_count += 1
            action = item.get("action")
            action = action if isinstance(action, dict) else {}
            action_type = str(action.get("type") or "unknown")
            action_types[action_type] += 1

            # Modern Responses payloads expose every query issued in
            # ``action.queries`` while retaining ``action.query`` as the
            # first/legacy query. Prefer the complete ordered list and only
            # fall back to the singular field for older payloads.
            raw_queries = action.get("queries")
            action_queries = (
                [
                    query
                    for query in raw_queries
                    if isinstance(query, str) and query
                ]
                if isinstance(raw_queries, list)
                else []
            )
            if not action_queries:
                query = action.get("query")
                if isinstance(query, str) and query:
                    action_queries.append(query)
            search_queries.extend(action_queries)

            raw_sources = action.get("sources")
            if isinstance(raw_sources, list):
                sources.extend(source for source in raw_sources if isinstance(source, dict))

            raw_results = item.get("results")
            if isinstance(raw_results, list):
                for rank, result in enumerate(raw_results):
                    if not isinstance(result, dict):
                        continue
                    search_results.append(
                        {
                            "rank": rank,
                            "search_call_index": web_search_call_count - 1,
                            "url": result.get("url"),
                            "title": result.get("title"),
                            "snippet": result.get("snippet"),
                            "type": result.get("type"),
                        }
                    )

        if item_type != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "output_text":
                continue
            block_text = block.get("text")
            if isinstance(block_text, str) and block_text:
                text = block_text
            annotations = block.get("annotations")
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                    continue
                nested = annotation.get("url_citation")
                citation = nested if isinstance(nested, dict) else annotation
                citations.append(
                    {
                        "title": citation.get("title"),
                        "url": citation.get("url"),
                        "start_index": _first_present(citation, "start_index", "start"),
                        "end_index": _first_present(citation, "end_index", "end"),
                    }
                )

    reasoning = response_raw.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    return {
        "text": text,
        "search_queries": search_queries,
        "sources": sources,
        "search_results": search_results,
        "citations": citations,
        "n_sources": len(sources),
        "n_search_results": len(search_results),
        "n_citations": len(citations),
        "web_search_call_count": web_search_call_count,
        "web_search_action_types": dict(sorted(action_types.items())),
        "response_status": response_raw.get("status"),
        "returned_model": response_raw.get("model"),
        "reasoning_effort": reasoning.get("effort"),
        "results_complete": len(search_results) >= len(sources),
    }


def normalize_payload(
    payload: dict[str, Any],
    *,
    query_meta: dict[str, Any] | None = None,
    source_relative_path: str = "",
) -> dict[str, Any]:
    """Build the common parsed envelope while retaining ``response_raw``."""

    response_raw = payload.get("response_raw")
    if not isinstance(response_raw, dict):
        raise ValueError("payload has no response_raw object")
    extracted = extract_response_raw(response_raw)

    source_meta = payload.get("_meta")
    source_meta = dict(source_meta) if isinstance(source_meta, dict) else {}
    meta = dict(source_meta)
    if "results_complete" in source_meta:
        meta["collection_results_complete"] = source_meta["results_complete"]
    meta.update(
        {
            "parse_schema_version": 1,
            "parsed": True,
            "source_raw_only": bool(source_meta.get("raw_only", False)),
            "raw_only": False,
            "source_relative_path": source_relative_path,
            "model": source_meta.get("model")
            or source_meta.get("requested_model")
            or extracted["returned_model"],
            "returned_model": source_meta.get("returned_model")
            or extracted["returned_model"],
            "reasoning_effort": source_meta.get("reasoning_effort")
            or extracted["reasoning_effort"],
            "response_status": extracted["response_status"],
            "results_complete": extracted["results_complete"],
        }
    )

    query_meta = query_meta or {}
    normalized = {
        "_meta": meta,
        "query": payload.get("query") or query_meta.get("query", ""),
        "language": payload.get("language") or query_meta.get("language", ""),
        "primary_intent": payload.get("primary_intent")
        or query_meta.get("primary_intent", ""),
        **extracted,
        "response_raw": response_raw,
    }
    return normalized


def _load_query_maps(dataset_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for subset in SUBDATASETS:
        path = dataset_root / subset / "query_list.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        queries = payload.get("queries", []) if isinstance(payload, dict) else []
        maps[subset] = {
            str(query.get("id", "")).zfill(3): query
            for query in queries
            if isinstance(query, dict)
        }
    return maps


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def parse_pool(
    source_dir: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path,
    *,
    write: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Parse a merged raw pool and return an auditable summary."""

    source = Path(source_dir)
    output = Path(output_dir)
    root = Path(dataset_root)
    query_maps = _load_query_maps(root)
    files = sorted(source.glob("v9_*/q*/run_*.json"))
    audit: dict[str, Any] = {
        "source_dir": str(source),
        "output_dir": str(output),
        "files": len(files),
        "outputs_written": 0,
        "json_errors": 0,
        "binding_errors": 0,
        "completed_responses": 0,
        "incomplete_responses": 0,
        "empty_text": 0,
        "empty_search_queries": 0,
        "empty_sources": 0,
        "results_incomplete": 0,
        "models": Counter(),
        "reasoning_efforts": Counter(),
        "total_search_queries": 0,
        "total_sources": 0,
        "total_search_results": 0,
        "total_citations": 0,
    }

    for path in files:
        relative = path.relative_to(source)
        if len(relative.parts) != 3:
            continue
        subset, qdir, _ = relative.parts
        base_qid = qdir.removeprefix("q").split("_", 1)[0].zfill(3)
        query_meta = query_maps.get(subset, {}).get(base_qid)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level payload is not an object")
            normalized = normalize_payload(
                payload,
                query_meta=query_meta,
                source_relative_path=str(relative),
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            audit["json_errors"] += 1
            if strict:
                raise ValueError(f"cannot parse {path}: {error}") from error
            continue

        meta = normalized["_meta"]
        binding_ok = bool(query_meta) and normalized["query"] == query_meta.get("query")
        binding_ok = binding_ok and str(meta.get("qid", qdir)) == qdir
        binding_ok = binding_ok and str(meta.get("subdataset", subset)) == subset
        if not binding_ok:
            audit["binding_errors"] += 1
            if strict:
                raise ValueError(f"query binding mismatch: {path}")

        completed = normalized["response_status"] == "completed"
        audit["completed_responses" if completed else "incomplete_responses"] += 1
        audit["empty_text"] += int(not normalized["text"])
        audit["empty_search_queries"] += int(not normalized["search_queries"])
        audit["empty_sources"] += int(not normalized["sources"])
        audit["results_incomplete"] += int(not normalized["results_complete"])
        audit["models"][str(normalized["returned_model"] or "")] += 1
        audit["reasoning_efforts"][str(normalized["reasoning_effort"] or "")] += 1
        audit["total_search_queries"] += len(normalized["search_queries"])
        audit["total_sources"] += normalized["n_sources"]
        audit["total_search_results"] += normalized["n_search_results"]
        audit["total_citations"] += normalized["n_citations"]

        if strict and (
            not normalized["text"]
            or not normalized["search_queries"]
            or not normalized["sources"]
        ):
            raise ValueError(f"missing required parsed content: {path}")
        if write:
            _atomic_write_json(output / relative, normalized)
            audit["outputs_written"] += 1

    audit["models"] = dict(sorted(audit["models"].items()))
    audit["reasoning_efforts"] = dict(sorted(audit["reasoning_efforts"].items()))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    audit = parse_pool(
        args.source_dir,
        args.output_dir,
        args.dataset_root,
        write=args.write,
        strict=args.strict,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
