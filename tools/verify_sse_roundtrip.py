#!/usr/bin/env python3
"""Reparse every sanitized SSE trace and compare analysis-relevant fields."""

from __future__ import annotations

import json
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

from export_processed_data import DATASETS, load_json, sanitize_web


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_thinking_parser():
    path = ROOT / "dataset-5.4-thinking" / "parse_sse.py"
    spec = importlib.util.spec_from_file_location("thinking_sse_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_thinking_subset(source: Path) -> int:
    parser = load_thinking_parser()
    query_payload = load_json(source / "query_list.json")
    queries = {
        str(item.get("id", "")).zfill(3): item.get("query", "")
        for item in query_payload.get("queries", [])
        if isinstance(item, dict)
    }
    checked = 0
    for path in sorted((source / "geo-web-crawl" / "chatgpt").glob("q*/*.txt")):
        qid = path.parent.name.removeprefix("q").zfill(3)
        stem_parts = path.stem.split("_done_", 1)
        timestamp = stem_parts[1] if len(stem_parts) == 2 else path.stem
        run_id = timestamp[5:] if len(timestamp) > 5 else timestamp
        connection = re.search(r"conn(\d+)", path.name)
        result = parser.extract_from_sse(path)
        result.update(
            {
                "query_id": qid,
                "conn_id": connection.group(1) if connection else "",
                "run_id": run_id,
                "original_query": queries.get(qid, ""),
            }
        )
        relative = Path(f"q{qid}/q{qid}_{run_id}_detail.json")
        expected = load_json(source / "search_detail" / relative)
        actual = sanitize_web(result)
        if actual != expected:
            raise RuntimeError(f"Round-trip mismatch: {source.name}/{relative}")
        checked += 1
    return checked


def main() -> None:
    from scripts.web_sse_parser import parse_subset

    checked = 0
    with tempfile.TemporaryDirectory(prefix="sse-roundtrip-") as directory:
        temporary_root = Path(directory)
        for dataset, config in DATASETS.items():
            for subset in config["subsets"]:
                source = ROOT / dataset / subset
                if dataset == "dataset-5.4-thinking":
                    checked += verify_thinking_subset(source)
                    continue
                output = temporary_root / dataset / subset
                audit = parse_subset(
                    source_dir=source,
                    output_dir=output,
                    query_list_path=source / "query_list.json",
                    write=True,
                    strict=True,
                    allow_mixed_output=False,
                )
                if audit["files"] != audit["outputs_written"]:
                    raise RuntimeError(
                        f"Incomplete round trip for {dataset}/{subset}: {audit}"
                    )
                for generated in sorted(output.glob("q*/*_detail.json")):
                    relative = generated.relative_to(output)
                    expected = load_json(source / "search_detail" / relative)
                    actual = sanitize_web(load_json(generated))
                    if actual != expected:
                        raise RuntimeError(
                            "Round-trip mismatch: "
                            f"{dataset}/{subset}/{relative.as_posix()}"
                        )
                    checked += 1
    print(json.dumps({"verified_sse_runs": checked}, indent=2))


if __name__ == "__main__":
    main()
