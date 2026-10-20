#!/usr/bin/env python3
"""Export analysis data and de-identified SSE traces for anonymous review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


DATASETS = {
    "dataset-5.2": {
        "subsets": (
            "v9_analysis",
            "v9_factual_lookup",
            "v9_info_synthesis",
            "v9_recommendation",
        ),
        "proxies": False,
    },
    "dataset-5.3-1600": {
        "subsets": (
            "v9_analysis",
            "v9_factual_lookup",
            "v9_info_synthesis",
            "v9_recommendation",
        ),
        "proxies": True,
    },
    "dataset-5.4-thinking": {
        "subsets": (
            "v10_analysis_5.4",
            "v10_factual_lookup_5.4",
            "v10_info_synthesis_5.4",
            "v10_recommendation_5.4",
        ),
        "proxies": False,
    },
    "dataset-5.5-1600": {
        "subsets": (
            "v9_analysis",
            "v9_factual_lookup",
            "v9_info_synthesis",
            "v9_recommendation",
        ),
        "proxies": True,
    },
}

WEB_FIELDS = (
    "search_queries",
    "actual_search_queries",
    "raw_search_queries",
    "search_query_source",
    "search_rounds",
    "search_query_records",
    "search_round_count",
    "search_query_count",
    "actual_search_query_count",
    "raw_search_query_count",
    "unique_search_query_count",
    "final_answer_is_partial",
    "total_search_result_occurrences",
    "total_candidate_urls",
    "total_all_urls",
    "total_cited",
    "total_cited_main",
    "total_cited_supporting",
    "total_uncited",
    "total_supporting_websites",
    "total_sources_footnote",
    "total_streaming_reference_urls",
    "citation_source",
    "url_counts_by_type",
    "all_urls",
    "query_id",
    "run_id",
    "original_query",
)

API_FIELDS = (
    "query",
    "language",
    "primary_intent",
    "search_queries",
    "sources",
    "search_results",
    "citations",
    "n_sources",
    "n_search_results",
    "n_citations",
    "web_search_call_count",
    "web_search_action_types",
    "response_status",
    "returned_model",
    "reasoning_effort",
    "results_complete",
)

RESULT_FILES = (
    "hierarchical_frequency_1600run.json",
    "hierarchical_frequency_1600run.md",
    "prediction_experiment_1600run.json",
    "prediction_experiment_1600run.md",
    "prediction_overall_rrf_ablation_1600run.json",
    "prediction_overall_rrf_ablation_1600run.md",
    "proxy_effectiveness_1600run.json",
    "proxy_effectiveness_1600run.md",
)

ENTITY_RESULT_DIR = "entity_runlevel_official_gpt41mini_v1"

JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
TOKEN_KEYS = {
    "access_token",
    "authorization",
    "cookie",
    "refresh_token",
    "resume_token",
    "session_token",
    "set-cookie",
    "token",
}
IDENTIFIER_KEYS = {
    "account_id",
    "client_id",
    "conversation_id",
    "debug_sonic_thread_id",
    "id",
    "message_id",
    "otrace_id",
    "parent_id",
    "request_id",
    "response_id",
    "session_id",
    "turn_exchange_id",
    "turn_trace_id",
    "user_id",
    "working_turn_id",
}
PRIVATE_CONTENT_TYPES = {
    "developer_content",
    "model_editable_context",
    "reasoning_recap",
    "thoughts",
}


class SseSanitizer:
    """Remove session metadata while preserving event structure and references."""

    def __init__(self) -> None:
        self._identifier_map: dict[str, str] = {}
        self._private_content_stream = False

    def pseudonymize(self, value: str) -> str:
        if value not in self._identifier_map:
            self._identifier_map[value] = f"anon-id-{len(self._identifier_map) + 1:04d}"
        return self._identifier_map[value]

    @staticmethod
    def is_identifier(value: str) -> bool:
        return bool(
            UUID_RE.fullmatch(value)
            or re.fullmatch(r"[0-9a-fA-F]{24,40}", value)
            or value.startswith(("conversation-", "msg_", "resp_", "thread_"))
        )

    def sanitize(self, value: Any, parent_key: str = "") -> Any:
        if isinstance(value, dict):
            author = value.get("author")
            role = author.get("role") if isinstance(author, dict) else None
            content = value.get("content")
            content_type = (
                content.get("content_type") if isinstance(content, dict) else None
            )
            if role is not None and isinstance(content, dict):
                self._private_content_stream = bool(
                    role in {"system", "developer"}
                    or content_type in PRIVATE_CONTENT_TYPES
                )

            patch_path = value.get("p")
            if (
                self._private_content_stream
                and isinstance(patch_path, str)
                and patch_path.startswith("/message/content")
            ):
                return {
                    key: (
                        "[redacted-private-content-delta]"
                        if key == "v" and isinstance(item, str)
                        else self.sanitize(item, str(key).lower())
                    )
                    for key, item in value.items()
                }

            output: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in TOKEN_KEYS:
                    output[key] = "[redacted-token]"
                elif normalized == "useragent":
                    output[key] = {"client_type": "web"}
                elif (
                    normalized in IDENTIFIER_KEYS
                    and isinstance(item, str)
                    and self.is_identifier(item)
                ):
                    output[key] = self.pseudonymize(item)
                elif key == "content" and (
                    role in {"system", "developer"}
                    or content_type in PRIVATE_CONTENT_TYPES
                ):
                    output[key] = {
                        "content_type": "redacted",
                        "parts": ["[redacted-private-content]"],
                    }
                else:
                    output[key] = self.sanitize(item, normalized)
            return output
        if isinstance(value, list):
            return [self.sanitize(item, parent_key) for item in value]
        if isinstance(value, str):
            sanitized = JWT_RE.sub("[redacted-token]", value)
            if parent_key in IDENTIFIER_KEYS and self.is_identifier(sanitized):
                return self.pseudonymize(sanitized)
            return sanitized
        return value


def sanitize_sse_file(source: Path, destination: Path) -> None:
    """Write a structurally equivalent SSE stream without private metadata."""

    sanitizer = SseSanitizer()
    output: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            output.append(line)
            continue
        raw_payload = line[6:]
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            output.append("data: " + JWT_RE.sub("[redacted-token]", raw_payload))
            continue
        sanitized = sanitizer.sanitize(payload)
        output.append(
            "data: "
            + json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def sanitize_query_list(payload: dict[str, Any]) -> dict[str, Any]:
    fields = ("id", "source", "row_idx", "primary_intent", "language", "query")
    return {
        "queries": [
            {key: item[key] for key in fields if key in item}
            for item in payload.get("queries") or []
            if isinstance(item, dict)
        ]
    }


def sanitize_web(payload: dict[str, Any]) -> dict[str, Any]:
    output = {key: payload[key] for key in WEB_FIELDS if key in payload}
    server = payload.get("server_meta") or {}
    stream = payload.get("stream_meta") or {}
    output["server_meta"] = {"model_slug": server.get("model_slug")}
    output["stream_meta"] = {
        key: stream.get(key)
        for key in (
            "capture_complete",
            "saw_done",
            "saw_message_stream_complete",
            "search_query_status",
        )
        if key in stream
    }
    rounds = []
    for item in output.get("search_rounds") or []:
        if not isinstance(item, dict):
            continue
        rounds.append(
            {
                key: item[key]
                for key in ("queries", "source", "round_index", "query_count")
                if key in item
            }
        )
    output["search_rounds"] = rounds
    return output


def sanitize_api(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in API_FIELDS if key in payload}


def export_subset(source: Path, destination: Path) -> tuple[int, int, int]:
    query_list = source / "query_list.json"
    if not query_list.exists():
        raise FileNotFoundError(query_list)
    write_json(destination / "query_list.json", sanitize_query_list(load_json(query_list)))
    parser_entry = source / "extract_query_urls.py"
    if parser_entry.exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parser_entry, destination / parser_entry.name)

    written = 0
    for path in sorted((source / "search_detail").glob("q*/*_detail.json")):
        relative = path.relative_to(source)
        write_json(destination / relative, sanitize_web(load_json(path)))
        written += 1

    sse_written = 0
    raw_root = source / "geo-web-crawl" / "chatgpt"
    for path in sorted(raw_root.glob("q*/*.txt")):
        sanitize_sse_file(path, destination / path.relative_to(source))
        sse_written += 1
    return written, sse_written, len((destination / "query_list.json").read_bytes())


def export_proxies(source: Path, destination: Path) -> dict[str, int]:
    counts = {"api_runs": 0, "google_snapshots": 0, "entity_caches": 0}
    api_source = source / "v9_api_pool" / "api_responses_parsed"
    api_destination = destination / "v9_api_pool" / "api_responses_parsed"
    for path in sorted(api_source.glob("v9_*/q*/run_*.json")):
        write_json(api_destination / path.relative_to(api_source), sanitize_api(load_json(path)))
        counts["api_runs"] += 1

    google_source = source / "v9_google_pool" / "google_pools"
    google_destination = destination / "v9_google_pool" / "google_pools"
    for path in sorted(google_source.glob("*.json")):
        write_json(google_destination / path.name, load_json(path))
        counts["google_snapshots"] += 1

    entity_source = source / "v9_trend_analysis" / "results" / "entity_llm_cache"
    entity_destination = (
        destination / "v9_trend_analysis" / "results" / "entity_llm_cache"
    )
    for path in sorted(entity_source.glob("*.json")):
        write_json(entity_destination / path.name, load_json(path))
        counts["entity_caches"] += 1
    return counts


def export_results(source_root: Path, output_root: Path) -> dict[str, int]:
    source = source_root / "result"
    destination = output_root / "result"
    destination.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:
        shutil.copyfile(source / name, destination / name)

    entity_source = source / ENTITY_RESULT_DIR
    entity_destination = destination / ENTITY_RESULT_DIR
    for name in ("manifest.json", "results.json", "results.md"):
        shutil.copyfile(entity_source / name, entity_destination / name)

    counts = {"entity_api_batches": 0, "entity_query_maps": 0, "entity_run_sets": 0}
    for path in sorted((entity_source / "api_batches").rglob("*.json")):
        payload = load_json(path)
        sanitized = {
            key: payload[key]
            for key in (
                "method_version",
                "input_hash",
                "base_url",
                "requested_model",
                "parsed",
            )
            if key in payload
        }
        write_json(
            entity_destination / path.relative_to(entity_source), sanitized
        )
        counts["entity_api_batches"] += 1

    for directory, count_key in (
        ("url_entity_maps", "entity_query_maps"),
        ("run_entity_sets", "entity_run_sets"),
    ):
        for path in sorted((entity_source / directory).rglob("*.json")):
            write_json(
                entity_destination / path.relative_to(entity_source),
                load_json(path),
            )
            counts[count_key] += 1
    return counts


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export_all(source_root: Path, output_root: Path) -> None:
    manifest: dict[str, Any] = {
        "format": "anonymous-analysis-ready-v2",
        "raw_sse_included": True,
        "raw_sse_sanitized": True,
        "raw_api_responses_included": False,
        "complete_answers_included": True,
        "datasets": {},
    }
    for dataset, config in DATASETS.items():
        source = source_root / dataset
        destination = output_root / dataset
        if not source.exists():
            raise FileNotFoundError(source)
        dataset_counts: dict[str, Any] = {"web_runs": 0, "subsets": {}}
        for subset in config["subsets"]:
            runs, sse_runs, _ = export_subset(source / subset, destination / subset)
            dataset_counts["subsets"][subset] = {
                "web_runs": runs,
                "sanitized_sse_runs": sse_runs,
            }
            dataset_counts["web_runs"] += runs
            dataset_counts.setdefault("sanitized_sse_runs", 0)
            dataset_counts["sanitized_sse_runs"] += sse_runs
        if config["proxies"]:
            dataset_counts.update(export_proxies(source, destination))
        dataset_counts["sha256"] = tree_digest(destination)
        manifest["datasets"][dataset] = dataset_counts
    manifest["results"] = export_results(source_root, output_root)
    write_json(output_root / "ARTIFACT_MANIFEST.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_all(args.source_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
