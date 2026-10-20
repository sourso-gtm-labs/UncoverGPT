#!/usr/bin/env python3
"""Shared ChatGPT Web SSE parser for GPT-5.2, GPT-5.3, and GPT-5.5 datasets.

The datasets expose the same logical search/citation information through a
mix of full-message snapshots and incremental ChatGPT patch events.  This
module reconstructs the final message state first, then applies one shared URL,
query, citation, and completeness contract to both model versions.

The legacy ``all_urls``/``cited_urls`` output fields remain available for the
existing analysis scripts.  New audit fields preserve the backend-rewritten
queries, decoded ``search(...)`` arguments, final answer, footnotes, and stream
completion evidence.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_QUERY_KEYS = {"utm_source", "srsltid"}
_CITE_TOKEN_RE = re.compile(
    r"turn(?P<turn>\d+)(?P<type>search|view|news|academia|product|video|image)(?P<index>\d+)"
)
_SEARCH_CALL_RE = re.compile(r"^\s*search\((?P<argument>.*)\)\s*;?\s*$", re.DOTALL)


def normalize_url(url: str) -> str:
    """Remove ChatGPT tracking parameters without collapsing repeated values."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        parsed = urlparse(url)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in TRACKING_QUERY_KEYS
        ]
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                urlencode(query, doseq=True),
                "",
            )
        )
    except (TypeError, ValueError):
        return url


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _container_for(next_token: str | None) -> Any:
    if next_token is not None and (next_token.isdigit() or next_token == "-"):
        return []
    return {}


def _append_value(current: Any, value: Any) -> Any:
    if current is None:
        return _deepcopy(value)
    if isinstance(current, str) and isinstance(value, str):
        return current + value
    if isinstance(current, list):
        if isinstance(value, list):
            current.extend(_deepcopy(value))
        else:
            current.append(_deepcopy(value))
        return current
    if isinstance(current, dict) and isinstance(value, dict):
        current.update(_deepcopy(value))
        return current
    return _deepcopy(value)


def _truncate_value(current: Any, length: Any) -> Any:
    """Apply ChatGPT's truncate operation to an existing string or list."""
    if isinstance(length, bool) or not isinstance(length, int):
        return current
    length = max(0, length)
    if isinstance(current, str):
        return current[:length]
    if isinstance(current, list):
        return current[:length]
    return current


def _apply_patch(root: dict[str, Any], operation: dict[str, Any]) -> None:
    """Apply the JSON-patch dialect used by ChatGPT Web SSE deltas."""
    path = operation.get("p")
    if not isinstance(path, str) or not path.startswith("/"):
        return
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/")]
    if not tokens:
        return

    parent: Any = root
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if isinstance(parent, dict):
            if token not in parent or parent[token] is None:
                parent[token] = _container_for(next_token)
            parent = parent[token]
        elif isinstance(parent, list) and token.isdigit():
            item_index = int(token)
            while len(parent) <= item_index:
                parent.append(None)
            if parent[item_index] is None:
                parent[item_index] = _container_for(next_token)
            parent = parent[item_index]
        else:
            return

    token = tokens[-1]
    op = operation.get("o", "replace")
    value = operation.get("v")

    if isinstance(parent, dict):
        if op == "remove":
            parent.pop(token, None)
        elif op == "append":
            parent[token] = _append_value(parent.get(token), value)
        elif op == "truncate":
            if token in parent:
                parent[token] = _truncate_value(parent[token], value)
        else:
            parent[token] = _deepcopy(value)
        return

    if not isinstance(parent, list):
        return
    if token == "-":
        if op not in {"remove", "truncate"}:
            parent.append(_deepcopy(value))
        return
    if not token.isdigit():
        return

    item_index = int(token)
    if op == "remove":
        if item_index < len(parent):
            parent.pop(item_index)
        return
    if op == "truncate":
        if item_index < len(parent):
            parent[item_index] = _truncate_value(parent[item_index], value)
        return

    while len(parent) <= item_index:
        parent.append(None)
    if op == "append":
        parent[item_index] = _append_value(parent[item_index], value)
    else:
        parent[item_index] = _deepcopy(value)


def _search_result_groups(root: dict[str, Any]) -> list[Any] | None:
    """Return the current message's mutable search-result group container."""
    message = root.get("message")
    if not isinstance(message, dict):
        return None
    metadata = message.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        message["metadata"] = metadata
    groups = metadata.setdefault("search_result_groups", [])
    if not isinstance(groups, list):
        groups = []
        metadata["search_result_groups"] = groups
    return groups


def _apply_sse_operation(root: dict[str, Any], operation: dict[str, Any]) -> None:
    """Apply one patch or GPT-5.2 bare search-result event.

    GPT-5.2 emits search results in three additional incremental forms:
    bare ``search_result_group``/``search_result`` objects, list-valued patches
    to ``search_result_groups``, and repeated list patches to a group's
    ``entries``.  Those list-valued patches are cumulative even when their
    operation is encoded as a replacement.
    """
    event_type = operation.get("type")
    if event_type == "search_result_group":
        groups = _search_result_groups(root)
        if groups is not None:
            groups.append(_deepcopy(operation))
        return
    if event_type == "search_result":
        groups = _search_result_groups(root)
        if groups is not None:
            groups.append(
                {
                    "domain": operation.get("attribution", ""),
                    "entries": [_deepcopy(operation)],
                }
            )
        return

    path = operation.get("p")
    value = operation.get("v")
    op = operation.get("o", "replace")
    if path == "" and op == "patch" and isinstance(value, list):
        for nested in value:
            if isinstance(nested, dict):
                _apply_sse_operation(root, nested)
        return
    if not isinstance(path, str):
        return

    normalized_path = path if path.startswith("/") else f"/{path}"
    normalized_path = normalized_path.replace(
        "/message/message/metadata/", "/message/metadata/"
    )
    if (
        normalized_path == "/message/metadata/search_result_groups"
        and isinstance(value, list)
    ):
        groups = _search_result_groups(root)
        if groups is not None:
            groups.extend(_deepcopy(value))
        return

    entries_match = re.fullmatch(
        r"/message/metadata/search_result_groups/(\d+)/entries",
        normalized_path,
    )
    if entries_match and isinstance(value, list):
        groups = _search_result_groups(root)
        if groups is None:
            return
        group_index = int(entries_match.group(1))
        while len(groups) <= group_index:
            groups.append({})
        if not isinstance(groups[group_index], dict):
            groups[group_index] = {}
        entries = groups[group_index].setdefault("entries", [])
        if not isinstance(entries, list):
            entries = []
            groups[group_index]["entries"] = entries
        entries.extend(_deepcopy(value))
        return

    _apply_patch(root, operation)


def _iter_sse_payloads(filepath: Path) -> Iterable[tuple[int, Any]]:
    with filepath.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if not raw:
                continue
            if raw == "[DONE]":
                yield line_number, "[DONE]"
                continue
            try:
                yield line_number, json.loads(raw)
            except json.JSONDecodeError:
                yield line_number, None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    if not isinstance(content, dict):
        return ""
    text = content.get("text")
    if isinstance(text, str):
        return text
    parts = content.get("parts") or []
    if isinstance(parts, list):
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def _parse_search_call(text: str) -> str | None:
    """Decode one legacy ``search(<JSON string>)`` tool call safely."""
    match = _SEARCH_CALL_RE.match(text or "")
    if not match:
        return None
    argument = match.group("argument").strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(argument)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(value, str):
            return value
    return None


def _reference_id(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"turn_index": None, "ref_type": "", "ref_index": None}
    return {
        "turn_index": value.get("turn_index"),
        "ref_type": value.get("ref_type", ""),
        "ref_index": value.get("ref_index"),
    }


def _reference_key(value: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (value.get("turn_index"), value.get("ref_type"), value.get("ref_index"))


def _tokens_from_matched_text(text: str) -> list[dict[str, Any]]:
    return [
        {
            "turn_index": int(match.group("turn")),
            "ref_type": match.group("type"),
            "ref_index": int(match.group("index")),
        }
        for match in _CITE_TOKEN_RE.finditer(text or "")
    ]


def _dedupe_dicts(items: Iterable[dict[str, Any]], key_function: Any) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = key_function(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _flatten_search_results(groups: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not isinstance(groups, list):
        return results
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not isinstance(entries, list):
            continue
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not entry.get("url"):
                continue
            ref_id = _reference_id(entry.get("ref_id"))
            results.append(
                {
                    "url": normalize_url(entry["url"]),
                    "raw_url": entry["url"],
                    "title": entry.get("title", ""),
                    "snippet": entry.get("snippet", ""),
                    "domain": group.get("domain", "") or entry.get("attribution", ""),
                    "attribution": entry.get("attribution", ""),
                    "ref_id": ref_id,
                    "ref_turn_index": ref_id.get("turn_index"),
                    "ref_type": ref_id.get("ref_type", ""),
                    "ref_index": ref_id.get("ref_index"),
                    "pub_date": entry.get("pub_date"),
                    "group_index": group_index,
                    "entry_index": entry_index,
                }
            )
    return results


def _collect_search_queries(
    message_records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    code_queries: list[str] = []
    seen_code_messages: set[Any] = set()
    rounds: list[dict[str, Any]] = []
    last_round: dict[str, Any] | None = None

    for record in message_records:
        message = record["message"]
        message_id = message.get("id") or ("event", record["event_index"])
        content = message.get("content") or {}
        if (
            isinstance(content, dict)
            and content.get("content_type") == "code"
            and message_id not in seen_code_messages
        ):
            seen_code_messages.add(message_id)
            query = _parse_search_call(_message_text(message))
            if query is not None:
                code_queries.append(query)

        metadata = message.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        search_query_metadata = metadata.get("search_model_queries") or {}
        if not isinstance(search_query_metadata, dict):
            continue
        query_values = search_query_metadata.get("queries") or []
        queries = [query for query in query_values if isinstance(query, str) and query]
        if not queries:
            continue

        current_round = {
            "event_index": record["event_index"],
            "create_time": message.get("create_time"),
            "queries": queries,
            "message_ids": [message.get("id", "")],
            "source": "search_model_queries",
        }
        duplicate = False
        if last_round and last_round["queries"] == queries:
            previous_time = last_round.get("create_time")
            current_time = current_round.get("create_time")
            duplicate = (
                isinstance(previous_time, (int, float))
                and isinstance(current_time, (int, float))
                and abs(current_time - previous_time) < 1.0
            )
        if duplicate:
            last_round["message_ids"].append(message.get("id", ""))
        else:
            rounds.append(current_round)
            last_round = current_round

    query_records: list[dict[str, Any]] = []
    for round_index, search_round in enumerate(rounds):
        search_round["round_index"] = round_index
        search_round["query_count"] = len(search_round["queries"])
        search_round.pop("event_index", None)
        for query_index, query in enumerate(search_round["queries"]):
            query_records.append(
                {
                    "round_index": round_index,
                    "query_index": query_index,
                    "query": query,
                    "source": "search_model_queries",
                }
            )
    return code_queries, rounds, query_records


def _collect_content_references(
    messages: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    footnote_sources: list[dict[str, Any]] = []
    streaming_refs: list[dict[str, Any]] = []
    finalized_streaming_refs: list[dict[str, Any]] = []
    supporting_websites: list[dict[str, Any]] = []

    for message in messages:
        metadata = message.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        references = metadata.get("content_references") or []
        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict):
                continue
            if reference.get("type") == "sources_footnote":
                sources = reference.get("sources") or []
                if not isinstance(sources, list):
                    continue
                for source in sources:
                    if not isinstance(source, dict) or not source.get("url"):
                        continue
                    footnote_sources.append(
                        {
                            "url": normalize_url(source["url"]),
                            "raw_url": source["url"],
                            "title": source.get("title", ""),
                            "attribution": source.get("attribution", ""),
                        }
                    )
                continue

            items = reference.get("items") or []
            if not isinstance(items, list):
                continue
            fallback_refs = _tokens_from_matched_text(reference.get("matched_text", ""))
            for item_index, item in enumerate(items):
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                raw_refs = item.get("refs") or []
                refs = (
                    [_reference_id(ref) for ref in raw_refs if isinstance(ref, dict)]
                    if isinstance(raw_refs, list)
                    else []
                )
                if not refs and item_index < len(fallback_refs):
                    refs = [fallback_refs[item_index]]
                parent_ref = refs[0] if refs else _reference_id(None)
                parent_url = normalize_url(item["url"])
                streaming_item = {
                    "url": parent_url,
                    "raw_url": item["url"],
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "attribution": item.get("attribution", ""),
                    "pub_date": item.get("pub_date"),
                    "ref_ids": refs,
                }
                streaming_refs.append(streaming_item)
                if reference.get("status") == "done":
                    finalized_streaming_refs.append(_deepcopy(streaming_item))

                raw_supporting = item.get("supporting_websites") or []
                if not isinstance(raw_supporting, list):
                    continue
                for supporting in raw_supporting:
                    if not isinstance(supporting, dict) or not supporting.get("url"):
                        continue
                    supporting_websites.append(
                        {
                            "url": normalize_url(supporting["url"]),
                            "raw_url": supporting["url"],
                            "title": supporting.get("title", ""),
                            "snippet": supporting.get("snippet", ""),
                            "attribution": supporting.get("attribution", ""),
                            "pub_date": supporting.get("pub_date"),
                            "parent_url": parent_url,
                            "parent_ref_id": parent_ref,
                            "parent_ref_turn_index": parent_ref.get("turn_index"),
                            "parent_ref_type": parent_ref.get("ref_type", ""),
                            "parent_ref_index": parent_ref.get("ref_index"),
                        }
                    )

    footnote_sources = _dedupe_dicts(footnote_sources, lambda item: item["url"])
    streaming_refs = _dedupe_dicts(
        streaming_refs,
        lambda item: (item["url"], tuple(_reference_key(ref) for ref in item["ref_ids"])),
    )
    finalized_streaming_refs = _dedupe_dicts(
        finalized_streaming_refs,
        lambda item: (item["url"], tuple(_reference_key(ref) for ref in item["ref_ids"])),
    )
    supporting_websites = _dedupe_dicts(
        supporting_websites, lambda item: (item["url"], item["parent_url"])
    )
    return footnote_sources, streaming_refs, finalized_streaming_refs, supporting_websites


def _build_url_records(
    search_results: list[dict[str, Any]],
    footnote_sources: list[dict[str, Any]],
    streaming_refs: list[dict[str, Any]],
    supporting_websites: list[dict[str, Any]],
    final_cited_urls: set[str],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    footnote_lookup = {item["url"]: item for item in footnote_sources}
    streaming_urls = {item["url"] for item in streaming_refs}

    for result in search_results:
        url = result["url"]
        if not url:
            continue
        ref_id = result["ref_id"]
        if url not in records:
            records[url] = {
                "url": url,
                "raw_url": result["raw_url"],
                "title": result["title"],
                "snippet": result["snippet"],
                "domain": result["domain"],
                "ref_turn_index": result["ref_turn_index"],
                "ref_type": result["ref_type"],
                "ref_index": result["ref_index"],
                "pub_date": result["pub_date"],
                "ref_ids": [],
                "occurrence_count": 0,
                "is_search_candidate": True,
                "source_presence": ["search_result_groups"],
            }
        record = records[url]
        record["occurrence_count"] += 1
        if _reference_key(ref_id) not in {_reference_key(item) for item in record["ref_ids"]}:
            record["ref_ids"].append(ref_id)

    for streaming in streaming_refs:
        url = streaming["url"]
        if not url:
            continue
        refs = streaming.get("ref_ids") or []
        first_ref = refs[0] if refs else _reference_id(None)
        if url not in records:
            records[url] = {
                "url": url,
                "raw_url": streaming["raw_url"],
                "title": streaming["title"],
                "snippet": streaming["snippet"],
                "domain": streaming["attribution"],
                "ref_turn_index": first_ref.get("turn_index"),
                "ref_type": first_ref.get("ref_type", ""),
                "ref_index": first_ref.get("ref_index"),
                "pub_date": streaming["pub_date"],
                "ref_ids": [],
                "occurrence_count": 0,
                "is_search_candidate": True,
                "source_presence": ["content_references"],
            }
        record = records[url]
        if "content_references" not in record["source_presence"]:
            record["source_presence"].append("content_references")
        for ref_id in refs:
            if _reference_key(ref_id) not in {
                _reference_key(item) for item in record["ref_ids"]
            }:
                record["ref_ids"].append(ref_id)

    for source in footnote_sources:
        url = source["url"]
        if not url:
            continue
        if url not in records:
            records[url] = {
                "url": url,
                "raw_url": source["raw_url"],
                "title": source["title"],
                "snippet": "",
                "domain": source["attribution"],
                "ref_turn_index": None,
                "ref_type": "",
                "ref_index": None,
                "pub_date": None,
                "ref_ids": [],
                "occurrence_count": 0,
                "is_search_candidate": False,
                "source_presence": ["sources_footnote"],
            }
        elif "sources_footnote" not in records[url]["source_presence"]:
            records[url]["source_presence"].append("sources_footnote")

    supporting_by_url: dict[str, dict[str, Any]] = {}
    for supporting in supporting_websites:
        url = supporting["url"]
        if not url:
            continue
        existing = supporting_by_url.get(url)
        if existing is None or (
            supporting["parent_url"] in final_cited_urls
            and existing["parent_url"] not in final_cited_urls
        ):
            supporting_by_url[url] = supporting

    for url, record in records.items():
        footnote = footnote_lookup.get(url, {})
        supporting = supporting_by_url.get(url)
        record["title"] = footnote.get("title") or record["title"]
        record["domain"] = footnote.get("attribution") or record["domain"]
        record["streaming_cited"] = url in streaming_urls
        record["in_supporting"] = supporting is not None
        if supporting:
            record["cited_type"] = "supporting"
            record["cited"] = (
                url in final_cited_urls or supporting["parent_url"] in final_cited_urls
            )
            record["parent_url"] = supporting["parent_url"]
            record["parent_ref_id"] = supporting["parent_ref_id"]
            record["parent_ref_index"] = supporting["parent_ref_index"]
        else:
            record["cited_type"] = "main"
            record["cited"] = url in final_cited_urls
            record["parent_url"] = None
            record["parent_ref_id"] = None
            record["parent_ref_index"] = None

    for url, supporting in supporting_by_url.items():
        if url in records:
            continue
        records[url] = {
            "url": url,
            "raw_url": supporting["raw_url"],
            "title": supporting["title"],
            "snippet": supporting["snippet"],
            "domain": supporting["attribution"],
            "ref_turn_index": None,
            "ref_type": "supporting",
            "ref_index": None,
            "pub_date": supporting["pub_date"],
            "ref_ids": [],
            "occurrence_count": 0,
            "is_search_candidate": False,
            "source_presence": ["supporting_websites"],
            "cited": supporting["parent_url"] in final_cited_urls,
            "cited_type": "supporting",
            "streaming_cited": url in streaming_urls,
            "in_supporting": True,
            "parent_url": supporting["parent_url"],
            "parent_ref_id": supporting["parent_ref_id"],
            "parent_ref_index": supporting["parent_ref_index"],
        }

    def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
        ref_index = item.get("ref_index")
        if isinstance(ref_index, int):
            return (0, ref_index, item["url"])
        return (1, 0, item["url"])

    return sorted(records.values(), key=sort_key)


def _select_final_answer(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], bool, bool]:
    """Select and join completed final-channel answer segments in stream order."""
    text_messages: list[dict[str, Any]] = []
    for message in messages:
        author = message.get("author") or {}
        content = message.get("content") or {}
        if author.get("role") != "assistant" or content.get("content_type") != "text":
            continue
        text = _message_text(message)
        if not text:
            continue
        metadata = message.get("metadata") or {}
        completed = message.get("status") == "finished_successfully" or (
            isinstance(metadata, dict) and metadata.get("finished_successfully") is True
        )
        text_messages.append(
            {
                "message_id": message.get("id", ""),
                "text": text,
                "channel": message.get("channel"),
                "status": message.get("status"),
                "end_turn": message.get("end_turn"),
                "completed": completed,
                "partial_capture": bool(
                    isinstance(metadata, dict) and metadata.get("partial_capture")
                ),
            }
        )

    completed_final = [
        item for item in text_messages if item["completed"] and item["channel"] == "final"
    ]
    completed_end_turn = [
        item for item in text_messages if item["completed"] and item["end_turn"] is True
    ]
    completed_any = [item for item in text_messages if item["completed"]]
    if completed_final:
        selected = completed_final
    elif completed_end_turn:
        selected = completed_end_turn
    elif completed_any:
        selected = [completed_any[-1]]
    elif text_messages:
        selected = [text_messages[-1]]
    else:
        selected = []

    segments = [
        {
            "message_id": item["message_id"],
            "text": item["text"],
            "channel": item["channel"],
            "status": item["status"],
            "end_turn": item["end_turn"],
        }
        for item in selected
    ]
    final_answer = "".join(item["text"] for item in selected)
    finished_successfully = bool(selected) and all(item["completed"] for item in selected)
    final_answer_is_partial = bool(selected) and (
        not finished_successfully or any(item["partial_capture"] for item in selected)
    )
    return final_answer, segments, final_answer_is_partial, finished_successfully


def extract_from_sse(filepath: str | Path) -> dict[str, Any]:
    """Parse one GPT-5.2, GPT-5.3, or GPT-5.5 ChatGPT Web SSE capture."""
    path = Path(filepath)
    message_records: list[dict[str, Any]] = []
    current_root: dict[str, Any] | None = None
    server_meta: dict[str, Any] = {}
    conversation_meta: dict[str, Any] = {}
    json_decode_errors = 0
    error_event_count = 0
    saw_done = False
    saw_stream_complete = False
    saw_conversation_detail = False
    saw_server_metadata = False
    saw_full_message = False
    first_payload_was_patch = False
    first_full_message_role = ""
    event_index = 0

    for line_number, payload in _iter_sse_payloads(path):
        event_index += 1
        if payload == "[DONE]":
            saw_done = True
            continue
        if payload is None:
            json_decode_errors += 1
            continue
        if not isinstance(payload, dict):
            continue

        event_type = payload.get("type")
        if event_type == "server_ste_metadata":
            saw_server_metadata = True
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            query_types = metadata.get("search_tool_query_types") or {}
            if not isinstance(query_types, dict):
                query_types = {}
            server_meta = {
                "tool_name": metadata.get("tool_name", ""),
                "tool_invoked": metadata.get("tool_invoked"),
                "is_search": metadata.get("is_search"),
                "model_slug": metadata.get("model_slug", ""),
                "default_model_slug": metadata.get("default_model_slug", ""),
                "thinking_effort": metadata.get("thinking_effort"),
                "search_tool_call_count": metadata.get("search_tool_call_count"),
                "has_system1": query_types.get("has_system1_search_query"),
                "has_system2": query_types.get("has_system2_search_query"),
                "did_auto_switch_to_reasoning": metadata.get("did_auto_switch_to_reasoning"),
                "plan_type": metadata.get("plan_type", ""),
                "turn_use_case": metadata.get("turn_use_case"),
                "cluster_region": metadata.get("cluster_region", ""),
                "request_id": metadata.get("request_id", ""),
                "message_id": metadata.get("message_id", ""),
                "conversation_id": payload.get("conversation_id", ""),
            }
        elif event_type == "message_stream_complete":
            saw_stream_complete = True
        elif event_type == "conversation_detail_metadata":
            saw_conversation_detail = True
            conversation_meta = {
                "conversation_id": payload.get("conversation_id", ""),
                "default_model_slug": payload.get("default_model_slug", ""),
            }

        error_code = payload.get("error_code")
        error_value = payload.get("error")
        if (
            event_type in {"error", "message_stream_error"}
            or (error_code is not None and error_code != "")
            or (error_value is not None and error_value != "")
        ):
            error_event_count += 1

        value = payload.get("v")
        if isinstance(value, dict) and isinstance(value.get("message"), dict):
            message = _deepcopy(value["message"])
            if not saw_full_message:
                first_full_message_role = ((message.get("author") or {}).get("role", ""))
            saw_full_message = True
            record = {
                "message": message,
                "event_index": event_index,
                "line_number": line_number,
            }
            message_records.append(record)
            current_root = {"message": message}
            continue

        if event_type in {"search_result_group", "search_result"}:
            operations = [payload]
        elif isinstance(payload.get("p"), str) and payload["p"].startswith("/"):
            operations = [payload]
        elif isinstance(value, list):
            operations = value
        else:
            continue

        if current_root is None:
            first_payload_was_patch = True
            orphan_message = {
                "id": "",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
                "metadata": {"partial_capture": True},
            }
            message_records.append(
                {
                    "message": orphan_message,
                    "event_index": event_index,
                    "line_number": line_number,
                }
            )
            current_root = {"message": orphan_message}

        for operation in operations:
            if isinstance(operation, dict):
                _apply_sse_operation(current_root, operation)

    messages = [record["message"] for record in message_records]
    code_search_queries, search_rounds, actual_query_records = _collect_search_queries(
        message_records
    )
    actual_search_queries = [record["query"] for record in actual_query_records]
    if actual_search_queries:
        search_queries = actual_search_queries
        search_query_source = "search_model_queries"
        search_query_records = actual_query_records
    else:
        search_queries = code_search_queries
        search_query_source = "search_code_fallback" if code_search_queries else "unavailable"
        search_rounds = [
            {
                "round_index": index,
                "queries": [query],
                "query_count": 1,
                "message_ids": [],
                "source": "search_code_fallback",
            }
            for index, query in enumerate(code_search_queries)
        ]
        search_query_records = [
            {
                "round_index": index,
                "query_index": 0,
                "query": query,
                "source": "search_code_fallback",
            }
            for index, query in enumerate(code_search_queries)
        ]

    search_results: list[dict[str, Any]] = []
    for message in messages:
        metadata = message.get("metadata") or {}
        if isinstance(metadata, dict):
            search_results.extend(_flatten_search_results(metadata.get("search_result_groups")))
    search_results = _dedupe_dicts(
        search_results,
        lambda item: (item["url"], _reference_key(item["ref_id"])),
    )

    (
        footnote_sources,
        streaming_refs,
        finalized_streaming_refs,
        supporting_websites,
    ) = _collect_content_references(messages)
    footnote_urls = {item["url"] for item in footnote_sources}
    finalized_urls = {item["url"] for item in finalized_streaming_refs}
    if footnote_urls:
        final_cited_urls = footnote_urls
        citation_source = "sources_footnote"
    elif saw_stream_complete and finalized_urls:
        final_cited_urls = finalized_urls
        citation_source = "grouped_webpages_fallback"
    else:
        final_cited_urls = set()
        citation_source = "none"

    all_urls = _build_url_records(
        search_results,
        footnote_sources,
        streaming_refs,
        supporting_websites,
        final_cited_urls,
    )

    (
        final_answer,
        final_answer_segments,
        final_answer_is_partial,
        finished_successfully,
    ) = _select_final_answer(messages)

    model = server_meta.get("model_slug", "")
    if not model:
        for message in messages:
            metadata = message.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("model_slug"):
                model = metadata["model_slug"]
                break
        server_meta["model_slug"] = model

    capture_complete = (
        saw_done
        and saw_stream_complete
        and saw_conversation_detail
        and saw_server_metadata
        and finished_successfully
        and json_decode_errors == 0
        and error_event_count == 0
    )
    type_counts = Counter(item.get("ref_type") or "unknown" for item in all_urls)
    total_cited_main = sum(
        1 for item in all_urls if item.get("cited") and item.get("cited_type") == "main"
    )
    total_cited_supporting = sum(
        1
        for item in all_urls
        if item.get("cited") and item.get("cited_type") == "supporting"
    )

    return {
        "server_meta": server_meta,
        "conversation_meta": conversation_meta,
        "stream_meta": {
            "saw_done": saw_done,
            "saw_message_stream_complete": saw_stream_complete,
            "saw_conversation_detail_metadata": saw_conversation_detail,
            "saw_server_ste_metadata": saw_server_metadata,
            "finished_successfully": finished_successfully,
            "json_decode_errors": json_decode_errors,
            "error_event_count": error_event_count,
            "capture_complete": capture_complete,
            "stream_started_mid_message": first_payload_was_patch,
            "first_full_message_role": first_full_message_role,
            "search_query_source": search_query_source,
        },
        "search_queries": search_queries,
        "actual_search_queries": actual_search_queries,
        "raw_search_queries": code_search_queries,
        "search_query_source": search_query_source,
        "search_rounds": search_rounds,
        "search_query_records": search_query_records,
        "search_round_count": len(search_rounds),
        "search_query_count": len(search_queries),
        "actual_search_query_count": len(actual_search_queries),
        "raw_search_query_count": len(code_search_queries),
        "unique_search_query_count": len(set(search_queries)),
        "final_answer": final_answer,
        "final_answer_segments": final_answer_segments,
        "final_answer_segment_count": len(final_answer_segments),
        "final_answer_is_partial": final_answer_is_partial,
        "total_search_result_occurrences": len(search_results),
        "total_candidate_urls": sum(
            1 for item in all_urls if item.get("is_search_candidate")
        ),
        "total_all_urls": len(all_urls),
        "total_cited": total_cited_main + total_cited_supporting,
        "total_cited_main": total_cited_main,
        "total_cited_supporting": total_cited_supporting,
        "total_uncited": sum(1 for item in all_urls if not item.get("cited")),
        "total_supporting_websites": len(supporting_websites),
        "total_sources_footnote": len(footnote_sources),
        "total_streaming_reference_urls": len({item["url"] for item in streaming_refs}),
        "citation_source": citation_source,
        "url_counts_by_type": dict(sorted(type_counts.items())),
        "sources_footnote": footnote_sources,
        "streaming_references": streaming_refs,
        "finalized_streaming_references": finalized_streaming_refs,
        "cited_urls": [item for item in all_urls if item.get("cited")],
        "supporting_websites": supporting_websites,
        "search_results": search_results,
        "all_urls": all_urls,
    }


def _load_queries(query_list_path: Path) -> dict[str, str]:
    if not query_list_path.exists():
        return {}
    with query_list_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("id", "")).zfill(3): item.get("query", "")
        for item in items
        if isinstance(item, dict)
    }


def _run_id(filepath: Path) -> str:
    parts = filepath.stem.split("_done_", 1)
    full_timestamp = parts[1] if len(parts) == 2 else filepath.stem
    return full_timestamp[5:] if len(full_timestamp) > 5 else full_timestamp


def _conn_id(filepath: Path) -> str:
    match = re.search(r"conn([^_]+)", filepath.name)
    return match.group(1) if match else ""


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
        temp_path = Path(stream.name)
    temp_path.replace(path)


def parse_subset(
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    query_list_path: str | Path | None = None,
    query_ids: Sequence[str] | None = None,
    *,
    write: bool = False,
    strict: bool = False,
    allow_mixed_output: bool = False,
) -> dict[str, Any]:
    """Parse one v9 subset, optionally writing legacy-compatible detail JSON."""
    source = Path(source_dir)
    output = Path(output_dir) if output_dir is not None else source / "search_detail"
    query_path = (
        Path(query_list_path) if query_list_path is not None else source / "query_list.json"
    )
    queries = _load_queries(query_path)
    requested = {
        str(query_id).removeprefix("q").zfill(3) for query_id in (query_ids or [])
    }
    chatgpt_dir = source / "geo-web-crawl" / "chatgpt"
    if not chatgpt_dir.is_dir():
        raise FileNotFoundError(f"SSE directory not found: {chatgpt_dir}")

    def selected_qdir(path: Path) -> bool:
        if not path.is_dir() or not path.name.startswith("q"):
            return False
        full_qid = path.name[1:].zfill(3)
        base_qid = full_qid.split("_", 1)[0].zfill(3)
        return not requested or full_qid in requested or base_qid in requested

    q_dirs = sorted(path for path in chatgpt_dir.iterdir() if selected_qdir(path))
    target_sources: dict[Path, Path] = {}
    tasks: list[tuple[Path, str, str, Path]] = []
    for q_dir in q_dirs:
        qid = q_dir.name[1:].zfill(3)
        for filepath in sorted(q_dir.glob("chatgpt_sse_conn*.txt")):
            run_id = _run_id(filepath)
            target = output / q_dir.name / f"q{qid}_{run_id}_detail.json"
            previous = target_sources.get(target)
            if previous is not None and previous != filepath:
                raise ValueError(
                    f"Output collision: {previous} and {filepath} both map to {target}"
                )
            target_sources[target] = filepath
            tasks.append((filepath, qid, run_id, target))

    existing_outputs = (
        set(output.glob("q*/*_detail.json")) if output.is_dir() else set()
    )
    planned_outputs = set(target_sources)
    unrelated_existing_outputs = existing_outputs - planned_outputs
    if write and unrelated_existing_outputs and not allow_mixed_output:
        raise ValueError(
            f"Output directory contains {len(unrelated_existing_outputs)} unrelated detail files: "
            f"{output}. Use an isolated --output-dir, or pass --allow-mixed-output "
            "when combining datasets is intentional."
        )

    audit: dict[str, Any] = {
        "source_dir": str(source),
        "output_dir": str(output),
        "files": len(tasks),
        "outputs_written": 0,
        "existing_output_files": len(existing_outputs),
        "would_create_outputs": len(planned_outputs - existing_outputs),
        "would_replace_outputs": len(planned_outputs & existing_outputs),
        "unrelated_existing_outputs": len(unrelated_existing_outputs),
        "complete": 0,
        "incomplete": 0,
        "json_decode_errors": 0,
        "empty_answers": 0,
        "empty_urls": 0,
        "query_sources": Counter(),
        "models": Counter(),
        "citation_sources": Counter(),
        "total_search_queries": 0,
        "total_all_urls": 0,
        "total_candidate_urls": 0,
        "total_cited": 0,
        "total_cited_main": 0,
        "total_cited_supporting": 0,
        "total_supporting_websites": 0,
        "total_sources_footnote": 0,
    }

    for filepath, qid, run_id, target in tasks:
        result = extract_from_sse(filepath)
        result["query_id"] = qid
        result["conn_id"] = _conn_id(filepath)
        result["run_id"] = run_id
        base_qid = qid.split("_", 1)[0].zfill(3)
        result["original_query"] = queries.get(qid) or queries.get(base_qid, "")

        stream_meta = result["stream_meta"]
        is_complete = bool(stream_meta["capture_complete"])
        audit["complete" if is_complete else "incomplete"] += 1
        audit["json_decode_errors"] += stream_meta["json_decode_errors"]
        audit["empty_answers"] += int(not result["final_answer"])
        audit["empty_urls"] += int(not result["all_urls"])
        audit["query_sources"][result["search_query_source"]] += 1
        audit["models"][result["server_meta"].get("model_slug", "")] += 1
        audit["citation_sources"][result["citation_source"]] += 1
        for key in (
            "total_all_urls",
            "total_candidate_urls",
            "total_cited",
            "total_cited_main",
            "total_cited_supporting",
            "total_supporting_websites",
            "total_sources_footnote",
        ):
            audit[key] += result[key]
        audit["total_search_queries"] += len(result["search_queries"])

        if strict and not is_complete:
            raise ValueError(f"Incomplete SSE capture: {filepath}")
        if write:
            _atomic_write_json(target, result)
            audit["outputs_written"] += 1

    audit["query_sources"] = dict(sorted(audit["query_sources"].items()))
    audit["models"] = dict(sorted(audit["models"].items()))
    audit["citation_sources"] = dict(sorted(audit["citation_sources"].items()))
    return audit


def main(default_source_dir: str | Path) -> None:
    """CLI used by the eight legacy dataset entry points."""
    default_source = Path(default_source_dir)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help="v9 subset containing geo-web-crawl and query_list.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="detail output root; audit display defaults to <source-dir>/search_detail",
    )
    parser.add_argument(
        "--query-list",
        type=Path,
        help="query_list.json path; defaults to <source-dir>/query_list.json",
    )
    parser.add_argument("--query-id", action="append", help="only parse this qID; repeatable")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="write detail JSON; requires an explicit --output-dir",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and audit without writing (the default)",
    )
    parser.add_argument(
        "--allow-mixed-output",
        action="store_true",
        help="allow unrelated existing detail files in the selected output directory",
    )
    parser.add_argument("--strict", action="store_true", help="fail on an incomplete SSE stream")
    args = parser.parse_args()

    if args.write and args.output_dir is None:
        parser.error("--write requires an explicit --output-dir")
    output_dir = args.output_dir or args.source_dir / "search_detail"
    audit = parse_subset(
        source_dir=args.source_dir,
        output_dir=output_dir,
        query_list_path=args.query_list,
        query_ids=args.query_id,
        write=args.write,
        strict=args.strict,
        allow_mixed_output=args.allow_mixed_output,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


__all__ = ["extract_from_sse", "main", "normalize_url", "parse_subset"]
