#!/usr/bin/env python3
"""Parse GPT-5.4 Thinking ChatGPT Web SSE captures.

GPT-5.4 Thinking can issue several ``web.run`` calls in one response. A search
call can contain multiple rewritten queries, and search calls can be mixed with
``product_query``, ``open``, ``click``, and ``find`` calls. Candidate pages are
collected from every model-visible web tool result and the final metadata pool.

The parser keeps the 5.3 ``all_urls``/``cited_urls`` compatibility fields while
also preserving search rounds, result occurrences, and composite reference IDs.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


BASE_DIR = Path(__file__).resolve().parent
TRACKING_QUERY_KEYS = {"utm_source", "srsltid"}
WEB_OPERATIONS = ("search_query", "product_query", "open", "click", "find")
_CITE_TOKEN_RE = re.compile(
    r"turn(?P<turn>\d+)(?P<type>search|view|news|academia|product|video|image)(?P<index>\d+)"
)


def normalize_url(url: str) -> str:
    """Remove ChatGPT tracking parameters without collapsing distinct URLs."""
    if not url:
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
    return [] if next_token is not None and (next_token.isdigit() or next_token == "-") else {}


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


def _apply_patch(root: dict[str, Any], operation: dict[str, Any]) -> None:
    """Apply the small JSON-patch dialect used by ChatGPT SSE deltas."""
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
        else:
            parent[token] = _deepcopy(value)
        return

    if not isinstance(parent, list):
        return
    if token == "-":
        if op != "remove":
            parent.append(_deepcopy(value))
        return
    if not token.isdigit():
        return
    item_index = int(token)
    while len(parent) <= item_index:
        parent.append(None)
    if op == "remove":
        if item_index < len(parent):
            parent.pop(item_index)
    elif op == "append":
        parent[item_index] = _append_value(parent[item_index], value)
    else:
        parent[item_index] = _deepcopy(value)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    if not isinstance(content, dict):
        return ""
    text = content.get("text")
    if isinstance(text, str):
        return text
    parts = content.get("parts") or []
    return "".join(part for part in parts if isinstance(part, str))


def _merge_server_metadata(
    target: dict[str, Any],
    metadata: dict[str, Any],
    *,
    authoritative: bool,
) -> None:
    """Merge model/search provenance without erasing earlier non-null values."""
    query_types = metadata.get("search_tool_query_types") or {}
    values = {
        "tool_name": metadata.get("tool_name"),
        "tool_invoked": metadata.get("tool_invoked"),
        "model_slug": metadata.get("model_slug"),
        "thinking_effort": metadata.get("thinking_effort"),
        "turn_mode": metadata.get("turn_mode"),
        "search_tool_call_count": metadata.get("search_tool_call_count"),
        "has_system1": query_types.get("has_system1_search_query"),
        "has_system2": query_types.get("has_system2_search_query"),
        "did_auto_switch_to_reasoning": metadata.get("did_auto_switch_to_reasoning"),
    }
    for key, value in values.items():
        if value is None or value == "":
            continue
        if authoritative or target.get(key) in {None, ""}:
            target[key] = value


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


def _parse_code_call(message: dict[str, Any], event_index: int) -> dict[str, Any]:
    raw_code = _message_text(message).strip()
    payload: dict[str, Any] | None = None
    parse_error = ""
    try:
        parsed = json.loads(raw_code)
        if isinstance(parsed, dict):
            payload = parsed
        else:
            parse_error = "code payload is not a JSON object"
    except json.JSONDecodeError as error:
        parse_error = str(error)

    operations = [key for key in WEB_OPERATIONS if payload and key in payload]
    if not operations and raw_code:
        operations = [
            key for key in WEB_OPERATIONS if re.search(rf'"{re.escape(key)}"\s*:', raw_code)
        ]
    if not operations:
        operations = ["unparsed"]
    return {
        "event_index": event_index,
        "message_id": message.get("id", ""),
        "recipient": message.get("recipient", ""),
        "operations": operations,
        "payload": payload,
        "raw_code": raw_code,
        "parse_error": parse_error,
    }


def _extract_search_rounds(
    metadata_batches: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    def call_queries(call: dict[str, Any]) -> list[dict[str, Any]]:
        payload = call.get("payload") or {}
        search_query = payload.get("search_query")
        if isinstance(search_query, list):
            return [
                {"query": item["q"], "domains": item.get("domains", [])}
                for item in search_query
                if isinstance(item, dict) and item.get("q")
            ]

        product_query = payload.get("product_query")
        product_search = product_query.get("search") if isinstance(product_query, dict) else None
        if not isinstance(product_search, list):
            return []
        records = []
        for item in product_search:
            if isinstance(item, str) and item:
                records.append({"query": item, "domains": []})
            elif isinstance(item, dict) and (item.get("q") or item.get("query")):
                records.append(
                    {
                        "query": item.get("q") or item.get("query"),
                        "domains": item.get("domains", []),
                    }
                )
        return records

    explicit_search_calls = [
        call for call in tool_calls if call_queries(call)
    ]
    unmatched_calls = list(explicit_search_calls)
    rounds: list[dict[str, Any]] = []

    for batch in metadata_batches:
        queries = batch["queries"]
        query_tuple = tuple(queries)
        matched_call = None
        for call in unmatched_calls:
            payload_queries = tuple(item["query"] for item in call_queries(call))
            if payload_queries == query_tuple:
                matched_call = call
                unmatched_calls.remove(call)
                break

        query_records = []
        payload_items = call_queries(matched_call) if matched_call else []
        for query_index, query in enumerate(queries):
            payload_item = payload_items[query_index] if query_index < len(payload_items) else {}
            query_records.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "domains": payload_item.get("domains", []),
                }
            )
        rounds.append(
            {
                "event_index": batch["event_index"],
                "source": "code_and_tool_metadata" if matched_call else "tool_metadata",
                "code_tool_call_index": matched_call.get("tool_call_index") if matched_call else None,
                "queries": query_records,
            }
        )

    for call in unmatched_calls:
        query_records = []
        for query_index, item in enumerate(call_queries(call)):
            query_records.append(
                {
                    "query_index": query_index,
                    "query": item["query"],
                    "domains": item.get("domains", []),
                }
            )
        if query_records:
            rounds.append(
                {
                    "event_index": call["event_index"],
                    "source": "code",
                    "code_tool_call_index": call["tool_call_index"],
                    "queries": query_records,
                }
            )

    rounds.sort(key=lambda item: item["event_index"])
    flat_records: list[dict[str, Any]] = []
    for round_index, search_round in enumerate(rounds):
        search_round["round_index"] = round_index
        search_round["query_count"] = len(search_round["queries"])
        for query in search_round["queries"]:
            query["round_index"] = round_index
            flat_records.append(query)
        search_round.pop("event_index", None)
    return rounds, flat_records


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
        references = metadata.get("content_references") or []
        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict):
                continue
            if reference.get("type") == "sources_footnote":
                for source in reference.get("sources", []):
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
                refs = [_reference_id(ref) for ref in item.get("refs", []) if isinstance(ref, dict)]
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
                for supporting in item.get("supporting_websites", []):
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


def _flatten_search_results(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        for entry_index, entry in enumerate(group.get("entries", [])):
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
                    "pub_date": entry.get("pub_date"),
                    "ref_id": ref_id,
                    "ref_turn_index": ref_id.get("turn_index"),
                    "ref_type": ref_id.get("ref_type", ""),
                    "ref_index": ref_id.get("ref_index"),
                    "rank_within_ref_turn": (
                        ref_id["ref_index"] + 1 if isinstance(ref_id.get("ref_index"), int) else None
                    ),
                    "group_index": group_index,
                    "entry_index": entry_index,
                }
            )
    return results


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

    # GPT-5.4 often omits already-cited pages from the final metadata pool. The
    # streamed grouped_webpages objects retain those candidates and their
    # composite reference IDs, so merge them just as the 5.3 parser does.
    for streaming in streaming_refs:
        url = streaming["url"]
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
        existing = supporting_by_url.get(supporting["url"])
        if existing is None or (
            supporting["parent_url"] in final_cited_urls
            and existing["parent_url"] not in final_cited_urls
        ):
            supporting_by_url[supporting["url"]] = supporting

    for url, record in records.items():
        footnote = footnote_lookup.get(url, {})
        supporting = supporting_by_url.get(url)
        record["title"] = footnote.get("title") or record["title"]
        record["domain"] = footnote.get("attribution") or record["domain"]
        record["streaming_cited"] = url in streaming_urls
        record["in_supporting"] = supporting is not None
        # Match the GPT-5.3/GPT-5.5 convention: membership in a
        # supporting_websites list determines cited_type, independently of
        # whether the same URL is also selected directly in the final citation
        # set. A supporting URL is cited when it or its selected parent survives.
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

    return list(records.values())


def extract_from_sse(filepath: str | Path) -> dict[str, Any]:
    """Extract one GPT-5.4 Thinking run from a raw SSE capture."""
    path = Path(filepath)
    messages: list[dict[str, Any]] = []
    current_root: dict[str, Any] | None = None
    server_meta: dict[str, Any] = {}
    final_metadata: dict[str, Any] = {}
    tool_search_results: list[dict[str, Any]] = []
    snapshot_search_results: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    code_message_ids: set[str] = set()
    metadata_batches: list[dict[str, Any]] = []
    last_batch: dict[str, Any] | None = None
    json_decode_errors = 0
    saw_done = False
    saw_stream_complete = False
    first_payload_was_patch = False
    first_full_message_role = ""
    saw_full_message = False
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
            metadata = payload.get("metadata") or {}
            _merge_server_metadata(server_meta, metadata, authoritative=True)
        elif event_type == "message_stream_complete":
            saw_stream_complete = True

        value = payload.get("v")
        if isinstance(value, dict) and isinstance(value.get("message"), dict):
            if not saw_full_message:
                first_full_message_role = (
                    (value["message"].get("author") or {}).get("role", "")
                )
            saw_full_message = True
            message = _deepcopy(value["message"])
            messages.append(message)
            current_root = {"message": message}

            metadata = message.get("metadata") or {}
            _merge_server_metadata(server_meta, metadata, authoritative=False)
            author = message.get("author") or {}
            if metadata.get("search_result_groups"):
                snapshot_results = _flatten_search_results(metadata["search_result_groups"])
                snapshot_search_results.extend(snapshot_results)
                if author.get("role") == "tool":
                    tool_search_results.extend(snapshot_results)
            query_values = (metadata.get("search_model_queries") or {}).get("queries") or []
            queries = [query for query in query_values if isinstance(query, str) and query]
            if queries:
                batch = {
                    "event_index": event_index,
                    "line_number": line_number,
                    "create_time": message.get("create_time"),
                    "queries": queries,
                    "message_ids": [message.get("id", "")],
                }
                duplicate = False
                if last_batch and last_batch["queries"] == queries:
                    previous_time = last_batch.get("create_time")
                    current_time = batch.get("create_time")
                    duplicate = (
                        isinstance(previous_time, (int, float))
                        and isinstance(current_time, (int, float))
                        and abs(current_time - previous_time) < 1.0
                    )
                if duplicate:
                    last_batch["message_ids"].append(message.get("id", ""))
                else:
                    metadata_batches.append(batch)
                    last_batch = batch

            content = message.get("content") or {}
            message_id = message.get("id", "")
            if content.get("content_type") == "code" and message_id not in code_message_ids:
                code_message_ids.add(message_id)
                call = _parse_code_call(message, event_index)
                call["tool_call_index"] = len(tool_calls)
                tool_calls.append(call)
            continue

        if isinstance(payload.get("p"), str) and payload["p"].startswith("/"):
            operations = [payload]
        elif isinstance(value, list):
            operations = value
        else:
            continue
        if not saw_full_message and current_root is None:
            first_payload_was_patch = True
            orphan_message = {
                "id": "",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
                "metadata": {"partial_capture": True},
            }
            messages.append(orphan_message)
            current_root = {"message": orphan_message}

        for operation in operations:
            if not isinstance(operation, dict):
                continue
            patch_path = operation.get("p", "")
            patch_value = operation.get("v")
            if (
                isinstance(patch_path, str)
                and patch_path.endswith("/message/metadata")
                and isinstance(patch_value, dict)
                and patch_value.get("search_result_groups") is not None
            ):
                final_metadata = _deepcopy(patch_value)
            if current_root is not None:
                _apply_patch(current_root, operation)

    search_rounds, query_records = _extract_search_rounds(metadata_batches, tool_calls)
    for index, call in enumerate(tool_calls):
        call["tool_call_index"] = index
        call.pop("event_index", None)
    final_groups = final_metadata.get("search_result_groups") or []
    final_search_results = _flatten_search_results(final_groups)
    # A final metadata snapshot can remap a tool result to a presentation ref
    # (typically turn 0). Keep both occurrences, while removing only exact
    # duplicate URL/ref tuples from repeated tool snapshots.
    search_results = _dedupe_dicts(
        [*snapshot_search_results, *final_search_results],
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

    final_answer = ""
    final_answer_is_partial = False
    for message in messages:
        author = message.get("author") or {}
        content = message.get("content") or {}
        if author.get("role") != "assistant" or content.get("content_type") != "text":
            continue
        text = _message_text(message)
        if text:
            final_answer = text
            final_answer_is_partial = bool((message.get("metadata") or {}).get("partial_capture"))

    type_counts = Counter(item.get("ref_type") or "unknown" for item in all_urls)
    total_cited_main = sum(
        1 for item in all_urls if item.get("cited") and item.get("cited_type") == "main"
    )
    total_cited_supporting = sum(
        1
        for item in all_urls
        if item.get("cited") and item.get("cited_type") == "supporting"
    )
    model = server_meta.get("model_slug", "")
    if not model:
        for message in messages:
            model = (message.get("metadata") or {}).get("model_slug", "")
            if model:
                break

    capture_is_partial = first_payload_was_patch or first_full_message_role not in {"system", "user"}
    query_status = "complete" if search_rounds else (
        "unavailable_partial_sse" if capture_is_partial else "unavailable"
    )
    operation_counts = Counter(
        operation for call in tool_calls for operation in call.get("operations", [])
    )

    return {
        "server_meta": {**server_meta, "model_slug": model},
        "stream_meta": {
            "saw_done": saw_done,
            "saw_message_stream_complete": saw_stream_complete,
            "stream_started_mid_message": first_payload_was_patch,
            "stream_started_mid_response": capture_is_partial,
            "first_full_message_role": first_full_message_role,
            "json_decode_errors": json_decode_errors,
            "candidate_pool_source": "message_metadata_union_final_metadata",
            "candidate_merge_sources": [
                "all_full_messages.metadata.search_result_groups",
                "final_metadata.search_result_groups",
                "final_answer.content_references",
            ],
            "search_query_status": query_status,
        },
        "search_rounds": search_rounds,
        "search_round_count": len(search_rounds),
        "search_query_records": query_records,
        "search_queries": [record["query"] for record in query_records],
        "search_query_count": len(query_records),
        "unique_search_query_count": len({record["query"] for record in query_records}),
        "web_tool_calls": tool_calls,
        "web_tool_call_count": len(tool_calls),
        "web_tool_operation_counts": dict(sorted(operation_counts.items())),
        "final_answer": final_answer,
        "final_answer_is_partial": final_answer_is_partial,
        "total_search_result_occurrences": len(search_results),
        "total_tool_search_result_occurrences": len(tool_search_results),
        "total_tool_candidate_urls": len({item["url"] for item in tool_search_results}),
        "total_snapshot_search_result_occurrences": len(snapshot_search_results),
        "total_snapshot_candidate_urls": len(
            {item["url"] for item in snapshot_search_results}
        ),
        "total_final_metadata_candidate_urls": len(
            {item["url"] for item in final_search_results}
        ),
        "total_streaming_reference_urls": len({item["url"] for item in streaming_refs}),
        "total_candidate_urls": sum(1 for item in all_urls if item["is_search_candidate"]),
        "total_all_urls": len(all_urls),
        "total_cited": total_cited_main + total_cited_supporting,
        "total_cited_main": total_cited_main,
        "total_cited_supporting": total_cited_supporting,
        "total_uncited": sum(1 for item in all_urls if not item.get("cited")),
        "total_supporting_websites": len(supporting_websites),
        "total_sources_footnote": len(footnote_sources),
        "citation_source": citation_source,
        "url_counts_by_type": dict(sorted(type_counts.items())),
        "sources_footnote": footnote_sources,
        "streaming_references": streaming_refs,
        "finalized_streaming_references": finalized_streaming_refs,
        "cited_urls": [item for item in all_urls if item.get("cited")],
        "supporting_websites": supporting_websites,
        "tool_search_results": tool_search_results,
        "final_metadata_search_results": final_search_results,
        "search_results": search_results,
        "all_urls": all_urls,
    }


def _load_queries(subset_dir: Path) -> dict[str, str]:
    query_list = subset_dir / "query_list.json"
    if not query_list.exists():
        return {}
    with query_list.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    return {
        str(item.get("id", "")).zfill(3): item.get("query", "")
        for item in items
        if isinstance(item, dict)
    }


def parse_dataset(base_dir: Path = BASE_DIR, subset_names: list[str] | None = None) -> dict[str, Any]:
    subset_dirs = sorted(path for path in base_dir.glob("v10_*_5.4") if path.is_dir())
    if subset_names:
        requested = set(subset_names)
        subset_dirs = [path for path in subset_dirs if path.name in requested]

    audit: dict[str, Any] = {
        "files": 0,
        "outputs": 0,
        "partial_sse": 0,
        "queries_unavailable": 0,
        "missing_candidates": 0,
        "missing_citations": 0,
        "models": Counter(),
        "subsets": {},
    }
    for subset_dir in subset_dirs:
        queries = _load_queries(subset_dir)
        chatgpt_dir = subset_dir / "geo-web-crawl" / "chatgpt"
        subset_files = 0
        for q_dir in sorted(path for path in chatgpt_dir.glob("q*") if path.is_dir()):
            qid = q_dir.name.removeprefix("q").zfill(3)
            output_dir = subset_dir / "search_detail" / q_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            for filepath in sorted(q_dir.glob("chatgpt_sse_conn*.txt")):
                subset_files += 1
                audit["files"] += 1
                result = extract_from_sse(filepath)
                stem_parts = filepath.stem.split("_done_", 1)
                full_timestamp = stem_parts[1] if len(stem_parts) == 2 else filepath.stem
                run_id = full_timestamp[5:] if len(full_timestamp) > 5 else full_timestamp
                connection_match = re.search(r"conn(\d+)", filepath.name)
                result.update(
                    {
                        "query_id": qid,
                        "conn_id": connection_match.group(1) if connection_match else "",
                        "run_id": run_id,
                        "original_query": queries.get(qid, ""),
                        "source_sse": str(filepath.relative_to(base_dir)),
                    }
                )
                output_file = output_dir / f"q{qid}_{run_id}_detail.json"
                with output_file.open("w", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                audit["outputs"] += 1
                model = result["server_meta"].get("model_slug") or "unknown"
                audit["models"][model] += 1
                audit["partial_sse"] += int(result["stream_meta"]["stream_started_mid_response"])
                audit["queries_unavailable"] += int(result["search_query_count"] == 0)
                audit["missing_candidates"] += int(result["total_candidate_urls"] == 0)
                audit["missing_citations"] += int(len(result["sources_footnote"]) == 0)
        audit["subsets"][subset_dir.name] = subset_files

    audit["models"] = dict(sorted(audit["models"].items()))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        action="append",
        help="Parse only this subset directory name; repeat to select several subsets.",
    )
    arguments = parser.parse_args()
    audit = parse_dataset(BASE_DIR, arguments.subset)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
