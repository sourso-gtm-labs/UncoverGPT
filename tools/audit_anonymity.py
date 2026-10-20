#!/usr/bin/env python3
"""Fail when the anonymous artifact contains identities or private metadata."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH_PARTS = {
    ".env",
    "api_responses_raw",
}
IGNORED_PATH_PARTS = {"__pycache__", ".pytest_cache", ".venv"}
FORBIDDEN_PATTERNS = {
    "local absolute path": re.compile(r"(?:/home/[^/\s]+|[A-Za-z]:\\\\Users\\\\[^\\\s]+)"),
    "credential": re.compile(
        r"(?:sk-[A-Za-z0-9_-]{20,}|ntn_[A-Za-z0-9_-]{20,}|olp_[A-Za-z0-9_-]{20,})"
    ),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
ALLOWED_EMAILS = {"anonymous@invalid"}
DATA_ROOTS = {"result", "chapter_code/generated"}
FORBIDDEN_DATA_KEYS = {
    "access_token",
    "account_id",
    "api_key",
    "authorization",
    "conversation_id",
    "cookie",
    "email",
    "response_id",
    "session_id",
    "system_fingerprint",
    "user_id",
}


def is_derived_data(relative: Path) -> bool:
    posix = relative.as_posix()
    return relative.parts[0].startswith("dataset-") or any(
        posix == root or posix.startswith(root + "/") for root in DATA_ROOTS
    )


def find_forbidden_data_keys(value: object, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_DATA_KEYS:
                found.add(normalized)
            find_forbidden_data_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            find_forbidden_data_keys(item, found)


def inspect_sse_payload(value: object, private_stream: bool) -> tuple[bool, bool]:
    """Track private message streams and detect unredacted content patches."""

    exposed_patch = False
    if isinstance(value, dict):
        author = value.get("author")
        content = value.get("content")
        if isinstance(author, dict) and isinstance(content, dict):
            role = author.get("role")
            content_type = content.get("content_type")
            private_stream = bool(
                role in {"system", "developer"}
                or content_type
                in {
                    "developer_content",
                    "model_editable_context",
                    "reasoning_recap",
                    "thoughts",
                    "redacted",
                }
            )
        patch_path = value.get("p")
        patch_value = value.get("v")
        if (
            private_stream
            and isinstance(patch_path, str)
            and patch_path.startswith("/message/content")
            and isinstance(patch_value, str)
            and patch_value != "[redacted-private-content-delta]"
        ):
            exposed_patch = True
        for item in value.values():
            private_stream, child_exposed = inspect_sse_payload(item, private_stream)
            exposed_patch = exposed_patch or child_exposed
    elif isinstance(value, list):
        for item in value:
            private_stream, child_exposed = inspect_sse_payload(item, private_stream)
            exposed_patch = exposed_patch or child_exposed
    return private_stream, exposed_patch


def has_exposed_private_sse_patch(text: str) -> bool:
    import json

    private_stream = False
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        private_stream, exposed = inspect_sse_payload(payload, private_stream)
        if exposed:
            return True
    return False


def main() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PATH_PARTS for part in relative.parts):
            continue
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            failures.append(f"forbidden path: {relative}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "geo-web-crawl" in relative.parts:
            if FORBIDDEN_PATTERNS["JWT"].search(text):
                failures.append(f"JWT: {relative}")
            if '"token":"[redacted-token]"' not in text:
                failures.append(f"missing token redaction: {relative}")
            if re.search(
                r'"(?:conversation_id|request_id|turn_exchange_id|turn_trace_id|working_turn_id)"'
                r':"(?!anon-id-)[^"]+"',
                text,
            ):
                failures.append(f"unsanitized SSE identifier: {relative}")
            if has_exposed_private_sse_patch(text):
                failures.append(f"unredacted private-content patch: {relative}")
            continue
        if is_derived_data(relative):
            if path.suffix.lower() == ".json":
                import json

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    failures.append(f"invalid JSON: {relative}")
                    continue
                found: set[str] = set()
                find_forbidden_data_keys(payload, found)
                if (
                    "api_batches" in relative.parts
                    and isinstance(payload, dict)
                    and "responses" in payload
                ):
                    found.add("responses")
                if found:
                    failures.append(
                        f"sensitive data key {sorted(found)}: {relative}"
                    )
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(text):
                if label == "email address" and match.group(0) in ALLOWED_EMAILS:
                    continue
                failures.append(f"{label}: {relative}")
                break
    if failures:
        raise SystemExit("Anonymity audit failed:\n" + "\n".join(failures))
    print("Anonymity audit passed.")


if __name__ == "__main__":
    main()
