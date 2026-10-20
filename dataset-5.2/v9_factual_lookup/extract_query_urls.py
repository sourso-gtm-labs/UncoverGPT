#!/usr/bin/env python3
"""Compatibility entry point for the shared GPT-5.2/GPT-5.3/GPT-5.5 parser."""

from pathlib import Path
import sys


ENTRY_PATH = Path(__file__).resolve()
REPO_ROOT = ENTRY_PATH.parents[1] if ENTRY_PATH.parent.name == "scripts" else ENTRY_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.web_sse_parser import (
    extract_from_sse,
    main as _shared_main,
    normalize_url,
    parse_subset,
)


def main() -> None:
    _shared_main(ENTRY_PATH.parent)


if __name__ == "__main__":
    main()
