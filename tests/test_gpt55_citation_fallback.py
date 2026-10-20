"""Regression tests for GPT-5.5 SSE citation finalization."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PARSER_PATH = (
    Path(__file__).parents[1]
    / "dataset-5.5-1600"
    / "v9_analysis"
    / "extract_query_urls.py"
)


def load_parser():
    spec = importlib.util.spec_from_file_location("gpt55_extract_query_urls", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_sse(path: Path, content_references: list, complete: bool) -> None:
    events = [
        {
            "v": {
                "message": {
                    "metadata": {"content_references": content_references},
                    "content": {},
                }
            }
        }
    ]
    if complete:
        events.append({"type": "message_stream_complete"})
    path.write_text(
        "".join(f"data: {json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )


def grouped_reference(*urls: str) -> dict:
    return {
        "type": "grouped_webpages",
        "status": "done",
        "items": [
            {
                "url": url,
                "title": url.rsplit("/", 1)[-1],
                "refs": [{"ref_type": "search", "ref_index": index}],
                "supporting_websites": (
                    [{"url": "https://support.example/evidence"}]
                    if index == 0
                    else []
                ),
            }
            for index, url in enumerate(urls)
        ],
    }


class CitationFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = load_parser()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.tmp_path = Path(self.temp_dir.name)

    def test_completed_grouped_webpages_fallback(self) -> None:
        sse_path = self.tmp_path / "fallback.txt"
        write_sse(
            sse_path,
            [grouped_reference("https://main.example/article")],
            True,
        )

        result = self.parser.extract_from_sse(str(sse_path))

        self.assertEqual(result["citation_source"], "grouped_webpages_fallback")
        self.assertEqual(result["total_cited_main"], 1)
        self.assertEqual(result["total_cited_supporting"], 1)
        self.assertEqual(
            {item["url"] for item in result["cited_urls"]},
            {
                "https://main.example/article",
                "https://support.example/evidence",
            },
        )

    def test_incomplete_response_does_not_fallback(self) -> None:
        sse_path = self.tmp_path / "incomplete.txt"
        write_sse(
            sse_path,
            [grouped_reference("https://main.example/article")],
            False,
        )

        result = self.parser.extract_from_sse(str(sse_path))

        self.assertEqual(result["citation_source"], "none")
        self.assertEqual(result["total_cited"], 0)

    def test_sources_footnote_remains_authoritative(self) -> None:
        sse_path = self.tmp_path / "footnote.txt"
        references = [
            grouped_reference(
                "https://main.example/retracted",
                "https://main.example/final",
            ),
            {
                "type": "sources_footnote",
                "sources": [{"url": "https://main.example/final"}],
            },
        ]
        write_sse(sse_path, references, True)

        result = self.parser.extract_from_sse(str(sse_path))

        self.assertEqual(result["citation_source"], "sources_footnote")
        self.assertEqual(
            [item["url"] for item in result["cited_urls"]],
            ["https://main.example/final"],
        )


if __name__ == "__main__":
    unittest.main()
