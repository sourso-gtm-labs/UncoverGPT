"""Regression tests for the unified GPT-5.3/GPT-5.5 API response parser."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.api_response_parser import extract_response_raw, parse_pool


def sample_response() -> dict:
    return {
        "id": "resp_test",
        "status": "completed",
        "model": "gpt-5.3-chat-latest",
        "reasoning": {"effort": "medium"},
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "rewritten query",
                    "queries": [
                        "rewritten query",
                        "second rewritten query",
                    ],
                    "sources": [{"type": "url", "url": "https://example.com"}],
                },
                "results": [
                    {
                        "type": "text_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "snippet": "Evidence",
                    }
                ],
            },
            {
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Answer",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "title": "Example",
                                "start_index": 1,
                                "end_index": 6,
                            }
                        ],
                    }
                ],
            },
        ],
    }


class ApiResponseParserTest(unittest.TestCase):
    def test_extracts_legacy_compatible_fields(self) -> None:
        parsed = extract_response_raw(sample_response())
        self.assertEqual(parsed["text"], "Answer")
        self.assertEqual(
            parsed["search_queries"],
            ["rewritten query", "second rewritten query"],
        )
        self.assertEqual(parsed["n_sources"], 1)
        self.assertEqual(parsed["n_search_results"], 1)
        self.assertEqual(parsed["n_citations"], 1)
        self.assertEqual(parsed["search_results"][0]["rank"], 0)
        self.assertTrue(parsed["results_complete"])

    def test_falls_back_to_legacy_singular_search_query(self) -> None:
        response = sample_response()
        response["output"][0]["action"].pop("queries")
        parsed = extract_response_raw(response)
        self.assertEqual(parsed["search_queries"], ["rewritten query"])

    def test_pool_dry_run_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "v9_analysis"
            subset.mkdir()
            (subset / "query_list.json").write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "id": "001",
                                "query": "Original",
                                "language": "en",
                                "primary_intent": "analysis",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "v9_factual_lookup",
                "v9_info_synthesis",
                "v9_recommendation",
            ):
                other = root / name
                other.mkdir()
                (other / "query_list.json").write_text(
                    json.dumps({"queries": []}), encoding="utf-8"
                )

            source = root / "raw"
            raw_file = source / "v9_analysis" / "q001" / "run_001_test.json"
            raw_file.parent.mkdir(parents=True)
            raw_file.write_text(
                json.dumps(
                    {
                        "_meta": {
                            "subdataset": "v9_analysis",
                            "qid": "q001",
                            "raw_only": True,
                        },
                        "query": "Original",
                        "response_raw": sample_response(),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "parsed"
            audit = parse_pool(source, output, root, write=False, strict=True)
            self.assertEqual(audit["files"], 1)
            self.assertFalse(output.exists())

            audit = parse_pool(source, output, root, write=True, strict=True)
            self.assertEqual(audit["outputs_written"], 1)
            parsed = json.loads((output / raw_file.relative_to(source)).read_text())
            self.assertEqual(parsed["query"], "Original")
            self.assertTrue(parsed["_meta"]["parsed"])
            self.assertFalse(parsed["_meta"]["raw_only"])
            self.assertTrue(parsed["_meta"]["source_raw_only"])


if __name__ == "__main__":
    unittest.main()
