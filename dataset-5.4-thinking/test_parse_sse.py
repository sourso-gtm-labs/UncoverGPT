#!/usr/bin/env python3
"""Regression tests for representative GPT-5.4 Thinking SSE variants."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("parse_5_4_sse", BASE_DIR / "parse_sse.py")
assert SPEC and SPEC.loader
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)

GPT55_SPEC = importlib.util.spec_from_file_location(
    "parse_5_5_sse",
    BASE_DIR.parent / "dataset-5.5-1600" / "v9_analysis" / "extract_query_urls.py",
)
assert GPT55_SPEC and GPT55_SPEC.loader
GPT55_PARSER = importlib.util.module_from_spec(GPT55_SPEC)
GPT55_SPEC.loader.exec_module(GPT55_PARSER)


class ParseSseTest(unittest.TestCase):
    def _parse_events(self, events: list[object]) -> dict:
        lines = []
        for event in events:
            payload = event if isinstance(event, str) else json.dumps(event)
            lines.append(f"data: {payload}\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.txt"
            path.write_text("".join(lines), encoding="utf-8")
            return PARSER.extract_from_sse(path)

    def test_multi_round_search_and_non_search_call(self) -> None:
        filepath = (
            BASE_DIR
            / "v10_info_synthesis_5.4/geo-web-crawl/chatgpt/q003"
            / "chatgpt_sse_conn003_done_2026-04-12T17-56-27.txt"
        )
        if not filepath.exists():
            self.skipTest("Private SSE fixture is not distributed with the code-only release")
        result = PARSER.extract_from_sse(filepath)

        self.assertEqual(result["server_meta"]["model_slug"], "gpt-5-4-thinking")
        self.assertEqual(result["server_meta"]["thinking_effort"], "standard")
        self.assertFalse(result["server_meta"]["did_auto_switch_to_reasoning"])
        self.assertEqual(result["search_round_count"], 11)
        self.assertEqual(result["search_query_count"], 44)
        self.assertEqual(result["web_tool_call_count"], 12)
        self.assertEqual(result["web_tool_operation_counts"], {"open": 1, "search_query": 11})
        self.assertEqual(
            result["total_search_result_occurrences"], len(result["search_results"])
        )
        self.assertTrue(
            all(
                {"turn_index", "ref_type", "ref_index"} <= set(item["ref_id"])
                for item in result["search_results"]
            )
        )

    def test_tool_metadata_recovers_hidden_search_calls(self) -> None:
        filepath = (
            BASE_DIR
            / "v10_recommendation_5.4/geo-web-crawl/chatgpt/q001"
            / "chatgpt_sse_conn001_done_2026-04-10T17-00-04.txt"
        )
        if not filepath.exists():
            self.skipTest("Private SSE fixture is not distributed with the code-only release")
        result = PARSER.extract_from_sse(filepath)

        self.assertEqual(result["web_tool_call_count"], 0)
        self.assertEqual(result["search_round_count"], 3)
        self.assertEqual(result["search_query_count"], 12)
        self.assertTrue(
            all(search_round["source"] == "tool_metadata" for search_round in result["search_rounds"])
        )

    def test_partial_capture_keeps_candidates_and_citations(self) -> None:
        result = self._parse_events(
            [
                {"p": "/message/content/parts/0", "o": "append", "v": "partial answer"},
                {
                    "p": "/message/metadata",
                    "o": "append",
                    "v": {
                        "search_result_groups": [
                            {
                                "domain": "example.com",
                                "entries": [
                                    {
                                        "url": "https://example.com/result",
                                        "title": "Result",
                                        "ref_id": {
                                            "turn_index": 0,
                                            "ref_type": "search",
                                            "ref_index": 0,
                                        },
                                    }
                                ],
                            }
                        ],
                        "content_references": [
                            {
                                "type": "sources_footnote",
                                "sources": [
                                    {
                                        "url": "https://example.com/result",
                                        "title": "Result",
                                        "attribution": "Example",
                                    }
                                ],
                            }
                        ],
                    },
                },
                "[DONE]",
            ]
        )

        self.assertTrue(result["stream_meta"]["stream_started_mid_response"])
        self.assertEqual(result["stream_meta"]["search_query_status"], "unavailable_partial_sse")
        self.assertEqual(result["search_query_count"], 0)
        self.assertGreater(result["total_candidate_urls"], 0)
        self.assertGreater(result["total_sources_footnote"], 0)
        self.assertTrue(result["final_answer_is_partial"])

    def test_url_normalization_only_removes_known_tracking(self) -> None:
        url = "HTTPS://Example.COM/a?utm_source=chatgpt.com&x=1&x=2#fragment"
        self.assertEqual(PARSER.normalize_url(url), "https://example.com/a?x=1&x=2")

    def test_top_level_scalar_patch_reconstructs_answer(self) -> None:
        result = self._parse_events(
            [
                {
                    "v": {
                        "message": {
                            "id": "answer",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Hel"]},
                            "metadata": {},
                        }
                    }
                },
                {"p": "/message/content/parts/0", "o": "append", "v": "lo"},
                "[DONE]",
            ]
        )
        self.assertEqual(result["final_answer"], "Hello")

    def test_candidate_pool_unions_tool_results_and_final_metadata(self) -> None:
        def group(url: str, turn: int, index: int) -> dict:
            return {
                "domain": "example.com",
                "entries": [
                    {
                        "url": url,
                        "title": url.rsplit("/", 1)[-1],
                        "ref_id": {
                            "turn_index": turn,
                            "ref_type": "search",
                            "ref_index": index,
                        },
                    }
                ],
            }

        result = self._parse_events(
            [
                {
                    "v": {
                        "message": {
                            "id": "tool",
                            "author": {"role": "tool", "name": "web.run"},
                            "content": {"content_type": "text", "parts": [""]},
                            "metadata": {
                                "search_result_groups": [group("https://example.com/early", 7, 0)]
                            },
                        }
                    }
                },
                {
                    "v": {
                        "message": {
                            "id": "answer",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["answer"]},
                            "metadata": {},
                        }
                    }
                },
                {
                    "p": "/message/metadata",
                    "v": {
                        "search_result_groups": [group("https://example.com/final", 0, 0)]
                    },
                },
                "[DONE]",
            ]
        )
        self.assertEqual(
            {item["url"] for item in result["all_urls"]},
            {"https://example.com/early", "https://example.com/final"},
        )
        self.assertEqual(result["total_tool_candidate_urls"], 1)
        self.assertEqual(result["total_final_metadata_candidate_urls"], 1)

    def test_supporting_label_matches_gpt53_gpt55_even_when_directly_footnoted(self) -> None:
        url = "https://example.com/cited"
        records = PARSER._build_url_records(
            [],
            [{"url": url, "raw_url": url, "title": "Cited", "attribution": "Example"}],
            [],
            [
                {
                    "url": url,
                    "raw_url": url,
                    "title": "Cited",
                    "snippet": "",
                    "attribution": "Example",
                    "pub_date": None,
                    "parent_url": "https://example.com/parent",
                    "parent_ref_id": {},
                    "parent_ref_index": 0,
                }
            ],
            {url},
        )
        self.assertTrue(records[0]["cited"])
        self.assertEqual(records[0]["cited_type"], "supporting")
        self.assertTrue(records[0]["in_supporting"])
        self.assertEqual(records[0]["parent_url"], "https://example.com/parent")

    def test_completed_grouped_webpages_fallback_matches_gpt55(self) -> None:
        main_url = "https://main.example/article"
        supporting_url = "https://support.example/evidence"
        result = self._parse_events(
            [
                {
                    "v": {
                        "message": {
                            "id": "answer",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["answer"]},
                            "metadata": {
                                "content_references": [
                                    {
                                        "type": "grouped_webpages",
                                        "status": "done",
                                        "items": [
                                            {
                                                "url": main_url,
                                                "refs": [
                                                    {
                                                        "turn_index": 0,
                                                        "ref_type": "search",
                                                        "ref_index": 0,
                                                    }
                                                ],
                                                "supporting_websites": [
                                                    {"url": supporting_url}
                                                ],
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    }
                },
                {"type": "message_stream_complete"},
                "[DONE]",
            ]
        )
        self.assertEqual(result["citation_source"], "grouped_webpages_fallback")
        self.assertEqual(
            {item["url"] for item in result["cited_urls"]},
            {main_url, supporting_url},
        )

    def test_incomplete_grouped_webpages_does_not_fallback(self) -> None:
        result = self._parse_events(
            [
                {
                    "v": {
                        "message": {
                            "id": "answer",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["answer"]},
                            "metadata": {
                                "content_references": [
                                    {
                                        "type": "grouped_webpages",
                                        "status": "done",
                                        "items": [
                                            {
                                                "url": "https://main.example/article",
                                                "refs": [
                                                    {
                                                        "turn_index": 0,
                                                        "ref_type": "search",
                                                        "ref_index": 0,
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    }
                },
                "[DONE]",
            ]
        )
        self.assertEqual(result["citation_source"], "none")
        self.assertEqual(result["total_cited"], 0)

    def test_sources_footnote_remains_authoritative(self) -> None:
        result = self._parse_events(
            [
                {
                    "v": {
                        "message": {
                            "id": "answer",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["answer"]},
                            "metadata": {
                                "content_references": [
                                    {
                                        "type": "grouped_webpages",
                                        "status": "done",
                                        "items": [
                                            {
                                                "url": "https://main.example/retracted",
                                                "refs": [
                                                    {
                                                        "turn_index": 0,
                                                        "ref_type": "search",
                                                        "ref_index": 0,
                                                    }
                                                ],
                                            },
                                            {
                                                "url": "https://main.example/final",
                                                "refs": [
                                                    {
                                                        "turn_index": 0,
                                                        "ref_type": "search",
                                                        "ref_index": 1,
                                                    }
                                                ],
                                            },
                                        ],
                                    },
                                    {
                                        "type": "sources_footnote",
                                        "sources": [
                                            {"url": "https://main.example/final"}
                                        ],
                                    },
                                ]
                            },
                        }
                    }
                },
                {"type": "message_stream_complete"},
                "[DONE]",
            ]
        )
        self.assertEqual(result["citation_source"], "sources_footnote")
        self.assertEqual(
            [item["url"] for item in result["cited_urls"]],
            ["https://main.example/final"],
        )

    def test_full_assistant_metadata_contributes_candidates(self) -> None:
        result = self._parse_events(
            [
                {
                    "v": {
                        "message": {
                            "id": "recap",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "reasoning_recap", "parts": []},
                            "metadata": {
                                "search_result_groups": [
                                    {
                                        "domain": "example.com",
                                        "entries": [
                                            {
                                                "url": "https://example.com/from-recap",
                                                "ref_id": {
                                                    "turn_index": 3,
                                                    "ref_type": "search",
                                                    "ref_index": 4,
                                                },
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    }
                },
                "[DONE]",
            ]
        )
        self.assertEqual(
            [item["url"] for item in result["all_urls"]],
            ["https://example.com/from-recap"],
        )

    def test_citation_contract_matches_gpt55(self) -> None:
        parent_url = "https://example.com/parent"
        direct_and_supporting_url = "https://example.com/direct-and-supporting"
        support_only_url = "https://example.com/support-only"
        content_references = [
            {
                "type": "grouped_webpages",
                "status": "done",
                "items": [
                    {
                        "url": parent_url,
                        "refs": [
                            {"turn_index": 0, "ref_type": "search", "ref_index": 0}
                        ],
                        "supporting_websites": [
                            {"url": direct_and_supporting_url},
                            {"url": support_only_url},
                        ],
                    },
                    {
                        "url": direct_and_supporting_url,
                        "refs": [
                            {"turn_index": 0, "ref_type": "search", "ref_index": 1}
                        ],
                    },
                ],
            },
            {
                "type": "sources_footnote",
                "sources": [
                    {"url": parent_url},
                    {"url": direct_and_supporting_url},
                ],
            },
        ]
        events = [
            {
                "v": {
                    "message": {
                        "id": "answer",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["answer"]},
                        "metadata": {"content_references": content_references},
                    }
                }
            },
            {"type": "message_stream_complete"},
            "[DONE]",
        ]
        lines = [
            f"data: {event if isinstance(event, str) else json.dumps(event)}\n"
            for event in events
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "citation-contract.txt"
            path.write_text("".join(lines), encoding="utf-8")
            result_54 = PARSER.extract_from_sse(path)
            result_55 = GPT55_PARSER.extract_from_sse(str(path))

        def citation_contract(result: dict) -> dict:
            return {
                "citation_source": result["citation_source"],
                "total_cited": result["total_cited"],
                "total_cited_main": result["total_cited_main"],
                "total_cited_supporting": result["total_cited_supporting"],
                "urls": {
                    item["url"]: {
                        "cited": item["cited"],
                        "cited_type": item["cited_type"],
                        "parent_url": item.get("parent_url"),
                    }
                    for item in result["all_urls"]
                },
            }

        self.assertEqual(citation_contract(result_54), citation_contract(result_55))

    def test_product_query_is_counted_as_a_search_round(self) -> None:
        call = {
            "event_index": 1,
            "tool_call_index": 0,
            "payload": {
                "product_query": {"search": ["eggs ten pack", "eggs twelve pack"]}
            },
        }
        rounds, records = PARSER._extract_search_rounds([], [call])
        self.assertEqual(len(rounds), 1)
        self.assertEqual([record["query"] for record in records], ["eggs ten pack", "eggs twelve pack"])

    def test_view_reference_token_fallback(self) -> None:
        self.assertEqual(
            PARSER._tokens_from_matched_text("cite turn12view3"),
            [{"turn_index": 12, "ref_type": "view", "ref_index": 3}],
        )


if __name__ == "__main__":
    unittest.main()
