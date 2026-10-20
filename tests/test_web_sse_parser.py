"""Shared GPT-5.2/GPT-5.3/GPT-5.5 Web SSE parser regression tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.web_sse_parser import extract_from_sse, parse_subset


ROOT = Path(__file__).parents[1]


def full_message(
    role: str,
    *,
    content_type: str = "text",
    text: str = "",
    metadata: dict | None = None,
    message_id: str = "message",
    create_time: float = 1.0,
) -> dict:
    content = {"content_type": content_type}
    if content_type == "code":
        content["text"] = text
    else:
        content["parts"] = [text]
    return {
        "v": {
            "message": {
                "id": message_id,
                "author": {"role": role},
                "content": content,
                "metadata": metadata or {},
                "create_time": create_time,
            }
        }
    }


def write_sse(path: Path, events: list[dict | str], *, done: bool = True) -> None:
    lines = []
    for event in events:
        if isinstance(event, str):
            lines.append(f"data: {event}\n")
        else:
            lines.append(f"data: {json.dumps(event)}\n")
    if done:
        lines.append("data: [DONE]\n")
    path.write_text("".join(lines), encoding="utf-8")


def grouped_reference(url: str, *, status: str = "done") -> dict:
    return {
        "type": "grouped_webpages",
        "status": status,
        "items": [
            {
                "url": url,
                "title": "main",
                "attribution": "Wikipedia",
                "refs": [
                    {"turn_index": 0, "ref_type": "search", "ref_index": 4},
                    {"turn_index": 0, "ref_type": "search", "ref_index": 9},
                ],
                "supporting_websites": [
                    {"url": "https://support.example/evidence", "title": "support"}
                ],
            }
        ],
    }


class SharedParserSyntheticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.tmp_path = Path(self.temp_dir.name)

    def test_uses_actual_queries_and_decodes_search_argument(self) -> None:
        sse_path = self.tmp_path / "queries.txt"
        write_sse(
            sse_path,
            [
                full_message(
                    "assistant",
                    content_type="code",
                    text=r'search("\u4f60\u597d\uff0c\"A\"")',
                    message_id="code",
                ),
                full_message(
                    "tool",
                    metadata={
                        "search_model_queries": {"queries": ["actual one", "actual two"]}
                    },
                    message_id="tool",
                    create_time=2.0,
                ),
                full_message(
                    "assistant",
                    text="answer",
                    metadata={"finished_successfully": True},
                    message_id="answer",
                    create_time=3.0,
                ),
                {
                    "type": "server_ste_metadata",
                    "metadata": {
                        "model_slug": "gpt-5-3-instant",
                        "tool_name": "SonicBrowserTool",
                        "tool_invoked": True,
                    },
                },
                {"type": "message_stream_complete"},
                {
                    "type": "conversation_detail_metadata",
                    "conversation_id": "conversation",
                    "default_model_slug": "gpt-5-3-instant",
                },
            ],
        )

        result = extract_from_sse(sse_path)

        self.assertEqual(result["raw_search_queries"], ['你好，"A"'])
        self.assertEqual(result["actual_search_queries"], ["actual one", "actual two"])
        self.assertEqual(result["search_queries"], ["actual one", "actual two"])
        self.assertEqual(result["search_query_source"], "search_model_queries")
        self.assertTrue(result["stream_meta"]["capture_complete"])

    def test_replays_list_append_and_truncate(self) -> None:
        sse_path = self.tmp_path / "patches.txt"
        main_url = "https://main.example/article?utm_source=chatgpt.com"
        write_sse(
            sse_path,
            [
                full_message(
                    "assistant",
                    text="answer",
                    metadata={"content_references": [grouped_reference(main_url)]},
                    message_id="answer",
                ),
                {
                    "v": [
                        {
                            "p": "/message/metadata/content_references/0/items/0/refs",
                            "o": "truncate",
                            "v": 1,
                        },
                        {
                            "p": "/message/metadata/content_references/0/items/0/attribution",
                            "o": "truncate",
                            "v": 4,
                        },
                        {
                            "p": "/message/metadata/content_references",
                            "o": "append",
                            "v": [
                                {
                                    "type": "sources_footnote",
                                    "sources": [{"url": main_url}],
                                }
                            ],
                        },
                    ]
                },
                {"type": "message_stream_complete"},
            ],
        )

        result = extract_from_sse(sse_path)

        self.assertEqual(result["citation_source"], "sources_footnote")
        self.assertEqual(result["streaming_references"][0]["attribution"], "Wiki")
        self.assertEqual(len(result["streaming_references"][0]["ref_ids"]), 1)
        self.assertEqual(result["total_all_urls"], 2)
        self.assertEqual(result["total_cited_main"], 1)
        self.assertEqual(result["total_cited_supporting"], 1)

    def test_replays_gpt52_bare_and_cumulative_search_result_events(self) -> None:
        sse_path = self.tmp_path / "gpt52_search_results.txt"

        def result(url: str, index: int) -> dict:
            return {
                "type": "search_result",
                "url": url,
                "title": f"result {index}",
                "attribution": "example",
                "ref_id": {
                    "turn_index": 0,
                    "ref_type": "search",
                    "ref_index": index,
                },
            }

        write_sse(
            sse_path,
            [
                full_message("tool", metadata={}, message_id="tool"),
                {
                    "v": [
                        {
                            "type": "search_result_group",
                            "domain": "example",
                            "entries": [result("https://example/one", 1)],
                        },
                        result("https://example/two", 2),
                        {
                            "p": "/message/metadata/search_result_groups",
                            "v": [
                                {
                                    "domain": "example",
                                    "entries": [result("https://example/three", 3)],
                                }
                            ],
                        },
                        {
                            "p": (
                                "/message/metadata/search_result_groups/2/entries"
                            ),
                            "v": [result("https://example/four", 4)],
                        },
                    ]
                },
                full_message(
                    "assistant",
                    text="answer",
                    metadata={"finished_successfully": True},
                    message_id="answer",
                    create_time=2.0,
                ),
                {
                    "type": "server_ste_metadata",
                    "metadata": {
                        "model_slug": "gpt-5-2-instant",
                        "tool_name": "FakeSonicBrowserTool",
                        "tool_invoked": True,
                    },
                },
                {"type": "message_stream_complete"},
                {"type": "conversation_detail_metadata"},
            ],
        )

        parsed = extract_from_sse(sse_path)

        self.assertEqual(
            {item["url"] for item in parsed["search_results"]},
            {
                "https://example/one",
                "https://example/two",
                "https://example/three",
                "https://example/four",
            },
        )
        self.assertEqual(parsed["total_candidate_urls"], 4)
        self.assertTrue(parsed["stream_meta"]["capture_complete"])

    def test_includes_footnote_only_url_without_marking_it_candidate(self) -> None:
        sse_path = self.tmp_path / "footnote_only.txt"
        write_sse(
            sse_path,
            [
                full_message(
                    "assistant",
                    text="answer",
                    metadata={
                        "content_references": [
                            {
                                "type": "sources_footnote",
                                "sources": [{"url": "https://footnote.example/only"}],
                            }
                        ]
                    },
                ),
                {"type": "message_stream_complete"},
            ],
        )

        result = extract_from_sse(sse_path)

        self.assertEqual(result["total_all_urls"], 1)
        self.assertEqual(result["total_candidate_urls"], 0)
        self.assertEqual(result["total_cited"], 1)
        self.assertFalse(result["all_urls"][0]["is_search_candidate"])

    def test_incomplete_stream_does_not_use_grouped_fallback(self) -> None:
        sse_path = self.tmp_path / "incomplete.txt"
        write_sse(
            sse_path,
            [
                full_message(
                    "assistant",
                    text="answer",
                    metadata={
                        "content_references": [
                            grouped_reference("https://main.example/article")
                        ]
                    },
                )
            ],
            done=False,
        )

        result = extract_from_sse(sse_path)

        self.assertEqual(result["citation_source"], "none")
        self.assertEqual(result["total_cited"], 0)
        self.assertFalse(result["stream_meta"]["capture_complete"])

    def test_json_error_is_reported_independently(self) -> None:
        sse_path = self.tmp_path / "bad_json.txt"
        write_sse(
            sse_path,
            [
                full_message("assistant", text="answer"),
                "{not-json}",
                {"type": "message_stream_complete"},
                {"type": "conversation_detail_metadata"},
            ],
        )

        result = extract_from_sse(sse_path)

        self.assertEqual(result["stream_meta"]["json_decode_errors"], 1)
        self.assertFalse(result["stream_meta"]["capture_complete"])

    def test_legacy_53_and_55_entries_export_the_shared_parser(self) -> None:
        for index, relative_path in enumerate(
            (
                "dataset-5.3-1600/v9_analysis/extract_query_urls.py",
                "dataset-5.5-1600/v9_analysis/extract_query_urls.py",
            )
        ):
            spec = importlib.util.spec_from_file_location(
                f"shared_parser_entry_{index}", ROOT / relative_path
            )
            self.assertIsNotNone(spec)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIs(module.extract_from_sse, extract_from_sse)

    def test_subset_parser_defaults_to_read_only_and_rejects_mixed_output(self) -> None:
        source = self.tmp_path / "source"
        raw_dir = source / "geo-web-crawl" / "chatgpt" / "q001"
        raw_dir.mkdir(parents=True)
        (source / "query_list.json").write_text(
            json.dumps({"queries": [{"id": "001", "query": "question"}]}),
            encoding="utf-8",
        )
        write_sse(
            raw_dir / "chatgpt_sse_conn001_done_2026-07-22T00-00-00.txt",
            [
                full_message(
                    "assistant",
                    text="answer",
                    metadata={"finished_successfully": True},
                ),
                {"type": "server_ste_metadata", "metadata": {"model_slug": "model"}},
                {"type": "message_stream_complete"},
                {"type": "conversation_detail_metadata"},
            ],
        )

        output = self.tmp_path / "output"
        audit = parse_subset(source, output_dir=output, strict=True)
        self.assertEqual(audit["outputs_written"], 0)
        self.assertFalse(output.exists())

        unrelated = output / "q999" / "q999_existing_detail.json"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrelated detail files"):
            parse_subset(source, output_dir=output, write=True, strict=True)


class SharedParserRealSampleTest(unittest.TestCase):
    def assert_sample(self, relative_path: str) -> dict:
        path = ROOT / relative_path
        if not path.exists():
            self.skipTest(f"formal SSE sample is not present: {path}")
        return extract_from_sse(path)

    def test_gpt53_unicode_truncate_sample(self) -> None:
        result = self.assert_sample(
            "dataset-5.3-1600/v9_recommendation/geo-web-crawl/chatgpt/q017/"
            "chatgpt_sse_conn017_done_2026-07-17T09-26-34.txt"
        )
        self.assertEqual(
            result["raw_search_queries"],
            ["推荐几部类似《Fear Street》带有青少年元素的英语恐怖片，发行在2000年以后"],
        )
        self.assertEqual(
            result["actual_search_queries"],
            ["2000s teen horror films similar to Fear Street high school slasher list"],
        )
        self.assertEqual(result["server_meta"]["model_slug"], "gpt-5-3-instant")
        self.assertEqual(result["total_all_urls"], 21)
        self.assertEqual(result["total_cited"], 5)
        self.assertEqual(result["total_cited_main"], 3)
        self.assertEqual(result["total_cited_supporting"], 2)
        self.assertEqual(result["total_supporting_websites"], 3)
        self.assertEqual(result["total_sources_footnote"], 3)
        self.assertEqual(result["final_answer_segment_count"], 2)
        self.assertEqual(len(result["final_answer"]), 2508)
        self.assertEqual(
            result["final_answer"],
            "".join(segment["text"] for segment in result["final_answer_segments"]),
        )

    def test_gpt52_bare_search_result_sample(self) -> None:
        result = self.assert_sample(
            "dataset/v11_5.2/web_crawl/chatgpt/an_001/"
            "chatgpt_sse_connan_001_done_2026-05-28T15-43-40.txt"
        )
        self.assertEqual(result["server_meta"]["model_slug"], "gpt-5-2-instant")
        self.assertEqual(result["total_candidate_urls"], 61)
        self.assertEqual(result["total_all_urls"], 61)
        self.assertEqual(result["total_sources_footnote"], 5)

    def test_gpt53_three_actual_queries_sample(self) -> None:
        result = self.assert_sample(
            "dataset-5.3-1600/v9_info_synthesis/geo-web-crawl/chatgpt/q015/"
            "chatgpt_sse_conn015_done_2026-07-18T05-01-03.txt"
        )
        self.assertEqual(len(result["actual_search_queries"]), 3)
        self.assertEqual(result["total_all_urls"], 27)
        self.assertEqual(result["total_cited"], 4)

    def test_gpt53_list_group_sample(self) -> None:
        result = self.assert_sample(
            "dataset-5.3-1600/v9_recommendation/geo-web-crawl/chatgpt/q018/"
            "chatgpt_sse_conn018_done_2026-07-18T05-43-34.txt"
        )
        self.assertEqual(result["actual_search_queries"], ["500元以内 二手手机 拍照 推荐 2025"])
        self.assertEqual(result["total_all_urls"], 8)
        self.assertEqual(result["total_cited"], 1)
        self.assertEqual(
            result["cited_urls"][0]["url"], "https://en.wikipedia.org/wiki/S%26P_500"
        )

    def test_gpt55_snapshot_sample(self) -> None:
        result = self.assert_sample(
            "dataset-5.5-1600/v9_recommendation/geo-web-crawl/chatgpt/q003/"
            "chatgpt_sse_conn003_done_2026-06-12T10-10-53.txt"
        )
        self.assertEqual(len(result["actual_search_queries"]), 2)
        self.assertEqual(result["server_meta"]["model_slug"], "gpt-5-5-instant")
        self.assertEqual(result["total_all_urls"], 40)
        self.assertEqual(result["total_candidate_urls"], 39)
        self.assertEqual(result["total_cited"], 11)
        self.assertEqual(result["total_supporting_websites"], 6)

    def test_gpt55_grouped_fallback_sample(self) -> None:
        result = self.assert_sample(
            "dataset-5.5-1600/v9_analysis/geo-web-crawl/chatgpt/q001/"
            "chatgpt_sse_conn001_done_2026-06-05T19-10-34.txt"
        )
        self.assertEqual(result["citation_source"], "grouped_webpages_fallback")
        self.assertEqual(result["total_sources_footnote"], 0)
        self.assertEqual(result["total_all_urls"], 14)
        self.assertEqual(result["total_candidate_urls"], 6)
        self.assertEqual(result["total_cited"], 14)


if __name__ == "__main__":
    unittest.main()
