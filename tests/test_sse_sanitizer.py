import json
import tempfile
import unittest
from pathlib import Path

from tools.export_processed_data import sanitize_sse_file


class SseSanitizerTest(unittest.TestCase):
    def test_removes_session_metadata_and_private_content(self) -> None:
        event = {
            "type": "resume_conversation_token",
            "token": "sample-resume-token",
            "conversation_id": "6a23af6e-66dc-83ec-8720-ef43d1ec7e40",
            "message": {
                "id": "f6cbc253-4891-422e-8d29-d851bffbde61",
                "author": {"role": "system"},
                "content": {"content_type": "text", "parts": ["private"]},
                "metadata": {
                    "request_id": "59c0ef91-ebb4-400b-b933-0a45f5a914eb",
                    "useragent": {"user_agent": "identifying browser"},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            destination = Path(directory) / "destination.txt"
            source.write_text(
                "event: delta\n"
                + "data: "
                + json.dumps(event)
                + "\n\n"
                + 'data: {"p":"/message/content/parts/0","o":"append","v":"private delta"}\n\n',
                encoding="utf-8",
            )
            sanitize_sse_file(source, destination)
            text = destination.read_text(encoding="utf-8")

        self.assertNotIn("sample-resume-token", text)
        self.assertNotIn("6a23af6e", text)
        self.assertNotIn("f6cbc253", text)
        self.assertNotIn("59c0ef91", text)
        self.assertNotIn('"parts":["private"]', text)
        self.assertNotIn("private delta", text)
        self.assertNotIn("identifying browser", text)
        self.assertIn('"token":"[redacted-token]"', text)
        self.assertIn('"conversation_id":"anon-id-', text)
        self.assertIn("[redacted-private-content]", text)
        self.assertIn("[redacted-private-content-delta]", text)


if __name__ == "__main__":
    unittest.main()
