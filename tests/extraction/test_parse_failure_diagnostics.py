import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractor.agent import ExtractorAgent, ExtractionFailureError
from extractor.schema_definitions import LibrarianUniversalHardware


class _AlwaysFailClient:
    def generate_structured(self, **kwargs):
        return None


class TestParseFailureDiagnostics(unittest.TestCase):
    def test_no_data_extracted_writes_diagnostics(self):
        agent = ExtractorAgent.__new__(ExtractorAgent)
        agent.client = _AlwaysFailClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_context = {
                "trace_dir": tmpdir,
                "batch_trace_file": os.path.join(tmpdir, "batch_trace.jsonl"),
                "parse_failures_file": os.path.join(tmpdir, "parse_failures.jsonl"),
                "validation_summary_file": os.path.join(tmpdir, "validation_summary.json"),
            }

            with self.assertRaises(ExtractionFailureError) as ctx:
                agent.extract_structured(
                    image=None,
                    prompt="Extract test",
                    response_model=LibrarianUniversalHardware,
                    domain="Hardware",
                    context_markdown="dummy content",
                    target_schema_name="LibrarianUniversalHardware",
                    target_schema_json={"title": "LibrarianUniversalHardware"},
                    trace_context=trace_context,
                )

            self.assertEqual(ctx.exception.details.get("reason"), "no_data_extracted")
            self.assertTrue(os.path.exists(trace_context["batch_trace_file"]))
            self.assertTrue(os.path.exists(trace_context["parse_failures_file"]))
            self.assertTrue(os.path.exists(trace_context["validation_summary_file"]))

            with open(trace_context["validation_summary_file"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("status"), "failed")
            self.assertEqual(summary.get("reason"), "no_data_extracted")


if __name__ == "__main__":
    unittest.main()
