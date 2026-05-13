import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractor.schema_engine import SchemaAuditor


class TestExplicitSchemaMode(unittest.TestCase):
    def test_build_target_schema_contract_explicit(self):
        explicit_schema = {
            "title": "RuntimeSchema",
            "type": "object",
            "properties": {
                "field_a": {"type": "string"}
            },
        }

        auditor = SchemaAuditor.__new__(SchemaAuditor)
        contract = auditor.build_target_schema_contract(discovery={}, explicit_schema=explicit_schema)

        self.assertEqual(contract["mode"], "explicit")
        self.assertEqual(contract["schema_family"], "explicit")
        self.assertEqual(contract["schema_title"], "RuntimeSchema")
        self.assertEqual(contract["schema_json"], explicit_schema)
        self.assertIsNone(contract["model_name"])


if __name__ == "__main__":
    unittest.main()
