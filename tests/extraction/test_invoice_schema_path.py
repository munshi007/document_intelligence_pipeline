import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractor.schema_definitions import LibrarianInvoiceRecord
from extractor.schema_registry import get_schema_family, get_schema_model


class TestInvoiceSchemaPath(unittest.TestCase):
    def test_corporate_maps_to_invoice_family(self):
        self.assertEqual(get_schema_family("Corporate"), "invoice_v1")

    def test_invoice_family_maps_to_invoice_model(self):
        model = get_schema_model("invoice_v1")
        self.assertIs(model, LibrarianInvoiceRecord)


if __name__ == "__main__":
    unittest.main()
