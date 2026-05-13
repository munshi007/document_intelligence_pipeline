import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractor.schema_definitions import LibrarianUniversalHardware
from extractor.schema_registry import get_schema_family, get_schema_model


class TestHardwareSchemaPath(unittest.TestCase):
    def test_industrial_maps_to_hardware_family(self):
        self.assertEqual(get_schema_family("Industrial"), "hardware_v1")

    def test_hardware_family_maps_to_hardware_model(self):
        model = get_schema_model("hardware_v1")
        self.assertIs(model, LibrarianUniversalHardware)


if __name__ == "__main__":
    unittest.main()
