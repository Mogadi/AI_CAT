"""Checks for Member 1 data functions. Uses a temporary CSV, not the issued file."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    SchemaError,
    prepare_dataset,
    write_data_report,
)

HEADER = (
    "record_id,plot_area_ha,rainfall_mm,soil_ph,seed_kg,"
    "distance_km,arrival_hour,actual_yield_kg,dispatch_attention"
)


def write_csv(folder: Path, body: str) -> Path:
    path = folder / "sample.csv"
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    return path


class DataPipelineTests(unittest.TestCase):
    def test_feature_matrix_excludes_identifier_and_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv(
                Path(folder),
                "R1,1.2,80,5.6,200,12,9,1400,0\n"
                "R2,0.8,100,6.1,150,20,14,900,1\n",
            )
            prepared = prepare_dataset(path, "AI-GXX")

        self.assertEqual(prepared.features.shape, (2, 6))
        self.assertEqual(prepared.feature_names, FEATURE_COLUMNS)
        self.assertNotIn("record_id", prepared.feature_names)
        self.assertNotIn("actual_yield_kg", prepared.feature_names)
        self.assertNotIn("dispatch_attention", prepared.feature_names)
        np.testing.assert_array_equal(prepared.regression_target, [1400.0, 900.0])
        np.testing.assert_array_equal(prepared.classification_target, [0, 1])
        self.assertEqual(prepared.report["row_count"], 2)
        self.assertEqual(prepared.report["feature_count"], 6)
        self.assertEqual(prepared.report["group_code"], "AI-GXX")
        self.assertEqual(prepared.report["missing_values"]["rainfall_mm"], 0)

    def test_sha256_matches_raw_file_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv(Path(folder), "R1,1,2,3,4,5,6,7,0\n")
            prepared = prepare_dataset(path, "AI-G03")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(prepared.report["sha256"], digest)

    def test_missing_and_duplicate_values_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv(
                Path(folder),
                "R1,1.0,80,5.5,100,10,8,500,1\n"
                "R1,1.0,80,5.5,100,10,8,500,1\n"
                "R3,,80,5.5,100,10,8,500,0\n",
            )
            prepared = prepare_dataset(path, "AI-G03")

        self.assertEqual(prepared.report["row_count"], 3)
        self.assertEqual(prepared.report["missing_values"]["plot_area_ha"], 1)
        self.assertEqual(prepared.report["duplicate_record_id_count"], 1)
        self.assertEqual(prepared.report["duplicate_row_count"], 1)
        self.assertEqual(prepared.report["incomplete_row_count"], 1)
        self.assertEqual(prepared.report["complete_row_count"], 2)
        self.assertEqual(prepared.features.shape, (2, 6))

    def test_unexpected_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text(HEADER + ",notes\nR1,1,2,3,4,5,6,7,0,extra\n", encoding="utf-8")
            with self.assertRaises(SchemaError):
                prepare_dataset(path, "AI-G03")

    def test_invalid_dispatch_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv(Path(folder), "R1,1,2,3,4,5,6,7,2\n")
            with self.assertRaises(SchemaError):
                prepare_dataset(path, "AI-G03")

    def test_report_json_contains_descriptive_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = write_csv(root, "R1,1,10,5,100,4,8,600,0\nR2,3,30,7,300,8,16,1000,1\n")
            prepared = prepare_dataset(path, "AI-G03")
            output = write_data_report(prepared.report, root / "data_report.json")
            saved = json.loads(output.read_text(encoding="utf-8"))

        rainfall = saved["descriptive_statistics"]["rainfall_mm"]
        self.assertEqual(rainfall["count"], 2)
        self.assertEqual(rainfall["min"], 10.0)
        self.assertEqual(rainfall["max"], 30.0)
        self.assertEqual(rainfall["mean"], 20.0)
        self.assertIn("std", rainfall)


if __name__ == "__main__":
    unittest.main()
