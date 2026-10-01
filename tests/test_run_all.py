"""Checks for Member 5's pipeline command. Uses a temporary CSV, not the issued file."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "record_id,plot_area_ha,rainfall_mm,soil_ph,seed_kg,"
    "distance_km,arrival_hour,actual_yield_kg,dispatch_attention"
)
ARTIFACTS = (
    "data_report.json",
    "regression_metrics.json",
    "regression_loss.png",
    "classification_metrics.json",
    "confusion_matrix.png",
    "clustering_metrics.json",
    "clusters.csv",
    "cluster_plot.png",
)


def write_batch(folder: Path, n_rows: int = 48) -> Path:
    generator = np.random.default_rng(3)
    lines = [HEADER]
    for index in range(n_rows):
        attention = index % 2
        area = 0.8 + attention + generator.normal(scale=0.05)
        rain = 70 + attention * 30 + generator.normal(scale=2)
        ph = 5.5 + generator.normal(scale=0.05)
        seed = 140 + attention * 40 + generator.normal(scale=3)
        distance = 8 + attention * 10 + generator.normal(scale=0.4)
        hour = 8 + (index % 6)
        yield_kg = 800 + area * 300 + generator.normal(scale=5)
        lines.append(
            f"R{index:03d},{area:.4f},{rain:.2f},{ph:.3f},{seed:.2f},{distance:.3f},{hour},{yield_kg:.2f},{attention}"
        )
    path = folder / "batch.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class RunAllTests(unittest.TestCase):
    def test_command_prints_group_code_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = write_batch(root)
            output = root / "artifacts"
            models = root / "models"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_all.py"),
                    "--data",
                    str(data),
                    "--output",
                    str(output),
                    "--group",
                    "AI-G03",
                    "--models",
                    str(models),
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            fingerprint = hashlib.sha256(data.read_bytes()).hexdigest()
            self.assertIn("Group code: AI-G03", completed.stdout)
            self.assertIn(f"Dataset SHA-256: {fingerprint}", completed.stdout)
            for name in ARTIFACTS:
                self.assertTrue((output / name).is_file(), name)
            report = json.loads((output / "data_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["sha256"], fingerprint)
            self.assertEqual(report["group_code"], "AI-G03")
            self.assertEqual(report["row_count"], 48)
            regression = json.loads((output / "regression_metrics.json").read_text(encoding="utf-8"))
            classification = json.loads((output / "classification_metrics.json").read_text(encoding="utf-8"))
            clustering = json.loads((output / "clustering_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(regression["random_seed"], 42)
            self.assertEqual(classification["random_seed"], 42)
            self.assertEqual(clustering["random_seed"], 42)
            self.assertNotIn("actual_yield_kg", clustering["feature_names"])
            self.assertNotIn("dispatch_attention", clustering["feature_names"])
            for name in ("regression_model.json", "classification_model.json", "clustering_model.json"):
                self.assertTrue((models / name).is_file(), name)

            rerun = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_all.py"),
                    "--data",
                    str(data),
                    "--output",
                    str(output),
                    "--group",
                    "AI-G03",
                    "--models",
                    str(models),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(rerun.returncode, 0, rerun.stderr)

    def test_bad_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "bad.csv"
            data.write_text("record_id,notes\nR1,extra\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_all.py"),
                    "--data",
                    str(data),
                    "--output",
                    str(root / "artifacts"),
                    "--group",
                    "AI-G03",
                    "--models",
                    str(root / "models"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("Error:", completed.stderr)
        self.assertNotIn("Dataset SHA-256:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
