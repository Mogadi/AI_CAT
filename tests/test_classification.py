"""Checks for Member 3 classification. Uses a synthetic table, not the issued CSV."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from classification import (  # noqa: E402
    RANDOM_SEED,
    ClassificationError,
    fit_classifier,
    run_classification,
)

FEATURE_NAMES = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]


def synthetic_attention(n_rows=120, seed=1):
    generator = np.random.default_rng(seed)
    features = generator.normal(loc=[1.2, 80, 5.8, 180, 12, 10], scale=[0.3, 12, 0.3, 30, 4, 2], size=(n_rows, 6))
    labels = np.zeros(n_rows, dtype=np.int64)
    labels[: n_rows // 2] = 1
    features[: n_rows // 2, 0] += 1.4
    features[: n_rows // 2, 4] += 8
    order = generator.permutation(n_rows)
    record_ids = np.array([f"R{index:03d}" for index in range(n_rows)])
    return features[order], labels[order], list(FEATURE_NAMES), record_ids[order]


class ClassificationTests(unittest.TestCase):
    def test_split_is_stratified_and_target_is_attention(self):
        features, labels, names, record_ids = synthetic_attention()
        fit = fit_classifier(features, labels, names, record_ids)

        self.assertTrue(fit.stratified)
        self.assertEqual(fit.random_seed, RANDOM_SEED)
        train_rate = fit.y_train.mean()
        full_rate = labels.mean()
        self.assertAlmostEqual(train_rate, full_rate, delta=0.08)
        self.assertNotIn("actual_yield_kg", fit.feature_names)
        self.assertNotIn("dispatch_attention", fit.feature_names)
        self.assertTrue(np.all((fit.proba_test >= 0) & (fit.proba_test <= 1)))

    def test_scaler_uses_training_rows_only(self):
        features, labels, names, _ = synthetic_attention()
        fit = fit_classifier(features, labels, names, random_seed=7)
        train_mean = features[fit.train_index].mean(axis=0)
        full_mean = features.mean(axis=0)
        np.testing.assert_allclose(fit.scaler.mean_, train_mean)
        self.assertFalse(np.allclose(fit.scaler.mean_, full_mean))

    def test_same_seed_repeats_the_coefficients(self):
        features, labels, names, _ = synthetic_attention()
        first = fit_classifier(features, labels, names, random_seed=11)
        second = fit_classifier(features, labels, names, random_seed=11)
        np.testing.assert_allclose(first.model.coef_, second.model.coef_)

    def test_yield_column_is_rejected(self):
        features, labels, names, _ = synthetic_attention(n_rows=40)
        leaked = np.column_stack([features, labels.astype(float)])
        with self.assertRaises(ClassificationError):
            fit_classifier(leaked, labels, names + ["actual_yield_kg"])

    def test_outputs_include_seed_metrics_and_confusion_matrix(self):
        features, labels, names, record_ids = synthetic_attention(n_rows=80)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metrics = run_classification(
                features,
                labels,
                names,
                "AI-G03",
                root / "artifacts",
                root / "models",
                record_ids=record_ids,
            )
            saved = json.loads((root / "artifacts" / "classification_metrics.json").read_text(encoding="utf-8"))
            model = json.loads((root / "models" / "classification_model.json").read_text(encoding="utf-8"))
            plot = root / "artifacts" / "confusion_matrix.png"
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 1000)

        self.assertEqual(saved["group_code"], "AI-G03")
        self.assertEqual(saved["random_seed"], RANDOM_SEED)
        self.assertEqual(saved["scaling"]["fit_on"], "training_features_only")
        self.assertEqual(saved["positive_class"], 1)
        self.assertEqual(saved["C"], 1.0)
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(name, saved["test_metrics"])
            self.assertGreaterEqual(saved["test_metrics"][name], 0)
            self.assertLessEqual(saved["test_metrics"][name], 1)
        matrix = saved["confusion_matrix"]["matrix"]
        self.assertEqual(len(matrix), 2)
        self.assertEqual(sum(sum(row) for row in matrix), saved["n_test"])
        self.assertEqual(len(saved["test_predictions"]), saved["n_test"])
        self.assertEqual(model["model_version"], metrics["model_version"])
        self.assertGreater(saved["test_metrics"]["f1"], 0.5)

    def test_single_class_training_is_rejected(self):
        features, _, names, _ = synthetic_attention(n_rows=20)
        labels = np.zeros(20, dtype=np.int64)
        with self.assertRaises(ClassificationError):
            fit_classifier(features, labels, names)


if __name__ == "__main__":
    unittest.main()
