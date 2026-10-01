"""Checks for Member 2 regression. Uses a synthetic table, not the issued CSV."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from regression import (  # noqa: E402
    RANDOM_SEED,
    RegressionError,
    batch_gradient,
    fit_linear_regression,
    mean_half_squared_error,
    run_regression,
)


def synthetic_harvest(n_rows=90, seed=0):
    generator = np.random.default_rng(seed)
    features = generator.normal(loc=[1.2, 80, 5.8, 180, 12, 10], scale=[0.4, 15, 0.4, 40, 6, 3], size=(n_rows, 6))
    coefficients = np.array([420.0, 2.5, -30.0, 1.1, -4.0, 3.0])
    target = 300.0 + features @ coefficients + generator.normal(scale=8.0, size=n_rows)
    names = [
        "plot_area_ha",
        "rainfall_mm",
        "soil_ph",
        "seed_kg",
        "distance_km",
        "arrival_hour",
    ]
    record_ids = np.array([f"R{index:03d}" for index in range(n_rows)])
    return features, target, names, record_ids


class RegressionTests(unittest.TestCase):
    def test_loss_falls_and_test_metrics_are_finite(self):
        features, target, names, record_ids = synthetic_harvest()
        fit = fit_linear_regression(features, target, names, record_ids)

        self.assertEqual(fit.random_seed, RANDOM_SEED)
        self.assertGreater(fit.loss_history[0], fit.loss_history[-1])
        self.assertEqual(len(fit.loss_history), fit.n_iterations + 1)
        self.assertEqual(len(fit.weights), 7)
        mae = float(np.mean(np.abs(fit.pred_test - fit.y_test)))
        r_squared = 1.0 - np.sum((fit.pred_test - fit.y_test) ** 2) / np.sum((fit.y_test - fit.y_test.mean()) ** 2)
        self.assertTrue(np.isfinite(mae))
        self.assertGreater(r_squared, 0.9)

    def test_scaler_uses_training_rows_only(self):
        features, target, names, _ = synthetic_harvest()
        fit = fit_linear_regression(features, target, names, random_seed=7)
        train_mean = features[fit.train_index].mean(axis=0)
        full_mean = features.mean(axis=0)
        np.testing.assert_allclose(fit.scaling_mean, train_mean)
        self.assertFalse(np.allclose(fit.scaling_mean, full_mean))

    def test_same_seed_repeats_the_weights(self):
        features, target, names, _ = synthetic_harvest()
        first = fit_linear_regression(features, target, names, random_seed=11, n_iterations=40)
        second = fit_linear_regression(features, target, names, random_seed=11, n_iterations=40)
        np.testing.assert_allclose(first.weights, second.weights)

    def test_analytic_gradient_matches_a_numerical_check(self):
        design = np.array([[1.0, 0.2, -0.4], [1.0, 1.5, 0.3], [1.0, -0.7, 0.8]])
        target = np.array([2.0, 0.5, 1.2])
        weights = np.array([0.4, -0.2, 0.7])
        analytic = batch_gradient(design, target, weights)
        numerical = np.zeros_like(weights)
        step = 1e-6
        for index in range(len(weights)):
            up = weights.copy()
            down = weights.copy()
            up[index] += step
            down[index] -= step
            numerical[index] = (
                mean_half_squared_error(design, target, up)
                - mean_half_squared_error(design, target, down)
            ) / (2 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-7)

    def test_outputs_include_seed_metrics_and_loss_plot(self):
        features, target, names, record_ids = synthetic_harvest(n_rows=40)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metrics = run_regression(
                features,
                target,
                names,
                "AI-G03",
                root / "artifacts",
                root / "models",
                record_ids=record_ids,
                n_iterations=50,
            )
            saved = json.loads((root / "artifacts" / "regression_metrics.json").read_text(encoding="utf-8"))
            model = json.loads((root / "models" / "regression_model.json").read_text(encoding="utf-8"))
            plot = root / "artifacts" / "regression_loss.png"
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 1000)

        self.assertEqual(saved["group_code"], "AI-G03")
        self.assertEqual(saved["random_seed"], RANDOM_SEED)
        self.assertEqual(saved["scaling"]["fit_on"], "training_features_only")
        self.assertIn("mae", saved["test_metrics"])
        self.assertIn("rmse", saved["test_metrics"])
        self.assertIn("r_squared", saved["test_metrics"])
        self.assertEqual(len(saved["test_predictions"]), saved["n_test"])
        self.assertEqual(len(saved["loss"]["history"]), saved["n_iterations"] + 1)
        self.assertEqual(model["model_version"], metrics["model_version"])
        self.assertNotIn("actual_yield_kg", saved["feature_names"])
        self.assertNotIn("dispatch_attention", saved["feature_names"])

    def test_bad_learning_rate_is_rejected(self):
        features, target, names, _ = synthetic_harvest(n_rows=10)
        with self.assertRaises(RegressionError):
            fit_linear_regression(features, target, names, learning_rate=0)

    def test_module_does_not_call_a_library_regressor(self):
        source = (SRC / "regression.py").read_text(encoding="utf-8")
        self.assertNotIn("sklearn", source)
        self.assertNotIn("LinearRegression", source)


if __name__ == "__main__":
    unittest.main()
