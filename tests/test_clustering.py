"""Checks for Member 4 clustering. Uses a synthetic table, not the issued CSV."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from clustering import (  # noqa: E402
    RANDOM_SEED,
    ClusteringError,
    _select_k,
    assign_cluster,
    fit_clusters,
    run_clustering,
)

FEATURE_NAMES = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]


def synthetic_profiles(n_rows=90, seed=2):
    generator = np.random.default_rng(seed)
    block = n_rows // 3
    centers = np.array(
        [
            [0.6, 40, 5.2, 80, 4, 8],
            [1.4, 90, 6.0, 200, 16, 11],
            [2.2, 140, 6.8, 320, 28, 15],
        ]
    )
    features = np.vstack(
        [
            generator.normal(loc=centers[0], scale=[0.05, 3, 0.05, 8, 0.6, 0.3], size=(block, 6)),
            generator.normal(loc=centers[1], scale=[0.05, 3, 0.05, 8, 0.6, 0.3], size=(block, 6)),
            generator.normal(loc=centers[2], scale=[0.05, 3, 0.05, 8, 0.6, 0.3], size=(n_rows - 2 * block, 6)),
        ]
    )
    record_ids = np.array([f"R{index:03d}" for index in range(n_rows)])
    return features, list(FEATURE_NAMES), record_ids


class ClusteringTests(unittest.TestCase):
    def test_every_row_is_labeled_from_inputs_only(self):
        features, names, record_ids = synthetic_profiles()
        fit = fit_clusters(features, names, record_ids)

        self.assertEqual(len(fit.labels), len(features))
        self.assertEqual(set(fit.silhouette_by_k), {2, 3, 4, 5})
        self.assertIn(fit.selected_k, {2, 3, 4, 5})
        self.assertTrue(set(fit.labels).issubset(set(range(fit.selected_k))))
        self.assertEqual(list(fit.record_ids), list(record_ids))
        self.assertNotIn("actual_yield_kg", fit.feature_names)
        self.assertNotIn("dispatch_attention", fit.feature_names)
        scores = {k: score for k, score in fit.silhouette_by_k.items() if score is not None}
        self.assertEqual(fit.selected_k, max(scores, key=lambda k: (scores[k], -k)))

    def test_scaler_uses_the_clustered_rows(self):
        features, names, _ = synthetic_profiles()
        fit = fit_clusters(features, names)
        np.testing.assert_allclose(fit.scaler.mean_, features.mean(axis=0))

    def test_targets_are_rejected(self):
        features, names, _ = synthetic_profiles(n_rows=30)
        leaked = np.column_stack([features, np.ones(len(features))])
        with self.assertRaises(ClusteringError):
            fit_clusters(leaked, names + ["dispatch_attention"])

    def test_same_seed_repeats_the_labels(self):
        features, names, _ = synthetic_profiles()
        first = fit_clusters(features, names, random_seed=11)
        second = fit_clusters(features, names, random_seed=11)
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_new_row_matches_the_nearest_saved_center(self):
        features, names, _ = synthetic_profiles()
        fit = fit_clusters(features, names, random_seed=RANDOM_SEED)
        assigned = assign_cluster(features[:5], fit)
        np.testing.assert_array_equal(assigned, fit.labels[:5])

    def test_tie_keeps_the_smaller_k(self):
        self.assertEqual(_select_k({2: 0.4, 3: 0.4, 4: 0.2, 5: None}), 2)

    def test_outputs_include_scores_labels_and_plot(self):
        features, names, record_ids = synthetic_profiles(n_rows=60)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metrics = run_clustering(
                features,
                names,
                "AI-G03",
                root / "artifacts",
                root / "models",
                record_ids=record_ids,
            )
            saved = json.loads((root / "artifacts" / "clustering_metrics.json").read_text(encoding="utf-8"))
            clusters = pd.read_csv(root / "artifacts" / "clusters.csv")
            model = json.loads((root / "models" / "clustering_model.json").read_text(encoding="utf-8"))
            plot = root / "artifacts" / "cluster_plot.png"
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 1000)

        self.assertEqual(saved["group_code"], "AI-G03")
        self.assertEqual(saved["random_seed"], RANDOM_SEED)
        self.assertEqual(saved["scaling"]["fit_on"], "all_clustered_input_rows")
        self.assertEqual(saved["row_count"], 60)
        self.assertEqual(len(clusters), 60)
        self.assertEqual(list(clusters["record_id"]), list(record_ids))
        self.assertIn("verified real-world category", saved["interpretation"])
        for k in ("2", "3", "4", "5"):
            self.assertIn(k, saved["k_selection"]["silhouette_scores"])
        self.assertEqual(model["selected_k"], metrics["k_selection"]["selected_k"])
        self.assertEqual(len(model["centers"]), saved["k_selection"]["selected_k"])


if __name__ == "__main__":
    unittest.main()
