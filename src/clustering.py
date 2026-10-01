"""Cluster collection rows from the input features only.

Owned by Member 4, Clustering and QA engineer:
Mohammed Osama Hasan (25/27014).

actual_yield_kg and dispatch_attention are never inputs. Features are
standardized, then k from 2 through 5 is scored with silhouette. The chosen
labels group similar measurements. They are not a verified real-world category.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
K_CANDIDATES = (2, 3, 4, 5)
MODEL_VERSION = "sklearn-kmeans-1"
FORBIDDEN_FEATURES = ("record_id", "actual_yield_kg", "dispatch_attention")
INTERPRETATION = (
    "Labels group similar input measurements. A cluster is not a verified real-world category."
)
MARKERS = ("o", "s", "^", "D", "X")


class ClusteringError(ValueError):
    """Raised when the feature table cannot be clustered."""


@dataclass(frozen=True)
class ClusteringFit:
    """Selected k-means model and a label for every input row."""

    scaler: StandardScaler
    centers: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    record_ids: np.ndarray
    scaled: np.ndarray
    silhouette_by_k: dict[int, float | None]
    selected_k: int
    random_seed: int


def fit_clusters(
    features: np.ndarray,
    feature_names: list[str],
    record_ids: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
) -> ClusteringFit:
    """Standardize every input row and keep the k with the best silhouette."""
    frame, ids = _validate_inputs(features, feature_names, record_ids)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame)
    scores = {
        k: _silhouette_for_k(scaled, k, random_seed)
        for k in K_CANDIDATES
        if k < len(frame)
    }
    selected_k = _select_k(scores)
    labels, centers = _fit_kmeans(scaled, selected_k, random_seed)
    return ClusteringFit(
        scaler=scaler,
        centers=centers,
        labels=labels.astype(np.int64),
        feature_names=list(frame.columns),
        record_ids=ids,
        scaled=scaled,
        silhouette_by_k=scores,
        selected_k=int(selected_k),
        random_seed=int(random_seed),
    )


def assign_cluster(features: np.ndarray, fit: ClusteringFit) -> np.ndarray:
    """Assign new rows to the nearest saved center in the scaled space."""
    frame = _feature_frame(features, fit.feature_names)
    scaled = fit.scaler.transform(frame)
    distances = np.linalg.norm(scaled[:, None, :] - fit.centers[None, :, :], axis=2)
    return distances.argmin(axis=1).astype(np.int64)


def build_clustering_metrics(fit: ClusteringFit, group_code: str) -> dict:
    """Build the clustering_metrics.json body from a completed fit."""
    code = str(group_code).strip()
    if not code:
        raise ClusteringError("group_code is required.")
    return {
        "group_code": code,
        "model_version": MODEL_VERSION,
        "random_seed": fit.random_seed,
        "row_count": int(len(fit.labels)),
        "feature_count": len(fit.feature_names),
        "feature_names": list(fit.feature_names),
        "scaling": {
            "method": "standardize",
            "fit_on": "all_clustered_input_rows",
            "mean": [_json_number(value) for value in fit.scaler.mean_],
            "scale": [_json_number(value) for value in fit.scaler.scale_],
        },
        "k_selection": {
            "candidates": [k for k in K_CANDIDATES if k in fit.silhouette_by_k],
            "silhouette_scores": {str(k): _json_number(score) for k, score in fit.silhouette_by_k.items()},
            "selected_k": fit.selected_k,
            "rule": "highest silhouette score; the smaller k wins a tie",
        },
        "interpretation": INTERPRETATION,
    }


def write_clustering_metrics(metrics: dict, output_path: Path) -> Path:
    """Write clustering_metrics.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return path


def write_clusters(fit: ClusteringFit, output_path: Path) -> Path:
    """Write one cluster label for every input record, in the same order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "record_id": fit.record_ids,
            "cluster_label": fit.labels,
        }
    )
    table.to_csv(path, index=False)
    return path


def write_cluster_plot(fit: ClusteringFit, output_path: Path) -> Path:
    """Write a two-dimensional view of the clusters. The fit itself uses all features."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = _display_coordinates(fit.scaled, fit.random_seed)
    plot_frame = pd.DataFrame(
        {
            "component_1": view[:, 0],
            "component_2": view[:, 1],
            "profile": [f"Profile {int(label) + 1}" for label in fit.labels],
        }
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    sns.scatterplot(
        data=plot_frame,
        x="component_1",
        y="component_2",
        hue="profile",
        style="profile",
        markers=list(MARKERS[: fit.selected_k]),
        ax=axis,
        s=36,
    )
    axis.set_title("Clusters from the input measures")
    axis.set_xlabel("Display component 1")
    axis.set_ylabel("Display component 2")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def save_clustering_model(fit: ClusteringFit, output_path: Path) -> Path:
    """Save the scaler and the centers so a later row can receive a label."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "random_seed": fit.random_seed,
        "selected_k": fit.selected_k,
        "feature_names": list(fit.feature_names),
        "scaling_mean": [_json_number(value) for value in fit.scaler.mean_],
        "scaling_scale": [_json_number(value) for value in fit.scaler.scale_],
        "centers": [[_json_number(value) for value in row] for row in fit.centers],
        "interpretation": INTERPRETATION,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_clustering(
    features: np.ndarray,
    feature_names: list[str],
    group_code: str,
    output_dir: Path,
    models_dir: Path,
    record_ids: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Fit the clusters and write the metrics, the label file, the plot, and the centers."""
    fit = fit_clusters(features, feature_names, record_ids=record_ids, random_seed=random_seed)
    metrics = build_clustering_metrics(fit, group_code)
    destination = Path(output_dir)
    write_clustering_metrics(metrics, destination / "clustering_metrics.json")
    write_clusters(fit, destination / "clusters.csv")
    write_cluster_plot(fit, destination / "cluster_plot.png")
    save_clustering_model(fit, Path(models_dir) / "clustering_model.json")
    return metrics


def _validate_inputs(features, feature_names, record_ids):
    frame = _feature_frame(features, feature_names)
    if len(frame) < 3:
        raise ClusteringError("Clustering needs at least three rows so k can start at 2.")
    if record_ids is None:
        ids = np.array([str(index) for index in range(len(frame))])
    else:
        ids = np.asarray(record_ids).astype(str)
        if len(ids) != len(frame):
            raise ClusteringError("record_ids must have one entry per feature row.")
    return frame, ids


def _feature_frame(features, feature_names: list[str]) -> pd.DataFrame:
    matrix = np.asarray(features, dtype=np.float64)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or len(names) == 0:
        raise ClusteringError("Feature matrix and feature names do not match.")
    blocked = [name for name in names if name in FORBIDDEN_FEATURES]
    if blocked:
        raise ClusteringError(f"These columns cannot be clustering inputs: {blocked}.")
    if not np.all(np.isfinite(matrix)):
        raise ClusteringError("Clustering features must be finite numbers.")
    return pd.DataFrame(matrix, columns=names)


def _fit_kmeans(scaled: np.ndarray, k: int, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_seed)
    labels = model.fit_predict(scaled)
    return labels, np.asarray(model.cluster_centers_, dtype=np.float64)


def _silhouette_for_k(scaled: np.ndarray, k: int, random_seed: int) -> float | None:
    if k >= len(scaled):
        return None
    labels, _ = _fit_kmeans(scaled, k, random_seed)
    if len(np.unique(labels)) < 2:
        return None
    score = float(silhouette_score(scaled, labels))
    if not np.isfinite(score):
        return None
    return score


def _select_k(scores: dict[int, float | None]) -> int:
    usable = {k: score for k, score in scores.items() if score is not None}
    if not usable:
        raise ClusteringError("No silhouette score could be computed for k from 2 through 5.")
    return max(usable, key=lambda k: (usable[k], -k))


def _display_coordinates(scaled: np.ndarray, random_seed: int) -> np.ndarray:
    if scaled.shape[1] == 1:
        return np.column_stack([scaled[:, 0], np.zeros(len(scaled))])
    components = min(2, scaled.shape[0], scaled.shape[1])
    view = PCA(n_components=components, random_state=random_seed).fit_transform(scaled)
    if view.shape[1] == 1:
        view = np.column_stack([view[:, 0], np.zeros(len(view))])
    return view


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number
