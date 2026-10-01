"""Interpretable classifier for dispatch_attention.

Owned by Member 3, Classification engineer:
Mojtaba Abdalitieef Ahmed (25/27660).

The target is the attention label. The six farm measures are the only inputs.
record_id and actual_yield_kg are never features. The scaler is fit on the
training rows only, and the seed is stored in the metrics file.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TEST_SIZE = 0.2
LOGISTIC_C = 1.0
MAX_ITER = 500
MODEL_VERSION = "sklearn-logreg-1"
POSITIVE_CLASS = 1
FORBIDDEN_FEATURES = ("record_id", "actual_yield_kg", "dispatch_attention")


class ClassificationError(ValueError):
    """Raised when the attention labels or the feature table cannot be trained."""


@dataclass(frozen=True)
class ClassificationFit:
    """Trained logistic regression and the rows used to score it."""

    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    classes: np.ndarray
    stratified: bool
    train_index: np.ndarray
    test_index: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pred_train: np.ndarray
    pred_test: np.ndarray
    proba_test: np.ndarray
    record_ids_test: np.ndarray
    random_seed: int
    test_size: float


def fit_classifier(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    record_ids: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> ClassificationFit:
    """Split, scale on the training rows, and fit one logistic regression."""
    frame, labels, ids = _validate_inputs(features, target, feature_names, record_ids)
    if not 0 < test_size < 1:
        raise ClassificationError("test_size must be between 0 and 1.")

    index = np.arange(len(frame))
    stratified = _can_stratify(labels, test_size)
    (
        x_train,
        x_test,
        y_train,
        y_test,
        id_train,
        id_test,
        train_index,
        test_index,
    ) = train_test_split(
        frame,
        labels,
        ids,
        index,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels if stratified else None,
    )
    del id_train
    if len(np.unique(y_train)) < 2:
        raise ClassificationError("Training rows need both attention labels, 0 and 1.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    model = LogisticRegression(C=LOGISTIC_C, solver="lbfgs", max_iter=MAX_ITER, random_state=random_seed)
    model.fit(train_scaled, y_train)

    return ClassificationFit(
        model=model,
        scaler=scaler,
        feature_names=list(frame.columns),
        classes=np.asarray(model.classes_),
        stratified=stratified,
        train_index=np.asarray(train_index),
        test_index=np.asarray(test_index),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        pred_train=model.predict(train_scaled).astype(np.int64),
        pred_test=model.predict(test_scaled).astype(np.int64),
        proba_test=_positive_probability(model, test_scaled),
        record_ids_test=np.asarray(id_test).astype(str),
        random_seed=int(random_seed),
        test_size=float(test_size),
    )


def predict_attention(features: np.ndarray, fit: ClassificationFit) -> tuple[np.ndarray, np.ndarray]:
    """Return the attention label and the probability of class 1."""
    frame = _feature_frame(features, fit.feature_names)
    scaled = fit.scaler.transform(frame)
    labels = fit.model.predict(scaled).astype(np.int64)
    probabilities = _positive_probability(fit.model, scaled)
    return labels, probabilities


def build_classification_metrics(fit: ClassificationFit, group_code: str) -> dict:
    """Build the classification_metrics.json body from a completed fit."""
    code = str(group_code).strip()
    if not code:
        raise ClassificationError("group_code is required.")

    matrix = confusion_matrix(fit.y_test, fit.pred_test, labels=[0, 1])
    return {
        "group_code": code,
        "model_version": MODEL_VERSION,
        "random_seed": fit.random_seed,
        "test_size": fit.test_size,
        "stratified": fit.stratified,
        "positive_class": POSITIVE_CLASS,
        "classifier": "logistic_regression",
        "C": LOGISTIC_C,
        "max_iter": MAX_ITER,
        "n_train": int(len(fit.train_index)),
        "n_test": int(len(fit.test_index)),
        "feature_names": list(fit.feature_names),
        "scaling": {
            "method": "standardize",
            "fit_on": "training_features_only",
            "mean": [_json_number(value) for value in fit.scaler.mean_],
            "scale": [_json_number(value) for value in fit.scaler.scale_],
        },
        "coefficients": {
            "intercept": _json_number(fit.model.intercept_[0]),
            "coefficients": [_json_number(value) for value in fit.model.coef_.ravel()],
        },
        "train_metrics": _score(fit.y_train, fit.pred_train),
        "test_metrics": _score(fit.y_test, fit.pred_test),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.astype(int).tolist(),
        },
        "test_predictions": [
            {
                "record_id": str(record_id),
                "actual_dispatch_attention": int(actual),
                "predicted_dispatch_attention": int(predicted),
                "probability_attention": _json_number(probability),
            }
            for record_id, actual, predicted, probability in zip(
                fit.record_ids_test,
                fit.y_test,
                fit.pred_test,
                fit.proba_test,
                strict=True,
            )
        ],
    }


def write_classification_metrics(metrics: dict, output_path: Path) -> Path:
    """Write classification_metrics.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return path


def write_confusion_matrix(matrix: list[list[int]], output_path: Path) -> Path:
    """Write confusion_matrix.png with both counts and the words 0 and 1."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(counts, cmap="Blues")
    axis.set_xticks([0, 1], ["0", "1"])
    axis.set_yticks([0, 1], ["0", "1"])
    axis.set_xlabel("Predicted dispatch_attention")
    axis.set_ylabel("Actual dispatch_attention")
    axis.set_title("Test confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(counts[row, column])), ha="center", va="center", color="#1A1A1A")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def save_classification_model(fit: ClassificationFit, output_path: Path) -> Path:
    """Save the logistic weights and the training-set scaler for later prediction."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "random_seed": fit.random_seed,
        "classifier": "logistic_regression",
        "C": LOGISTIC_C,
        "positive_class": POSITIVE_CLASS,
        "feature_names": list(fit.feature_names),
        "classes": [int(value) for value in fit.classes],
        "intercept": _json_number(fit.model.intercept_[0]),
        "coefficients": [_json_number(value) for value in fit.model.coef_.ravel()],
        "scaling_mean": [_json_number(value) for value in fit.scaler.mean_],
        "scaling_scale": [_json_number(value) for value in fit.scaler.scale_],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_classification(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    group_code: str,
    output_dir: Path,
    models_dir: Path,
    record_ids: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> dict:
    """Fit the classifier and write the metrics JSON, the matrix plot, and the weights."""
    fit = fit_classifier(
        features,
        target,
        feature_names,
        record_ids=record_ids,
        random_seed=random_seed,
        test_size=test_size,
    )
    metrics = build_classification_metrics(fit, group_code)
    destination = Path(output_dir)
    write_classification_metrics(metrics, destination / "classification_metrics.json")
    write_confusion_matrix(metrics["confusion_matrix"]["matrix"], destination / "confusion_matrix.png")
    save_classification_model(fit, Path(models_dir) / "classification_model.json")
    return metrics


def _validate_inputs(features, target, feature_names, record_ids):
    names = [str(name) for name in feature_names]
    frame = _feature_frame(features, names)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(labels) != len(frame):
        raise ClassificationError("Each feature row needs one dispatch_attention label.")
    if len(frame) < 2:
        raise ClassificationError("Classification needs at least two complete rows.")
    if not np.all(np.isfinite(labels)):
        raise ClassificationError("dispatch_attention labels must be finite numbers.")
    invalid = sorted({int(value) for value in labels if value not in (0.0, 1.0)})
    if invalid:
        raise ClassificationError(f"dispatch_attention must be 0 or 1. Found: {invalid}.")
    labels = labels.astype(np.int64)
    if record_ids is None:
        ids = np.array([str(index) for index in range(len(frame))])
    else:
        ids = np.asarray(record_ids).astype(str)
        if len(ids) != len(frame):
            raise ClassificationError("record_ids must have one entry per feature row.")
    return frame, labels, ids


def _feature_frame(features, feature_names: list[str]) -> pd.DataFrame:
    matrix = np.asarray(features, dtype=np.float64)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or len(names) == 0:
        raise ClassificationError("Feature matrix and feature names do not match.")
    blocked = [name for name in names if name in FORBIDDEN_FEATURES]
    if blocked:
        raise ClassificationError(f"These columns cannot be classification features: {blocked}.")
    if not np.all(np.isfinite(matrix)):
        raise ClassificationError("Classification features must be finite numbers.")
    return pd.DataFrame(matrix, columns=names)


def _can_stratify(labels: np.ndarray, test_size: float) -> bool:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 2:
        return False
    n_test = int(round(len(labels) * test_size))
    n_test = min(max(n_test, 1), len(labels) - 1)
    return n_test >= len(classes) and (len(labels) - n_test) >= len(classes)


def _positive_probability(model: LogisticRegression, scaled: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(scaled)
    column = list(model.classes_).index(POSITIVE_CLASS)
    return probabilities[:, column]


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "accuracy": _json_number(accuracy_score(actual, predicted)),
        "precision": _json_number(precision_score(actual, predicted, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": _json_number(recall_score(actual, predicted, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1": _json_number(f1_score(actual, predicted, pos_label=POSITIVE_CLASS, zero_division=0)),
    }


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number
