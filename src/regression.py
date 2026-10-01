"""NumPy linear regression trained with batch gradient descent.

Owned by Member 2, Regression engineer:
Ehab Fakhralden Mohamed Hamid (25/27950).

The section does not use a library regression estimator. Feature means and
standard deviations are computed on the training rows only. The fixed seed is
stored in the metrics file. The loss and the update are derived in
notes/regression_derivation.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 42
LEARNING_RATE = 0.05
N_ITERATIONS = 800
TEST_SIZE = 0.2
MODEL_VERSION = "numpy-gd-1"


class RegressionError(ValueError):
    """Raised when the regression inputs or the training run are unusable."""


@dataclass(frozen=True)
class RegressionFit:
    """Weights and the training record needed to score and plot the model."""

    weights: np.ndarray
    feature_names: list[str]
    scaling_mean: np.ndarray
    scaling_std: np.ndarray
    loss_history: list[float]
    train_index: np.ndarray
    test_index: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pred_train: np.ndarray
    pred_test: np.ndarray
    record_ids_test: np.ndarray
    random_seed: int
    learning_rate: float
    n_iterations: int
    test_size: float


def fit_linear_regression(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    record_ids: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
    learning_rate: float = LEARNING_RATE,
    n_iterations: int = N_ITERATIONS,
    test_size: float = TEST_SIZE,
) -> RegressionFit:
    """Split, scale on the training rows, and run batch gradient descent."""
    matrix, yields, names, ids = _validate_inputs(features, target, feature_names, record_ids)
    _validate_training_choices(learning_rate, n_iterations, test_size)

    train_index, test_index = _split_indices(len(matrix), test_size, random_seed)
    x_train = matrix[train_index]
    x_test = matrix[test_index]
    y_train = yields[train_index]
    y_test = yields[test_index]

    scaling_mean, scaling_std = _fit_standardizer(x_train)
    train_ready = _design_matrix(_apply_standardizer(x_train, scaling_mean, scaling_std))
    test_ready = _design_matrix(_apply_standardizer(x_test, scaling_mean, scaling_std))

    weights, loss_history = _batch_gradient_descent(train_ready, y_train, learning_rate, n_iterations)
    pred_train = train_ready @ weights
    pred_test = test_ready @ weights
    if not np.all(np.isfinite(pred_test)) or not np.all(np.isfinite(weights)):
        raise RegressionError("Gradient descent produced non-finite weights or predictions.")

    return RegressionFit(
        weights=weights,
        feature_names=names,
        scaling_mean=scaling_mean,
        scaling_std=scaling_std,
        loss_history=loss_history,
        train_index=train_index,
        test_index=test_index,
        y_train=y_train,
        y_test=y_test,
        pred_train=pred_train,
        pred_test=pred_test,
        record_ids_test=ids[test_index],
        random_seed=int(random_seed),
        learning_rate=float(learning_rate),
        n_iterations=int(n_iterations),
        test_size=float(test_size),
    )


def predict_yield(features: np.ndarray, fit: RegressionFit) -> np.ndarray:
    """Predict harvest weight in kilograms for rows shaped like the training features."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(fit.feature_names):
        raise RegressionError(
            f"Expected {len(fit.feature_names)} features, got shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise RegressionError("Prediction features must be finite numbers.")
    ready = _design_matrix(_apply_standardizer(matrix, fit.scaling_mean, fit.scaling_std))
    return ready @ fit.weights


def build_regression_metrics(fit: RegressionFit, group_code: str) -> dict:
    """Build the regression_metrics.json body from a completed fit."""
    code = str(group_code).strip()
    if not code:
        raise RegressionError("group_code is required.")

    return {
        "group_code": code,
        "model_version": MODEL_VERSION,
        "random_seed": fit.random_seed,
        "learning_rate": fit.learning_rate,
        "n_iterations": fit.n_iterations,
        "test_size": fit.test_size,
        "n_train": int(len(fit.train_index)),
        "n_test": int(len(fit.test_index)),
        "feature_names": list(fit.feature_names),
        "scaling": {
            "method": "standardize",
            "fit_on": "training_features_only",
            "ddof": 0,
            "mean": [_json_number(value) for value in fit.scaling_mean],
            "std": [_json_number(value) for value in fit.scaling_std],
        },
        "weights": {
            "intercept": _json_number(fit.weights[0]),
            "coefficients": [_json_number(value) for value in fit.weights[1:]],
        },
        "loss": {
            "definition": "half mean squared error on the training rows",
            "initial": _json_number(fit.loss_history[0]),
            "final": _json_number(fit.loss_history[-1]),
            "history": [_json_number(value) for value in fit.loss_history],
        },
        "train_metrics": _score(fit.y_train, fit.pred_train),
        "test_metrics": _score(fit.y_test, fit.pred_test),
        "test_predictions": [
            {
                "record_id": str(record_id),
                "actual_yield_kg": _json_number(actual),
                "predicted_yield_kg": _json_number(predicted),
            }
            for record_id, actual, predicted in zip(fit.record_ids_test, fit.y_test, fit.pred_test, strict=True)
        ],
    }


def write_regression_metrics(metrics: dict, output_path: Path) -> Path:
    """Write regression_metrics.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return path


def write_loss_plot(loss_history: list[float], output_path: Path) -> Path:
    """Write regression_loss.png from the training loss history."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.arange(len(loss_history))
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(iterations, loss_history, color="#243B53", linewidth=1.6)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Training loss")
    axis.set_title("Batch gradient descent")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def save_regression_model(fit: RegressionFit, output_path: Path) -> Path:
    """Save the weights and the training-set scaler for later prediction."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "random_seed": fit.random_seed,
        "learning_rate": fit.learning_rate,
        "n_iterations": fit.n_iterations,
        "feature_names": list(fit.feature_names),
        "intercept": _json_number(fit.weights[0]),
        "coefficients": [_json_number(value) for value in fit.weights[1:]],
        "scaling_mean": [_json_number(value) for value in fit.scaling_mean],
        "scaling_std": [_json_number(value) for value in fit.scaling_std],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_regression(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    group_code: str,
    output_dir: Path,
    models_dir: Path,
    record_ids: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
    learning_rate: float = LEARNING_RATE,
    n_iterations: int = N_ITERATIONS,
    test_size: float = TEST_SIZE,
) -> dict:
    """Fit the model and write the metrics JSON, the loss plot, and the saved weights."""
    fit = fit_linear_regression(
        features,
        target,
        feature_names,
        record_ids=record_ids,
        random_seed=random_seed,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        test_size=test_size,
    )
    metrics = build_regression_metrics(fit, group_code)
    destination = Path(output_dir)
    write_regression_metrics(metrics, destination / "regression_metrics.json")
    write_loss_plot(fit.loss_history, destination / "regression_loss.png")
    save_regression_model(fit, Path(models_dir) / "regression_model.json")
    return metrics


def mean_half_squared_error(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Half the mean squared error. This is the loss minimised by the update."""
    errors = design @ weights - target
    return float(0.5 * np.mean(errors ** 2))


def batch_gradient(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gradient of the half mean squared error for one full batch."""
    errors = design @ weights - target
    return (design.T @ errors) / len(target)


def _batch_gradient_descent(
    design: np.ndarray,
    target: np.ndarray,
    learning_rate: float,
    n_iterations: int,
) -> tuple[np.ndarray, list[float]]:
    weights = np.zeros(design.shape[1], dtype=np.float64)
    history = [mean_half_squared_error(design, target, weights)]
    for _ in range(n_iterations):
        weights = weights - learning_rate * batch_gradient(design, target, weights)
        history.append(mean_half_squared_error(design, target, weights))
    return weights, history


def _validate_inputs(features, target, feature_names, record_ids):
    matrix = np.asarray(features, dtype=np.float64)
    yields = np.asarray(target, dtype=np.float64).reshape(-1)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or len(names) == 0:
        raise RegressionError("Feature matrix and feature names do not match.")
    if len(yields) != len(matrix):
        raise RegressionError("Each feature row needs one yield target.")
    if len(matrix) < 2:
        raise RegressionError("Regression needs at least two complete rows.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(yields)):
        raise RegressionError("Regression inputs must be finite numbers.")
    if record_ids is None:
        ids = np.array([str(index) for index in range(len(matrix))])
    else:
        ids = np.asarray(record_ids).astype(str)
        if len(ids) != len(matrix):
            raise RegressionError("record_ids must have one entry per feature row.")
    return matrix, yields, names, ids


def _validate_training_choices(learning_rate: float, n_iterations: int, test_size: float) -> None:
    if learning_rate <= 0:
        raise RegressionError("learning_rate must be positive.")
    if n_iterations < 1:
        raise RegressionError("n_iterations must be at least 1.")
    if not 0 < test_size < 1:
        raise RegressionError("test_size must be between 0 and 1.")


def _split_indices(n_rows: int, test_size: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_test = int(round(n_rows * test_size))
    n_test = min(max(n_test, 1), n_rows - 1)
    order = np.random.default_rng(random_seed).permutation(n_rows)
    return order[n_test:], order[:n_test]


def _fit_standardizer(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _design_matrix(scaled_features: np.ndarray) -> np.ndarray:
    ones = np.ones((len(scaled_features), 1), dtype=np.float64)
    return np.hstack([ones, scaled_features])


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = predicted - actual
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {
        "mae": _json_number(mae),
        "rmse": _json_number(rmse),
        "r_squared": _json_number(r_squared),
    }


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number
