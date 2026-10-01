"""Load, validate, and vectorize the harvest dataset.

Owned by Member 1, Data and UX lead:
Nuzha ZainEl-Abdeen Mohammed Ismail (25/27419).

The issued CSV is hashed before any cleaning. record_id is never a feature.
actual_yield_kg and dispatch_attention stay out of the feature matrix so later
stages can train without leaking the targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IDENTIFIER_COLUMN = "record_id"
REGRESSION_TARGET = "actual_yield_kg"
CLASSIFICATION_TARGET = "dispatch_attention"

FEATURE_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]

NUMERIC_COLUMNS = FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]

EXPECTED_COLUMNS = [IDENTIFIER_COLUMN, *NUMERIC_COLUMNS]

STAT_NAMES = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")


class SchemaError(ValueError):
    """Raised when the CSV does not match the published harvest schema."""


@dataclass(frozen=True)
class PreparedDataset:
    """Arrays for one validated file, plus the data-report dictionary."""

    record_ids: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    regression_target: np.ndarray
    classification_target: np.ndarray
    report: dict


def file_sha256(csv_path: Path) -> str:
    """Return the SHA-256 hex digest of the raw dataset file."""
    digest = hashlib.sha256()
    with csv_path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Read the issued CSV. Column names are trimmed. record_id stays text."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path, dtype={IDENTIFIER_COLUMN: "string"})
    frame.columns = [str(name).strip() for name in frame.columns]
    return frame


def validate_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Check column names and types. Return columns in the published order."""
    found = list(frame.columns)
    missing = [name for name in EXPECTED_COLUMNS if name not in found]
    unexpected = [name for name in found if name not in EXPECTED_COLUMNS]
    if missing or unexpected:
        raise SchemaError(
            "CSV columns do not match the harvest schema. "
            f"Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
        )

    checked = frame.loc[:, EXPECTED_COLUMNS].copy()
    checked[IDENTIFIER_COLUMN] = checked[IDENTIFIER_COLUMN].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")

    present_labels = checked[CLASSIFICATION_TARGET].dropna()
    invalid_labels = sorted(
        {
            _json_number(value)
            for value in present_labels.to_numpy()
            if float(value) not in (0.0, 1.0)
        }
    )
    if invalid_labels:
        raise SchemaError(
            f"{CLASSIFICATION_TARGET} must be 0 or 1. Found: {invalid_labels}."
        )

    return checked


def prepare_dataset(csv_path: Path, group_code: str) -> PreparedDataset:
    """Hash, validate, and vectorize one CSV. Incomplete rows stay out of the matrix."""
    path = Path(csv_path)
    code = str(group_code).strip()
    if not code:
        raise ValueError("group_code is required.")

    fingerprint = file_sha256(path)
    checked = validate_schema(load_csv(path))
    report = build_data_report(checked, group_code=code, sha256=fingerprint)

    complete = _complete_rows(checked)
    features = complete.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    record_ids = complete[IDENTIFIER_COLUMN].astype(str).to_numpy()
    regression_target = complete[REGRESSION_TARGET].to_numpy(dtype=np.float64)
    classification_target = complete[CLASSIFICATION_TARGET].to_numpy(dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise SchemaError("Feature matrix must have one column per input feature.")
    if IDENTIFIER_COLUMN in FEATURE_COLUMNS:
        raise SchemaError("record_id must not be used as a model feature.")

    return PreparedDataset(
        record_ids=record_ids,
        features=features,
        feature_names=list(FEATURE_COLUMNS),
        regression_target=regression_target,
        classification_target=classification_target,
        report=report,
    )


def build_data_report(frame: pd.DataFrame, group_code: str, sha256: str) -> dict:
    """Build the data_report.json body from an already validated frame."""
    missing_values = {
        column: int(frame[column].isna().sum()) for column in EXPECTED_COLUMNS
    }
    duplicate_record_id_count = int(frame[IDENTIFIER_COLUMN].duplicated().sum())
    duplicate_row_count = int(frame.duplicated().sum())
    incomplete_row_count = int((~_complete_mask(frame)).sum())

    return {
        "group_code": group_code,
        "sha256": sha256,
        "row_count": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "identifier_column": IDENTIFIER_COLUMN,
        "regression_target": REGRESSION_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "missing_values": missing_values,
        "duplicate_record_id_count": duplicate_record_id_count,
        "duplicate_row_count": duplicate_row_count,
        "incomplete_row_count": incomplete_row_count,
        "complete_row_count": int(len(frame) - incomplete_row_count),
        "descriptive_statistics": {
            column: _numeric_stats(frame[column].to_numpy(dtype=np.float64))
            for column in NUMERIC_COLUMNS
        },
    }


def write_data_report(report: dict, output_path: Path) -> Path:
    """Write data_report.json. Parent folders are created when needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _complete_mask(frame: pd.DataFrame) -> pd.Series:
    model_columns = FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
    return frame[model_columns].notna().all(axis=1) & frame[IDENTIFIER_COLUMN].notna()


def _complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[_complete_mask(frame)].reset_index(drop=True)


def _numeric_stats(values: np.ndarray) -> dict:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return {name: (0 if name == "count" else None) for name in STAT_NAMES}

    std = float(np.std(clean, ddof=1)) if clean.size > 1 else None
    q25, median, q75 = np.percentile(clean, [25, 50, 75])
    stats = {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": std,
        "min": float(np.min(clean)),
        "25%": float(q25),
        "50%": float(median),
        "75%": float(q75),
        "max": float(np.max(clean)),
    }
    return {name: _json_number(stats[name]) for name in STAT_NAMES}


def _json_number(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
