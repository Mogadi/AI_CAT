"""Run the harvest decision pipeline from the command line.

Owned by Member 5, Reproducibility and release lead:
Lazarus Simboya Ira Inyasio (25/28180).

The documented command is:

python run_all.py --data data/AI_A1_G03.csv --output artifacts/ --group AI-G03

Setup, members, and the clean-run record are in README.md and notes/clean_run.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from classification import ClassificationError, run_classification  # noqa: E402
from clustering import ClusteringError, run_clustering  # noqa: E402
from data_pipeline import SchemaError, prepare_dataset, write_data_report  # noqa: E402
from regression import RegressionError, run_regression  # noqa: E402

RANDOM_SEED = 42

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


class PipelineError(RuntimeError):
    """Raised when the issued file cannot be turned into the three decisions."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the harvest decision pipeline and write its artifacts.")
    parser.add_argument("--data", required=True, help="Path to the group CSV.")
    parser.add_argument("--output", required=True, help="Folder for the generated artifacts.")
    parser.add_argument("--group", required=True, help="Group code, for example AI-G03.")
    parser.add_argument("--models", default=str(ROOT / "models"), help="Folder for the saved model files.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed recorded in every metrics file.")
    return parser


def run_pipeline(data_path: Path, output_dir: Path, group_code: str, models_dir: Path, seed: int) -> dict:
    """Prepare one CSV and write the data, regression, classification, and clustering files."""
    prepared = prepare_dataset(data_path, group_code)
    if prepared.report["complete_row_count"] < 3:
        raise PipelineError("At least three complete rows are required to fit the three decisions.")

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    write_data_report(prepared.report, output_dir / "data_report.json")
    run_regression(
        prepared.features,
        prepared.regression_target,
        prepared.feature_names,
        group_code,
        output_dir,
        models_dir,
        record_ids=prepared.record_ids,
        random_seed=seed,
    )
    run_classification(
        prepared.features,
        prepared.classification_target,
        prepared.feature_names,
        group_code,
        output_dir,
        models_dir,
        record_ids=prepared.record_ids,
        random_seed=seed,
    )
    run_clustering(
        prepared.features,
        prepared.feature_names,
        group_code,
        output_dir,
        models_dir,
        record_ids=prepared.record_ids,
        random_seed=seed,
    )
    return prepared.report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_pipeline(
            Path(args.data),
            Path(args.output),
            args.group,
            Path(args.models),
            args.seed,
        )
    except (SchemaError, RegressionError, ClassificationError, ClusteringError, PipelineError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Group code: {report['group_code']}")
    print(f"Dataset SHA-256: {report['sha256']}")
    print(f"Rows: {report['row_count']}")
    print(f"Complete rows: {report['complete_row_count']}")
    print("Artifacts:")
    for name in ARTIFACTS:
        print(f"  {Path(args.output) / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
