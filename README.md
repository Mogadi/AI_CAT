# HarvestLink decision pipeline

SWE 3513 Assignment 1, group code **AI-G03**.

Musanze HarvestLink Cooperative. The command below loads the group CSV and writes the data report, the yield model, the dispatch-attention model, and the operating profiles.

Repository: https://github.com/Mogadi/AI_CAT

The pipeline code verified before this README was commit `bf2b5cdcb41d3aa1aa66f13fe2b3742751031ac0`. After this README is committed, the Moodle hash is the new `git rev-parse HEAD`.

## Group

| Role | Name | Registration |
|---|---|---|
| Member 1 — Data and UX lead | Nuzha ZainEl-Abdeen Mohammed Ismail | 25/27419 |
| Member 2 — Regression engineer | Ehab Fakhralden Mohamed Hamid | 25/27950 |
| Member 3 — Classification engineer | Mojtaba Abdalitieef Ahmed | 25/27660 |
| Member 4 — Clustering and QA engineer. Group leader | Mohammed Osama Hasan | 25/27014 |
| Member 5 — Reproducibility and release lead | Lazarus Simboya Ira Inyasio | 25/28180 |
| Prediction and demonstration lead | Feras Saifaddin Ismail Mohammed | 25/27916 |

## Setup

Tested with Python 3.14.6 on Windows. From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the lecturer CSV at `data/AI_A1_G03.csv`. Do not rename its columns.

## Run

```powershell
python run_all.py --data data/AI_A1_G03.csv --output artifacts/ --group AI-G03
```

The command prints the group code and the SHA-256 of the CSV bytes, then overwrites the files in `artifacts/` and `models/`. The random seed is 42 unless `--seed` is passed. `--models` defaults to `models`.

## Expected outputs

`artifacts/`

- `data_report.json`
- `regression_metrics.json`
- `regression_loss.png`
- `classification_metrics.json`
- `confusion_matrix.png`
- `clustering_metrics.json`
- `clusters.csv`
- `cluster_plot.png`

`models/`

- `regression_model.json`
- `classification_model.json`
- `clustering_model.json`

A CSV with the wrong columns stops before those files are treated as results. The message starts with `Error:` and the process exits with code 1.

## Known limitations

- The lecturer CSV is not in this repository. The clean-run record used a temporary file with the same columns, then deleted it.
- `predict.py` is not in this release.
- The staff dashboard is the design in `design/AI_A1_G03_UIUX.pdf`. This command does not open a window.
- Cluster labels group similar input measurements. They are not a verified real-world category.
- Yield is an estimate from six measures. Dispatch attention is a probability cut at 0.5 for label 1. The officer’s decision is not made by this command.
- Do not submit `.venv`, `__pycache__`, or a hand-written metrics file. The artifact files must come from the command above.
