# Clean-run record

Owner: Member 5, Reproducibility and release lead, Lazarus Simboya Ira Inyasio (25/28180).

Date: 1 October 2026. Machine: Windows. Python 3.14.6.

The pipeline commit under test was `bf2b5cdcb41d3aa1aa66f13fe2b3742751031ac0`. Dependencies were installed from `requirements.txt` into a new virtual environment at `%TEMP%\aicat-clean-run`, not into the project folder.

## Install

```powershell
python -m venv $env:TEMP\aicat-clean-run
& "$env:TEMP\aicat-clean-run\Scripts\python.exe" -m pip install -r requirements.txt
```

The install finished with exit code 0. Imported versions: NumPy 2.5.1, and the requirements pins for pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.0, and seaborn 0.13.2.

## Command

The lecturer file `data/AI_A1_G03.csv` is not in the repository. This run used a temporary 48-row file with the published columns, outside the project, and then discarded it. Its fingerprint must not be entered on Moodle.

```powershell
python run_all.py --data <temporary.csv> --output <temporary\artifacts> --group AI-G03 --models <temporary\models>
```

Exit code 0.

```text
Group code: AI-G03
Dataset SHA-256: e77cb229cce96326e399b2bfa459e2e09eff289d70dd95c3a2edaaf72405a938
Rows: 48
Complete rows: 48
```

That SHA-256 matched the temporary file bytes. It is not the issued dataset.

Files written, with sizes in bytes:

| File | Bytes |
|---|---|
| data_report.json | 2622 |
| regression_metrics.json | 23706 |
| regression_loss.png | 21984 |
| classification_metrics.json | 3300 |
| confusion_matrix.png | 18630 |
| clustering_metrics.json | 1174 |
| clusters.csv | 409 |
| cluster_plot.png | 28541 |
| regression_model.json | 802 |
| classification_model.json | 876 |
| clustering_model.json | 1049 |

## Rejected file

A two-column CSV was passed to the same command. Exit code 1. No fingerprint was printed.

```text
Error: CSV columns do not match the harvest schema. Missing: ['plot_area_ha', 'rainfall_mm', 'soil_ph', 'seed_kg', 'distance_km', 'arrival_hour', 'actual_yield_kg', 'dispatch_attention']. Unexpected: ['notes'].
```

## Still required before submission

Run the same command on `data/AI_A1_G03.csv` when the lecturer file is present. The fingerprint from that run is the one that belongs in Moodle. Then set the README hash to `git rev-parse HEAD` of the commit that contains the submitted code.
