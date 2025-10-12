# MUGA LAB Undergraduate Research – Results Directory

This directory serves as the **primary output location** for all
**MLflow experiment runs**, model artifacts, and logged metrics.

Each experiment stage (Baseline Tuning, Calibration, Distillation, etc.)
automatically writes its logs to the MLflow backend under this directory.

---

## Directory Overview

| Folder / File | Description |
|----------------|-------------|
| **`mlruns/`** | Default MLflow tracking directory containing all experiment runs and models. |
| **`README.md`** | This documentation file. |

---

## How MLflow Organizes Data

Each subfolder in `mlruns/` corresponds to an MLflow experiment:

| MLflow Experiment | Script Source | Output Content |
|-------------------|----------------|----------------|
| `Baseline_Tuning` | `01_baseline_tuning.py` | Hyperparameter trials, validation metrics, trained models. |
| `Temperature_Scaling` | `02_temperature_scaling.py` | Optimal temperature \(T^*\), calibration metrics (ECE, MCS, NLL). |
| `Distillation_Experiment` | `03_distillation_experiment.py` | Teacher and student model metrics and artifacts. |
| `Cross_Architecture_Evaluation` | `04_cross_architecture_eval.py` | Comparative metrics across architectures. |
| `Calibration_Summary` | `05_calibration_summary.py` | Aggregated summaries and exported reports. |
| `Seed_Sensitivity` | `utils/seed_sensitivity_utils.py` | Multi-seed reproducibility and robustness logs. |

---

## Typical Workflow

To visualize results locally:
```bash
mlflow ui --backend-store-uri ./mlruns
```
