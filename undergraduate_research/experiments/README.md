# MUGA LAB Undergraduate Research – Experimental Pipeline

This directory contains the **core experimental workflow** for
the MUGA LAB undergraduate research program on
**Calibration and Distillation of Modern Non-Image Neural Networks**.

Each file represents a distinct and reproducible experimental stage,
designed to align with corresponding sections of the undergraduate thesis
(Methodology → Results → Discussion).

All experiments are fully tracked using **MLflow 3.0** and
compatible with **CUDA**, **MPS (Apple Silicon)**, and **CPU** backends.

---

## 📂 Directory Overview

| File | Stage | Description |
|------|--------|-------------|
| **`01_baseline_tuning.py`** | Stage 1 | Tunes a baseline MLP using **Optuna** or **DEHB**, logs best configuration to MLflow. |
| **`02_temperature_scaling.py`** | Stage 2 | Applies post-hoc **temperature scaling calibration** to the tuned model, optimizes \( T^* \), and evaluates calibration metrics (ECE, MCS, NLL). |
| **`03_distillation_experiment.py`** | Stage 3 | Performs **knowledge distillation** from the calibrated teacher to a smaller student model, logging teacher and student performance. |
| **`04_cross_architecture_eval.py`** | Stage 4 | Compares calibration and accuracy across multiple architectures (Baseline, Calibrated, Distilled). Generates LaTeX-ready tables. |
| **`05_calibration_summary.py`** | Stage 5 | Aggregates metrics from all experiments, computes mean ± std, and exports **CSV + LaTeX** summary tables for thesis reporting. |
| **`__init__.py`** | — | Marks directory as a Python module for structured imports. |

---

## Workflow Summary

The MUGA LAB undergraduate pipeline follows a **five-stage methodology**:

1. **Model Optimization (Stage 1)**  
   Optimize an MLP model on tabular data using Optuna or DEHB.  
   Output: best hyperparameters + model artifact logged to MLflow.

2. **Calibration (Stage 2)**  
   Calibrate the tuned model using temperature scaling.  
   Output: optimal temperature \(T^*\), improved ECE and NLL metrics.

3. **Distillation (Stage 3)**  
   Distill calibrated teacher knowledge into a smaller student network.  
   Output: distilled student model + teacher/student metrics.

4. **Cross-Architecture Comparison (Stage 4)**  
   Evaluate multiple trained models across architectures or training regimes.  
   Output: comparative LaTeX table of calibration and accuracy metrics.

5. **Summary and Reporting (Stage 5)**  
   Consolidate all experiments into a final quantitative report.  
   Output: CSV + LaTeX tables for the Results & Discussion chapters.

---

## Integration with Other MUGA LAB Modules

| Dependency | Description |
|-------------|-------------|
| **`mlp_tuner_tabular_mlflow.py`** | Core MLP tuning logic and MLflow integration. |
| **`calibration_metrics.py`** | Implements Expected Calibration Error (ECE), Miscalibration Score (MCS), and NLL. |
| **`reliability_diagram_utils.py`** | Generates reliability diagrams and logs them to MLflow. |
| **`../utils/seed_sensitivity_utils.py`** | Provides multi-seed reproducibility utilities for robustness analysis. |

---

## Execution Guide

### Example Pipeline
```bash
# 1. Baseline tuning
python 01_baseline_tuning.py --train ../../data/train.csv --target label --search optuna

# 2. Calibration
python 02_temperature_scaling.py --model_uri runs:/<RUN_ID>/model --train ../../data/train.csv --target label

# 3. Distillation
python 03_distillation_experiment.py --teacher_uri runs:/<RUN_ID>/model --train ../../data/train.csv --target label

# 4. Cross-architecture evaluation
python 04_cross_architecture_eval.py \
  --models runs:/<BASELINE>/model runs:/<CALIBRATED>/model runs:/<DISTILLED>/model \
  --labels Baseline Calibrated Distilled \
  --train ../../data/train.csv --target label

# 5. Summary aggregation
python 05_calibration_summary.py --mlflow_uri ../../results/mlruns --output_dir ../../reports/summary
