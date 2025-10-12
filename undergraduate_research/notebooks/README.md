# MUGA LAB Undergraduate Research – Analysis Notebooks

**Model Understanding and Generative Alignment Laboratory (MUGA LAB)**  
Department of Mathematics • Ateneo de Manila University 
BS Applied Mathematics (Data Science Track)

---

## Overview

This folder contains Jupyter notebooks that provide **visual, metric-based, and reproducibility analyses**
for undergraduate research projects in **Calibration**, **Distillation**, and **Interpretability**.

Each notebook complements a Python experiment script from
[`../experiments/`](../experiments/) and is designed to be
*publication-ready, MLflow-tracked, and pedagogically structured*.

---

## Notebook Collection

| Notebook | Purpose | Key Dependencies |
|-----------|----------|------------------|
| **`calibration_analysis.ipynb`** | Computes and visualizes calibration metrics (ECE, MCS, NLL). Compares pre- and post-temperature-scaled models. | `calibration_metrics.py`, `reliability_diagram_utils.py` |
| **`seed_sensitivity_analysis.ipynb`** | Evaluates model stability across random seeds using variance analysis and reliability diagrams. | `seed_sensitivity_utils.py` |
| *(future)* **`distillation_inspection.ipynb`** | Visualizes teacher–student transfer quality and calibration changes. | `03_distillation_experiment.py` |
| *(future)* **`interpretability_analysis.ipynb`** | Links SHAP or gradient-based attributions with calibration metrics. | `interpretability_utils.py` |

---

## ⚙️ Environment Requirements

- Python ≥ 3.10  
- PyTorch ≥ 2.0  
- Optuna or DEHB  
- MLflow 3.0  
- Pandas, NumPy, Matplotlib  
- (optional) Captum, SHAP for interpretability notebooks  

Activate your environment:

```bash
conda activate mugalab
jupyter lab
