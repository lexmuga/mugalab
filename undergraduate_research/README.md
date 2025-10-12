# MUGA LAB Undergraduate Research Framework

**Model Understanding and Generative Alignment Laboratory (MUGA LAB)**  
Department of Mathematics, Ateneo de Manila University 
Undergraduate Program in **BS Applied Mathematics (Data Science Track)**

---

## About

This repository hosts the **MUGA LAB Undergraduate Research Framework** —  
a reproducible, modular, and MLflow-tracked platform for studying **Calibration**, **Distillation**, 
and **Interpretability** in **Modern Non-Image Neural Networks** (e.g., MLPs, Transformers for tabular 
and text data).

The framework provides all core experiment scripts, metrics, visualization utilities, and reporting 
tools needed for conducting undergraduate theses in Data Science.

---

## Research Vision

> “Understanding how neural networks represent, calibrate, and transfer uncertainty.”

MUGA LAB aims to bridge **mathematical rigor** and **applied machine learning research**.
Our undergraduate theses focus on designing **interpretable**, **reproducible**, and **calibrated**
predictive models for real-world, non-image datasets.

---

## Repository Structure

| Folder | Description |
|--------|--------------|
| **`experiments/`** | Sequential experiment scripts (01–05) for tuning, calibration, distillation, and summary. |
| **`utils/`** | Reproducibility tools such as multi-seed evaluation and variance analysis. |
| **`results/`** | MLflow experiment tracking folder (contains metrics, models, logs). |
| **`reports/`** | Thesis-ready outputs (CSV and LaTeX summaries). |
| **`notebooks/`** | Visualization and calibration analysis notebooks. |

---

## Core Research Areas

### 1. **Calibration of Neural Networks**
- Temperature scaling and post-hoc calibration.
- Reliability diagrams and uncertainty quantification.
- Comparison of ECE, NLL, and miscalibration metrics.
- Sensitivity of calibration across random seeds.

**Sample Thesis Topics:**
- *“Evaluating the Effect of Random Initialization on Calibration Metrics in Neural Networks.”*  
- *“A Comparative Study of Post-Hoc Calibration Methods for Tabular Deep Models.”*  
- *“Quantifying the Trade-Off Between Accuracy and Calibration in Shallow MLPs.”*

---

### 2. **Distillation and Model Compression**
- Knowledge distillation between teacher–student architectures.
- Transfer of calibrated uncertainty through distillation.
- Cross-architecture calibration analysis.

**Sample Thesis Topics:**
- *“Distilling Calibrated Neural Networks: Preserving Uncertainty in Student Models.”*  
- *“Effects of Temperature and Loss Weighting on Distillation of Tabular Models.”*  
- *“Model Compression with Calibration-Aware Distillation.”*

---

### 3. **Interpretability and Explainability**
- Feature attribution in calibrated networks (e.g., SHAP, Integrated Gradients).  
- Calibration-aware explanations for reliability under uncertainty.  
- Post-hoc interpretability of distilled student models.

**Sample Thesis Topics:**
- *“Interpretability under Calibration: An Analysis of SHAP Values on Calibrated MLPs.”*  
- *“Explaining Model Confidence: Feature Attribution in Calibrated Tabular Networks.”*  
- *“How Distillation Affects Interpretability in Non-Image Neural Networks.”*

---

### 4. **Evaluation and Robustness**
- Multi-seed and cross-dataset reproducibility.  
- Statistical variance of calibration metrics.  
- Calibration–robustness trade-offs in noisy environments.

**Sample Thesis Topics:**
- *“Variance-Aware Evaluation of Neural Network Calibration across Random Seeds.”*  
- *“Cross-Dataset Robustness of Calibrated MLP Models.”*  
- *“Quantifying Reproducibility in Post-Hoc Calibration Experiments.”*

---

## Tools and Technologies

| Component | Description |
|------------|-------------|
| **PyTorch 2.x** | Core deep learning framework for all experiments. |
| **Optuna / DEHB** | Hyperparameter optimization libraries. |
| **MLflow 3.0** | Unified experiment tracking and artifact storage. |
| **Pandas / NumPy / Matplotlib** | Data processing and visualization stack. |
| **LaTeX Export Tools** | Automatic generation of publication-ready tables. |

---

##  Workflow Overview

The MUGA LAB pipeline supports the following reproducible stages:

| Stage | Script | Description |
|-------|---------|-------------|
| 01 | `01_baseline_tuning.py` | Tune baseline MLP hyperparameters (Optuna or DEHB). |
| 02 | `02_temperature_scaling.py` | Perform post-hoc temperature calibration. |
| 03 | `03_distillation_experiment.py` | Distill knowledge from calibrated teacher to student model. |
| 04 | `04_cross_architecture_eval.py` | Compare calibration across multiple architectures. |
| 05 | `05_calibration_summary.py` | Aggregate and export summary results (CSV + LaTeX). |

Supporting modules:
- `utils/seed_sensitivity_utils.py`: Multi-seed reproducibility analysis.
- `reports/`: LaTeX-ready results for thesis inclusion.

---

## Recruitment Information (2026–2027 Cohort)

MUGA LAB will open **undergraduate research slots** for  
**BS Applied Mathematics (Data Science track)** students interested in:

| Track | Focus | Expected Output |
|--------|--------|----------------|
| **Calibration & Uncertainty** | Measuring and improving neural network calibration. | Empirical study + calibration toolkit. |
| **Distillation & Model Efficiency** | Teacher–student compression and uncertainty transfer. | Comparative evaluation + MLflow results. |
| **Interpretability under Calibration** | Linking calibration with explainable AI. | Analysis + visualization notebook. |
| **Reproducibility & Evaluation** | Statistical evaluation under random seeds. | Sensitivity report + variance analysis. |

Interested students will work in 2–3 member teams and use this repository as the **official research framework**.

To apply, prepare:
- 1–page concept note
- Preferred topic and data type (tabular or text)
- Commitment to weekly mentoring sessions (10–12 weeks)

---

## Citation Note

> “All undergraduate research projects are implemented within the
> MUGA LAB Undergraduate Research Framework,
> ensuring reproducible experimentation and transparent reporting
> for calibration, distillation, and interpretability studies.”

---

## Maintainer Notes

- Keep experiment scripts synchronized with the MLflow backend.  
- Document new utilities under `utils/`.  
- Commit generated reports (CSV, LaTeX) to `reports/` for archival.  
- Ensure all code is publication-safe (no emojis, clear docstrings, PEP 8 compliant).  
