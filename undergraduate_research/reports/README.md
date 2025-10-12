# MUGA LAB Undergraduate Research – Reports Directory

This directory contains all **automatically generated outputs** from
the MUGA LAB Undergraduate Research Experimental Pipeline,
including calibration, distillation, and cross-architecture analyses.

These reports are **exported automatically** by the experiment scripts
(e.g., `04_cross_architecture_eval.py`, `05_calibration_summary.py`)
and are designed for **direct inclusion in thesis documents** (LaTeX tables, CSV summaries).

---

## Directory Overview

| Folder / File | Description |
|----------------|-------------|
| **`tables/`** | Contains LaTeX-formatted tables for inclusion in the Results chapter (e.g., `cross_architecture_eval_table.tex`). |
| **`summary/`** | Contains aggregate summaries in both CSV and LaTeX formats (e.g., `calibration_summary.csv`, `calibration_summary.tex`). |
| **`README.md`** | This documentation file. |

---

## Generated Artifacts

| Source Script | Output File | Description |
|----------------|--------------|-------------|
| `04_cross_architecture_eval.py` | `tables/cross_architecture_eval_table.tex` | LaTeX table comparing accuracy and calibration metrics across Baseline, Calibrated, and Distilled models. |
| `05_calibration_summary.py` | `summary/calibration_summary.csv`, `summary/calibration_summary.tex` | Aggregated statistics (mean ± std) of all experiments across random seeds. |
| `utils/seed_sensitivity_utils.py` | `summary/seed_sensitivity_results.csv` | Multi-seed reproducibility and sensitivity analysis results. |

---

## Thesis Integration Guide

These files are designed for **direct LaTeX import** in undergraduate thesis documents.

Example:
```latex
\input{reports/tables/cross_architecture_eval_table.tex}
```
