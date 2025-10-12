# MUGA LAB Undergraduate Research Utilities

This directory contains **supporting modules** for reproducible experimentation,
calibration, and sensitivity analysis within the **MUGA LAB Undergraduate Research Framework**.

These utilities are **not experiment stages** (unlike `01_`–`05_` scripts).
They provide reusable infrastructure for evaluation, logging, and robustness assessment.

---

## Directory Overview

| File | Purpose |
|------|----------|
| **`seed_sensitivity_utils.py`** | Provides multi-seed evaluation and variance analysis tools for reproducibility studies. |
| **`__init__.py`** | Marks this folder as a Python package for relative imports. |

---

## Integration in Experiments

Import these utilities inside experiment scripts (e.g., `03_distillation_experiment.py`):

```python
from mugalab.undergraduate_research.utils.seed_sensitivity_utils import (
    evaluate_across_seeds,
    evaluate_metrics_across_seeds,
    export_seed_results,
)
```
