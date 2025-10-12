# =====================================================
# helpers_indexing.py
# =====================================================
# Utility functions for loading, applying, and validating
# master split indices in the Justice Dataset experiments.
#
# This helper ensures consistent indexing and reproducibility
# across traditional and embedding vectorization workflows.
# =====================================================

import os
import joblib
import pandas as pd

# Default location for saved indices
MASTER_SPLIT_PATH = "Artifacts/master_split_indices.joblib"


# =====================================================
# 1. Load Master Splits
# =====================================================
def load_master_splits(path: str = MASTER_SPLIT_PATH):
    """
    Load the master split dictionary containing indices
    for regression, binary, and multiclass tasks.

    Args:
        path : str
            Path to the saved .joblib split dictionary.

    Returns:
        dict : {
            "regression": {...},
            "binary": {...},
            "multiclass": {...}
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Master split not found at: {path}\n"
            "→ Please run generate_master_split.py first."
        )
    splits = joblib.load(path)
    print(f"Loaded master split indices from: {path}")
    return splits


# =====================================================
# 2. Apply Split Indices to DataFrame
# =====================================================
def apply_split(df: pd.DataFrame, split_indices: dict, text_col: str, label_col: str):
    """
    Slice a DataFrame into train/val/test subsets based on master split indices.

    Args:
        df : pandas.DataFrame
            Dataset containing at least text_col and label_col.
        split_indices : dict
            Dictionary with keys train_idx, val_idx, test_idx.
        text_col : str
            Name of the text column (e.g., 'facts_clean').
        label_col : str
            Name of the label column (e.g., 'first_party_winner').

    Returns:
        dict : {
            "X_train", "X_val", "X_test",
            "y_train", "y_val", "y_test"
        }
    """
    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in DataFrame: {missing}")

    splits = {
        "X_train": df.loc[split_indices["train_idx"], text_col],
        "X_val":   df.loc[split_indices["val_idx"], text_col],
        "X_test":  df.loc[split_indices["test_idx"], text_col],
        "y_train": df.loc[split_indices["train_idx"], label_col],
        "y_val":   df.loc[split_indices["val_idx"], label_col],
        "y_test":  df.loc[split_indices["test_idx"], label_col],
    }

    # Optional: Ensure index alignment consistency
    for key, series in splits.items():
        splits[key] = series.reset_index(drop=True)

    print(f"Applied split indices for {label_col} (train/val/test)")
    return splits


# =====================================================
# 3. Validate Split Integrity
# =====================================================
def validate_split_integrity(master_splits: dict, df: pd.DataFrame = None):
    """
    Validate the integrity of the master splits:
      - Ensures no index overlaps between train/val/test.
      - Ensures all indices are within dataset bounds (if df provided).

    Args:
        master_splits : dict
            Master split dictionary as loaded from joblib.
        df : pandas.DataFrame, optional
            Original dataset to check index coverage.

    Returns:
        pd.DataFrame : summary table of validation results.
    """
    records = []
    for task, split in master_splits.items():
        train_idx, val_idx, test_idx = (
            set(split["train_idx"]),
            set(split["val_idx"]),
            set(split["test_idx"]),
        )

        total = len(train_idx | val_idx | test_idx)
        overlap = bool(
            (train_idx & val_idx) or (train_idx & test_idx) or (val_idx & test_idx)
        )
        out_of_bounds = False

        if df is not None:
            max_idx = df.index.max()
            if any(i > max_idx for i in train_idx | val_idx | test_idx):
                out_of_bounds = True

        records.append({
            "task": task,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "total_unique": total,
            "has_overlap": overlap,
            "out_of_bounds": out_of_bounds
        })

    summary = pd.DataFrame(records)
    print("Split integrity validation summary:")
    print(summary.to_string(index=False))
    return summary


# =====================================================
# 4. Example Usage (for course notebooks)
# =====================================================
if __name__ == "__main__":
    # For demonstration in teaching notebooks
    splits = load_master_splits()
    validate_split_integrity(splits)