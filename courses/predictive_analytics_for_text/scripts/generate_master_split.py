# ======================================================
# generate_master_split.py
# version v1.1
# ======================================================
# Purpose:
#   Creates and saves master split indices for
#   regression, binary, and multiclass tasks.
# ======================================================

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DATA_PATH = "../Justice_Dataset/justice.csv"
MASTER_SPLIT_PATH = "Artifacts/master_split_indices.joblib"
RANDOM_STATE = 103
TEST_SIZE = 0.30
VAL_SIZE = 0.50  # 15% val + 15% test overall


def stratified_splits(X, y, test_size=0.3, val_size=0.5, random_state=103):
    """Generate stratified train/val/test indices."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(sss.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(sss_val.split(X_temp, y_temp))

    val_indices = X_temp.iloc[val_idx].index
    test_indices = X_temp.iloc[test_idx].index

    return {
        "train_idx": X_train.index,
        "val_idx": val_indices,
        "test_idx": test_indices
    }


def create_master_split(data_path=DATA_PATH, random_state=RANDOM_STATE):
    """Creates master splits for regression, binary, and multiclass tasks."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, index_col=0)

    required_cols = ["facts", "term", "first_party_winner", "issue_area"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ------------------------------
    # 1. Regression subset
    # ------------------------------
    df["term_year"] = pd.to_numeric(df["term"], errors="coerce")
    regression_df = df[["facts", "term_year"]].dropna()

    X = regression_df["facts"]
    y = regression_df["term_year"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=random_state
    )

    regression_indices = {
        "train_idx": X_train.index,
        "val_idx": X_val.index,
        "test_idx": X_test.index
    }

    # ------------------------------
    # 2. Binary classification
    # ------------------------------
    binary_df = df[["facts", "first_party_winner"]].dropna()
    binary_indices = stratified_splits(binary_df["facts"], binary_df["first_party_winner"],
                                       test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=random_state)

    # ------------------------------
    # 3. Multiclass classification
    # ------------------------------
    multiclass_df = df[["facts", "issue_area"]].dropna()
    merge_map = {"Private Action": "Miscellaneous", "Interstate Relations": "Miscellaneous"}
    multiclass_df["issue_area"] = multiclass_df["issue_area"].replace(merge_map)

    multiclass_indices = stratified_splits(multiclass_df["facts"], multiclass_df["issue_area"],
                                           test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=random_state)

    # ------------------------------
    # Save all splits together
    # ------------------------------
    master_splits = {
        "regression": regression_indices,
        "binary": binary_indices,
        "multiclass": multiclass_indices
    }

    os.makedirs(os.path.dirname(MASTER_SPLIT_PATH), exist_ok=True)
    joblib.dump(master_splits, MASTER_SPLIT_PATH)
    print(f"Master splits (3 tasks) saved to: {MASTER_SPLIT_PATH}")


def ensure_master_split():
    if os.path.exists(MASTER_SPLIT_PATH):
        print(f"Master split already exists at: {MASTER_SPLIT_PATH}")
    else:
        print("Master split not found. Generating new split...")
        create_master_split()


if __name__ == "__main__":
    ensure_master_split()
