# -----------------------------------
# Lean Vectorization Helpers
# -----------------------------------

import re
import unicodedata
import html
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# -----------------------------------
# 1. Basic Text Cleaning
# -----------------------------------
def clean_text(text):
    """Lightweight cleaning: normalize Unicode, strip HTML, lowercase."""
    if not isinstance(text, str):
        return ""
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    # Unescape HTML
    text = html.unescape(text)
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------------
# 2. Train/Val/Test Split (70/15/15)
# -----------------------------------
def stratified_split(X, y, random_state=2025):
    """Stratified 70/15/15 split for classification tasks."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# -----------------------------------
# 3. Vectorization
# -----------------------------------
def vectorize_text(X_train, X_val, X_test, method="count", max_features=5000):
    """
    Vectorize text with Count or TF-IDF.
    """
    if method == "count":
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    else:
        raise ValueError("method must be 'count' or 'tfidf'")
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)
    X_test_vec  = vectorizer.transform(X_test)
    
    return vectorizer, X_train_vec, X_val_vec, X_test_vec

## -----------------------------------
# 4. Save / Load Artifacts
# -----------------------------------
def save_results(vectorizer, X_train, y_train, X_val, y_val, X_test, y_test, path):
    """
    Save fitted vectorizer and dataset splits to a .joblib file.
    The saved bundle can later be reloaded for model training,
    feature importance analysis, or SHAP explainability.

    Parameters
    ----------
    vectorizer : sklearn Vectorizer
        The fitted CountVectorizer or TfidfVectorizer instance.
    X_train, X_val, X_test : scipy.sparse.csr_matrix
        The vectorized feature matrices for train/val/test.
    y_train, y_val, y_test : pd.Series or np.array
        The corresponding labels.
    path : str
        Full file path (including filename) to save the bundle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        "vectorizer": vectorizer,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }, path)
    print(f"Saved vectorizer and splits to: {path}")


def load_results(path):
    """
    Load a previously saved .joblib bundle (vectorizer + splits).
    """
    return joblib.load(path)
# -----------------------------------------------------------------------------
# 5. Coverage Curve Utilities
# -----------------------------------------------------------------------------

def compute_coverage_curve(
    texts: pd.Series, 
    max_features_range: List[int],
    vectorizer_kwargs: Dict, 
    vectorizer_type: str = "count"
) -> Tuple[List[int], List[float]]:
    """
    Compute coverage curve for different max_features values.
    
    Args:
        texts: Series of preprocessed text documents
        max_features_range: List of max_features values to test
        vectorizer_kwargs: Common vectorizer parameters (e.g. stop_words, ngram_range)
        vectorizer_type: "count" or "tfidf"
        
    Returns:
        (max_features_list, coverage_list)
    """
    coverages = []
    for max_feat in max_features_range:
        if vectorizer_type == "count":
            vectorizer = CountVectorizer(max_features=max_feat, **vectorizer_kwargs)
        else:
            vectorizer = TfidfVectorizer(max_features=max_feat, **vectorizer_kwargs)
        
        X = vectorizer.fit_transform(texts)
        coverage = (X.getnnz(axis=1) > 0).mean()
        coverages.append(coverage)
    return max_features_range, coverages


def select_max_features(
    texts: pd.Series,
    variant_name: str,
    vectorizer_type: str = "tfidf",
    coverage_target: float = 0.95,
    max_feat_range: List[int] = None,
    vectorizer_kwargs: Dict = None,
    plot: bool = True
) -> int:
    """
    Select the smallest max_features achieving at least `coverage_target`.
    
    Args:
        texts: Preprocessed text documents
        variant_name: Name of the task/variant (for logging)
        vectorizer_type: "count" or "tfidf"
        coverage_target: Desired coverage fraction (default=0.95)
        max_feat_range: List of max_features to test
        vectorizer_kwargs: Extra arguments for vectorizer
        plot: Whether to show coverage curve
    
    Returns:
        int: chosen max_features
    """
    #if max_feat_range is None:
    #    if variant_name == "Regression TF-IDF":
    #        max_feat_range = list(range(50, 11000, 700))
    #    else:
    #        max_feat_range = list(range(50, 11000, 700))
    if max_feat_range is None:
        max_feat_range = list(range(50, 11000, 700))
        
    if vectorizer_kwargs is None:
        vectorizer_kwargs = {"stop_words": "english"}
    
    feats, coverages = compute_coverage_curve(
        texts, max_feat_range, vectorizer_kwargs, vectorizer_type
    )

    
    if plot:
        plt.plot(feats, coverages, marker="o")
        plt.axhline(y=coverage_target, color="red", linestyle="--")
        #plt.axvline(x=m, color="green",linestyle="--")
        plt.xlabel("max_features")
        plt.ylabel("Coverage")
        plt.title(f"Coverage Curve ({variant_name})")
        plt.grid(True)
        plt.show()
    
    for m, c in zip(feats, coverages):
        if c >= coverage_target:
            print(f"Selected max_features={m} for {variant_name} ({c:.3%} coverage)")
            print("="*80)
            return m
    
    
    print(f"No value reached target {coverage_target:.3%}, using {feats[-1]}")
    return feats[-1]


