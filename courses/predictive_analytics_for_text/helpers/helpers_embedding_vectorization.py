# -----------------------------------------------------------------------------
# Embedding_Vectorization_Helpers_v4.py (patched full version)
# -----------------------------------------------------------------------------

import os
import warnings
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Try safe import of gensim
try:
    from gensim.models import Word2Vec, FastText, KeyedVectors
    import gensim.downloader as api
    _GENSIM_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        "gensim not available or incompatible with numpy. "
        "Only non-gensim embeddings will work. "
        f"Details: {str(e)}"
    )
    _GENSIM_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1. Train/Val/Test Split
# -----------------------------------------------------------------------------
def create_train_val_test_split(X, y, task_type="binary", test_size=0.15, val_size=0.15, random_state=42):
    """
    Create stratified or random splits depending on task.
    """
    if task_type in ["binary", "multiclass"]:
        stratify = y
    else:
        stratify = None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=stratify, random_state=random_state
    )
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        stratify=y_temp if stratify is not None else None,
        random_state=random_state
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }


# -----------------------------------------------------------------------------
# 2. Embedding Vectorization
# -----------------------------------------------------------------------------
def vectorize_embeddings(splits, variant="word2vec_avg", vector_size=300):
    """
    Vectorize text using one of 8 embedding strategies:
      - word2vec_avg, word2vec_tfidf
      - fasttext_avg, fasttext_tfidf
      - glove_avg, glove_tfidf
      - trainable_word2vec, trainable_fasttext
    """
    if not _GENSIM_AVAILABLE and ("word2vec" in variant or "fasttext" in variant or "glove" in variant):
        raise ImportError(
            f"Variant '{variant}' requires gensim, but gensim is not available or incompatible."
        )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # --------------------------------------------------------
    # Helper: train TF-IDF for weighted embeddings
    # --------------------------------------------------------
    tfidf = TfidfVectorizer()
    tfidf.fit(X_train)
    idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    def embed_avg(texts, model):
        return np.array([
            np.mean([model.wv[w] for w in doc.split() if w in model.wv] or [np.zeros(vector_size)], axis=0)
            for doc in texts
        ])

    def embed_tfidf(texts, model):
        return np.array([
            np.average(
                [model.wv[w] for w in doc.split() if w in model.wv] or [np.zeros(vector_size)],
                axis=0,
                weights=[idf_dict.get(w, 1.0) for w in doc.split() if w in model.wv] or None
            )
            for doc in texts
        ])

    # --------------------------------------------------------
    # Select embedding variant
    # --------------------------------------------------------
    if variant == "word2vec_avg":
        model = Word2Vec([doc.split() for doc in X_train], vector_size=vector_size, min_count=2, workers=4)
        vectorizer = lambda texts: embed_avg(texts, model)

    elif variant == "word2vec_tfidf":
        model = Word2Vec([doc.split() for doc in X_train], vector_size=vector_size, min_count=2, workers=4)
        vectorizer = lambda texts: embed_tfidf(texts, model)

    elif variant == "fasttext_avg":
        model = FastText([doc.split() for doc in X_train], vector_size=vector_size, min_count=2, workers=4)
        vectorizer = lambda texts: embed_avg(texts, model)

    elif variant == "fasttext_tfidf":
        model = FastText([doc.split() for doc in X_train], vector_size=vector_size, min_count=2, workers=4)
        vectorizer = lambda texts: embed_tfidf(texts, model)

    elif variant == "glove_avg":
        glove = api.load("glove-wiki-gigaword-300")  # pre-trained
        vectorizer = lambda texts: np.array([
            np.mean([glove[w] for w in doc.split() if w in glove] or [np.zeros(300)], axis=0)
            for doc in texts
        ])

    elif variant == "glove_tfidf":
        glove = api.load("glove-wiki-gigaword-300")
        vectorizer = lambda texts: np.array([
            np.average(
                [glove[w] for w in doc.split() if w in glove] or [np.zeros(300)],
                axis=0,
                weights=[idf_dict.get(w, 1.0) for w in doc.split() if w in glove] or None
            )
            for doc in texts
        ])

    elif variant == "trainable_fasttext_avg":
        model = Word2Vec([doc.split() for doc in X_train], vector_size=vector_size, min_count=1, workers=4, epochs=20)
        vectorizer = lambda texts: embed_avg(texts, model)

    elif variant == "trainable_fasttext_tfidf":
        model = FastText([doc.split() for doc in X_train], vector_size=vector_size, min_count=1, workers=4, epochs=20)
        vectorizer = lambda texts: embed_tfidf(texts, model)

    else:
        raise ValueError(f"Unknown variant '{variant}'")

    # --------------------------------------------------------
    # Apply vectorizer
    # --------------------------------------------------------
    results = {
        "X_train": vectorizer(X_train),
        "X_val": vectorizer(X_val),
        "X_test": vectorizer(X_test),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "variant": variant,
    }
    return results


# -----------------------------------------------------------------------------
# 3. Save Results
# -----------------------------------------------------------------------------
#def save_results(results, filename, out_dir="../Results"):
#    os.makedirs(out_dir, exist_ok=True)
#    filepath = os.path.join(out_dir, filename)
#    joblib.dump(results, filepath)
#    print(f"Saved {results['variant']} results to {filepath}")

# -----------------------------------------------------------------------------
# 3. Save Results
# -----------------------------------------------------------------------------


def save_results(results, filename, task_type, out_dir="Artifacts"):
    import os, joblib
    os.makedirs(f"{out_dir}/{task_type}", exist_ok=True)
    filepath = os.path.join(out_dir, task_type, f"{task_type}_{results['variant']}_300f.joblib")
    joblib.dump(results, filepath)
    print(f"Saved {results['variant']} results to {filepath}")

def load_results(path):
    return joblib.load(path)
