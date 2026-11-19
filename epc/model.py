"""Loading the trained XGBoost classifier and predicting a phenotype.

The model file is only ever read. It is checked against a known SHA-256 before loading, and
the server refuses to start if it does not match.
"""
import hashlib
import threading
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

MODEL_SHA256 = "7ecbb9fdac0d1d54d088b3de22759a19ccb485ddcee999eee2db9d3e64a213d1"   # Mice.model
N_FEATURES = 4
THRESHOLD = 0.5            # share of windows classed as outlier at or above which a mouse is "Outlier"

_predict_lock = threading.Lock()


class ModelChecksumError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path, expected_sha256=MODEL_SHA256):
    """Load the classifier once, after checking the file has not changed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ModelChecksumError(f"{path} has SHA-256 {digest}, expected {expected_sha256}")
    model = XGBClassifier()
    model.load_model(str(path))
    if model.n_features_in_ != N_FEATURES:
        raise ModelChecksumError(f"{path} expects {model.n_features_in_} features, not {N_FEATURES}")
    return model


def classify(model, features):
    """Classify each window, then the mouse by majority of windows.

    `features` is the table from features.extract_features. Returns the phenotype label,
    the share of windows classed as outlier, the per-window classes and the per-window
    outlier probabilities.
    """
    X = np.asarray(features, dtype=np.float64)
    with _predict_lock:
        window_class = model.predict(X).astype(int)
        window_prob = model.predict_proba(X)[:, 1]
    share = float(window_class.mean())
    label = "Outlier" if share >= THRESHOLD else "Normal"
    return label, share, window_class, window_prob
