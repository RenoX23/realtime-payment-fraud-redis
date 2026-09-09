"""Baseline classifiers demonstrating why naive accuracy fails on imbalanced fraud datasets."""

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class DummyFraudClassifier:
    """Naive baseline predicting non-fraud 100% of the time.

    Demonstrates that ~99.9% accuracy can be achieved while catching 0% of fraud,
    resulting in catastrophic financial losses for payment processors.
    """

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyFraudClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        # 99.9% prob of legit, 0.1% prob of fraud (or 1.0, 0.0)
        probs = np.zeros((n, 2))
        probs[:, 0] = 1.0
        probs[:, 1] = 0.0
        return probs


class LogisticRegressionBaseline:
    """Cost-sensitive linear baseline with balanced class weighting."""

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class RandomForestBaseline:
    """Bagged tree ensemble baseline with balanced subsampling."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            max_depth=12,
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
