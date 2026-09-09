"""Unsupervised Isolation Forest anomaly detector for zero-day fraud pattern detection."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


class FraudIsolationForest:
    """Unsupervised anomaly detection engine identifying novel out-of-distribution transactions.

    Trained on normal transaction distributions to detect novel zero-day attacks
    unseen by the supervised gradient booster.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.01,
        max_samples: float = 0.8,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted: bool = False
        # Calibration bounds
        self.score_min: float = -0.5
        self.score_max: float = 0.2

    def fit(self, X: np.ndarray) -> "FraudIsolationForest":
        """Fit Isolation Forest on background transactions and calculate calibration percentiles."""
        self.model.fit(X)
        self.is_fitted = True

        # Calculate decision function statistics for normalization
        raw_scores = self.model.decision_function(X)
        # 1st percentile and 99th percentile for robust normalization
        self.score_min = float(np.percentile(raw_scores, 1))
        self.score_max = float(np.percentile(raw_scores, 99))
        return self

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated anomaly risk scores in [0.0, 1.0].

        Higher scores indicate greater deviation from normal transaction manifold.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring.")

        # decision_function: lower (negative) values indicate greater abnormality
        raw_scores = self.model.decision_function(X)

        # Invert and normalize so that highest anomaly = 1.0, normal = 0.0
        normalized = (self.score_max - raw_scores) / max(1e-6, (self.score_max - self.score_min))
        calibrated = np.clip(normalized, 0.0, 1.0)
        return calibrated

    def save(self, filepath: Path) -> None:
        """Serialize fitted model and calibration parameters."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
                "score_min": self.score_min,
                "score_max": self.score_max,
                "params": {
                    "n_estimators": self.n_estimators,
                    "contamination": self.contamination,
                    "max_samples": self.max_samples,
                    "random_state": self.random_state,
                },
            },
            filepath
        )

    @classmethod
    def load(cls, filepath: Path) -> "FraudIsolationForest":
        """Load serialized model artifact."""
        if not filepath.exists():
            raise FileNotFoundError(f"Anomaly model artifact not found at {filepath}")
        artifact = joblib.load(filepath)
        params = artifact["params"]
        instance = cls(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            max_samples=params["max_samples"],
            random_state=params["random_state"]
        )
        instance.model = artifact["model"]
        instance.is_fitted = artifact["is_fitted"]
        instance.score_min = artifact.get("score_min", -0.5)
        instance.score_max = artifact.get("score_max", 0.2)
        return instance
