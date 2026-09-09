"""Data preprocessing, temporal splitting, and feature pipeline serialization."""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

from src.data.schemas import FeatureVector

# Complete ordered list of features expected by ML models
MODEL_FEATURE_NAMES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_risk_score",
    "channel_risk_score",
    "device_trust_score",
    "distance_from_home_km",
    "distance_from_last_tx_km",
    "seconds_since_last_tx",
    "tx_count_past_5m",
    "tx_count_past_1h",
    "tx_sum_amount_past_24h",
    "ratio_to_avg_amount_24h",
    "is_international",
]


class DataPreprocessor:
    """Production preprocessing pipeline ensuring zero temporal leakage.

    Fits scaling parameters strictly on training historical partitions and
    transforms both batch dataframes and single real-time feature vectors.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or MODEL_FEATURE_NAMES
        self.scaler = RobustScaler()
        self.is_fitted = False

    def temporal_train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.75,
        val_ratio: float = 0.10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset chronologically to strictly mirror real-time model deployment.

        Fraud models must never use random k-fold cross-validation or random splits
        because future fraud patterns would leak into historical training data.
        """
        sorted_df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(sorted_df)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = sorted_df.iloc[:train_end].copy()
        val_df = sorted_df.iloc[train_end:val_end].copy()
        test_df = sorted_df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def fit(self, X: Any) -> "DataPreprocessor":
        """Fit scaler on training features only."""
        if isinstance(X, pd.DataFrame):
            features_subset = X[self.feature_names].values
        else:
            features_subset = np.asarray(X)
        self.scaler.fit(features_subset)
        self.is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform dataframe or ndarray features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before calling transform().")
        if isinstance(X, pd.DataFrame):
            features_subset = X[self.feature_names].values
        else:
            features_subset = np.asarray(X)
        return self.scaler.transform(features_subset)

    def fit_transform(self, X: Any) -> np.ndarray:
        """Fit scaler and transform in one step."""
        return self.fit(X).transform(X)

    def transform_single(self, vector: FeatureVector) -> np.ndarray:
        """Transform a single incoming real-time feature vector for low-latency inference."""
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before calling transform_single().")

        vector_dict = vector.model_dump()
        raw_vals = np.array([[vector_dict[name] for name in self.feature_names]], dtype=np.float64)
        return self.scaler.transform(raw_vals)

    def save(self, filepath: Path) -> None:
        """Serialize fitted preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            },
            filepath
        )

    @classmethod
    def load(cls, filepath: Path) -> "DataPreprocessor":
        """Load preprocessor from serialized artifact."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        data = joblib.load(filepath)
        preprocessor = cls(feature_names=data["feature_names"])
        preprocessor.scaler = data["scaler"]
        preprocessor.is_fitted = data["is_fitted"]
        return preprocessor
