"""LightGBM supervised fraud classification model optimized for PR-AUC on imbalanced transaction streams."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import lightgbm as lgb
import joblib


class FraudLightGBM:
    """Production LightGBM classifier with cost-sensitive scaling and PR-AUC early stopping."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = 10.0,
        random_state: int = 42,
        decision_threshold: float = 0.50
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self.model: Optional[lgb.LGBMClassifier] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30
    ) -> "FraudLightGBM":
        """Fit LightGBM with calibrated class imbalance weight and validation monitoring."""
        if self.scale_pos_weight is not None:
            spw = self.scale_pos_weight
        else:
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            spw = float(min(20.0, max(1.0, neg_count / max(1, pos_count * 5))))

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            objective="binary",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )

        callbacks = []
        eval_metric = "average_precision"

        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            try:
                # Modern LightGBM argument
                self.model.fit(
                    X_train,
                    y_train,
                    eval_X=X_val,
                    eval_y=y_val,
                    eval_metric=eval_metric,
                    callbacks=callbacks
                )
            except (TypeError, ValueError):
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=eval_metric,
                    callbacks=callbacks
                )
        else:
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 1D array of calibrated fraud probabilities P(is_fraud=1)."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before predict_proba.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary classification using tuned decision threshold."""
        thresh = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importances sorted by contribution."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before extracting importances.")
        importances = self.model.feature_importances_
        feature_importance_map = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }
        return dict(sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True))

    def save(self, filepath: Path) -> None:
        """Serialize model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
                "decision_threshold": self.decision_threshold,
                "params": {
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "num_leaves": self.num_leaves,
                    "max_depth": self.max_depth,
                    "min_child_samples": self.min_child_samples,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "scale_pos_weight": self.scale_pos_weight,
                    "random_state": self.random_state,
                }
            },
            filepath
        )

    @classmethod
    def load(cls, filepath: Path) -> "FraudLightGBM":
        """Load model from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        artifact = joblib.load(filepath)
        params = artifact["params"]
        instance = cls(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=params.get("scale_pos_weight", 10.0),
            random_state=params["random_state"],
            decision_threshold=artifact.get("decision_threshold", 0.5)
        )
        instance.model = artifact["model"]
        instance.is_fitted = artifact["is_fitted"]
        return instance
