"""Hybrid ensembling combining supervised gradient boosting and unsupervised anomaly scoring."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import joblib

from src.data.schemas import (
    FeatureVector,
    ScoringResult,
    DecisionType,
)
from src.data.preprocessor import DataPreprocessor
from src.models.supervised import FraudLightGBM
from src.models.anomaly import FraudIsolationForest


class HybridFraudEnsemble:
    """Ensemble fusing supervised known-fraud probabilities with unsupervised zero-day novelty scores.

    Formula:
        Risk = (w_supervised * P_lgbm) + (w_anomaly * S_isolation)
    """

    def __init__(
        self,
        supervised_model: Optional[FraudLightGBM] = None,
        anomaly_model: Optional[FraudIsolationForest] = None,
        supervised_weight: float = 0.70,
        anomaly_weight: float = 0.30,
        approved_threshold: float = 0.30,
        declined_threshold: float = 0.75,
    ):
        self.supervised_model = supervised_model
        self.anomaly_model = anomaly_model
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight
        self.approved_threshold = approved_threshold
        self.declined_threshold = declined_threshold

        # Validate weights sum to 1.0
        total_weight = self.supervised_weight + self.anomaly_weight
        if not np.isclose(total_weight, 1.0):
            self.supervised_weight /= total_weight
            self.anomaly_weight /= total_weight

    def score(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[DecisionType]]:
        """Compute batch scores and decisions across all input rows.

        Returns:
            Tuple of (supervised_scores, anomaly_scores, hybrid_scores, decisions)
        """
        if self.supervised_model is None or not self.supervised_model.is_fitted:
            raise ValueError("Supervised model must be provided and fitted.")
        if self.anomaly_model is None or not self.anomaly_model.is_fitted:
            raise ValueError("Anomaly model must be provided and fitted.")

        sup_scores = self.supervised_model.predict_proba(X)
        ano_scores = self.anomaly_model.score_anomaly(X)

        hybrid_scores = (self.supervised_weight * sup_scores) + (self.anomaly_weight * ano_scores)
        hybrid_scores = np.clip(hybrid_scores, 0.0, 1.0)

        decisions = []
        for s in hybrid_scores:
            if s >= self.declined_threshold:
                decisions.append(DecisionType.DECLINED)
            elif s >= self.approved_threshold:
                decisions.append(DecisionType.MANUAL_REVIEW)
            else:
                decisions.append(DecisionType.APPROVED)

        return sup_scores, ano_scores, hybrid_scores, decisions

    def score_single(
        self,
        vector: FeatureVector,
        preprocessor: DataPreprocessor,
        transaction_id: str,
        user_id: str,
        amount: float,
        latency_ms: float = 0.0
    ) -> ScoringResult:
        """Score single transaction feature vector with sub-30ms performance and explainability."""
        start_time = time.perf_counter()

        X = preprocessor.transform_single(vector)
        sup_score = float(self.supervised_model.predict_proba(X)[0])
        ano_score = float(self.anomaly_model.score_anomaly(X)[0])

        hybrid_score = round(
            float(np.clip(
                (self.supervised_weight * sup_score) + (self.anomaly_weight * ano_score),
                0.0,
                1.0
            )),
            4
        )

        if hybrid_score >= self.declined_threshold:
            decision = DecisionType.DECLINED
        elif hybrid_score >= self.approved_threshold:
            decision = DecisionType.MANUAL_REVIEW
        else:
            decision = DecisionType.APPROVED

        # Extract attribution reasons
        reasons = []
        if sup_score >= 0.70:
            reasons.append("HIGH_SUPERVISED_FRAUD_PROBABILITY")
        if ano_score >= 0.65:
            reasons.append("UNSUPERVISED_OUT_OF_DISTRIBUTION_ANOMALY")
        if vector.tx_count_past_5m >= 4:
            reasons.append("HIGH_TRANSACTION_VELOCITY_BURST")
        if vector.distance_from_last_tx_km >= 400 and vector.seconds_since_last_tx <= 1800:
            reasons.append("IMPOSSIBLE_GEOGRAPHIC_TRAVEL_VELOCITY")
        if vector.ratio_to_avg_amount_24h >= 6.0:
            reasons.append("ANOMALOUS_AMOUNT_SPIKE_VS_HISTORY")
        if vector.device_trust_score <= 0.25:
            reasons.append("UNTRUSTED_OR_NEW_DEVICE_FINGERPRINT")
        if vector.is_international and vector.channel_risk_score >= 0.5:
            reasons.append("HIGH_RISK_CROSS_BORDER_CHANNEL")

        if not reasons and decision != DecisionType.APPROVED:
            reasons.append("ELEVATED_MULTIVARIATE_RISK_COMPOSITE")

        total_latency = latency_ms + ((time.perf_counter() - start_time) * 1000.0)

        return ScoringResult(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            supervised_score=round(sup_score, 4),
            anomaly_score=round(ano_score, 4),
            hybrid_risk_score=hybrid_score,
            decision=decision,
            latency_ms=round(total_latency, 2),
            reasons=reasons,
            metadata={
                "weights": {
                    "supervised": self.supervised_weight,
                    "anomaly": self.anomaly_weight,
                },
                "thresholds": {
                    "approved": self.approved_threshold,
                    "declined": self.declined_threshold,
                }
            }
        )

    def save(self, directory: Path) -> None:
        """Persist both ensemble components and configuration."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.supervised_model:
            self.supervised_model.save(directory / "lightgbm_fraud_model.joblib")
        if self.anomaly_model:
            self.anomaly_model.save(directory / "isolation_forest.joblib")

        config = {
            "supervised_weight": self.supervised_weight,
            "anomaly_weight": self.anomaly_weight,
            "approved_threshold": self.approved_threshold,
            "declined_threshold": self.declined_threshold,
        }
        joblib.dump(config, directory / "ensemble_config.joblib")

    @classmethod
    def load(cls, directory: Path) -> "HybridFraudEnsemble":
        """Load fitted models and assemble hybrid pipeline."""
        sup_path = directory / "lightgbm_fraud_model.joblib"
        ano_path = directory / "isolation_forest.joblib"
        cfg_path = directory / "ensemble_config.joblib"

        supervised_model = FraudLightGBM.load(sup_path) if sup_path.exists() else None
        anomaly_model = FraudIsolationForest.load(ano_path) if ano_path.exists() else None

        config = joblib.load(cfg_path) if cfg_path.exists() else {}

        return cls(
            supervised_model=supervised_model,
            anomaly_model=anomaly_model,
            supervised_weight=config.get("supervised_weight", 0.70),
            anomaly_weight=config.get("anomaly_weight", 0.30),
            approved_threshold=config.get("approved_threshold", 0.30),
            declined_threshold=config.get("declined_threshold", 0.75),
        )
