"""Unit tests for unsupervised anomaly detection and hybrid ensembling."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.data.schemas import (
    FeatureVector,
    DecisionType,
)
from src.data.preprocessor import DataPreprocessor, MODEL_FEATURE_NAMES
from src.models.supervised import FraudLightGBM
from src.models.anomaly import FraudIsolationForest
from src.models.ensemble import HybridFraudEnsemble


@pytest.fixture
def trained_models():
    """Create and fit both supervised and anomaly models on synthetic benchmark data."""
    rng = np.random.default_rng(42)
    n_features = len(MODEL_FEATURE_NAMES)

    # 1000 normal transactions
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(1000, n_features))
    y_normal = np.zeros(1000, dtype=int)

    # 20 known fraud samples (high values on first 3 features)
    X_fraud = rng.normal(loc=4.0, scale=1.0, size=(20, n_features))
    y_fraud = np.ones(20, dtype=int)

    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([y_normal, y_fraud])

    # Fit LightGBM
    lgbm = FraudLightGBM(n_estimators=30, learning_rate=0.1, random_state=42)
    lgbm.fit(X, y)

    # Fit Isolation Forest strictly on normal transactions
    iso = FraudIsolationForest(n_estimators=50, random_state=42)
    iso.fit(X_normal)

    return lgbm, iso


def test_isolation_forest_calibrated_scoring(trained_models):
    """Verify Isolation Forest outputs bounded scores in [0.0, 1.0]."""
    _, iso = trained_models
    assert iso.is_fitted is True

    # Normal points should have low anomaly scores
    X_inliers = np.zeros((10, len(MODEL_FEATURE_NAMES)))
    scores_inliers = iso.score_anomaly(X_inliers)
    assert np.all((scores_inliers >= 0.0) & (scores_inliers <= 1.0))
    assert np.mean(scores_inliers) < 0.60

    # Extreme outliers (10 standard deviations out) should have high anomaly scores
    X_outliers = np.ones((5, len(MODEL_FEATURE_NAMES))) * 10.0
    scores_outliers = iso.score_anomaly(X_outliers)
    assert np.all(scores_outliers >= 0.70)


def test_hybrid_ensemble_scoring_math(trained_models):
    """Verify weighted ensemble arithmetic: 0.7 * supervised + 0.3 * anomaly."""
    lgbm, iso = trained_models
    ensemble = HybridFraudEnsemble(
        supervised_model=lgbm,
        anomaly_model=iso,
        supervised_weight=0.70,
        anomaly_weight=0.30,
        approved_threshold=0.30,
        declined_threshold=0.75,
    )

    X_test = np.random.randn(20, len(MODEL_FEATURE_NAMES))
    sup_scores, ano_scores, hybrid_scores, decisions = ensemble.score(X_test)

    expected_hybrid = np.clip((0.70 * sup_scores) + (0.30 * ano_scores), 0.0, 1.0)
    np.testing.assert_allclose(hybrid_scores, expected_hybrid, atol=1e-5)
    assert len(decisions) == 20
    assert all(isinstance(d, DecisionType) for d in decisions)


def test_hybrid_ensemble_catches_zero_day_anomaly(trained_models):
    """Verify hybrid ensemble detects out-of-distribution zero-day anomaly missed by supervised model alone.

    Acceptance Criteria for Phase 2:
    Ensemble detects synthetic out-of-distribution anomaly patterns missed by supervised model alone.
    """
    lgbm, iso = trained_models
    ensemble = HybridFraudEnsemble(
        supervised_model=lgbm,
        anomaly_model=iso,
        supervised_weight=0.60,
        anomaly_weight=0.40,
        approved_threshold=0.30,
        declined_threshold=0.70,
    )

    # Construct zero-day attack: normal on features 0-2 (so LightGBM thinks it's legit),
    # but wildly abnormal on features 5-10 (unseen multivariate distribution)
    n_features = len(MODEL_FEATURE_NAMES)
    zero_day_vector = np.zeros((1, n_features))
    # Features 0-2 remain 0.0 (looks completely normal to supervised model)
    # Features 5-10 are 8 standard deviations away
    zero_day_vector[0, 5:] = 8.0

    sup_score = float(lgbm.predict_proba(zero_day_vector)[0])
    ano_score = float(iso.score_anomaly(zero_day_vector)[0])

    _, _, hybrid_score, decision = ensemble.score(zero_day_vector)
    hybrid_val = float(hybrid_score[0])
    decision_val = decision[0]

    # Supervised model alone thought it was legitimate (< 0.25)
    assert sup_score < 0.25, f"Expected supervised model to miss zero-day, got {sup_score}"

    # Isolation forest detects the anomaly (> 0.70)
    assert ano_score >= 0.70, f"Expected Isolation Forest to flag zero-day anomaly, got {ano_score}"

    # Hybrid ensemble elevates the transaction into review/declined status
    assert hybrid_val >= 0.30, f"Expected hybrid score >= 0.30, got {hybrid_val}"
    assert decision_val in [DecisionType.MANUAL_REVIEW, DecisionType.DECLINED]


def test_ensemble_single_scoring_and_serialization(trained_models):
    """Verify single feature vector scoring with latency and rule attribution."""
    lgbm, iso = trained_models
    ensemble = HybridFraudEnsemble(supervised_model=lgbm, anomaly_model=iso)

    preprocessor = DataPreprocessor()
    # Fake fit
    preprocessor.fit_transform(np.zeros((10, len(MODEL_FEATURE_NAMES)), dtype=float))

    sample_vector = FeatureVector(
        amount=1200.0,
        hour_of_day=3,
        day_of_week=1,
        merchant_risk_score=0.8,
        channel_risk_score=0.7,
        device_trust_score=0.1,  # Should trigger UNTRUSTED_OR_NEW_DEVICE_FINGERPRINT
        distance_from_home_km=600.0,
        distance_from_last_tx_km=700.0,
        seconds_since_last_tx=300.0,  # 5 min ago + 700km -> IMPOSSIBLE_GEOGRAPHIC_TRAVEL_VELOCITY
        tx_count_past_5m=6,  # Should trigger HIGH_TRANSACTION_VELOCITY_BURST
        tx_count_past_1h=12,
        tx_sum_amount_past_24h=3500.0,
        ratio_to_avg_amount_24h=8.5,  # Should trigger ANOMALOUS_AMOUNT_SPIKE_VS_HISTORY
        is_international=1,
    )

    result = ensemble.score_single(
        vector=sample_vector,
        preprocessor=preprocessor,
        transaction_id="tx_test_001",
        user_id="usr_001",
        amount=1200.0
    )

    assert result.transaction_id == "tx_test_001"
    assert result.latency_ms > 0.0
    assert len(result.reasons) >= 3
    assert "HIGH_TRANSACTION_VELOCITY_BURST" in result.reasons
    assert "IMPOSSIBLE_GEOGRAPHIC_TRAVEL_VELOCITY" in result.reasons
    assert "ANOMALOUS_AMOUNT_SPIKE_VS_HISTORY" in result.reasons
    assert "UNTRUSTED_OR_NEW_DEVICE_FINGERPRINT" in result.reasons

    # Serialization test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir) / "models"
        ensemble.save(tmp_dir)

        loaded_ensemble = HybridFraudEnsemble.load(tmp_dir)
        assert loaded_ensemble.supervised_model is not None
        assert loaded_ensemble.anomaly_model is not None
        assert loaded_ensemble.supervised_weight == 0.70
