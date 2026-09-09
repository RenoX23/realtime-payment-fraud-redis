"""Unit tests for ML models, baselines, evaluator, and PR-AUC criteria."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.data.generator import TransactionDataGenerator
from src.data.preprocessor import DataPreprocessor, MODEL_FEATURE_NAMES
from src.models.evaluator import ModelEvaluator
from src.models.baselines import (
    DummyFraudClassifier,
    LogisticRegressionBaseline,
    RandomForestBaseline,
)
from src.models.supervised import FraudLightGBM


@pytest.fixture
def sample_data():
    """Generate small reproducible dataset for fast model testing."""
    gen = TransactionDataGenerator(num_users=20, random_seed=42)
    df = gen.generate_dataset(num_samples=1000, fraud_rate=0.03)
    prep = DataPreprocessor()
    train_df, val_df, test_df = prep.temporal_train_test_split(df, train_ratio=0.7, val_ratio=0.15)
    X_train = prep.fit_transform(train_df)
    y_train = train_df["is_fraud"].values
    X_val = prep.transform(val_df)
    y_val = val_df["is_fraud"].values
    X_test = prep.transform(test_df)
    y_test = test_df["is_fraud"].values
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_evaluator_metrics_and_financial_loss():
    """Verify evaluator calculates correct PR-AUC, confusion matrix, and dollar loss."""
    evaluator = ModelEvaluator(cost_fn=200.0, cost_fp=10.0)
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    metrics = evaluator.compute_metrics(y_true, y_probs, threshold=0.5)
    assert metrics["pr_auc"] > 0.8
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["confusion_matrix"]["false_negatives"] == 0
    assert metrics["financial_loss_usd"] == 0.0

    # With a higher threshold creating False Negatives
    metrics_high_thresh = evaluator.compute_metrics(y_true, y_probs, threshold=0.85)
    assert metrics_high_thresh["confusion_matrix"]["false_negatives"] == 1
    assert metrics_high_thresh["financial_loss_usd"] == 200.0


def test_evaluator_threshold_optimizers():
    """Verify optimal threshold search functions find sensible boundaries."""
    evaluator = ModelEvaluator(cost_fn=250.0, cost_fp=12.0)
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.65, 0.7, 0.85, 0.9])

    opt_thresh, best_f1 = evaluator.find_optimal_threshold_f1(y_true, y_probs)
    assert 0.3 < opt_thresh <= 0.65
    assert best_f1 == 1.0

    cost_thresh, min_cost = evaluator.find_optimal_threshold_cost(y_true, y_probs)
    assert 0.3 < cost_thresh <= 0.65
    assert min_cost == 0.0


def test_precision_at_k_and_recall_at_precision():
    """Verify Top-K precision and recall at fixed precision."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    p_at_2 = evaluator.compute_precision_at_k(y_true, y_probs, k=2)
    assert p_at_2 == 1.0

    recall_at_95 = evaluator.compute_recall_at_precision(y_true, y_probs, target_precision=0.95)
    assert recall_at_95 == 1.0


def test_dummy_classifier_high_accuracy_zero_pr_auc():
    """Demonstrate naive dummy classifier fails to detect fraud."""
    dummy = DummyFraudClassifier()
    X = np.random.randn(100, 5)
    y = np.zeros(100)
    y[:2] = 1  # 2% fraud

    dummy.fit(X, y)
    probs = dummy.predict_proba(X)[:, 1]
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y, probs, threshold=0.5)

    # Catches 0% of fraud
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["confusion_matrix"]["false_negatives"] == 2


def test_lightgbm_training_and_serialization(sample_data):
    """Verify LightGBM fits, predicts probabilities, and serializes cleanly."""
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    model = FraudLightGBM(n_estimators=30, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert model.is_fitted is True

    # Predictions
    probs = model.predict_proba(X_test)
    assert len(probs) == len(X_test)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = model.predict(X_test, threshold=0.4)
    assert set(preds).issubset({0, 1})

    # Feature importances
    importances = model.get_feature_importances(MODEL_FEATURE_NAMES)
    assert len(importances) == len(MODEL_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in importances.values())

    # Serialization roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_file = Path(tmpdir) / "model.joblib"
        model.save(tmp_model_file)
        assert tmp_model_file.exists()

        loaded_model = FraudLightGBM.load(tmp_model_file)
        assert loaded_model.is_fitted is True
        loaded_probs = loaded_model.predict_proba(X_test)
        np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)
