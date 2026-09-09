"""Unit tests for data schemas, synthetic transaction generation, and preprocessing pipeline."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.data.schemas import (
    TransactionPayload,
    FeatureVector,
    MerchantCategory,
    PaymentChannel,
)
from src.data.generator import TransactionDataGenerator
from src.data.preprocessor import DataPreprocessor, MODEL_FEATURE_NAMES


def test_transaction_payload_validation():
    """Verify Pydantic enforces positive amount and valid types."""
    # Valid payload
    payload = TransactionPayload(
        transaction_id="tx_12345",
        user_id="usr_00001",
        amount=125.50,
        timestamp=1700000000.0,
        merchant_category=MerchantCategory.ELECTRONICS,
        channel=PaymentChannel.ONLINE,
        device_trust_score=0.85,
        distance_from_home_km=12.4,
        is_international=False,
    )
    assert payload.amount == 125.50
    assert payload.merchant_category == MerchantCategory.ELECTRONICS

    # Negative amount should raise ValidationError
    with pytest.raises(ValidationError):
        TransactionPayload(
            transaction_id="tx_invalid",
            user_id="usr_00001",
            amount=-50.0,
            timestamp=1700000000.0,
            merchant_category=MerchantCategory.GROCERY,
            channel=PaymentChannel.POS_CHIP,
            device_trust_score=0.9,
            distance_from_home_km=5.0,
        )


def test_transaction_generator_imbalance_and_temporal_order():
    """Verify synthetic generator generates extreme class imbalance in strict chronological order."""
    generator = TransactionDataGenerator(num_users=50, random_seed=42)
    df = generator.generate_dataset(num_samples=2000, fraud_rate=0.002)

    assert len(df) == 2000
    assert "is_fraud" in df.columns
    assert "timestamp" in df.columns

    # Verify chronological order
    timestamps = df["timestamp"].values
    assert np.all(np.diff(timestamps) >= 0), "Timestamps must be strictly non-decreasing"

    # Verify fraud rate within realistic tolerance
    fraud_count = df["is_fraud"].sum()
    assert fraud_count > 0, "Must contain at least 1 fraud sample"
    assert fraud_count <= 20, f"Fraud count ({fraud_count}) should reflect ~0.2% rate"


def test_temporal_split_zero_leakage():
    """Verify temporal train/test split avoids future-to-past leakage."""
    generator = TransactionDataGenerator(num_users=50, random_seed=42)
    df = generator.generate_dataset(num_samples=1000, fraud_rate=0.01)

    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.temporal_train_test_split(df, train_ratio=0.7, val_ratio=0.15)

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    # Ensure zero temporal overlap
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_preprocessor_fit_transform_and_serialization():
    """Verify preprocessor transforms features and serializes accurately."""
    generator = TransactionDataGenerator(num_users=50, random_seed=42)
    df = generator.generate_dataset(num_samples=500, fraud_rate=0.01)

    preprocessor = DataPreprocessor()
    transformed_batch = preprocessor.fit_transform(df)

    assert transformed_batch.shape == (500, len(MODEL_FEATURE_NAMES))
    assert not np.isnan(transformed_batch).any()

    # Single vector transformation
    single_vector = FeatureVector(
        amount=250.0,
        hour_of_day=14,
        day_of_week=3,
        merchant_risk_score=0.65,
        channel_risk_score=0.60,
        device_trust_score=0.2,
        distance_from_home_km=150.0,
        is_international=1,
        tx_count_past_5m=4,
        tx_count_past_1h=8,
        tx_sum_amount_past_24h=1200.0,
        ratio_to_avg_amount_24h=4.5,
        distance_from_last_tx_km=85.0,
        seconds_since_last_tx=45.0,
    )
    single_transformed = preprocessor.transform_single(single_vector)
    assert single_transformed.shape == (1, len(MODEL_FEATURE_NAMES))

    # Serialization roundtrip test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "preprocessor.joblib"
        preprocessor.save(tmp_path)
        assert tmp_path.exists()

        loaded = DataPreprocessor.load(tmp_path)
        assert loaded.is_fitted is True
        roundtrip_transformed = loaded.transform_single(single_vector)
        np.testing.assert_allclose(single_transformed, roundtrip_transformed)
