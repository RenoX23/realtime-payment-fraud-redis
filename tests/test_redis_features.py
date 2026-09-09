"""Unit tests for Redis feature store, rolling window aggregations, and feature assembly."""

import time
import pytest

from src.data.schemas import (
    TransactionPayload,
    MerchantCategory,
    PaymentChannel,
)
from src.features.redis_store import RedisFeatureStore, InMemoryFeatureStoreFallback
from src.features.assembler import FeatureAssembler


def test_in_memory_feature_store_fallback():
    """Verify fallback in-memory store computes sliding windows accurately."""
    store = InMemoryFeatureStoreFallback()
    user_id = "test_usr_001"
    now = 1700000000.0

    # First transaction
    store.record_transaction(user_id, "tx_1", now - 600, 100.0, 10.0)  # 10 min ago
    store.record_transaction(user_id, "tx_2", now - 120, 50.0, 12.0)   # 2 min ago
    store.record_transaction(user_id, "tx_3", now - 30, 200.0, 15.0)   # 30 sec ago

    feats = store.get_features(user_id, now, current_amount=75.0, current_dist=18.0)

    # In past 5m: tx_2 and tx_3 -> count = 2
    assert feats["tx_count_past_5m"] == 2
    # In past 1h: tx_1, tx_2, tx_3 -> count = 3
    assert feats["tx_count_past_1h"] == 3
    # Sum in past 24h: 100 + 50 + 200 = 350.0
    assert feats["tx_sum_amount_past_24h"] == 350.0
    # Last transaction was tx_3 (30 sec ago, dist 15.0 vs 18.0)
    assert feats["seconds_since_last_tx"] == 30.0
    assert feats["distance_from_last_tx_km"] == 3.0


def test_redis_feature_store_client_sliding_windows():
    """Verify live Redis feature store (or connected backend) executes sliding window pipeline."""
    store = RedisFeatureStore()
    user_id = f"test_usr_{int(time.time())}"
    now = time.time()

    # Flush before test
    store.flush_user(user_id)

    # Initial query on empty user
    feats_empty, lat_ms = store.get_features(user_id, now, current_amount=120.0, current_dist=5.0)
    assert feats_empty["tx_count_past_5m"] == 0
    assert feats_empty["tx_count_past_1h"] == 0
    assert feats_empty["tx_sum_amount_past_24h"] == 0.0
    assert lat_ms < 20.0  # Must be fast

    # Add transactions
    store.record_transaction(user_id, "tx_101", now - 200, 150.0, 10.0)
    store.record_transaction(user_id, "tx_102", now - 60, 250.0, 25.0)

    feats_populated, lat_ms2 = store.get_features(user_id, now, current_amount=50.0, current_dist=30.0)
    assert feats_populated["tx_count_past_5m"] == 2
    assert feats_populated["tx_count_past_1h"] == 2
    assert feats_populated["tx_sum_amount_past_24h"] == 400.0
    assert feats_populated["distance_from_last_tx_km"] == 5.0
    assert feats_populated["seconds_since_last_tx"] == pytest.approx(60.0, abs=2.0)

    # Clean up
    store.flush_user(user_id)


def test_feature_assembler_vector_generation():
    """Verify feature assembler compiles all static and dynamic features."""
    store = RedisFeatureStore()
    assembler = FeatureAssembler(feature_store=store)

    payload = TransactionPayload(
        transaction_id="tx_test_asm_1",
        user_id="usr_asm_001",
        amount=340.50,
        timestamp=1710000000.0,
        merchant_category=MerchantCategory.JEWELRY,
        channel=PaymentChannel.ONLINE,
        device_trust_score=0.45,
        distance_from_home_km=85.0,
        is_international=True,
    )

    vector, redis_latency = assembler.assemble(payload)

    assert vector.amount == 340.50
    assert vector.merchant_risk_score == 0.80
    assert vector.channel_risk_score == 0.60
    assert vector.device_trust_score == 0.45
    assert vector.is_international == 1
    assert redis_latency >= 0.0

    # Commit
    assembler.commit(payload)
    # Next transaction should see past history
    payload2 = TransactionPayload(
        transaction_id="tx_test_asm_2",
        user_id="usr_asm_001",
        amount=100.0,
        timestamp=1710000100.0,  # 100 sec later
        merchant_category=MerchantCategory.GROCERY,
        channel=PaymentChannel.POS_CHIP,
        device_trust_score=0.9,
        distance_from_home_km=10.0,
        is_international=False,
    )
    vector2, _ = assembler.assemble(payload2)
    assert vector2.tx_count_past_5m >= 1
    assert vector2.tx_sum_amount_past_24h >= 340.50
