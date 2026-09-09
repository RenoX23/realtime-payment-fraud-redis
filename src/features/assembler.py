"""Feature assembler combining static transaction payload attributes with real-time Redis features."""

from datetime import datetime, timezone
from typing import Tuple, Optional
import pandas as pd

from src.data.schemas import (
    TransactionPayload,
    FeatureVector,
    MerchantCategory,
    PaymentChannel,
)
from src.data.generator import MERCHANT_RISK_PRIORS, CHANNEL_RISK_PRIORS
from src.features.redis_store import RedisFeatureStore


class FeatureAssembler:
    """Assembles full model-ready feature vectors from incoming payloads and Redis state."""

    def __init__(self, feature_store: Optional[RedisFeatureStore] = None):
        self.feature_store = feature_store or RedisFeatureStore()

    def assemble(self, payload: TransactionPayload) -> Tuple[FeatureVector, float]:
        """Assemble static and dynamic features for scoring.

        Returns:
            Tuple of (FeatureVector, redis_lookup_latency_ms)
        """
        # Temporal decomposition (UTC)
        dt = datetime.fromtimestamp(payload.timestamp, tz=timezone.utc)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()

        # Risk priors
        merchant_risk = MERCHANT_RISK_PRIORS.get(payload.merchant_category.value, 0.20)
        channel_risk = CHANNEL_RISK_PRIORS.get(payload.channel.value, 0.20)

        # Dynamic rolling features from Redis feature store
        redis_features, redis_latency = self.feature_store.get_features(
            user_id=payload.user_id,
            current_ts=payload.timestamp,
            current_amount=payload.amount,
            current_dist=payload.distance_from_home_km
        )

        vector = FeatureVector(
            amount=payload.amount,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            merchant_risk_score=merchant_risk,
            channel_risk_score=channel_risk,
            device_trust_score=payload.device_trust_score,
            distance_from_home_km=payload.distance_from_home_km,
            is_international=1 if payload.is_international else 0,
            tx_count_past_5m=redis_features["tx_count_past_5m"],
            tx_count_past_1h=redis_features["tx_count_past_1h"],
            tx_sum_amount_past_24h=redis_features["tx_sum_amount_past_24h"],
            ratio_to_avg_amount_24h=redis_features["ratio_to_avg_amount_24h"],
            distance_from_last_tx_km=redis_features["distance_from_last_tx_km"],
            seconds_since_last_tx=redis_features["seconds_since_last_tx"],
        )

        return vector, redis_latency

    def commit(self, payload: TransactionPayload) -> None:
        """Persist transaction event into Redis sliding window."""
        self.feature_store.record_transaction(
            user_id=payload.user_id,
            transaction_id=payload.transaction_id,
            timestamp=payload.timestamp,
            amount=payload.amount,
            distance_from_home_km=payload.distance_from_home_km,
        )
