"""Pydantic data schemas for transaction payloads, feature vectors, and scoring responses."""

from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class DecisionType(str, Enum):
    """Transaction disposition decision tier."""
    APPROVED = "APPROVED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DECLINED = "DECLINED"


class MerchantCategory(str, Enum):
    """Categorization of merchant domain."""
    GROCERY = "grocery"
    RESTAURANT = "restaurant"
    ELECTRONICS = "electronics"
    JEWELRY = "jewelry"
    TRAVEL = "travel"
    GAMING = "gaming"
    CRYPTO = "crypto"
    UTILITIES = "utilities"
    HEALTH = "health"
    GENERAL_RETAIL = "general_retail"


class PaymentChannel(str, Enum):
    """Channel through which transaction occurred."""
    ONLINE = "online"
    POS_CHIP = "pos_chip"
    POS_CONTACTLESS = "pos_contactless"
    ATM = "atm"
    P2P = "p2p"


class TransactionPayload(BaseModel):
    """Raw incoming transaction payload from payment gateway or simulator."""

    transaction_id: str = Field(..., description="Unique transaction UUID")
    user_id: str = Field(..., description="Cardholder or account identifier")
    amount: float = Field(..., gt=0, description="Transaction monetary amount in USD")
    timestamp: float = Field(..., description="Unix timestamp (epoch in seconds)")
    merchant_category: MerchantCategory = Field(..., description="Merchant domain category")
    channel: PaymentChannel = Field(..., description="Payment presentation channel")
    device_trust_score: float = Field(
        ..., ge=0.0, le=1.0, description="Device fingerprint trust score between 0.0 and 1.0"
    )
    distance_from_home_km: float = Field(
        ..., ge=0.0, description="Haversine distance from cardholder primary residence (km)"
    )
    is_international: bool = Field(False, description="Flag indicating cross-border transaction")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount must be strictly positive")
        return round(v, 2)


class FeatureVector(BaseModel):
    """Engineered feature vector consumed by hybrid ML scoring pipeline."""

    amount: float
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    merchant_risk_score: float
    channel_risk_score: float
    device_trust_score: float
    distance_from_home_km: float
    is_international: int  # 0 or 1
    # Real-Time Rolling Features (Computed / Cached in Redis)
    tx_count_past_5m: int = 0
    tx_count_past_1h: int = 0
    tx_sum_amount_past_24h: float = 0.0
    ratio_to_avg_amount_24h: float = 1.0
    distance_from_last_tx_km: float = 0.0
    seconds_since_last_tx: float = 86400.0


class ScoringResult(BaseModel):
    """Scored transaction output with confidence, ensemble breakdown, and telemetry."""

    transaction_id: str
    user_id: str
    amount: float
    supervised_score: float = Field(..., ge=0.0, le=1.0, description="LightGBM probability")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Isolation Forest anomaly score")
    hybrid_risk_score: float = Field(..., ge=0.0, le=1.0, description="Weighted ensemble risk score")
    decision: DecisionType
    latency_ms: float = Field(..., ge=0.0, description="End-to-end scoring latency in milliseconds")
    reasons: list[str] = Field(default_factory=list, description="Rule or feature attribution flags")
    metadata: Dict[str, Any] = Field(default_factory=dict)
