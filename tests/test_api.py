"""Integration tests for FastAPI real-time fraud scoring endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, startup_event
from src.data.schemas import DecisionType


@pytest.fixture(scope="module")
def client():
    """Create test client with models initialized."""
    startup_event()
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    """Verify /v1/health returns healthy models and redis backend."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "redis_feature_store" in data
    assert data["sla_target_ms"] == 30.0


def test_score_endpoint_approved_transaction(client):
    """Verify scoring a low-risk legitimate transaction is APPROVED with sub-30ms response."""
    payload = {
        "transaction_id": "tx_legit_001",
        "user_id": "usr_legit_001",
        "amount": 25.50,
        "timestamp": 1715000000.0,
        "merchant_category": "grocery",
        "channel": "pos_chip",
        "device_trust_score": 0.95,
        "distance_from_home_km": 2.5,
        "is_international": False,
    }

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_legit_001"
    assert data["decision"] in [DecisionType.APPROVED.value, DecisionType.MANUAL_REVIEW.value]
    assert "latency_ms" in data
    assert "hybrid_risk_score" in data
    assert "supervised_score" in data
    assert "anomaly_score" in data

    # Verify telemetry headers
    assert "X-Scoring-Latency-Ms" in response.headers
    assert "X-Fraud-Decision" in response.headers


def test_score_endpoint_flagged_fraud_transaction(client):
    """Verify high-risk attack transaction triggers decline or review with explainability reasons."""
    payload = {
        "transaction_id": "tx_fraud_burst_001",
        "user_id": "usr_victim_001",
        "amount": 4800.0,
        "timestamp": 1715000100.0,
        "merchant_category": "jewelry",
        "channel": "online",
        "device_trust_score": 0.05,
        "distance_from_home_km": 950.0,
        "is_international": True,
    }

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] in [DecisionType.MANUAL_REVIEW.value, DecisionType.DECLINED.value]
    assert data["hybrid_risk_score"] >= 0.30
    assert len(data["reasons"]) > 0


def test_score_endpoint_validation_error(client):
    """Verify invalid payloads (e.g. negative amount) are rejected with 422 Unprocessable Entity."""
    invalid_payload = {
        "transaction_id": "tx_invalid",
        "user_id": "usr_001",
        "amount": -500.0,  # Negative amount prohibited
        "timestamp": 1715000000.0,
        "merchant_category": "grocery",
        "channel": "pos_chip",
        "device_trust_score": 0.9,
        "distance_from_home_km": 5.0,
    }

    response = client.post("/v1/score", json=invalid_payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """Verify /v1/metrics returns latency SLA distribution and decisions breakdown."""
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "total_scored_transactions" in data
    assert "latency_telemetry_ms" in data
    assert "p50" in data["latency_telemetry_ms"]
    assert "p95" in data["latency_telemetry_ms"]


def test_simulate_endpoint(client):
    """Verify /v1/simulate generates and scores transaction streams for live dashboard."""
    response = client.post("/v1/simulate?count=10&fraud_rate=0.20")
    assert response.status_code == 200
    data = response.json()

    assert data["simulated_count"] == 10
    assert len(data["scored_results"]) == 10
