"""FastAPI low-latency transaction fraud scoring microservice."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from src.config import settings
from src.data.schemas import (
    TransactionPayload,
    ScoringResult,
    DecisionType,
)
from src.data.preprocessor import DataPreprocessor
from src.features.redis_store import RedisFeatureStore
from src.features.assembler import FeatureAssembler
from src.models.ensemble import HybridFraudEnsemble
from src.data.generator import TransactionDataGenerator

# Global runtime state
state: Dict[str, Any] = {
    "preprocessor": None,
    "ensemble": None,
    "feature_store": None,
    "assembler": None,
    "latencies_ms": [],
    "decision_counts": {
        DecisionType.APPROVED.value: 0,
        DecisionType.MANUAL_REVIEW.value: 0,
        DecisionType.DECLINED.value: 0,
    },
    "total_scored": 0,
}


def startup_event():
    """Load model artifacts and initialize Redis connection on server start."""
    feature_store = RedisFeatureStore()
    state["feature_store"] = feature_store
    state["assembler"] = FeatureAssembler(feature_store=feature_store)

    prep_path = settings.MODEL_DIR / "preprocessor.joblib"
    if prep_path.exists():
        state["preprocessor"] = DataPreprocessor.load(prep_path)
    else:
        state["preprocessor"] = DataPreprocessor()

    state["ensemble"] = HybridFraudEnsemble.load(settings.MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for startup and shutdown events."""
    startup_event()
    yield


# App initialization
app = FastAPI(
    title="Real-Time Payment Fraud Scoring Engine",
    description="Sub-30ms hybrid fraud detection microservice powered by LightGBM, Isolation Forest, and Redis feature caching.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)


@app.get("/v1/health", status_code=status.HTTP_200_OK)
def health_check():
    """System health check, model readiness, and Redis latency inspection."""
    feature_store: RedisFeatureStore = state.get("feature_store")
    redis_health = feature_store.health_check() if feature_store else {"status": "uninitialized"}

    ensemble: HybridFraudEnsemble = state.get("ensemble")
    models_ready = (
        ensemble is not None
        and ensemble.supervised_model is not None
        and ensemble.anomaly_model is not None
    )

    return {
        "status": "healthy" if models_ready else "degraded",
        "models_loaded": {
            "lightgbm_supervised": ensemble.supervised_model is not None if ensemble else False,
            "isolation_forest": ensemble.anomaly_model is not None if ensemble else False,
            "preprocessor": state.get("preprocessor") is not None and state["preprocessor"].is_fitted,
        },
        "redis_feature_store": redis_health,
        "sla_target_ms": 30.0,
    }


@app.post("/v1/score", response_model=ScoringResult, status_code=status.HTTP_200_OK)
def score_transaction(payload: TransactionPayload, response: Response):
    """Score incoming transaction payload with sub-30ms latency SLA."""
    t_start = time.perf_counter()

    assembler: FeatureAssembler = state.get("assembler")
    ensemble: HybridFraudEnsemble = state.get("ensemble")
    preprocessor: DataPreprocessor = state.get("preprocessor")

    if not assembler or not ensemble or not preprocessor:
        raise HTTPException(status_code=503, detail="Scoring engine models are not loaded.")

    # 1. Assemble features from payload + Redis sliding window
    vector, redis_latency = assembler.assemble(payload)

    # 2. Run Hybrid Scoring Pipeline
    result = ensemble.score_single(
        vector=vector,
        preprocessor=preprocessor,
        transaction_id=payload.transaction_id,
        user_id=payload.user_id,
        amount=payload.amount,
        latency_ms=redis_latency,
    )

    # 3. Commit transaction to Redis sliding window
    assembler.commit(payload)

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    result.latency_ms = total_latency_ms

    # Update telemetry state
    state["latencies_ms"].append(total_latency_ms)
    state["total_scored"] += 1
    state["decision_counts"][result.decision.value] += 1

    # Attach performance telemetry headers
    response.headers["X-Scoring-Latency-Ms"] = str(total_latency_ms)
    response.headers["X-Redis-Lookup-Ms"] = str(redis_latency)
    response.headers["X-Fraud-Decision"] = result.decision.value

    return result


@app.post("/v1/batch-score", response_model=List[ScoringResult])
def batch_score(payloads: List[TransactionPayload]):
    """Batch scoring for bulk processing or backtesting streams."""
    results = []
    assembler: FeatureAssembler = state.get("assembler")
    ensemble: HybridFraudEnsemble = state.get("ensemble")
    preprocessor: DataPreprocessor = state.get("preprocessor")

    if not assembler or not ensemble or not preprocessor:
        raise HTTPException(status_code=503, detail="Models not initialized.")

    for p in payloads:
        vector, redis_ms = assembler.assemble(p)
        res = ensemble.score_single(
            vector=vector,
            preprocessor=preprocessor,
            transaction_id=p.transaction_id,
            user_id=p.user_id,
            amount=p.amount,
            latency_ms=redis_ms
        )
        assembler.commit(p)
        results.append(res)

    return results


@app.get("/v1/metrics")
def get_metrics():
    """Operational telemetry, decision distribution, and latency SLA percentiles."""
    latencies = state.get("latencies_ms", [])
    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        mean_lat = float(np.mean(latencies))
    else:
        p50 = p95 = p99 = mean_lat = 0.0

    return {
        "total_scored_transactions": state["total_scored"],
        "decisions": state["decision_counts"],
        "latency_telemetry_ms": {
            "mean": round(mean_lat, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "sla_breached_over_30ms": sum(1 for l in latencies if l > 30.0),
        }
    }


@app.post("/v1/simulate")
def simulate_stream(count: int = 20, fraud_rate: float = 0.10):
    """Generate simulated transactions and return scored results for UI streaming."""
    generator = TransactionDataGenerator(num_users=20, random_seed=int(time.time() % 1000))
    df = generator.generate_dataset(num_samples=count, fraud_rate=fraud_rate)

    results = []
    assembler: FeatureAssembler = state.get("assembler")
    ensemble: HybridFraudEnsemble = state.get("ensemble")
    preprocessor: DataPreprocessor = state.get("preprocessor")

    for _, row in df.iterrows():
        p = TransactionPayload(
            transaction_id=row["transaction_id"],
            user_id=row["user_id"],
            amount=float(row["amount"]),
            timestamp=float(row["timestamp"]),
            merchant_category=row["merchant_category"],
            channel=row["channel"],
            device_trust_score=float(row["device_trust_score"]),
            distance_from_home_km=float(row["distance_from_home_km"]),
            is_international=bool(row["is_international"]),
        )
        vector, redis_ms = assembler.assemble(p)
        res = ensemble.score_single(
            vector=vector,
            preprocessor=preprocessor,
            transaction_id=p.transaction_id,
            user_id=p.user_id,
            amount=p.amount,
            latency_ms=redis_ms
        )
        assembler.commit(p)
        results.append(res.model_dump())

    return {"simulated_count": count, "scored_results": results}
