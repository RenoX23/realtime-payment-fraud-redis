"""High-throughput latency benchmarking script profiling Redis feature lookups and hybrid model scoring."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure root in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.data.schemas import TransactionPayload, MerchantCategory, PaymentChannel
from src.data.preprocessor import DataPreprocessor
from src.features.redis_store import RedisFeatureStore
from src.features.assembler import FeatureAssembler
from src.models.ensemble import HybridFraudEnsemble
from src.data.generator import TransactionDataGenerator


def benchmark_latency(
    num_iterations: int = 500,
    output_file: Path = settings.MODEL_DIR / "latency_benchmark.json"
) -> Dict[str, Any]:
    """Profile latency distribution across 500 real-time transaction scoring cycles.

    Verifies the sub-28ms end-to-end SLA requirement in PROJECT.md.
    """
    print("==================================================================")
    print("PHASE 3: REDIS FEATURE STORE & REAL-TIME LATENCY BENCHMARK")
    print("==================================================================")

    # 1. Initialize Components
    print("\n[1/4] Connecting to Redis feature store and loading hybrid ensemble...")
    feature_store = RedisFeatureStore()
    health = feature_store.health_check()
    print(f"      Redis Backend: {health.get('backend')} | Status: {health.get('status')} | Ping: {health.get('ping_ms', 0)}ms")

    assembler = FeatureAssembler(feature_store=feature_store)
    preprocessor = DataPreprocessor.load(settings.MODEL_DIR / "preprocessor.joblib")
    ensemble = HybridFraudEnsemble.load(settings.MODEL_DIR)

    # Warmup pipeline with 10 dummy iterations
    print("\n[2/4] Warming up JIT and model execution caches (10 cycles)...")
    generator = TransactionDataGenerator(num_users=20, random_seed=42)
    warmup_df = generator.generate_dataset(num_samples=10, fraud_rate=0.1)
    for _, row in warmup_df.iterrows():
        payload = TransactionPayload(
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
        vec, _ = assembler.assemble(payload)
        ensemble.score_single(vec, preprocessor, payload.transaction_id, payload.user_id, payload.amount)

    # 2. Generate Real-time Test Payloads
    print(f"\n[3/4] Benchmarking {num_iterations} sequential payment transactions against Redis feature store...")
    test_df = generator.generate_dataset(num_samples=num_iterations, fraud_rate=0.05)

    redis_latencies: List[float] = []
    inference_latencies: List[float] = []
    commit_latencies: List[float] = []
    end_to_end_latencies: List[float] = []

    for _, row in test_df.iterrows():
        payload = TransactionPayload(
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

        t_start = time.perf_counter()

        # Step A: Redis feature lookup & assembly
        t_redis_start = time.perf_counter()
        vector, redis_ms = assembler.assemble(payload)
        redis_latencies.append(redis_ms)

        # Step B: Model inference (LightGBM + Isolation Forest)
        t_inf_start = time.perf_counter()
        result = ensemble.score_single(
            vector=vector,
            preprocessor=preprocessor,
            transaction_id=payload.transaction_id,
            user_id=payload.user_id,
            amount=payload.amount,
            latency_ms=redis_ms
        )
        t_inf_end = time.perf_counter()
        inference_latencies.append((t_inf_end - t_inf_start) * 1000.0)

        # Step C: Commit transaction to Redis sliding window
        t_commit_start = time.perf_counter()
        assembler.commit(payload)
        commit_latencies.append((time.perf_counter() - t_commit_start) * 1000.0)

        t_end = time.perf_counter()
        end_to_end_latencies.append((t_end - t_start) * 1000.0)

    # 3. Compute Percentiles
    p50_e2e = float(np.percentile(end_to_end_latencies, 50))
    p90_e2e = float(np.percentile(end_to_end_latencies, 90))
    p95_e2e = float(np.percentile(end_to_end_latencies, 95))
    p99_e2e = float(np.percentile(end_to_end_latencies, 99))
    mean_e2e = float(np.mean(end_to_end_latencies))

    p50_redis = float(np.percentile(redis_latencies, 50))
    p95_redis = float(np.percentile(redis_latencies, 95))

    p50_inf = float(np.percentile(inference_latencies, 50))
    p95_inf = float(np.percentile(inference_latencies, 95))

    p50_commit = float(np.percentile(commit_latencies, 50))
    p95_commit = float(np.percentile(commit_latencies, 95))

    print("\n[4/4] Latency Benchmark Profile (Milliseconds):")
    print("------------------------------------------------------------------")
    print(f"Pipeline Stage            | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms)")
    print("------------------------------------------------------------------")
    print(f"Redis Feature Lookup      | {p50_redis:8.2f} | {np.percentile(redis_latencies, 90):8.2f} | {p95_redis:8.2f} | {np.percentile(redis_latencies, 99):8.2f}")
    print(f"Hybrid ML Inference       | {p50_inf:8.2f} | {np.percentile(inference_latencies, 90):8.2f} | {p95_inf:8.2f} | {np.percentile(inference_latencies, 99):8.2f}")
    print(f"Redis Window Write/Commit | {p50_commit:8.2f} | {np.percentile(commit_latencies, 90):8.2f} | {p95_commit:8.2f} | {np.percentile(commit_latencies, 99):8.2f}")
    print("------------------------------------------------------------------")
    print(f"END-TO-END TOTAL LATENCY  | {p50_e2e:8.2f} | {p90_e2e:8.2f} | {p95_e2e:8.2f} | {p99_e2e:8.2f}")
    print("------------------------------------------------------------------")
    print(f"Mean Latency:             {mean_e2e:.2f} ms")
    print(f"Target SLA:               < 28.0 ms")

    # Check Acceptance Criteria
    assert p95_e2e < 28.0, f"End-to-end P95 latency {p95_e2e:.2f}ms exceeded 28ms SLA!"
    print("\n>>> ACCEPTANCE CRITERIA MET: End-to-end latency clocked at < 28ms (P95: {:.2f}ms)! <<<\n".format(p95_e2e))

    benchmark_data = {
        "iterations": num_iterations,
        "backend": health.get("backend"),
        "metrics_ms": {
            "end_to_end": {
                "mean": round(mean_e2e, 2),
                "p50": round(p50_e2e, 2),
                "p90": round(p90_e2e, 2),
                "p95": round(p95_e2e, 2),
                "p99": round(p99_e2e, 2),
            },
            "redis_lookup": {
                "p50": round(p50_redis, 2),
                "p95": round(p95_redis, 2),
            },
            "hybrid_inference": {
                "p50": round(p50_inf, 2),
                "p95": round(p95_inf, 2),
            },
            "redis_commit": {
                "p50": round(p50_commit, 2),
                "p95": round(p95_commit, 2),
            },
        },
        "sla_target_ms": 28.0,
        "sla_passed": p95_e2e < 28.0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved benchmark telemetry report to: {output_file}\n")

    return benchmark_data


if __name__ == "__main__":
    benchmark_latency()
