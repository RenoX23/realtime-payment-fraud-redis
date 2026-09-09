"""In-memory Redis feature store maintaining real-time rolling transaction aggregations with TTL."""

import json
import time
from typing import Dict, Any, Tuple, Optional, List
import redis

from src.config import settings


class InMemoryFeatureStoreFallback:
    """High-performance in-memory fallback mimicking Redis when no Redis daemon is accessible."""

    def __init__(self):
        self._windows: Dict[str, List[Tuple[float, float, float]]] = {}  # uid -> [(ts, amount, dist)]
        self._last_tx: Dict[str, Dict[str, float]] = {}

    def get_features(
        self,
        user_id: str,
        current_ts: float,
        current_amount: float,
        current_dist: float
    ) -> Dict[str, Any]:
        history = self._windows.get(user_id, [])
        cutoff_24h = current_ts - 86400
        cutoff_1h = current_ts - 3600
        cutoff_5m = current_ts - 300

        # Purge entries older than 24h
        valid_history = [entry for entry in history if entry[0] >= cutoff_24h]
        self._windows[user_id] = valid_history

        count_5m = sum(1 for entry in valid_history if entry[0] >= cutoff_5m)
        count_1h = sum(1 for entry in valid_history if entry[0] >= cutoff_1h)
        count_24h = len(valid_history)
        sum_24h = sum(entry[1] for entry in valid_history)

        avg_24h = (sum_24h / count_24h) if count_24h > 0 else current_amount
        ratio_to_avg = round(current_amount / max(5.0, avg_24h), 2)

        last_info = self._last_tx.get(user_id)
        if last_info:
            sec_since_last = max(1.0, current_ts - last_info["ts"])
            dist_last = abs(current_dist - last_info["dist"])
        else:
            sec_since_last = 86400.0
            dist_last = 0.0

        return {
            "tx_count_past_5m": count_5m,
            "tx_count_past_1h": count_1h,
            "tx_sum_amount_past_24h": round(sum_24h, 2),
            "ratio_to_avg_amount_24h": ratio_to_avg,
            "distance_from_last_tx_km": round(dist_last, 2),
            "seconds_since_last_tx": round(sec_since_last, 1),
        }

    def record_transaction(
        self,
        user_id: str,
        tx_id: str,
        ts: float,
        amount: float,
        dist: float
    ) -> None:
        if user_id not in self._windows:
            self._windows[user_id] = []
        self._windows[user_id].append((ts, amount, dist))
        self._last_tx[user_id] = {"ts": ts, "dist": dist, "amount": amount}

    def clear(self) -> None:
        self._windows.clear()
        self._last_tx.clear()


class RedisFeatureStore:
    """Production Redis client managing sliding window counters and features with sub-5ms latency."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        socket_timeout: Optional[float] = None
    ):
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db if db is not None else settings.REDIS_DB
        self.password = password or settings.REDIS_PASSWORD
        self.socket_timeout = socket_timeout or settings.REDIS_SOCKET_TIMEOUT

        self.client: Optional[redis.Redis] = None
        self.fallback = InMemoryFeatureStoreFallback()
        self.use_fallback = False

        self._connect()

    def _connect(self) -> None:
        """Attempt connection to Redis cluster or standalone instance."""
        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.socket_timeout,
                decode_responses=True
            )
            client.ping()
            self.client = client
            self.use_fallback = False
        except Exception:
            self.client = None
            self.use_fallback = True

    def health_check(self) -> Dict[str, Any]:
        """Check Redis connectivity and return roundtrip ping latency."""
        if self.use_fallback or self.client is None:
            return {
                "status": "degraded_fallback",
                "backend": "in_memory",
                "ping_ms": 0.05,
                "connected": False
            }
        try:
            t0 = time.perf_counter()
            self.client.ping()
            ping_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return {
                "status": "healthy",
                "backend": "redis",
                "ping_ms": ping_ms,
                "connected": True
            }
        except Exception as e:
            return {
                "status": "error",
                "backend": "in_memory_fallback",
                "error": str(e),
                "connected": False
            }

    def get_features(
        self,
        user_id: str,
        current_ts: float,
        current_amount: float,
        current_dist: float
    ) -> Tuple[Dict[str, Any], float]:
        """Retrieve real-time rolling window features with sub-5ms latency SLA.

        Returns:
            Tuple of (features_dict, lookup_latency_ms)
        """
        t0 = time.perf_counter()

        if self.use_fallback or self.client is None:
            feats = self.fallback.get_features(user_id, current_ts, current_amount, current_dist)
            lookup_ms = (time.perf_counter() - t0) * 1000.0
            return feats, round(lookup_ms, 3)

        try:
            pipe = self.client.pipeline()
            win_key = f"fraud:user:window:{user_id}"
            last_key = f"fraud:user:last:{user_id}"

            cutoff_24h = current_ts - 86400
            cutoff_1h = current_ts - 3600
            cutoff_5m = current_ts - 300

            # 1. Clean entries older than 24 hours
            pipe.zremrangebyscore(win_key, "-inf", cutoff_24h)
            # 2. Get 5-minute count
            pipe.zcount(win_key, cutoff_5m, "+inf")
            # 3. Get 1-hour count
            pipe.zcount(win_key, cutoff_1h, "+inf")
            # 4. Get all entries in past 24 hours (scores + values)
            pipe.zrangebyscore(win_key, cutoff_24h, "+inf")
            # 5. Fetch last transaction hash
            pipe.hgetall(last_key)

            results = pipe.execute()

            count_5m = int(results[1])
            count_1h = int(results[2])
            entries_24h = results[3]
            last_tx_data = results[4]

            # Parse amounts from 24h entries: stored as "amount:dist:tx_id"
            sum_24h = 0.0
            count_24h = len(entries_24h)
            for entry_str in entries_24h:
                try:
                    amt_str = entry_str.split(":")[0]
                    sum_24h += float(amt_str)
                except (ValueError, IndexError):
                    pass

            avg_24h = (sum_24h / count_24h) if count_24h > 0 else current_amount
            ratio_to_avg = round(current_amount / max(5.0, avg_24h), 2)

            if last_tx_data and "ts" in last_tx_data:
                sec_since_last = max(1.0, current_ts - float(last_tx_data["ts"]))
                prev_dist = float(last_tx_data.get("dist", current_dist))
                dist_last = abs(current_dist - prev_dist)
            else:
                sec_since_last = 86400.0
                dist_last = 0.0

            lookup_ms = (time.perf_counter() - t0) * 1000.0

            features = {
                "tx_count_past_5m": count_5m,
                "tx_count_past_1h": count_1h,
                "tx_sum_amount_past_24h": round(sum_24h, 2),
                "ratio_to_avg_amount_24h": ratio_to_avg,
                "distance_from_last_tx_km": round(dist_last, 2),
                "seconds_since_last_tx": round(sec_since_last, 1),
            }
            return features, round(lookup_ms, 3)

        except Exception:
            # Automatic graceful fallback if Redis query encounters socket timeout or error
            feats = self.fallback.get_features(user_id, current_ts, current_amount, current_dist)
            lookup_ms = (time.perf_counter() - t0) * 1000.0
            return feats, round(lookup_ms, 3)

    def record_transaction(
        self,
        user_id: str,
        transaction_id: str,
        timestamp: float,
        amount: float,
        distance_from_home_km: float
    ) -> None:
        """Update Redis rolling window and user state atomically with TTL."""
        if self.use_fallback or self.client is None:
            self.fallback.record_transaction(
                user_id, transaction_id, timestamp, amount, distance_from_home_km
            )
            return

        try:
            pipe = self.client.pipeline()
            win_key = f"fraud:user:window:{user_id}"
            last_key = f"fraud:user:last:{user_id}"

            member_val = f"{amount}:{distance_from_home_km}:{transaction_id}"
            # Add to sorted set with timestamp score
            pipe.zadd(win_key, {member_val: timestamp})
            # Set TTL to 24 hours (86,400 seconds)
            pipe.expire(win_key, 86400)

            # Update last transaction hash
            pipe.hset(
                last_key,
                mapping={
                    "ts": str(timestamp),
                    "dist": str(distance_from_home_km),
                    "amount": str(amount),
                    "tx_id": transaction_id,
                }
            )
            pipe.expire(last_key, 86400)
            pipe.execute()

        except Exception:
            # Fallback on failure
            self.fallback.record_transaction(
                user_id, transaction_id, timestamp, amount, distance_from_home_km
            )

    def flush_user(self, user_id: str) -> None:
        """Clear user state (useful for test resets)."""
        if self.client:
            try:
                self.client.delete(f"fraud:user:window:{user_id}", f"fraud:user:last:{user_id}")
            except Exception:
                pass
        self.fallback.clear()
