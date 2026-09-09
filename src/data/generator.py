"""Synthetic financial transaction generator simulating realistic payment streams with extreme class imbalance."""

import time
import uuid
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.data.schemas import MerchantCategory, PaymentChannel

# Risk priors for domain encoding
MERCHANT_RISK_PRIORS = {
    MerchantCategory.GROCERY.value: 0.05,
    MerchantCategory.RESTAURANT.value: 0.10,
    MerchantCategory.HEALTH.value: 0.08,
    MerchantCategory.UTILITIES.value: 0.04,
    MerchantCategory.GENERAL_RETAIL.value: 0.15,
    MerchantCategory.TRAVEL.value: 0.40,
    MerchantCategory.ELECTRONICS.value: 0.65,
    MerchantCategory.GAMING.value: 0.70,
    MerchantCategory.JEWELRY.value: 0.80,
    MerchantCategory.CRYPTO.value: 0.90,
}

CHANNEL_RISK_PRIORS = {
    PaymentChannel.POS_CHIP.value: 0.05,
    PaymentChannel.POS_CONTACTLESS.value: 0.10,
    PaymentChannel.ATM.value: 0.25,
    PaymentChannel.P2P.value: 0.45,
    PaymentChannel.ONLINE.value: 0.60,
}


class TransactionDataGenerator:
    """Generates synthetic high-fidelity financial payment transactions.

    Models realistic customer spending behavior, merchant category distributions,
    temporal dynamics, and low-prevalence (~0.15%) multi-vector fraud attacks.
    """

    def __init__(
        self,
        num_users: int = 1000,
        random_seed: int = 42
    ):
        self.num_users = num_users
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Pre-assign persistent user profiles (mean spending, home base)
        self.user_ids = [f"usr_{i:05d}" for i in range(num_users)]
        self.user_mean_amounts = self.rng.lognormal(mean=3.5, sigma=0.6, size=num_users)  # ~$30-$150 avg
        self.user_profile_map = {
            uid: float(amt) for uid, amt in zip(self.user_ids, self.user_mean_amounts)
        }

    def generate_dataset(
        self,
        num_samples: int = 100_000,
        fraud_rate: float = 0.0015,
        start_time: Optional[float] = None
    ) -> pd.DataFrame:
        """Generate a chronologically sorted transaction dataset with extreme class imbalance.

        Args:
            num_samples: Total number of transactions to generate.
            fraud_rate: Target fraction of fraudulent transactions (default: 0.15%).
            start_time: Unix epoch starting time. Defaults to 30 days prior to current time.

        Returns:
            pd.DataFrame with all feature columns and binary 'is_fraud' label.
        """
        if start_time is None:
            # 30 days prior to now
            start_time = time.time() - (30 * 86400)

        num_frauds = int(num_samples * fraud_rate)
        num_legit = num_samples - num_frauds

        # 1. Generate timestamps (Poisson arrival process for chronological ordering)
        # Average interval ~ 25 seconds between transactions
        time_intervals = self.rng.exponential(scale=25.0, size=num_samples)
        timestamps = start_time + np.cumsum(time_intervals)

        # 2. Assign fraud indices uniformly or in clustered bursts across the timeline
        fraud_indices = set(self.rng.choice(num_samples, size=num_frauds, replace=False))

        # Assign user IDs
        assigned_users = self.rng.choice(self.user_ids, size=num_samples)

        categories = list(MerchantCategory)
        cat_values = [c.value for c in categories]
        cat_probs_legit = [0.25, 0.20, 0.08, 0.02, 0.05, 0.05, 0.02, 0.15, 0.08, 0.10]
        cat_probs_fraud = [0.02, 0.03, 0.30, 0.25, 0.15, 0.10, 0.12, 0.01, 0.01, 0.01]

        channels = list(PaymentChannel)
        chan_values = [c.value for c in channels]
        chan_probs_legit = [0.35, 0.30, 0.20, 0.08, 0.07]
        chan_probs_fraud = [0.75, 0.03, 0.02, 0.10, 0.10]

        data = []
        user_history = {}  # uid -> (last_time, last_dist, count_1h, count_5m, sum_24h)

        for i in range(num_samples):
            ts = timestamps[i]
            uid = assigned_users[i]
            user_mean = self.user_profile_map[uid]
            is_fraud = 1 if i in fraud_indices else 0

            # Compute temporal attributes
            dt = pd.to_datetime(ts, unit="s")
            hour_of_day = dt.hour
            day_of_week = dt.dayofweek

            # Initialize user tracking if first appearance
            if uid not in user_history:
                user_history[uid] = {
                    "last_ts": ts - 86400,
                    "tx_times": [],
                    "tx_amounts": []
                }

            hist = user_history[uid]
            prev_ts = hist["last_ts"]
            sec_since_last = max(1.0, ts - prev_ts)

            # Clean sliding window history (keep last 24h)
            cutoff_24h = ts - 86400
            cutoff_1h = ts - 3600
            cutoff_5m = ts - 300

            valid_recent = [
                (t, a) for t, a in zip(hist["tx_times"], hist["tx_amounts"]) if t >= cutoff_24h
            ]
            hist["tx_times"] = [t for t, _ in valid_recent]
            hist["tx_amounts"] = [a for _, a in valid_recent]

            count_24h = len(hist["tx_times"])
            count_1h = sum(1 for t in hist["tx_times"] if t >= cutoff_1h)
            count_5m = sum(1 for t in hist["tx_times"] if t >= cutoff_5m)
            sum_24h = sum(hist["tx_amounts"])
            avg_24h = (sum_24h / count_24h) if count_24h > 0 else user_mean

            if not is_fraud:
                # Legitimate transaction behavior
                category = self.rng.choice(cat_values, p=cat_probs_legit)
                channel = self.rng.choice(chan_values, p=chan_probs_legit)
                amount = float(self.rng.lognormal(mean=np.log(user_mean), sigma=0.5))
                amount = max(1.50, round(amount, 2))
                device_trust = float(np.clip(self.rng.beta(a=8, b=2), 0.0, 1.0))
                distance_home = float(self.rng.exponential(scale=12.0))
                distance_last = float(self.rng.exponential(scale=5.0))
                is_intl = 1 if self.rng.random() < 0.03 else 0
            else:
                # Fraudulent transaction behaviors (multi-attack vectors)
                attack_type = self.rng.choice(["velocity_burst", "high_amount_jewelry", "geo_leap", "untrusted_online"])

                category = self.rng.choice(cat_values, p=cat_probs_fraud)
                channel = self.rng.choice(chan_values, p=chan_probs_fraud)
                is_intl = 1 if self.rng.random() < 0.35 else 0

                if attack_type == "velocity_burst":
                    count_5m += self.rng.integers(3, 8)
                    count_1h += self.rng.integers(5, 12)
                    sec_since_last = float(self.rng.uniform(5, 60))
                    amount = float(user_mean * self.rng.uniform(1.5, 4.0))
                    device_trust = float(self.rng.uniform(0.1, 0.5))
                    distance_home = float(self.rng.uniform(10, 80))
                    distance_last = float(self.rng.uniform(0.1, 5.0))
                elif attack_type == "high_amount_jewelry":
                    amount = float(user_mean * self.rng.uniform(8.0, 35.0))
                    device_trust = float(self.rng.uniform(0.05, 0.4))
                    distance_home = float(self.rng.uniform(50, 400))
                    distance_last = float(self.rng.uniform(20, 200))
                    category = self.rng.choice([MerchantCategory.JEWELRY.value, MerchantCategory.ELECTRONICS.value, MerchantCategory.CRYPTO.value])
                    channel = PaymentChannel.ONLINE.value
                elif attack_type == "geo_leap":
                    # Impossible physical leap
                    sec_since_last = float(self.rng.uniform(60, 600))  # 1 to 10 mins ago
                    distance_last = float(self.rng.uniform(800, 3500))  # Cross-continent
                    distance_home = distance_last
                    amount = float(user_mean * self.rng.uniform(2.0, 8.0))
                    device_trust = float(self.rng.uniform(0.0, 0.3))
                else:  # untrusted_online
                    amount = float(user_mean * self.rng.uniform(3.0, 10.0))
                    device_trust = float(self.rng.uniform(0.0, 0.2))
                    channel = PaymentChannel.ONLINE.value
                    distance_home = float(self.rng.uniform(100, 1500))
                    distance_last = float(self.rng.uniform(50, 500))

                amount = max(10.0, round(amount, 2))

            # Ratio to user's 24h rolling average
            ratio_to_avg = round(amount / max(5.0, avg_24h), 2)

            # Update user history
            hist["last_ts"] = ts
            hist["tx_times"].append(ts)
            hist["tx_amounts"].append(amount)

            # Record
            data.append({
                "transaction_id": str(uuid.UUID(int=self.rng.integers(0, 2**63))),
                "transaction_id": str(uuid.UUID(int=int(self.rng.integers(0, 2**63)))),
                "user_id": uid,
                "timestamp": ts,
                "amount": amount,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "merchant_category": category,
                "channel": channel,
                "merchant_risk_score": MERCHANT_RISK_PRIORS.get(category, 0.2),
                "channel_risk_score": CHANNEL_RISK_PRIORS.get(channel, 0.2),
                "device_trust_score": round(device_trust, 3),
                "distance_from_home_km": round(distance_home, 2),
                "distance_from_last_tx_km": round(distance_last, 2),
                "seconds_since_last_tx": round(sec_since_last, 1),
                "tx_count_past_5m": count_5m,
                "tx_count_past_1h": count_1h,
                "tx_sum_amount_past_24h": round(sum_24h, 2),
                "ratio_to_avg_amount_24h": ratio_to_avg,
                "is_international": is_intl,
                "is_fraud": is_fraud
            })

        df = pd.DataFrame(data)
        # Sort chronologically to mirror production streaming ingestion
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
