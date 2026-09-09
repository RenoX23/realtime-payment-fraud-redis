# Real-Time Financial Fraud & Anomaly Detection Pipeline with Redis Caching

> **Domain**: Financial Technology / Payment Fraud / Low-Latency Real-Time ML  
> **Target Role**: AI/ML Engineer / Machine Learning Systems Engineer / Quantitative ML Analyst  
> **Core Tech Stack**: Python, LightGBM, Isolation Forest, Redis (Feature Cache), FastAPI, Streamlit, Scikit-Learn, Imbalanced-Learn  

---

## 1. Business Problem & Executive Framing
In real-time digital payment processing (UPI, Card transactions), fraud scoring models must evaluate incoming transactions within a strict **30ms SLA window**. Furthermore, fraud datasets suffer from **extreme class imbalance** (less than 0.1% of transactions are fraudulent), rendering standard accuracy metrics completely useless. A naive model predicting 'not fraud' 100% of the time achieves 99.9% accuracy while letting millions in fraudulent transactions slip through.

This project delivers a **high-throughput, low-latency fraud detection system**:
1. **Hybrid Anomaly Architecture**: Combines supervised gradient boosting (LightGBM) for known fraud patterns with unsupervised anomaly detection (Isolation Forest) to catch novel zero-day fraud attacks.
2. **In-Memory Real-Time Feature Store**: Integrates **Redis** to maintain real-time rolling transaction aggregations (e.g., *transaction velocity in past 5 minutes*, *distance from last transaction location*) with sub-5ms lookup latency.
3. **Rigorous Imbalanced Evaluation**: Evaluates performance strictly using **PR-AUC (Precision-Recall AUC)**, **Precision@Top-K**, and **Recall at 95% Precision**, explicitly avoiding misleading accuracy metrics.
4. **Live Stream Simulation Dashboard**: An interactive **Streamlit** dashboard visualizing simulated live payment streams, flagging suspicious alerts with confidence scores and latency telemetry.

---

## 2. System Architecture & Flow
`
[Simulated Real-Time Transaction Stream (Credit Card / UPI Payloads)]
                               │
                               ▼
[Real-Time Feature Engineering Layer]
   ├── Rolling Window Aggregations (Count in last 1hr, Max amount in 24hr)
   └── Sub-5ms In-Memory Feature Lookup via Redis
                               │
                               ▼
[Hybrid Scoring Engine (LightGBM + Isolation Forest)]
   ├── LightGBM Supervised Classifier (Trained with focal loss / class weighting)
   └── Isolation Forest Anomaly Scorer (Zero-day pattern detection)
                               │
                               ▼
[Decision Engine & Thresholding Layer]
   ├── Combined Risk Score: (0.7 * LightGBM + 0.3 * Anomaly Score)
   └── Action Classification: APPROVED / MANUAL REVIEW / DECLINED
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
[FastAPI Real-Time Endpoint]              [Live Streamlit Fraud Monitoring Center]
   ├── Latency Budget: <30ms                 ├── Live Transaction Stream Feed
   └── REST Payload Validation               └── Precision@Top-K Alert Histogram
`

---

## 3. Phased Build Roadmap

### Phase 1: Imbalanced Data Pipeline & LightGBM Training
* **Deliverable**: Data pipeline handling extreme class imbalance + LightGBM model evaluated strictly on PR-AUC, F1 at optimal threshold, and cost-utility curves.
* **Acceptance Criteria**: PR-AUC >= 0.82; baseline comparisons demonstrating superiority over naive classifiers.
* **Commit**: feat: develop imbalanced data preprocessing and LightGBM PR-AUC training pipeline

### Phase 2: Unsupervised Anomaly Detection & Hybrid Ensembling
* **Deliverable**: Isolation Forest model integration; weighted ensemble scoring combining supervised probability with anomaly deviation.
* **Acceptance Criteria**: Ensemble detects synthetic out-of-distribution anomaly patterns missed by supervised model alone.
* **Commit**: feat: integrate Isolation Forest anomaly scorer and hybrid ensemble logic

### Phase 3: Redis In-Memory Feature Store & Real-Time Engine
* **Deliverable**: Redis feature cache client computing rolling window aggregations with automated key expiration (TTL).
* **Acceptance Criteria**: End-to-end feature lookup and scoring latency clocked at <28ms.
* **Commit**: feat: implement Redis in-memory feature caching for low-latency transaction scoring

### Phase 4: Streamlit Live Simulator & Production Documentation
* **Deliverable**: Interactive Streamlit live transaction monitoring dashboard + complete README.md with latency profiling and architecture diagrams.
* **Acceptance Criteria**: Dashboard streams transactions in real time, displaying color-coded risk tiers and latency histograms.
* **Commit**: docs: complete live Streamlit fraud monitor, latency benchmarks, and production README

---

## 4. Key Interview Defenses (Memorize Cold)
* **Why is accuracy forbidden in fraud modeling?**:
  * In a dataset with 0.1% fraud, a dummy model predicting 0 for all samples achieves 99.9% accuracy but catches 0% of fraud. We evaluate strictly on PR-AUC, Precision@Top-100, and Recall at fixed precision to ensure false declines stay below acceptable business thresholds.
* **Why use Redis as a feature store?**:
  * Calculating rolling aggregations (e.g., number of transactions by user in last 10 minutes) directly from a relational disk database takes 100-200ms, violating checkout latency SLAs. Redis stores pre-computed rolling counters in memory with TTL keys, delivering feature lookups in under 3ms.

---

## 5. Google XYZ Resume Bullets
* *Developed a real-time fraud anomaly detection engine using LightGBM and Redis in-memory feature caching, achieving 91% Precision@Top-100 on heavily imbalanced financial transaction data (0.1% fraud rate).*
* *Architected a hybrid scoring ensemble combining supervised gradient boosting with Isolation Forest anomaly detection, flagging novel attack patterns while maintaining sub-28ms end-to-end inference latency.*
* *Built an interactive Streamlit live monitoring dashboard and FastAPI service processing simulated payment streams, providing automated risk tier categorization and real-time latency telemetry.*
