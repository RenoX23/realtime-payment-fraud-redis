"""End-to-end training and benchmark pipeline for imbalanced fraud classification."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import settings
from src.data.schemas import DecisionType
from src.data.generator import TransactionDataGenerator
from src.data.preprocessor import DataPreprocessor, MODEL_FEATURE_NAMES
from src.models.supervised import FraudLightGBM
from src.models.anomaly import FraudIsolationForest
from src.models.ensemble import HybridFraudEnsemble
from src.models.baselines import (
    DummyFraudClassifier,
    LogisticRegressionBaseline,
    RandomForestBaseline,
)
from src.models.evaluator import ModelEvaluator


def run_training_pipeline(
    num_samples: int = 100_000,
    fraud_rate: float = 0.002,
    output_dir: Path = settings.MODEL_DIR,
    save_sample_data: bool = True
) -> Dict[str, Any]:
    """Execute complete Phase 1 imbalanced data and model training pipeline.

    Args:
        num_samples: Number of transactions to simulate.
        fraud_rate: Prevalence of fraudulent transactions (e.g. 0.002 = 0.2%).
        output_dir: Directory where trained artifacts will be saved.
        save_sample_data: Whether to save sample CSV in data/ directory.

    Returns:
        Dictionary containing comparative evaluation metrics across all models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("PHASE 1: IMBALANCED DATA PIPELINE & LIGHTGBM TRAINING PIPELINE")
    print("==================================================================")

    # 1. Generate Dataset
    print(f"\n[1/6] Generating {num_samples:,} synthetic transactions (Fraud Rate: {fraud_rate * 100:.2f}%)...")
    generator = TransactionDataGenerator(num_users=1000, random_seed=42)
    df = generator.generate_dataset(num_samples=num_samples, fraud_rate=fraud_rate)

    total_frauds = int(df["is_fraud"].sum())
    print(f"      Total records: {len(df):,} | Fraudulent transactions: {total_frauds:,} ({total_frauds / len(df) * 100:.3f}%)")

    if save_sample_data:
        sample_path = settings.DATA_DIR / "sample_transactions.csv"
        # Save first 5,000 samples for inspectability
        df.head(5000).to_csv(sample_path, index=False)
        print(f"      Saved sample dataset to: {sample_path}")

    # 2. Temporal Train/Val/Test Split
    print("\n[2/6] Performing chronological temporal train/validation/test split...")
    preprocessor = DataPreprocessor(feature_names=MODEL_FEATURE_NAMES)
    train_df, val_df, test_df = preprocessor.temporal_train_test_split(df, train_ratio=0.70, val_ratio=0.15)

    print(f"      Train set: {len(train_df):,} records ({train_df['is_fraud'].sum()} frauds)")
    print(f"      Val set:   {len(val_df):,} records ({val_df['is_fraud'].sum()} frauds)")
    print(f"      Test set:  {len(test_df):,} records ({test_df['is_fraud'].sum()} frauds)")

    # 3. Fit Preprocessing Pipeline
    print("\n[3/6] Fitting robust scaler on historical training split (zero future leakage)...")
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["is_fraud"].values

    X_val = preprocessor.transform(val_df)
    y_val = val_df["is_fraud"].values

    X_test = preprocessor.transform(test_df)
    y_test = test_df["is_fraud"].values

    preprocessor_path = output_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"      Fitted preprocessor saved to: {preprocessor_path}")

    # 4. Train Baselines for Rigorous Benchmark
    print("\n[4/6] Training baseline models (Dummy Naive, Logistic Regression, Random Forest)...")
    evaluator = ModelEvaluator(
        cost_fn=settings.COST_FALSE_NEGATIVE,
        cost_fp=settings.COST_FALSE_POSITIVE
    )

    # A. Dummy Classifier
    dummy = DummyFraudClassifier()
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    dummy_metrics = evaluator.compute_metrics(y_test, dummy_probs, threshold=0.5)

    # B. Logistic Regression (Balanced)
    log_reg = LogisticRegressionBaseline(random_state=42)
    log_reg.fit(X_train, y_train)
    log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
    log_reg_metrics = evaluator.compute_metrics(y_test, log_reg_probs, threshold=0.5)

    # C. Random Forest (Balanced Subsample)
    rf = RandomForestBaseline(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluator.compute_metrics(y_test, rf_probs, threshold=0.5)

    # 5. Train LightGBM Classifier with PR-AUC Objective
    print("\n[5/6] Training LightGBM with calibrated scale_pos_weight & PR-AUC early stopping...")
    lgbm = FraudLightGBM(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=10.0,
        random_state=42
    )
    lgbm.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=30)

    lgbm_test_probs = lgbm.predict_proba(X_test)

    # Threshold optimization on validation set to prevent test leakage
    lgbm_val_probs = lgbm.predict_proba(X_val)
    optimal_thresh_f1, best_val_f1 = evaluator.find_optimal_threshold_f1(y_val, lgbm_val_probs)
    optimal_thresh_cost, min_val_cost = evaluator.find_optimal_threshold_cost(y_val, lgbm_val_probs)

    # Clamp decision threshold within operational bounds [0.15, 0.85]
    clamped_thresh = float(np.clip(optimal_thresh_f1, 0.15, 0.85))

    print(f"      Optimal F1 Threshold (from Val set):   {optimal_thresh_f1:.4f} (Val F1: {best_val_f1:.4f})")
    print(f"      Operational Decision Threshold:       {clamped_thresh:.4f}")
    print(f"      Cost-Minimizing Threshold:            {optimal_thresh_cost:.4f}")

    # Set decision threshold
    lgbm.decision_threshold = clamped_thresh
    lgbm_metrics = evaluator.compute_metrics(y_test, lgbm_test_probs, threshold=clamped_thresh)

    # Calculate Top-K metrics
    p_at_50 = evaluator.compute_precision_at_k(y_test, lgbm_test_probs, k=50)
    p_at_100 = evaluator.compute_precision_at_k(y_test, lgbm_test_probs, k=100)
    recall_at_95_p = evaluator.compute_recall_at_precision(y_test, lgbm_test_probs, target_precision=0.95)

    lgbm_metrics["precision_at_50"] = p_at_50
    lgbm_metrics["precision_at_100"] = p_at_100
    lgbm_metrics["recall_at_95_precision"] = recall_at_95_p

    # Save model artifact
    model_path = output_dir / "lightgbm_fraud_model.joblib"
    lgbm.save(model_path)
    print(f"      Trained LightGBM model saved to: {model_path}")

    # Feature Importances
    feature_importances = lgbm.get_feature_importances(MODEL_FEATURE_NAMES)

    # 6. Train Isolation Forest Anomaly Scorer (Phase 2)
    print("\n[6/7] Training unsupervised Isolation Forest on normal transactions for zero-day detection...")
    # Train on normal transactions only
    X_train_normal = X_train[y_train == 0]
    iso_forest = FraudIsolationForest(
        n_estimators=50,
        contamination=0.01,
        random_state=42
    )
    iso_forest.fit(X_train_normal)
    iso_path = output_dir / "isolation_forest.joblib"
    iso_forest.save(iso_path)
    print(f"      Trained Isolation Forest saved to: {iso_path}")

    # Build and persist Hybrid Ensemble
    ensemble = HybridFraudEnsemble(
        supervised_model=lgbm,
        anomaly_model=iso_forest,
        supervised_weight=0.70,
        anomaly_weight=0.30,
        approved_threshold=0.30,
        declined_threshold=0.75,
    )
    ensemble.save(output_dir)
    print(f"      Hybrid Ensemble configuration saved to: {output_dir / 'ensemble_config.joblib'}")

    # Evaluate Hybrid Ensemble on Test Set
    _, ano_test_scores, hybrid_test_scores, _ = ensemble.score(X_test)
    hybrid_metrics = evaluator.compute_metrics(y_test, hybrid_test_scores, threshold=0.30)
    hybrid_metrics["precision_at_100"] = evaluator.compute_precision_at_k(y_test, hybrid_test_scores, k=100)

    # Zero-Day Novelty Anomaly Stress Test (Phase 2 Acceptance Criteria)
    print("\n[7/7] Executing Zero-Day Out-of-Distribution (OOD) Novelty Attack Benchmark...")
    # Generate synthetic zero-day attacks that look normal to LightGBM but deviate multivariately
    num_zero_day = 50
    zero_day_X = np.zeros((num_zero_day, len(MODEL_FEATURE_NAMES)))
    # Features 0-3 (amount, hour, dow, merchant) are set to normal baseline values
    # Features 6-11 (distance hops, velocity burst combinations) are shifted heavily
    zero_day_X[:, 6] = np.random.uniform(5.0, 10.0, size=num_zero_day)  # Distance home
    zero_day_X[:, 7] = np.random.uniform(6.0, 12.0, size=num_zero_day)  # Distance last tx
    zero_day_X[:, 8] = np.random.uniform(-1.0, -0.8, size=num_zero_day) # Seconds since last (very rapid)
    zero_day_X[:, 9] = np.random.uniform(4.0, 8.0, size=num_zero_day)   # 5m velocity

    zd_sup_scores = lgbm.predict_proba(zero_day_X)
    zd_ano_scores = iso_forest.score_anomaly(zero_day_X)
    _, _, zd_hybrid_scores, zd_decisions = ensemble.score(zero_day_X)

    zd_caught_by_sup = int(np.sum(zd_sup_scores >= 0.50))
    zd_caught_by_ano = int(np.sum(zd_ano_scores >= 0.65))
    zd_caught_by_hybrid = int(sum(1 for d in zd_decisions if d in [DecisionType.MANUAL_REVIEW, DecisionType.DECLINED]))

    print(f"      Simulated Zero-Day Attacks:               {num_zero_day}")
    print(f"      Flagged by Supervised Model Alone:       {zd_caught_by_sup} / {num_zero_day} ({zd_caught_by_sup/num_zero_day*100:.1f}%)")
    print(f"      Flagged by Isolation Forest Anomaly:     {zd_caught_by_ano} / {num_zero_day} ({zd_caught_by_ano/num_zero_day*100:.1f}%)")
    print(f"      Flagged by Hybrid Decision Engine:       {zd_caught_by_hybrid} / {num_zero_day} ({zd_caught_by_hybrid/num_zero_day*100:.1f}%)")

    assert zd_caught_by_hybrid > zd_caught_by_sup, "Ensemble must catch more zero-day attacks than supervised model alone!"
    print("\n>>> ACCEPTANCE CRITERIA MET: Hybrid ensemble successfully detected novel zero-day anomalies! <<<")

    # 7. Comparative Benchmark Report
    print("\nBenchmark Results Comparison (Strict Imbalance Evaluation):")
    comparison_table = [
        {
            "Model": "Dummy (Always 0)",
            "PR-AUC": dummy_metrics["pr_auc"],
            "ROC-AUC": dummy_metrics["roc_auc"],
            "F1-Score": dummy_metrics["f1_score"],
            "Financial Loss ($)": f"${dummy_metrics['financial_loss_usd']:,.2f}",
        },
        {
            "Model": "Logistic Regression (Balanced)",
            "PR-AUC": log_reg_metrics["pr_auc"],
            "ROC-AUC": log_reg_metrics["roc_auc"],
            "F1-Score": log_reg_metrics["f1_score"],
            "Financial Loss ($)": f"${log_reg_metrics['financial_loss_usd']:,.2f}",
        },
        {
            "Model": "Random Forest (Balanced Subsample)",
            "PR-AUC": rf_metrics["pr_auc"],
            "ROC-AUC": rf_metrics["roc_auc"],
            "F1-Score": rf_metrics["f1_score"],
            "Financial Loss ($)": f"${rf_metrics['financial_loss_usd']:,.2f}",
        },
        {
            "Model": "LightGBM (Supervised)",
            "PR-AUC": lgbm_metrics["pr_auc"],
            "ROC-AUC": lgbm_metrics["roc_auc"],
            "F1-Score": lgbm_metrics["f1_score"],
            "Financial Loss ($)": f"${lgbm_metrics['financial_loss_usd']:,.2f}",
        },
        {
            "Model": "Hybrid Ensemble (LightGBM + IsoForest)",
            "PR-AUC": hybrid_metrics["pr_auc"],
            "ROC-AUC": hybrid_metrics["roc_auc"],
            "F1-Score": hybrid_metrics["f1_score"],
            "Financial Loss ($)": f"${hybrid_metrics['financial_loss_usd']:,.2f}",
        },
    ]

    report_df = pd.DataFrame(comparison_table)
    print("\n" + report_df.to_string(index=False))

    print("\nKey Performance Indicators (Hybrid Pipeline):")
    print(f"  * LightGBM PR-AUC:        {lgbm_metrics['pr_auc']:.4f} (Target >= {settings.TARGET_PR_AUC})")
    print(f"  * Hybrid PR-AUC:          {hybrid_metrics['pr_auc']:.4f}")
    print(f"  * Precision@Top-100:      {p_at_100 * 100:.1f}%")
    print(f"  * Recall at 95% Prec:     {recall_at_95_p * 100:.1f}%")
    print(f"  * Zero-Day Catch Rate:    {zd_caught_by_hybrid / num_zero_day * 100:.1f}%")
    print(f"  * Financial Loss Saved:   ${dummy_metrics['financial_loss_usd'] - lgbm_metrics['financial_loss_usd']:,.2f} vs Naive")

    # Persist metrics summary
    metrics_report = {
        "dataset_summary": {
            "total_samples": num_samples,
            "fraud_rate": fraud_rate,
            "test_samples": len(test_df),
            "test_frauds": int(y_test.sum()),
        },
        "models": {
            "dummy": dummy_metrics,
            "logistic_regression": log_reg_metrics,
            "random_forest": rf_metrics,
            "lightgbm": lgbm_metrics,
            "hybrid_ensemble": hybrid_metrics,
        },
        "zero_day_benchmark": {
            "num_attacks": num_zero_day,
            "supervised_caught": zd_caught_by_sup,
            "anomaly_caught": zd_caught_by_ano,
            "hybrid_caught": zd_caught_by_hybrid,
        },
        "feature_importances": feature_importances,
    }

    report_path = output_dir / "metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Full benchmark metrics report saved to: {report_path}\n")

    return metrics_report


if __name__ == "__main__":
    run_training_pipeline()
