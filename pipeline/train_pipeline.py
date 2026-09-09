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
from src.data.generator import TransactionDataGenerator
from src.data.preprocessor import DataPreprocessor, MODEL_FEATURE_NAMES
from src.models.supervised import FraudLightGBM
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

    # 6. Comparative Benchmark Report
    print("\n[6/6] Benchmark Results Comparison (Strict Imbalance Evaluation):")
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
            "Model": "LightGBM (Ours)",
            "PR-AUC": lgbm_metrics["pr_auc"],
            "ROC-AUC": lgbm_metrics["roc_auc"],
            "F1-Score": lgbm_metrics["f1_score"],
            "Financial Loss ($)": f"${lgbm_metrics['financial_loss_usd']:,.2f}",
        },
    ]

    report_df = pd.DataFrame(comparison_table)
    print("\n" + report_df.to_string(index=False))

    print("\nKey Performance Indicators (LightGBM):")
    print(f"  * PR-AUC:                 {lgbm_metrics['pr_auc']:.4f} (Target >= {settings.TARGET_PR_AUC})")
    print(f"  * Precision@Top-100:      {p_at_100 * 100:.1f}%")
    print(f"  * Recall at 95% Prec:     {recall_at_95_p * 100:.1f}%")
    print(f"  * Optimal Threshold:      {optimal_thresh_f1:.4f}")
    print(f"  * Financial Loss Saved:   ${dummy_metrics['financial_loss_usd'] - lgbm_metrics['financial_loss_usd']:,.2f} vs Naive")

    # Verify Acceptance Criteria
    assert lgbm_metrics["pr_auc"] >= settings.TARGET_PR_AUC, (
        f"PR-AUC {lgbm_metrics['pr_auc']} did not meet target threshold of {settings.TARGET_PR_AUC}!"
    )
    print("\n>>> ACCEPTANCE CRITERIA MET: PR-AUC >= 0.82 verified! <<<")

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
