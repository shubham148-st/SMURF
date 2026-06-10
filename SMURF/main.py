"""
SMURF Hackathon — Main Pipeline
=================================
AI/ML-Based Classification of Suspicious Mule Accounts

End-to-end pipeline that:
1. Loads and preprocesses the hackathon DataSet.csv (3,924 features)
2. Engineers and selects the most discriminative features
3. Trains multiple classifiers (XGBoost, LightGBM, RF, Ensemble)
4. Runs unsupervised anomaly detection (Isolation Forest, LOF)
5. Combines into risk scores with tiered alert generation
6. Runs ablation study comparing all models
7. Outputs evaluation plots and risk alerts

Usage:
    python main.py                    # Full pipeline
    python main.py --mode train       # Train only
    python main.py --mode ablation    # Ablation study only
    python main.py --mode quick       # Quick run with key features only

Reuses from SMURF AML_smurf:
    - FocalLoss (model.py → focal_loss_objective)
    - Class imbalance handling (train.py → ratio-based subsampling)
    - Ablation methodology (alabation.py → multi-model comparison)
    - Evaluation metrics (train.py → F1, Precision, Recall)
"""

import argparse
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def run_full_pipeline(use_key_features_only=False, run_anomaly=True, run_ablation_flag=True):
    """
    Execute the complete mule account classification pipeline.
    """
    from preprocess import prepare_data
    from feature_engineering import run_feature_engineering
    from model import build_xgboost, build_lightgbm, build_stacking_ensemble
    from train import (
        train_single_model, train_with_cross_validation,
        run_ablation_study, save_model,
    )
    from anomaly_detection import run_anomaly_detection
    from risk_scoring import compute_risk_scores, generate_risk_report
    from evaluate import plot_model_comparison, plot_feature_importance

    pipeline_start = time.time()

    print("\n" + "█" * 60)
    print("█  SMURF — Suspicious Mule Account Classification")
    print("█  AI/ML Hackathon Pipeline")
    print("█" * 60)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Data Preprocessing
    # ═══════════════════════════════════════════════════════════
    X_train, X_test, y_train, y_test, feature_names, encoders = prepare_data(
        use_undersampling=False,
        use_key_features_only=use_key_features_only,
    )

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Feature Engineering
    # ═══════════════════════════════════════════════════════════
    if not use_key_features_only:
        X_train_eng, X_test_eng, selected_features, importance_report = run_feature_engineering(
            X_train, X_test, y_train, auto_select=True, top_n=80,
        )
    else:
        X_train_eng, X_test_eng = X_train, X_test
        selected_features = feature_names
        importance_report = {}

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Train Primary Model (XGBoost with Focal Loss)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("█  PHASE 3: PRIMARY MODEL TRAINING")
    print("█" * 60)

    xgb_model, xgb_es = build_xgboost(y_train, use_focal_loss=False)
    trained_xgb, xgb_proba, xgb_report = train_single_model(
        xgb_model, X_train_eng, y_train, X_test_eng, y_test,
        model_name="XGBoost Primary",
        early_stopping_rounds=xgb_es,
    )
    save_model(trained_xgb, "XGBoost_Primary")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Train LightGBM
    # ═══════════════════════════════════════════════════════════
    lgb_model = build_lightgbm(y_train)
    trained_lgb, lgb_proba, lgb_report = train_single_model(
        lgb_model, X_train_eng, y_train, X_test_eng, y_test,
        model_name="LightGBM",
    )
    save_model(trained_lgb, "LightGBM")

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Train Stacking Ensemble
    # ═══════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("█  PHASE 5: STACKING ENSEMBLE")
    print("█" * 60)

    ensemble_model = build_stacking_ensemble(y_train)
    trained_ensemble, ensemble_proba, ensemble_report = train_single_model(
        ensemble_model, X_train_eng, y_train, X_test_eng, y_test,
        model_name="Stacking Ensemble",
    )
    save_model(trained_ensemble, "Stacking_Ensemble")

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Anomaly Detection
    # ═══════════════════════════════════════════════════════════
    if run_anomaly:
        anomaly_results = run_anomaly_detection(X_train_eng, X_test_eng, y_test)
        anomaly_scores = anomaly_results["combined_scores"]
    else:
        anomaly_scores = None

    # ═══════════════════════════════════════════════════════════
    # PHASE 7: Risk Scoring & Alert Generation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("█  PHASE 7: RISK SCORING & ALERTS")
    print("█" * 60)

    # Use the best model's probabilities (ensemble)
    best_proba = ensemble_proba

    risk_scores = compute_risk_scores(
        supervised_proba=best_proba,
        anomaly_scores=anomaly_scores,
    )

    risk_report = generate_risk_report(
        risk_scores=risk_scores,
        y_true=y_test,
        account_indices=np.array(y_test.index),
    )

    # ═══════════════════════════════════════════════════════════
    # PHASE 8: Ablation Study (Model Comparison)
    # ═══════════════════════════════════════════════════════════
    if run_ablation_flag:
        ablation_results = run_ablation_study(X_train_eng, X_test_eng, y_train, y_test, runs=2)
        plot_model_comparison(ablation_results)

    # ═══════════════════════════════════════════════════════════
    # PHASE 9: Feature Importance Visualization
    # ═══════════════════════════════════════════════════════════
    if hasattr(trained_xgb, "feature_importances_"):
        import pandas as pd
        fi = pd.DataFrame({
            "feature": X_train_eng.columns,
            "importance": trained_xgb.feature_importances_,
        }).sort_values("importance", ascending=False)
        plot_feature_importance(fi, "XGBoost Primary")

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - pipeline_start
    print("\n" + "█" * 60)
    print("█  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"\n  Model Performance Summary:")
    print(f"    XGBoost:          F1={xgb_report['optimal_metrics']['f1_score']:.4f}  AUC={xgb_report['optimal_metrics']['roc_auc']:.4f}")
    print(f"    LightGBM:         F1={lgb_report['optimal_metrics']['f1_score']:.4f}  AUC={lgb_report['optimal_metrics']['roc_auc']:.4f}")
    print(f"    Stacking Ensemble: F1={ensemble_report['optimal_metrics']['f1_score']:.4f}  AUC={ensemble_report['optimal_metrics']['roc_auc']:.4f}")

    if risk_report:
        print(f"\n  Risk Alerts Generated: {len(risk_report['alerts'])}")
        print(f"  Tier Distribution: {risk_report['tier_distribution']}")

    print(f"\n  Outputs saved to: outputs/")
    print(f"  Models saved to:  saved_models/")
    print("█" * 60 + "\n")

    return {
        "xgb": (trained_xgb, xgb_report),
        "lgb": (trained_lgb, lgb_report),
        "ensemble": (trained_ensemble, ensemble_report),
        "risk_report": risk_report,
    }


def run_quick_mode():
    """Quick run using only the 18 bank-specified key features."""
    print("\n[Quick Mode] Using only bank-specified key features (18 features)")
    return run_full_pipeline(use_key_features_only=True, run_anomaly=False, run_ablation_flag=False)


def run_train_only():
    """Train primary models without ablation study."""
    return run_full_pipeline(run_anomaly=True, run_ablation_flag=False)


def run_ablation_only():
    """Run ablation study only."""
    from preprocess import prepare_data
    from feature_engineering import run_feature_engineering
    from train import run_ablation_study
    from evaluate import plot_model_comparison

    X_train, X_test, y_train, y_test, features, enc = prepare_data()
    X_train_eng, X_test_eng, selected, _ = run_feature_engineering(X_train, X_test, y_train)

    results = run_ablation_study(X_train_eng, X_test_eng, y_train, y_test, runs=3)
    plot_model_comparison(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SMURF — Suspicious Mule Account Classification Pipeline"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "train", "ablation", "quick"],
        help="Pipeline execution mode (default: full)",
    )
    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "train":
        run_train_only()
    elif args.mode == "ablation":
        run_ablation_only()
    elif args.mode == "quick":
        run_quick_mode()
