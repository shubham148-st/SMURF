"""
Training Pipeline
==================
Orchestrates model training with cross-validation, early stopping,
and comprehensive logging.

Reuses patterns from AML_smurf/train.py:
- train_and_evaluate() structure
- FocalLoss integration
- Evaluation metric logging format

Reuses patterns from AML_smurf/alabation.py:
- Multi-model ablation study with repeated runs
- Tabular result formatting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import time
import os

from config import (
    N_CV_FOLDS, RANDOM_STATE, MODEL_DIR, CLASSIFICATION_THRESHOLD,
    ABLATION_RUNS,
)
from model import get_all_models, build_xgboost, build_lightgbm, build_stacking_ensemble
from evaluate import (
    compute_metrics, find_optimal_threshold, print_evaluation_report,
    generate_full_report,
)


def train_single_model(model, X_train, y_train, X_test, y_test,
                        model_name="Model", early_stopping_rounds=None):
    """
    Train a single model and evaluate.
    
    Structure reused from AML_smurf/train.py train_and_evaluate():
        model.train()
        for epoch in range(epochs):
            ...
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(test_out) > 0.4).cpu().numpy()
            f1 = f1_score(test_y, preds)
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 60}")
    
    start = time.time()
    
    # Handle XGBoost early stopping
    if early_stopping_rounds is not None and hasattr(model, "fit"):
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        except TypeError:
            # Fallback if model doesn't support eval_set
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    print(f"[Train] Training completed in {elapsed:.1f}s")
    
    # Get probability predictions
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test).astype(float)
    
    # Evaluate (reusing SMURF metric computation pattern)
    report = generate_full_report(y_test, y_pred_proba, model_name)
    
    return model, y_pred_proba, report


def train_with_cross_validation(model_factory, X, y, model_name="Model", n_folds=N_CV_FOLDS):
    """
    Train with stratified k-fold cross-validation.
    Returns out-of-fold predictions for stacking.
    """
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION: {model_name} ({n_folds}-Fold)")
    print(f"{'=' * 60}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    oof_predictions = np.zeros(len(y))
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  --- Fold {fold_idx + 1}/{n_folds} ---")
        
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]
        
        model = model_factory()
        
        try:
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
        except TypeError:
            model.fit(X_fold_train, y_fold_train)
        
        if hasattr(model, "predict_proba"):
            fold_preds = model.predict_proba(X_fold_val)[:, 1]
        else:
            fold_preds = model.predict(X_fold_val).astype(float)
        
        oof_predictions[val_idx] = fold_preds
        
        metrics = compute_metrics(y_fold_val, fold_preds, threshold=CLASSIFICATION_THRESHOLD)
        fold_metrics.append(metrics)
        
        # Print fold results (SMURF-style output)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Aggregate fold results
    avg_metrics = {
        key: np.mean([m[key] for m in fold_metrics])
        for key in ["precision", "recall", "f1_score", "roc_auc"]
    }
    std_metrics = {
        key: np.std([m[key] for m in fold_metrics])
        for key in ["precision", "recall", "f1_score", "roc_auc"]
    }
    
    print(f"\n{'─' * 50}")
    print(f"  CV RESULTS ({n_folds}-Fold Average):")
    for key in ["precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {key:>12s}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    print(f"{'─' * 50}")
    
    return oof_predictions, avg_metrics, std_metrics


def run_ablation_study(X_train, X_test, y_train, y_test, runs=ABLATION_RUNS):
    """
    Run ablation study comparing all models.
    
    Directly reuses the ablation methodology from AML_smurf/alabation.py:
        for name, model_fn in models_to_test:
            f1s, precs, recs = [], [], []
            for _ in range(runs):
                model = model_fn()
                p, r, f = alabation(model, balanced_data.clone(), epochs=100)
                precs.append(p); recs.append(r); f1s.append(f)
            
            p_mean, p_std = np.mean(precs), np.std(precs)
            ...
    """
    print("\n" + "=" * 70)
    print(f"ABLATION STUDY: MULE ACCOUNT CLASSIFICATION ({runs} Runs)")
    print("=" * 70)
    
    all_models = get_all_models(y_train)
    results = {}
    
    print("\n| Model | Precision (Mean±SD) | Recall (Mean±SD) | F1-Score (Mean±SD) | AUC (Mean±SD) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for name, (model_template, early_stop) in all_models.items():
        f1s, precs, recs, aucs = [], [], [], []
        
        for run in range(runs):
            # Re-initialize model for fair trial (same as SMURF ablation)
            models = get_all_models(y_train)
            model, es = models[name]
            
            try:
                if es is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False,
                    )
                else:
                    model.fit(X_train, y_train)
            except Exception as e:
                print(f"  [WARN] {name} run {run+1} failed: {e}")
                continue
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.predict(X_test).astype(float)
            
            metrics = compute_metrics(y_test, y_proba)
            precs.append(metrics["precision"])
            recs.append(metrics["recall"])
            f1s.append(metrics["f1_score"])
            aucs.append(metrics["roc_auc"])
        
        if len(f1s) > 0:
            p_m, p_s = np.mean(precs), np.std(precs)
            r_m, r_s = np.mean(recs), np.std(recs)
            f_m, f_s = np.mean(f1s), np.std(f1s)
            a_m, a_s = np.mean(aucs), np.std(aucs)
            
            print(f"| {name} | {p_m:.3f}±{p_s:.3f} | {r_m:.3f}±{r_s:.3f} | {f_m:.3f}±{f_s:.3f} | {a_m:.3f}±{a_s:.3f} |")
            
            results[name] = {
                "precision": p_m, "recall": r_m, "f1_score": f_m, "roc_auc": a_m,
                "precision_std": p_s, "recall_std": r_s, "f1_std": f_s, "auc_std": a_s,
            }
    
    print("=" * 70)
    return results


def save_model(model, model_name, save_dir=MODEL_DIR):
    """Save trained model to disk."""
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(save_dir, f"{safe_name}.joblib")
    joblib.dump(model, path)
    print(f"[Train] Model saved: {path}")
    return path


def load_model(model_name, save_dir=MODEL_DIR):
    """Load trained model from disk."""
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(save_dir, f"{safe_name}.joblib")
    model = joblib.load(path)
    print(f"[Train] Model loaded: {path}")
    return model
