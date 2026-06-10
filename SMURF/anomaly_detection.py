"""
Anomaly Detection Module
=========================
Unsupervised anomaly detection for mule account identification.
Provides complementary signals to the supervised classifiers.

These anomaly scores can be used as:
1. Standalone detectors for novel/unseen fraud patterns
2. Additional features for the supervised ensemble
3. Second-opinion validation of classification results
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import time

from config import ISOLATION_FOREST_PARAMS, KEY_FEATURES, RANDOM_STATE


def train_isolation_forest(X_train, feature_subset=None):
    """
    Train Isolation Forest for unsupervised anomaly detection.
    
    Isolation Forest works by randomly selecting features and split values,
    then measuring how many splits are needed to isolate each point.
    Mule accounts with unusual transaction patterns will be isolated faster.
    """
    print("\n[AnomalyDet] Training Isolation Forest...")
    start = time.time()
    
    if feature_subset is not None:
        available = [f for f in feature_subset if f in X_train.columns]
        X_subset = X_train[available]
    else:
        X_subset = X_train
    
    # Standardize features for distance-based methods
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    iso_forest = IsolationForest(**ISOLATION_FOREST_PARAMS)
    iso_forest.fit(X_scaled)
    
    elapsed = time.time() - start
    print(f"[AnomalyDet] Isolation Forest trained in {elapsed:.1f}s")
    
    return iso_forest, scaler, (feature_subset or list(X_train.columns))


def predict_anomaly_scores(model, scaler, X, feature_cols):
    """
    Generate anomaly scores from trained Isolation Forest.
    
    Returns scores in [0, 1] range where:
    - Higher score → more anomalous (more likely mule account)
    - Lower score → more normal (legitimate account)
    """
    available = [f for f in feature_cols if f in X.columns]
    X_subset = X[available]
    X_scaled = scaler.transform(X_subset)
    
    # decision_function returns negative scores for anomalies
    raw_scores = model.decision_function(X_scaled)
    
    # Normalize to [0, 1] — invert so anomalies have HIGH scores
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    
    if max_score - min_score > 0:
        normalized = 1 - (raw_scores - min_score) / (max_score - min_score)
    else:
        normalized = np.zeros_like(raw_scores)
    
    return normalized


def train_lof(X_train, feature_subset=None, n_neighbors=20):
    """
    Train Local Outlier Factor for density-based anomaly detection.
    LOF captures accounts that exist in sparse regions of the feature space,
    which may indicate unusual mule account behavior.
    
    Note: LOF is transductive — we fit and predict in one step for training data,
    and use novelty detection mode for new data.
    """
    print("[AnomalyDet] Training Local Outlier Factor...")
    start = time.time()
    
    if feature_subset is not None:
        available = [f for f in feature_subset if f in X_train.columns]
        X_subset = X_train[available]
    else:
        X_subset = X_train
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=0.05,
        novelty=True,  # Enable predict() for new data
        n_jobs=-1,
    )
    lof.fit(X_scaled)
    
    elapsed = time.time() - start
    print(f"[AnomalyDet] LOF trained in {elapsed:.1f}s")
    
    return lof, scaler


def run_anomaly_detection(X_train, X_test, y_test=None):
    """
    Full anomaly detection pipeline.
    
    1. Train Isolation Forest on key features
    2. Train LOF on key features
    3. Combine anomaly scores
    4. Optionally evaluate against known labels
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    
    # Use bank-specified key features for anomaly detection
    key_available = [f for f in KEY_FEATURES if f in X_train.columns]
    
    if len(key_available) < 3:
        print("[AnomalyDet] Too few key features available, using all features")
        key_available = None
    
    # 1. Isolation Forest
    iso_model, iso_scaler, iso_features = train_isolation_forest(X_train, key_available)
    iso_scores_train = predict_anomaly_scores(iso_model, iso_scaler, X_train, iso_features)
    iso_scores_test = predict_anomaly_scores(iso_model, iso_scaler, X_test, iso_features)
    
    # 2. LOF
    lof_model, lof_scaler = train_lof(X_train, key_available)
    
    # LOF scores for test data
    lof_features = key_available if key_available else list(X_train.columns)
    available_lof = [f for f in lof_features if f in X_test.columns]
    X_test_lof = lof_scaler.transform(X_test[available_lof])
    lof_raw = lof_model.decision_function(X_test_lof)
    
    # Normalize LOF scores
    lof_min, lof_max = lof_raw.min(), lof_raw.max()
    if lof_max - lof_min > 0:
        lof_scores_test = 1 - (lof_raw - lof_min) / (lof_max - lof_min)
    else:
        lof_scores_test = np.zeros_like(lof_raw)
    
    # 3. Combined anomaly score (weighted average)
    combined_scores = 0.6 * iso_scores_test + 0.4 * lof_scores_test
    
    # 4. Evaluation against known labels (if available)
    if y_test is not None:
        from sklearn.metrics import roc_auc_score
        
        iso_auc = roc_auc_score(y_test, iso_scores_test)
        lof_auc = roc_auc_score(y_test, lof_scores_test)
        combined_auc = roc_auc_score(y_test, combined_scores)
        
        print(f"\n[AnomalyDet] Unsupervised Detection AUC:")
        print(f"  Isolation Forest: {iso_auc:.4f}")
        print(f"  LOF:              {lof_auc:.4f}")
        print(f"  Combined:         {combined_auc:.4f}")
    
    print("=" * 60 + "\n")
    
    return {
        "iso_model": iso_model,
        "iso_scaler": iso_scaler,
        "iso_features": iso_features,
        "lof_model": lof_model,
        "lof_scaler": lof_scaler,
        "iso_scores_test": iso_scores_test,
        "lof_scores_test": lof_scores_test,
        "combined_scores": combined_scores,
    }
