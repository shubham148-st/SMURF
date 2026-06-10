"""
Feature Engineering Module
===========================
Performs feature selection, importance analysis, and engineers new
features from the 3,924-column hackathon dataset.

Leverages domain knowledge from the SMURF AML project:
- Transaction velocity patterns (structuring detection)
- Amount distribution anomalies
- Behavioral deviation metrics
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
import time

from config import KEY_FEATURES, TARGET_COL, RANDOM_STATE

warnings.filterwarnings("ignore")


def compute_feature_importance_rf(X_train, y_train, top_n=50):
    """
    Compute feature importance using a quick Random Forest.
    Returns sorted importance DataFrame.
    """
    print(f"[FeatureEng] Computing RF feature importance (top {top_n})...")
    start = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    elapsed = time.time() - start
    print(f"[FeatureEng] RF importance computed in {elapsed:.1f}s")
    print(f"[FeatureEng] Top 10 features:")
    for _, row in importances.head(10).iterrows():
        marker = " ★" if row["feature"] in KEY_FEATURES else ""
        print(f"  {row['feature']:>10s}: {row['importance']:.4f}{marker}")
    
    return importances.head(top_n)


def compute_mutual_information(X_train, y_train, top_n=50):
    """
    Compute mutual information scores between features and target.
    Captures non-linear dependencies that RF importance might miss.
    """
    print(f"[FeatureEng] Computing mutual information scores...")
    start = time.time()
    
    mi_scores = mutual_info_classif(
        X_train, y_train,
        discrete_features="auto",
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )
    
    mi_df = pd.DataFrame({
        "feature": X_train.columns,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False)
    
    elapsed = time.time() - start
    print(f"[FeatureEng] MI scores computed in {elapsed:.1f}s")
    
    return mi_df.head(top_n)


def select_features_combined(X_train, y_train, top_n=100):
    """
    Combine multiple feature selection methods:
    1. Random Forest importance
    2. Mutual Information
    3. Bank-specified key features (always included)
    
    Returns the union of top features from all methods.
    """
    print("\n" + "=" * 60)
    print("FEATURE SELECTION (Multi-Method Ensemble)")
    print("=" * 60)
    
    # Method 1: RF Importance
    rf_top = compute_feature_importance_rf(X_train, y_train, top_n=top_n)
    rf_features = set(rf_top["feature"].tolist())
    
    # Method 2: Mutual Information
    mi_top = compute_mutual_information(X_train, y_train, top_n=top_n)
    mi_features = set(mi_top["feature"].tolist())
    
    # Method 3: Bank-specified key features (always included)
    key_available = set(f for f in KEY_FEATURES if f in X_train.columns)
    
    # Union of all methods
    selected = rf_features | mi_features | key_available
    
    # Also include missingness indicators if present
    missing_indicators = [c for c in X_train.columns if c.endswith("_is_missing")]
    selected.update(missing_indicators)
    
    selected = sorted(selected)
    
    print(f"\n[FeatureEng] Feature selection summary:")
    print(f"  RF top-{top_n}:          {len(rf_features)} features")
    print(f"  MI top-{top_n}:          {len(mi_features)} features")
    print(f"  Bank key features:   {len(key_available)} features")
    print(f"  Missingness flags:   {len(missing_indicators)} features")
    print(f"  Combined (union):    {len(selected)} features")
    
    # Report overlap
    overlap = rf_features & mi_features
    print(f"  RF ∩ MI overlap:     {len(overlap)} features")
    
    return selected, rf_top, mi_top


def engineer_interaction_features(df, feature_cols=None):
    """
    Engineer interaction features from the bank-specified key features.
    These capture cross-feature patterns common in mule account behavior:
    
    Domain insights from SMURF AML research:
    - Ratio features capture structuring (splitting large amounts into smaller ones)
    - Difference features capture velocity changes
    - Product features capture combined risk signals
    """
    print("[FeatureEng] Engineering interaction features...")
    
    key_available = [f for f in KEY_FEATURES if f in df.columns]
    new_features = []
    
    if len(key_available) < 2:
        print("[FeatureEng] Not enough key features for interactions, skipping.")
        return df, new_features
    
    # Pairwise ratios between adjacent key features (structuring indicators)
    for i in range(len(key_available) - 1):
        f1, f2 = key_available[i], key_available[i + 1]
        ratio_col = f"{f1}_div_{f2}"
        denominator = df[f2].replace(0, np.nan)
        df[ratio_col] = (df[f1] / denominator).fillna(0)
        # Clip extreme ratios to prevent numerical instability
        df[ratio_col] = df[ratio_col].clip(-100, 100)
        new_features.append(ratio_col)
    
    # Statistical aggregations across key features (behavioral profiles)
    key_values = df[key_available]
    df["key_features_mean"] = key_values.mean(axis=1)
    df["key_features_std"] = key_values.std(axis=1).fillna(0)
    df["key_features_min"] = key_values.min(axis=1)
    df["key_features_max"] = key_values.max(axis=1)
    df["key_features_range"] = df["key_features_max"] - df["key_features_min"]
    df["key_features_skew"] = key_values.skew(axis=1).fillna(0)
    
    # Non-zero count across key features (activity level indicator)
    df["key_features_nonzero"] = (key_values != 0).sum(axis=1)
    
    # Negative value count (anomaly indicator — legitimate accounts rarely go negative)
    df["key_features_negatives"] = (key_values < 0).sum(axis=1)
    
    stats_features = [
        "key_features_mean", "key_features_std", "key_features_min",
        "key_features_max", "key_features_range", "key_features_skew",
        "key_features_nonzero", "key_features_negatives",
    ]
    new_features.extend(stats_features)
    
    print(f"[FeatureEng] Created {len(new_features)} interaction/statistical features")
    return df, new_features


def run_feature_engineering(X_train, X_test, y_train, auto_select=True, top_n=100):
    """
    Full feature engineering pipeline.
    
    1. Engineer interaction features on both train and test
    2. Optionally auto-select best features using multi-method ensemble
    3. Return reduced feature sets
    
    Returns
    -------
    X_train_eng, X_test_eng : DataFrames with engineered features
    selected_features : list of selected feature names
    importance_report : dict with RF and MI importance DataFrames
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # 1. Engineer new features
    X_train, new_train = engineer_interaction_features(X_train.copy())
    X_test, new_test = engineer_interaction_features(X_test.copy())
    
    if auto_select:
        # 2. Feature selection on the augmented feature set
        selected, rf_imp, mi_imp = select_features_combined(X_train, y_train, top_n=top_n)
        
        # Ensure new engineered features are included
        selected = list(set(selected) | set(new_train))
        selected = [f for f in selected if f in X_train.columns]
        
        X_train_eng = X_train[selected]
        X_test_eng = X_test[selected]
        
        importance_report = {"rf": rf_imp, "mi": mi_imp}
    else:
        selected = list(X_train.columns)
        X_train_eng = X_train
        X_test_eng = X_test
        importance_report = {}
    
    print(f"\n[FeatureEng] Final feature count: {len(selected)}")
    print("=" * 60 + "\n")
    
    return X_train_eng, X_test_eng, selected, importance_report


if __name__ == "__main__":
    from preprocess import prepare_data
    
    X_train, X_test, y_train, y_test, features, enc = prepare_data()
    X_train_eng, X_test_eng, selected, report = run_feature_engineering(
        X_train, X_test, y_train, auto_select=True, top_n=80
    )
    print(f"Selected {len(selected)} features for modeling.")
