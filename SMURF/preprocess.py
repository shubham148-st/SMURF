"""
Data Preprocessing Pipeline
============================
Loads the hackathon DataSet.csv, handles missing values (NA),
encodes categorical features, and prepares train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import time

from config import (
    DATA_PATH, TARGET_COL, KEY_FEATURES,
    RANDOM_STATE, TEST_SIZE, NORMAL_TO_FRAUD_RATIO,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def load_raw_data(path=None):
    """
    Load the raw hackathon CSV.
    The dataset has ~3,924 anonymized features (F1–F3924) with
    mixed numeric/categorical types and many NA values.
    """
    path = path or DATA_PATH
    print(f"[Preprocess] Loading dataset from: {path}")
    start = time.time()

    df = pd.read_csv(path, na_values=["NA", "na", "N/A", ""])
    
    elapsed = time.time() - start
    print(f"[Preprocess] Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns in {elapsed:.1f}s")
    return df


def identify_column_types(df, target_col=TARGET_COL):
    """
    Separate columns into numeric and categorical based on dtype.
    Some columns may look numeric but contain string values (e.g., "Savings").
    """
    feature_cols = [c for c in df.columns if c != target_col]
    
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            numeric_cols.append(col)
        else:
            # Try to coerce — if mostly numeric with some strings, 
            # it's a mixed column we handle specially
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_null_original = df[col].notna().sum()
            non_null_coerced = coerced.notna().sum()
            
            if non_null_original > 0 and (non_null_coerced / max(non_null_original, 1)) > 0.5:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
    
    print(f"[Preprocess] Column types — Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
    return numeric_cols, categorical_cols


def handle_missing_values(df, numeric_cols, categorical_cols):
    """
    Impute missing values:
    - Numeric columns: median imputation (robust to outliers in financial data)
    - Categorical columns: mode imputation + 'UNKNOWN' fallback
    
    Also creates binary missingness indicators for key features,
    since missingness itself can be a fraud signal.
    """
    print("[Preprocess] Handling missing values...")
    
    # Create missingness indicators for key features (feature engineering insight)
    for col in KEY_FEATURES:
        if col in df.columns:
            missing_col = f"{col}_is_missing"
            df[missing_col] = df[col].isna().astype(int)
    
    # Numeric imputation — median is robust for skewed financial distributions
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # Categorical imputation
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
            df[col] = df[col].fillna(fill_val)
    
    return df


def encode_categoricals(df, categorical_cols):
    """
    Label-encode categorical columns.
    Returns the dataframe and a dict of fitted encoders for inverse transform.
    """
    print(f"[Preprocess] Encoding {len(categorical_cols)} categorical columns...")
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    
    return df, encoders


def undersample_majority(df, target_col=TARGET_COL, ratio=NORMAL_TO_FRAUD_RATIO):
    """
    Undersample the majority (legitimate) class.
    Reuses the ratio-based subsampling concept from AML_smurf/train.py subsample_graph().
    
    In the original SMURF:
        num_normal_to_keep = min(num_fraud * normal_to_fraud_ratio, len(normal_indices))
    
    Here we apply the same logic at the row (account) level instead of edge level.
    """
    fraud = df[df[target_col] == 1]
    normal = df[df[target_col] == 0]
    
    num_fraud = len(fraud)
    num_normal_to_keep = min(num_fraud * ratio, len(normal))
    
    normal_sampled = normal.sample(n=num_normal_to_keep, random_state=RANDOM_STATE)
    
    balanced = pd.concat([fraud, normal_sampled]).sample(frac=1.0, random_state=RANDOM_STATE)
    
    print(f"[Preprocess] Undersampled: {len(fraud)} fraud + {num_normal_to_keep} normal = {len(balanced)} total")
    return balanced.reset_index(drop=True)


def prepare_data(use_undersampling=False, use_key_features_only=False):
    """
    Full preprocessing pipeline: load → type detection → impute → encode → split.
    
    Parameters
    ----------
    use_undersampling : bool
        If True, apply majority-class undersampling before splitting.
    use_key_features_only : bool
        If True, restrict features to the 18 bank-specified key features only.
    
    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames/Series
    feature_names : list of feature column names
    encoders : dict of fitted LabelEncoders
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Load
    df = load_raw_data()
    
    # 2. Target extraction & validation
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset!")
    
    print(f"[Preprocess] Target distribution:")
    target_counts = df[TARGET_COL].value_counts()
    for val, count in target_counts.items():
        pct = count / len(df) * 100
        print(f"  Class {val}: {count:,} ({pct:.2f}%)")
    
    # 3. Column type identification
    numeric_cols, categorical_cols = identify_column_types(df)
    
    # 4. Missing value imputation
    df = handle_missing_values(df, numeric_cols, categorical_cols)
    
    # 5. Categorical encoding
    df, encoders = encode_categoricals(df, categorical_cols)
    
    # 6. Optional undersampling (inspired by SMURF subsample_graph)
    if use_undersampling:
        df = undersample_majority(df)
    
    # 7. Feature selection
    if use_key_features_only:
        # Use only the 18 bank-specified features + any missingness indicators
        available_key = [f for f in KEY_FEATURES if f in df.columns]
        missing_indicators = [c for c in df.columns if c.endswith("_is_missing")]
        feature_cols = available_key + missing_indicators
        print(f"[Preprocess] Using {len(feature_cols)} key features (+ missingness indicators)")
    else:
        feature_cols = [c for c in df.columns if c != TARGET_COL]
    
    # 8. Final type coercion — ensure everything is numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    X = df[feature_cols]
    y = df[TARGET_COL]
    
    # 9. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"[Preprocess] Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")
    print(f"[Preprocess] Features: {X_train.shape[1]:,}")
    print("=" * 60 + "\n")
    
    return X_train, X_test, y_train, y_test, feature_cols, encoders


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, enc = prepare_data()
    print(f"Ready — {len(features)} features, {len(X_train)} train, {len(X_test)} test")
