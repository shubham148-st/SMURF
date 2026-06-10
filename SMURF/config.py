"""
Central configuration for the Mule Account Classification pipeline.
All paths, feature lists, hyperparameters, and constants live here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "AML_smurf", "dataset", "DataSet.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Target Variable ─────────────────────────────────────────────────────────
TARGET_COL = "F3924"

# ─── Bank-Specified Key Features (from problem statement) ────────────────────
# These are the features commonly used by banks for fraud detection.
KEY_FEATURES = [
    "F115", "F321", "F527", "F531", "F670",
    "F1692", "F2082", "F2122", "F2582", "F2678",
    "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]

# ─── Training Configuration ──────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# ─── Class Imbalance Handling ─────────────────────────────────────────────────
NORMAL_TO_FRAUD_RATIO = 10  # For undersampling the majority class
SMOTE_SAMPLING_STRATEGY = 0.5  # Target minority ratio after SMOTE

# ─── Focal Loss Parameters (reused from AML_smurf/model.py FocalLoss) ────────
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# ─── XGBoost Hyperparameters ─────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
}

# ─── LightGBM Hyperparameters ────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# ─── Anomaly Detection ───────────────────────────────────────────────────────
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.05,  # Expected fraud proportion
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ─── Evaluation Thresholds ───────────────────────────────────────────────────
# Reused from AML_smurf/train.py (sigmoid threshold 0.4)
CLASSIFICATION_THRESHOLD = 0.4

# ─── Risk Score Tiers ────────────────────────────────────────────────────────
RISK_TIERS = {
    "CRITICAL": 0.85,
    "HIGH":     0.65,
    "MEDIUM":   0.40,
    "LOW":      0.0,
}

# ─── Ablation Study Runs (reused from AML_smurf/alabation.py methodology) ───
ABLATION_RUNS = 3
