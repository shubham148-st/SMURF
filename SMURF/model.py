"""
Model Definitions for Mule Account Classification
===================================================
Contains tabular ML models and custom loss functions.


- FocalLoss: Adapted from PyTorch to sklearn/xgboost-compatible gradient/hessian
- Class imbalance strategy: scale_pos_weight computed from data distribution
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, RANDOM_STATE,
    FOCAL_ALPHA, FOCAL_GAMMA,
)


# ─── Focal Loss for XGBoost ──────────────────────────────────────────────────
# Directly adapted from AML_smurf/model.py FocalLoss class.
# Original PyTorch version:
#   bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#   pt = torch.exp(-bce_loss)
#   focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#
# Below is the gradient/hessian formulation required by XGBoost custom objectives.

def focal_loss_objective(y_true, y_pred, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """
    Focal Loss custom objective for XGBoost.
    Modulates cross-entropy to focus learning on hard-to-classify mule accounts,
    addressing the extreme class imbalance in AML datasets.
    
    Reused from: AML_smurf/model.py → FocalLoss class (adapted from PyTorch to numpy)
    
    Parameters
    ----------
    y_true : array-like, true labels (0 or 1)
    y_pred : array-like, raw predictions (logits)
    alpha : float, weighting factor for positive class (default from SMURF: 0.75)
    gamma : float, focusing parameter (default from SMURF: 2.0)
    
    Returns
    -------
    gradient, hessian : arrays for XGBoost optimization
    """
    # Sigmoid activation (equivalent to torch.sigmoid in SMURF)
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    
    # Focal modulation terms
    y = y_true
    pt = np.where(y == 1, p, 1 - p)
    alpha_t = np.where(y == 1, alpha, 1 - alpha)
    
    # Gradient: d(FocalLoss)/d(y_pred)
    focal_weight = alpha_t * (1 - pt) ** gamma
    grad = focal_weight * (
        gamma * (1 - pt) ** (gamma - 1) * pt * np.log(np.maximum(pt, 1e-7)) * (2 * y - 1)
        + (1 - pt) ** gamma * (p - y)
    )
    
    # Hessian: d²(FocalLoss)/d(y_pred)²
    hess = focal_weight * (
        (1 - pt) ** gamma * p * (1 - p)
        + gamma * (1 - pt) ** (gamma - 1) * pt * (1 - pt) * np.abs(2 * y - 1)
    )
    hess = np.maximum(hess, 1e-7)  # Ensure positive-definiteness
    
    return grad, hess


def focal_loss_metric(y_true, y_pred, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """Focal loss evaluation metric for XGBoost early stopping."""
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    
    pt = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    loss = -alpha_t * (1 - pt) ** gamma * np.log(pt)
    return "focal_loss", float(np.mean(loss))


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_xgboost(y_train=None, use_focal_loss=True):
    """
    Build XGBoost classifier with optional focal loss.
    Computes scale_pos_weight from training data distribution
    (similar to the class ratio concept in SMURF subsample_graph).
    """
    params = XGBOOST_PARAMS.copy()
    
    if y_train is not None and not use_focal_loss:
        # Auto-compute class weight — same concept as SMURF's normal_to_fraud_ratio
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        params["scale_pos_weight"] = neg_count / max(pos_count, 1)
    
    early_stopping = params.pop("early_stopping_rounds", None)
    
    model = xgb.XGBClassifier(**params)
    
    if use_focal_loss:
        model.set_params(objective=focal_loss_objective)
    
    return model, early_stopping


def build_lightgbm(y_train=None):
    """Build LightGBM classifier with automatic class balancing."""
    params = LIGHTGBM_PARAMS.copy()
    params["is_unbalance"] = True
    
    return lgb.LGBMClassifier(**params)


def build_random_forest(y_train=None):
    """Build Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gradient_boosting():
    """Build sklearn GradientBoosting (slower but different bias)."""
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )


def build_mlp():
    """Build a Multi-Layer Perceptron for neural-net diversity in ensemble."""
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=300,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    )


def build_stacking_ensemble(y_train=None):
    """
    Build a stacking ensemble combining multiple model types.
    
    Architecture inspired by SMURF ablation methodology:
    Test diverse model families, then stack their predictions.
    
    Base learners: XGBoost, LightGBM, Random Forest
    Meta-learner: Logistic Regression (calibrated probabilities)
    """
    xgb_model, _ = build_xgboost(y_train, use_focal_loss=False)
    xgb_model.set_params(n_estimators=200, early_stopping_rounds=None)
    
    base_estimators = [
        ("xgboost", xgb_model),
        ("lightgbm", build_lightgbm(y_train)),
        ("random_forest", build_random_forest(y_train)),
    ]
    
    meta_learner = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def build_voting_ensemble(y_train=None):
    """Soft-voting ensemble of diverse classifiers."""
    xgb_model, _ = build_xgboost(y_train, use_focal_loss=False)
    xgb_model.set_params(n_estimators=200, early_stopping_rounds=None)
    
    estimators = [
        ("xgboost", xgb_model),
        ("lightgbm", build_lightgbm(y_train)),
        ("random_forest", build_random_forest(y_train)),
    ]
    
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )


# ─── Model Registry ──────────────────────────────────────────────────────────

def get_all_models(y_train=None):
    """
    Returns a dict of all available models.
    Used by the ablation study (reused from AML_smurf/alabation.py methodology).
    """
    xgb_focal, xgb_es = build_xgboost(y_train, use_focal_loss=True)
    xgb_standard, xgb_es2 = build_xgboost(y_train, use_focal_loss=False)
    
    return {
        "XGBoost (Focal Loss)": (xgb_focal, xgb_es),
        "XGBoost (Standard)": (xgb_standard, xgb_es2),
        "LightGBM": (build_lightgbm(y_train), None),
        "Random Forest": (build_random_forest(y_train), None),
        "Gradient Boosting": (build_gradient_boosting(), None),
        "Stacking Ensemble": (build_stacking_ensemble(y_train), None),
    }
