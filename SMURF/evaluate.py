"""
Evaluation Module
==================
Comprehensive evaluation metrics, visualizations, and reporting.

Reuses metric patterns from AML_smurf/train.py:
- F1, Precision, Recall computation
- Threshold optimization (SMURF used 0.4 instead of default 0.5)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import (
    CLASSIFICATION_THRESHOLD, RISK_TIERS, OUTPUT_DIR,
)


def compute_metrics(y_true, y_pred_proba, threshold=CLASSIFICATION_THRESHOLD):
    """
    Compute comprehensive classification metrics.
    
    Reused from AML_smurf/train.py evaluation block:
        preds = (torch.sigmoid(test_out) > 0.4).cpu().numpy()
        f1 = f1_score(test_y, preds)
        prec = precision_score(test_y, preds, zero_division=0)
        rec = recall_score(test_y, preds, zero_division=0)
    
    Extended with AUC-ROC, PR-AUC, and confusion matrix.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    
    return metrics


def find_optimal_threshold(y_true, y_pred_proba, metric="f1"):
    """
    Find the threshold that maximizes the chosen metric.
    SMURF used a fixed 0.4 threshold — here we search systematically.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score


def print_evaluation_report(model_name, metrics):
    """
    Print a formatted evaluation report.
    Follows the format from AML_smurf/train.py:
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
    """
    print(f"\n{'─' * 50}")
    print(f"  Model: {model_name}")
    print(f"  Threshold: {metrics['threshold']:.2f}")
    print(f"{'─' * 50}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"{'─' * 50}")
    
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:>6d}  FP={cm[0][1]:>6d}")
    print(f"    FN={cm[1][0]:>6d}  TP={cm[1][1]:>6d}")
    print(f"{'─' * 50}")


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    save_path = save_path or os.path.join(OUTPUT_DIR, f"roc_{model_name.replace(' ', '_').lower()}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] ROC curve saved: {save_path}")
    return save_path


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """Plot Precision-Recall curve with AP score."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"{model_name} (AP = {ap:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    save_path = save_path or os.path.join(OUTPUT_DIR, f"pr_{model_name.replace(' ', '_').lower()}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] PR curve saved: {save_path}")
    return save_path


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Suspicious"],
        yticklabels=["Legitimate", "Suspicious"],
        ax=ax, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    
    save_path = save_path or os.path.join(OUTPUT_DIR, f"cm_{model_name.replace(' ', '_').lower()}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Confusion matrix saved: {save_path}")
    return save_path


def plot_feature_importance(importance_df, model_name="Model", top_n=20, save_path=None):
    """Plot horizontal bar chart of feature importances."""
    top = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#FF5722" if f in ["F115", "F321", "F527", "F531", "F670",
              "F1692", "F2082", "F2122", "F2582", "F2678",
              "F2737", "F2956", "F3043", "F3836", "F3887",
              "F3889", "F3891", "F3894"] else "#2196F3"
              for f in top["feature"]]
    
    ax.barh(range(len(top)), top["importance"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}\n(Orange = Bank Key Features)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    
    save_path = save_path or os.path.join(OUTPUT_DIR, f"importance_{model_name.replace(' ', '_').lower()}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Feature importance plot saved: {save_path}")
    return save_path


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison of multiple models across metrics.
    Inspired by SMURF ablation study output format:
        | Model | Precision | Recall | F1-Score |
    """
    models = list(results_dict.keys())
    metrics_names = ["precision", "recall", "f1_score", "roc_auc"]
    
    data = {m: [results_dict[model].get(m, 0) for model in models] for m in metrics_names}
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        ax.bar(x + i * width, data[metric], width, label=metric.replace("_", " ").title(), color=color)
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Ablation Study", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    
    save_path = save_path or os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Model comparison saved: {save_path}")
    return save_path


def generate_full_report(y_true, y_pred_proba, model_name="Model"):
    """
    Run all evaluation steps and return comprehensive results.
    """
    # Find optimal threshold
    opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_pred_proba, metric="f1")
    
    # Compute metrics at both default and optimal thresholds
    metrics_default = compute_metrics(y_true, y_pred_proba, threshold=CLASSIFICATION_THRESHOLD)
    metrics_optimal = compute_metrics(y_true, y_pred_proba, threshold=opt_threshold)
    
    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'=' * 60}")
    
    print(f"\n  [Default Threshold = {CLASSIFICATION_THRESHOLD}]")
    print_evaluation_report(model_name, metrics_default)
    
    print(f"\n  [Optimal Threshold = {opt_threshold:.2f} → F1 = {opt_f1:.4f}]")
    print_evaluation_report(f"{model_name} (Optimized)", metrics_optimal)
    
    # Generate plots
    y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)
    roc_path = plot_roc_curve(y_true, y_pred_proba, model_name)
    pr_path = plot_precision_recall_curve(y_true, y_pred_proba, model_name)
    cm_path = plot_confusion_matrix(y_true, y_pred_opt, model_name)
    
    return {
        "default_metrics": metrics_default,
        "optimal_metrics": metrics_optimal,
        "optimal_threshold": opt_threshold,
        "plots": {"roc": roc_path, "pr": pr_path, "cm": cm_path},
    }
