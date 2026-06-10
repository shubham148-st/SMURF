"""
Risk Scoring & Alert Generation
=================================
Generates risk scores and intelligent alerts for suspicious accounts.

Combines supervised classification probabilities with unsupervised
anomaly scores to produce a unified risk assessment.

Risk Tiers (from config.py):
  CRITICAL ≥ 0.85 — Immediate investigation required
  HIGH     ≥ 0.65 — Flag for review within 24 hours
  MEDIUM   ≥ 0.40 — Enhanced monitoring recommended
  LOW      < 0.40 — Normal activity
"""

import numpy as np
import pandas as pd
import os

from config import RISK_TIERS, OUTPUT_DIR


def compute_risk_scores(supervised_proba, anomaly_scores=None,
                        supervised_weight=0.7, anomaly_weight=0.3):
    """
    Combine supervised classification probability with unsupervised anomaly score
    into a unified risk score.
    
    Parameters
    ----------
    supervised_proba : array-like
        Probability of being a mule account from supervised classifier (0 to 1).
    anomaly_scores : array-like, optional
        Anomaly score from unsupervised model (0 to 1).
    supervised_weight : float
        Weight for supervised component (default: 0.7).
    anomaly_weight : float
        Weight for anomaly component (default: 0.3).
    
    Returns
    -------
    risk_scores : array of combined risk scores (0 to 1)
    """
    if anomaly_scores is not None:
        risk_scores = (
            supervised_weight * supervised_proba +
            anomaly_weight * anomaly_scores
        )
    else:
        risk_scores = supervised_proba
    
    # Clip to [0, 1]
    risk_scores = np.clip(risk_scores, 0, 1)
    
    return risk_scores


def assign_risk_tier(score):
    """Map a risk score to a human-readable tier label."""
    for tier, threshold in sorted(RISK_TIERS.items(), key=lambda x: -x[1]):
        if score >= threshold:
            return tier
    return "LOW"


def generate_alerts(risk_scores, account_indices=None, threshold=None):
    """
    Generate intelligent alerts for accounts exceeding risk thresholds.
    
    Returns a DataFrame with:
    - Account index
    - Risk score
    - Risk tier
    - Alert message
    """
    if threshold is None:
        threshold = RISK_TIERS.get("MEDIUM", 0.40)
    
    if account_indices is None:
        account_indices = np.arange(len(risk_scores))
    
    alerts = []
    for idx, score in zip(account_indices, risk_scores):
        if score >= threshold:
            tier = assign_risk_tier(score)
            
            if tier == "CRITICAL":
                msg = f"🔴 CRITICAL: Account {idx} — Risk {score:.2f}. Immediate investigation required."
            elif tier == "HIGH":
                msg = f"🟠 HIGH: Account {idx} — Risk {score:.2f}. Flag for priority review."
            elif tier == "MEDIUM":
                msg = f"🟡 MEDIUM: Account {idx} — Risk {score:.2f}. Enhanced monitoring recommended."
            else:
                msg = f"🟢 LOW: Account {idx} — Risk {score:.2f}. Normal activity."
            
            alerts.append({
                "account_index": idx,
                "risk_score": round(score, 4),
                "risk_tier": tier,
                "alert_message": msg,
            })
    
    alerts_df = pd.DataFrame(alerts)
    
    if len(alerts_df) > 0:
        alerts_df = alerts_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    
    return alerts_df


def generate_risk_report(risk_scores, y_true=None, account_indices=None):
    """
    Generate comprehensive risk scoring report.
    
    Returns
    -------
    report : dict with risk distribution and alerts
    """
    print("\n" + "=" * 60)
    print("RISK SCORING REPORT")
    print("=" * 60)
    
    # Distribution across tiers
    tiers = [assign_risk_tier(s) for s in risk_scores]
    tier_counts = pd.Series(tiers).value_counts()
    
    print(f"\n  Risk Tier Distribution:")
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(risk_scores) * 100
        print(f"    {tier:>10s}: {count:>6d} accounts ({pct:>5.1f}%)")
    
    print(f"\n  Total accounts: {len(risk_scores):,}")
    print(f"  Flagged (≥ MEDIUM): {sum(1 for t in tiers if t in ['CRITICAL', 'HIGH', 'MEDIUM']):,}")
    
    # Generate alerts for MEDIUM and above
    alerts_df = generate_alerts(risk_scores, account_indices)
    
    print(f"\n  Alerts generated: {len(alerts_df)}")
    
    if len(alerts_df) > 0:
        print(f"\n  Top 10 Highest-Risk Accounts:")
        for _, row in alerts_df.head(10).iterrows():
            print(f"    {row['alert_message']}")
    
    # If true labels are available, measure alignment
    if y_true is not None:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, risk_scores)
        print(f"\n  Risk Score AUC vs True Labels: {auc:.4f}")
        
        # Detection rate at each tier
        y_arr = np.array(y_true)
        for tier in ["CRITICAL", "HIGH", "MEDIUM"]:
            tier_mask = np.array([assign_risk_tier(s) == tier for s in risk_scores])
            if tier_mask.sum() > 0:
                fraud_in_tier = y_arr[tier_mask].sum()
                total_in_tier = tier_mask.sum()
                print(f"    {tier}: {int(fraud_in_tier)}/{total_in_tier} flagged accounts are truly suspicious ({fraud_in_tier/total_in_tier*100:.1f}%)")
    
    # Save alerts to CSV
    if len(alerts_df) > 0:
        alerts_path = os.path.join(OUTPUT_DIR, "risk_alerts.csv")
        alerts_df.to_csv(alerts_path, index=False)
        print(f"\n  Alerts saved to: {alerts_path}")
    
    # Save full risk scores
    scores_df = pd.DataFrame({
        "account_index": account_indices if account_indices is not None else np.arange(len(risk_scores)),
        "risk_score": risk_scores,
        "risk_tier": tiers,
    })
    scores_path = os.path.join(OUTPUT_DIR, "risk_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"  Full scores saved to: {scores_path}")
    
    print("=" * 60 + "\n")
    
    return {
        "tier_distribution": dict(tier_counts),
        "alerts": alerts_df,
        "scores_df": scores_df,
    }
