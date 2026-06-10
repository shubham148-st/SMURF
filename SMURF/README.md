# SMURF — Suspicious Mule Account Classification

> AI/ML-Based Classification of Suspicious Mule Accounts  
> Hackathon Solution built on the SMURF AML research framework

## Architecture

```
SMURF/
├── config.py               # Configuration, paths, hyperparameters
├── preprocess.py            # Data loading, NA handling, encoding, splitting
├── feature_engineering.py   # Feature selection (RF + MI + Bank features)
├── model.py                 # XGBoost, LightGBM, RF, Ensemble + Focal Loss
├── train.py                 # Training pipeline, cross-validation, ablation
├── anomaly_detection.py     # Isolation Forest + LOF anomaly scoring
├── risk_scoring.py          # Risk tiers & intelligent alert generation
├── evaluate.py              # Metrics, ROC/PR curves, confusion matrices
├── main.py                  # Pipeline orchestrator
├── outputs/                 # Plots, reports, risk alerts (auto-created)
└── saved_models/            # Trained model artifacts (auto-created)
```

## Reuse from SMURF AML Research

| Component | Original (AML_smurf) | Adapted (Hackathon) |
|---|---|---|
| **Focal Loss** | `model.py` → PyTorch FocalLoss | `model.py` → XGBoost custom objective |
| **Class Imbalance** | `train.py` → `subsample_graph()` | `preprocess.py` → `undersample_majority()` |
| **Ablation Study** | `alabation.py` → multi-model comparison | `train.py` → `run_ablation_study()` |
| **Evaluation** | `train.py` → F1/Precision/Recall at 0.4 threshold | `evaluate.py` → comprehensive metrics + threshold optimization |

## Quick Start

```bash
# Full pipeline (all features + ablation)
python main.py --mode full

# Quick test with 18 bank key features only
python main.py --mode quick

# Train models only (no ablation)
python main.py --mode train

# Ablation study only
python main.py --mode ablation
```

## Dataset

- **Source**: `AML_smurf/dataset/DataSet.csv`
- **Shape**: ~3,924 anonymized features per account
- **Target**: `F3924` (binary: 0 = legitimate, 1 = suspicious/mule)
- **Key Features** (bank-specified): F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894

## Pipeline Outputs

1. **Trained Models**: XGBoost, LightGBM, Stacking Ensemble saved to `saved_models/`
2. **Evaluation Plots**: ROC curves, PR curves, confusion matrices in `outputs/`
3. **Risk Alerts**: CSV with risk-scored accounts and tier labels in `outputs/risk_alerts.csv`
4. **Ablation Study**: Model comparison table and bar chart
5. **Feature Importance**: Ranked feature importance with bank key features highlighted

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
joblib
```
