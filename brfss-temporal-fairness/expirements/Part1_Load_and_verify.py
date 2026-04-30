"""
Part 1 of 5: Load data, train XGBoost on 2015, verify metrics match paper.
Run: python part1_load_and_verify.py
"""

import pandas as pd
import numpy as np
import time
import os
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = 'data/processed'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FEATURES = [
    # Demographics (7)
    'sex', 'race_black', 'race_hispanic', 'race_other',
    'age_group', 'income_group', 'education',
    # Health (9)
    'bmi', 'heart_attack', 'heart_disease', 'stroke',
    'exercise', 'ever_smoked',
    'general_health', 'physical_health_days', 'mental_health_days'
]

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 70)
print("PART 1: LOAD DATA AND VERIFY MODEL")
print("=" * 70)

print("\nLoading data...")
df_2015 = pd.read_csv(os.path.join(DATA_DIR, 'brfss_2015_clean.csv'))
df_2020 = pd.read_csv(os.path.join(DATA_DIR, 'brfss_2020_clean.csv'))
df_2022 = pd.read_csv(os.path.join(DATA_DIR, 'brfss_2022_clean.csv'))

print(f"  2015: {len(df_2015):,} rows  ({df_2015['diabetes'].mean()*100:.2f}% diabetes)")
print(f"  2020: {len(df_2020):,} rows  ({df_2020['diabetes'].mean()*100:.2f}% diabetes)")
print(f"  2022: {len(df_2022):,} rows  ({df_2022['diabetes'].mean()*100:.2f}% diabetes)")

# Verify features exist
missing = [f for f in MODEL_FEATURES if f not in df_2015.columns]
if missing:
    print(f"\n  ERROR: Missing features: {missing}")
    exit(1)
else:
    print(f"\n  All {len(MODEL_FEATURES)} features present in all years.")

# =============================================================================
# TRAIN XGBOOST ON 2015
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST ON 2015 DATA")
print("=" * 70)

X_train = df_2015[MODEL_FEATURES]
y_train = df_2015['diabetes']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n  Scale pos weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

start = time.time()
model.fit(X_train, y_train)
print(f"  Trained in {time.time() - start:.1f}s")

# =============================================================================
# VERIFY METRICS MATCH PAPER
# =============================================================================

print("\n" + "=" * 70)
print("VERIFICATION: DO METRICS MATCH THE PAPER?")
print("=" * 70)

results = {}
for year, df in [('2015', df_2015), ('2020', df_2020), ('2022', df_2022)]:
    X = df[MODEL_FEATURES]
    y = df['diabetes']
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    m = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc': roc_auc_score(y, y_proba),
    }
    results[year] = m

    label = '(train)' if year == '2015' else f'(+{int(year)-2015} yrs)'
    print(f"\n  {year} {label}:")
    print(f"    Accuracy:  {m['accuracy']*100:.2f}%")
    print(f"    Precision: {m['precision']*100:.2f}%")
    print(f"    Recall:    {m['recall']*100:.2f}%")
    print(f"    F1:        {m['f1']*100:.2f}%")
    print(f"    AUC-ROC:   {m['auc']*100:.2f}%")

# Compare to paper values
print("\n" + "-" * 70)
print("PAPER COMPARISON (Table 2 / Section 5.1):")
print("-" * 70)
print(f"  Paper says 2022 Accuracy: ~71.1%   You got: {results['2022']['accuracy']*100:.1f}%")
print(f"  Paper says 2022 F1:       ~43.3%   You got: {results['2022']['f1']*100:.1f}%")
print(f"  Paper says 2022 Recall:   ~73.1%   You got: {results['2022']['recall']*100:.1f}%")
print(f"  Paper says 2022 AUC:      ~79.6%   You got: {results['2022']['auc']*100:.1f}%")

acc_match = abs(results['2022']['accuracy']*100 - 71.1) < 2.0
f1_match = abs(results['2022']['f1']*100 - 43.3) < 2.0
if acc_match and f1_match:
    print("\n  PASS: Metrics are consistent with paper. Safe to proceed.")
else:
    print("\n  WARNING: Metrics differ from paper. Check hyperparameters.")

# =============================================================================
# SAVE MODEL AND DATA FOR PARTS 2-5
# =============================================================================

print("\n" + "=" * 70)
print("SAVING FOR NEXT PARTS")
print("=" * 70)

# Save model
model_path = os.path.join(OUTPUT_DIR, 'xgb_2015_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  Model saved: {model_path}")

# Save dataframes as pickle for fast loading in subsequent parts
for year, df in [('2015', df_2015), ('2020', df_2020), ('2022', df_2022)]:
    path = os.path.join(OUTPUT_DIR, f'df_{year}.pkl')
    df.to_pickle(path)
print(f"  DataFrames saved to {OUTPUT_DIR}/df_YYYY.pkl")

# Save feature list
import json
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump({'model_features': MODEL_FEATURES}, f)
print(f"  Config saved: {OUTPUT_DIR}/config.json")

print("\n" + "=" * 70)
print("PART 1 COMPLETE - Run part2_bootstrap_cis.py next")
print("=" * 70)