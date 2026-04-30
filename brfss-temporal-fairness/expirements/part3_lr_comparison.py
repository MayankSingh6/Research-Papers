"""
Part 3 of 5: Logistic Regression intersectional comparison.
Run: python part3_lr_comparison.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD FROM PART 1
# =============================================================================

OUTPUT_DIR = 'results'

print("=" * 70)
print("PART 3: LOGISTIC REGRESSION INTERSECTIONAL COMPARISON")
print("=" * 70)

print("\nLoading model and data from Part 1...")

with open(os.path.join(OUTPUT_DIR, 'xgb_2015_model.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

MODEL_FEATURES = config['model_features']

df_2015 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2015.pkl'))
df_2022 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2022.pkl'))

print("  Loaded successfully.")

# =============================================================================
# HELPERS
# =============================================================================

def get_race_mask(df, race_name):
    if race_name == 'White':
        return (df['race_black'] == 0) & (df['race_hispanic'] == 0) & (df['race_other'] == 0)
    elif race_name == 'Black':
        return df['race_black'] == 1
    elif race_name == 'Hispanic':
        return df['race_hispanic'] == 1
    else:
        return df['race_other'] == 1


def subgroup_f1(df, y_pred, age_code, race_name):
    age_mask = df['age_group'] == age_code
    race_mask = get_race_mask(df, race_name)
    mask = age_mask & race_mask
    if mask.sum() == 0:
        return np.nan, 0
    yt = df.loc[mask, 'diabetes'].values
    yp = y_pred[mask]
    if yt.sum() == 0:
        return np.nan, mask.sum()
    return f1_score(yt, yp, zero_division=0), mask.sum()


def subgroup_recall(df, y_pred, age_code, race_name):
    age_mask = df['age_group'] == age_code
    race_mask = get_race_mask(df, race_name)
    mask = age_mask & race_mask
    if mask.sum() == 0:
        return np.nan
    yt = df.loc[mask, 'diabetes'].values
    yp = y_pred[mask]
    if yt.sum() == 0:
        return np.nan
    return recall_score(yt, yp, zero_division=0)


KEY_INTERSECTIONS = [
    (1, 'White', '18-24'),
    (1, 'Black', '18-24'),
    (1, 'Hispanic', '18-24'),
    (13, 'White', '80+'),
    (13, 'Black', '80+'),
    (13, 'Hispanic', '80+'),
]

AGE_LABELS = {
    1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
    5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
    9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'
}


def subgroup_f1_age(df, y_pred, age_code):
    mask = df['age_group'] == age_code
    if mask.sum() == 0:
        return np.nan, 0
    yt = df.loc[mask, 'diabetes'].values
    yp = y_pred[mask]
    if yt.sum() == 0:
        return np.nan, mask.sum()
    return f1_score(yt, yp, zero_division=0), mask.sum()

# =============================================================================
# TRAIN LOGISTIC REGRESSION ON 2015
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION ON 2015 DATA")
print("=" * 70)

X_train = df_2015[MODEL_FEATURES]
y_train = df_2015['diabetes']
X_test = df_2022[MODEL_FEATURES]
y_test = df_2022['diabetes']

start = time.time()
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train, y_train)
print(f"  Trained in {time.time() - start:.1f}s")

# =============================================================================
# OVERALL PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE ON 2022 TEST SET")
print("=" * 70)

print(f"\n  {'Model':<15} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'AUC':>8}")
print("  " + "-" * 55)

for name, mdl in [('XGBoost', xgb_model), ('Log. Regression', lr_model)]:
    yp = mdl.predict(X_test)
    ypr = mdl.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, yp)
    f1 = f1_score(y_test, yp)
    rec = recall_score(y_test, yp)
    auc = roc_auc_score(y_test, ypr)
    print(f"  {name:<15} {acc*100:>9.1f}% {f1*100:>7.1f}% {rec*100:>7.1f}% {auc*100:>7.1f}%")

# =============================================================================
# INTERSECTIONAL COMPARISON (KEY SUBGROUPS)
# =============================================================================

print("\n" + "=" * 70)
print("INTERSECTIONAL COMPARISON (KEY SUBGROUPS)")
print("=" * 70)

# Generate predictions for both models on both years
xgb_pred_2015 = xgb_model.predict(df_2015[MODEL_FEATURES])
xgb_pred_2022 = xgb_model.predict(X_test)
lr_pred_2015 = lr_model.predict(df_2015[MODEL_FEATURES])
lr_pred_2022 = lr_model.predict(X_test)

results = []

for age_code, race_name, age_label in KEY_INTERSECTIONS:
    xgb_f1_15, _ = subgroup_f1(df_2015, xgb_pred_2015, age_code, race_name)
    xgb_f1_22, n = subgroup_f1(df_2022, xgb_pred_2022, age_code, race_name)
    lr_f1_15, _ = subgroup_f1(df_2015, lr_pred_2015, age_code, race_name)
    lr_f1_22, _ = subgroup_f1(df_2022, lr_pred_2022, age_code, race_name)

    xgb_rec_22 = subgroup_recall(df_2022, xgb_pred_2022, age_code, race_name)
    lr_rec_22 = subgroup_recall(df_2022, lr_pred_2022, age_code, race_name)

    xgb_change = (xgb_f1_22 - xgb_f1_15) * 100
    lr_change = (lr_f1_22 - lr_f1_15) * 100

    row = {
        'Age': age_label, 'Race': race_name, 'N_2022': n,
        'XGB_F1_2015': xgb_f1_15, 'XGB_F1_2022': xgb_f1_22, 'XGB_Change_pp': xgb_change,
        'LR_F1_2015': lr_f1_15, 'LR_F1_2022': lr_f1_22, 'LR_Change_pp': lr_change,
        'XGB_Recall_2022': xgb_rec_22, 'LR_Recall_2022': lr_rec_22,
    }
    results.append(row)

    print(f"\n  {age_label} {race_name} (N={n:,}):")
    print(f"    XGBoost:  F1 {xgb_f1_15*100:.1f}% -> {xgb_f1_22*100:.1f}% ({xgb_change:+.1f}pp)  Recall: {xgb_rec_22*100:.1f}%")
    print(f"    Log.Reg:  F1 {lr_f1_15*100:.1f}% -> {lr_f1_22*100:.1f}% ({lr_change:+.1f}pp)  Recall: {lr_rec_22*100:.1f}%")

# =============================================================================
# FULL AGE-STRATIFIED COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("FULL AGE-STRATIFIED COMPARISON (ALL 13 AGE GROUPS)")
print("=" * 70)

print(f"\n  {'Age':<8} {'N':>8} {'XGB_F1':>8} {'LR_F1':>8} {'XGB_Chg':>9} {'LR_Chg':>9}")
print("  " + "-" * 55)

age_results = []
for age_code in range(1, 14):
    label = AGE_LABELS[age_code]

    xf15, _ = subgroup_f1_age(df_2015, xgb_pred_2015, age_code)
    xf22, n = subgroup_f1_age(df_2022, xgb_pred_2022, age_code)
    lf15, _ = subgroup_f1_age(df_2015, lr_pred_2015, age_code)
    lf22, _ = subgroup_f1_age(df_2022, lr_pred_2022, age_code)

    xgb_chg = (xf22 - xf15) * 100
    lr_chg = (lf22 - lf15) * 100

    print(f"  {label:<8} {n:>8,} {xf22*100:>7.1f}% {lf22*100:>7.1f}% {xgb_chg:>+8.1f}pp {lr_chg:>+8.1f}pp")

    age_results.append({
        'Age': label, 'Age_Code': age_code, 'N_2022': n,
        'XGB_F1_2015': xf15, 'XGB_F1_2022': xf22, 'XGB_Change_pp': xgb_chg,
        'LR_F1_2015': lf15, 'LR_F1_2022': lf22, 'LR_Change_pp': lr_chg,
    })

# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, 'lr_vs_xgboost_intersectional.csv')
results_df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

age_df = pd.DataFrame(age_results)
age_path = os.path.join(OUTPUT_DIR, 'lr_vs_xgboost_age_stratified.csv')
age_df.to_csv(age_path, index=False)
print(f"  Saved: {age_path}")

# Save LR model for Part 4 if needed
lr_path = os.path.join(OUTPUT_DIR, 'lr_2015_model.pkl')
with open(lr_path, 'wb') as f:
    pickle.dump(lr_model, f)
print(f"  LR model saved: {lr_path}")

# =============================================================================
# KEY FINDING
# =============================================================================

print("\n" + "=" * 70)
print("KEY FINDING")
print("=" * 70)

yb_xgb = results_df[(results_df['Age']=='18-24') & (results_df['Race']=='Black')]['XGB_Change_pp'].values[0]
yb_lr = results_df[(results_df['Age']=='18-24') & (results_df['Race']=='Black')]['LR_Change_pp'].values[0]

print(f"\n  Young Black adults (18-24):")
print(f"    XGBoost degradation: {yb_xgb:+.1f}pp")
print(f"    Log.Reg degradation: {yb_lr:+.1f}pp")

if abs(yb_lr) > 10:
    print(f"    -> BOTH models show severe degradation -> DATA-DRIVEN problem")
elif abs(yb_lr) > 5:
    print(f"    -> Both degrade but LR less severely -> partially data-driven, partially model-specific")
else:
    print(f"    -> LR degrades much less -> degradation is amplified by nonlinear models")

print("\n" + "=" * 70)
print("PART 3 COMPLETE - Run part4_retraining.py next")
print("=" * 70)