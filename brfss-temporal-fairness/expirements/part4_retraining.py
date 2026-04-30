"""
Part 4 of 5: Retraining mitigation experiment.
Train XGBoost on 2015+2020 combined, test on 2022, compare to original.
Run: python part4_retraining.py
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
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD FROM PART 1
# =============================================================================

OUTPUT_DIR = 'results'

print("=" * 70)
print("PART 4: RETRAINING MITIGATION EXPERIMENT")
print("=" * 70)

print("\nLoading model and data from Part 1...")

with open(os.path.join(OUTPUT_DIR, 'xgb_2015_model.pkl'), 'rb') as f:
    xgb_original = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

MODEL_FEATURES = config['model_features']

df_2015 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2015.pkl'))
df_2020 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2020.pkl'))
df_2022 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2022.pkl'))

print("  Loaded successfully.")

# =============================================================================
# HELPERS
# =============================================================================

AGE_LABELS = {
    1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
    5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
    9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'
}

KEY_INTERSECTIONS = [
    (1, 'White', '18-24'),
    (1, 'Black', '18-24'),
    (1, 'Hispanic', '18-24'),
    (13, 'White', '80+'),
    (13, 'Black', '80+'),
    (13, 'Hispanic', '80+'),
]


def get_race_mask(df, race_name):
    if race_name == 'White':
        return (df['race_black'] == 0) & (df['race_hispanic'] == 0) & (df['race_other'] == 0)
    elif race_name == 'Black':
        return df['race_black'] == 1
    elif race_name == 'Hispanic':
        return df['race_hispanic'] == 1
    else:
        return df['race_other'] == 1


def subgroup_metrics(df, y_pred, age_code, race_name):
    age_mask = df['age_group'] == age_code
    race_mask = get_race_mask(df, race_name)
    mask = age_mask & race_mask
    n = mask.sum()
    if n == 0:
        return {'n': 0, 'f1': np.nan, 'recall': np.nan, 'n_diabetic': 0}
    yt = df.loc[mask, 'diabetes'].values
    yp = y_pred[mask]
    n_diab = int(yt.sum())
    if n_diab == 0:
        return {'n': n, 'f1': np.nan, 'recall': np.nan, 'n_diabetic': 0}
    return {
        'n': n,
        'n_diabetic': n_diab,
        'f1': f1_score(yt, yp, zero_division=0),
        'recall': recall_score(yt, yp, zero_division=0),
    }


def age_metrics(df, y_pred, age_code):
    mask = df['age_group'] == age_code
    n = mask.sum()
    if n == 0:
        return {'n': 0, 'f1': np.nan, 'recall': np.nan}
    yt = df.loc[mask, 'diabetes'].values
    yp = y_pred[mask]
    if yt.sum() == 0:
        return {'n': n, 'f1': np.nan, 'recall': np.nan}
    return {
        'n': n,
        'f1': f1_score(yt, yp, zero_division=0),
        'recall': recall_score(yt, yp, zero_division=0),
    }

# =============================================================================
# TRAIN RETRAINED MODEL ON 2015 + 2020
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST ON 2015+2020 COMBINED DATA")
print("=" * 70)

df_combined = pd.concat([df_2015, df_2020], ignore_index=True)
X_train_combined = df_combined[MODEL_FEATURES]
y_train_combined = df_combined['diabetes']

print(f"\n  2015 only:   {len(df_2015):,} rows  ({df_2015['diabetes'].mean()*100:.2f}% diabetes)")
print(f"  2015+2020:   {len(df_combined):,} rows  ({y_train_combined.mean()*100:.2f}% diabetes)")
print(f"  Test (2022): {len(df_2022):,} rows  ({df_2022['diabetes'].mean()*100:.2f}% diabetes)")

scale_pos_weight = (y_train_combined == 0).sum() / (y_train_combined == 1).sum()
print(f"\n  Scale pos weight (combined): {scale_pos_weight:.2f}")

start = time.time()
xgb_retrained = XGBClassifier(
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
xgb_retrained.fit(X_train_combined, y_train_combined)
print(f"  Trained in {time.time() - start:.1f}s")

# =============================================================================
# OVERALL COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE ON 2022")
print("=" * 70)

X_test = df_2022[MODEL_FEATURES]
y_test = df_2022['diabetes']

print(f"\n  {'Model':<28} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'AUC':>8}")
print("  " + "-" * 60)

for name, mdl in [('Original (2015)', xgb_original), ('Retrained (2015+2020)', xgb_retrained)]:
    yp = mdl.predict(X_test)
    ypr = mdl.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, yp)
    f1 = f1_score(y_test, yp)
    rec = recall_score(y_test, yp)
    auc = roc_auc_score(y_test, ypr)
    print(f"  {name:<28} {acc*100:>9.1f}% {f1*100:>7.1f}% {rec*100:>7.1f}% {auc*100:>7.1f}%")

# =============================================================================
# INTERSECTIONAL COMPARISON (KEY SUBGROUPS)
# =============================================================================

print("\n" + "=" * 70)
print("INTERSECTIONAL COMPARISON (KEY SUBGROUPS)")
print("=" * 70)

orig_pred_2015 = xgb_original.predict(df_2015[MODEL_FEATURES])
orig_pred_2022 = xgb_original.predict(X_test)
retr_pred_2022 = xgb_retrained.predict(X_test)

results = []

for age_code, race_name, age_label in KEY_INTERSECTIONS:
    m_orig_15 = subgroup_metrics(df_2015, orig_pred_2015, age_code, race_name)
    m_orig_22 = subgroup_metrics(df_2022, orig_pred_2022, age_code, race_name)
    m_retr_22 = subgroup_metrics(df_2022, retr_pred_2022, age_code, race_name)

    orig_change = (m_orig_22['f1'] - m_orig_15['f1']) * 100
    improvement = (m_retr_22['f1'] - m_orig_22['f1']) * 100

    row = {
        'Age': age_label, 'Race': race_name,
        'N_2022': m_orig_22['n'],
        'N_diabetic_2022': m_orig_22['n_diabetic'],
        'Original_F1_2015': m_orig_15['f1'],
        'Original_F1_2022': m_orig_22['f1'],
        'Original_Drift_pp': orig_change,
        'Retrained_F1_2022': m_retr_22['f1'],
        'Improvement_pp': improvement,
        'Original_Recall_2022': m_orig_22['recall'],
        'Retrained_Recall_2022': m_retr_22['recall'],
    }
    results.append(row)

    print(f"\n  {age_label} {race_name} (N={m_orig_22['n']:,}, {m_orig_22['n_diabetic']} diabetic):")
    print(f"    Original:  {m_orig_15['f1']*100:.1f}% -> {m_orig_22['f1']*100:.1f}% ({orig_change:+.1f}pp drift)")
    print(f"    Retrained: -> {m_retr_22['f1']*100:.1f}% ({improvement:+.1f}pp vs original 2022)")
    print(f"    Recall:    {m_orig_22['recall']*100:.1f}% -> {m_retr_22['recall']*100:.1f}%")

# =============================================================================
# FULL AGE-STRATIFIED COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("FULL AGE-STRATIFIED COMPARISON (ALL 13 AGE GROUPS)")
print("=" * 70)

print(f"\n  {'Age':<8} {'N':>8} {'Orig_F1':>9} {'Retr_F1':>9} {'Diff':>8} {'Orig_Rec':>9} {'Retr_Rec':>9}")
print("  " + "-" * 70)

age_results = []
for age_code in range(1, 14):
    label = AGE_LABELS[age_code]
    m_orig = age_metrics(df_2022, orig_pred_2022, age_code)
    m_retr = age_metrics(df_2022, retr_pred_2022, age_code)

    if m_orig['n'] == 0:
        continue

    diff = (m_retr['f1'] - m_orig['f1']) * 100

    print(f"  {label:<8} {m_orig['n']:>8,} {m_orig['f1']*100:>8.1f}% {m_retr['f1']*100:>8.1f}% "
          f"{diff:>+7.1f}pp {m_orig['recall']*100:>8.1f}% {m_retr['recall']*100:>8.1f}%")

    age_results.append({
        'Age': label, 'Age_Code': age_code, 'N_2022': m_orig['n'],
        'Original_F1': m_orig['f1'], 'Retrained_F1': m_retr['f1'], 'F1_Diff_pp': diff,
        'Original_Recall': m_orig['recall'], 'Retrained_Recall': m_retr['recall'],
    })

# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, 'retraining_experiment.csv')
results_df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

age_df = pd.DataFrame(age_results)
age_path = os.path.join(OUTPUT_DIR, 'retraining_age_stratified.csv')
age_df.to_csv(age_path, index=False)
print(f"  Saved: {age_path}")

# Save retrained model
retr_path = os.path.join(OUTPUT_DIR, 'xgb_retrained_model.pkl')
with open(retr_path, 'wb') as f:
    pickle.dump(xgb_retrained, f)
print(f"  Retrained model saved: {retr_path}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for _, r in results_df[results_df['Age'] == '18-24'].iterrows():
    race = r['Race']
    orig = r['Original_F1_2022']
    retr = r['Retrained_F1_2022']
    imp = r['Improvement_pp']
    baseline = r['Original_F1_2015']
    print(f"\n  Young {race} (18-24):")
    print(f"    2015 baseline:   {baseline*100:.1f}%")
    print(f"    2022 original:   {orig*100:.1f}%")
    print(f"    2022 retrained:  {retr*100:.1f}% ({imp:+.1f}pp improvement)")
    recovered = (retr - orig) / (baseline - orig) * 100 if baseline != orig else 0
    print(f"    Recovery:        {recovered:.0f}% of lost performance recovered")

print("\n" + "=" * 70)
print("PART 4 COMPLETE - Run part5_save_all.py next")
print("=" * 70)