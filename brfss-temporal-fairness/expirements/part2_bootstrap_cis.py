"""
Part 2 of 5: Bootstrap Confidence Intervals for Table 3.
Run: python part2_bootstrap_cis.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD FROM PART 1
# =============================================================================

OUTPUT_DIR = 'results'

print("=" * 70)
print("PART 2: BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

print("\nLoading model and data from Part 1...")

with open(os.path.join(OUTPUT_DIR, 'xgb_2015_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

MODEL_FEATURES = config['model_features']

df_2015 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2015.pkl'))
df_2022 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_2022.pkl'))

print("  Loaded successfully.")

# =============================================================================
# CONFIG
# =============================================================================

N_BOOTSTRAP = 1000
SEED = 42

KEY_INTERSECTIONS = [
    (1, 'White', '18-24'),
    (1, 'Black', '18-24'),
    (1, 'Hispanic', '18-24'),
    (13, 'White', '80+'),
    (13, 'Black', '80+'),
    (13, 'Hispanic', '80+'),
]

# =============================================================================
# HELPER FUNCTIONS
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


def get_subgroup(df, y_pred, age_code, race_name):
    age_mask = df['age_group'] == age_code
    race_mask = get_race_mask(df, race_name)
    mask = age_mask & race_mask
    return df.loc[mask, 'diabetes'].values, y_pred[mask], mask.sum()


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute 95% bootstrap CI for a metric."""
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return np.nan, np.nan

    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.sum() == 0:
            continue
        scores.append(metric_fn(yt, yp, zero_division=0))

    if len(scores) < 100:
        return np.nan, np.nan

    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

# =============================================================================
# COMPUTE BOOTSTRAP CIs
# =============================================================================

print("\n" + "=" * 70)
print(f"COMPUTING BOOTSTRAP CIs ({N_BOOTSTRAP} resamples per subgroup)")
print("=" * 70)

y_pred_2015 = model.predict(df_2015[MODEL_FEATURES])
y_pred_2022 = model.predict(df_2022[MODEL_FEATURES])

results = []

for age_code, race_name, age_label in KEY_INTERSECTIONS:
    print(f"\n  {age_label} {race_name}...")

    # 2015
    yt_15, yp_15, n_15 = get_subgroup(df_2015, y_pred_2015, age_code, race_name)
    f1_15 = f1_score(yt_15, yp_15, zero_division=0) if yt_15.sum() > 0 else np.nan
    rec_15 = recall_score(yt_15, yp_15, zero_division=0) if yt_15.sum() > 0 else np.nan
    f1_15_lo, f1_15_hi = bootstrap_ci(yt_15, yp_15, f1_score)
    rec_15_lo, rec_15_hi = bootstrap_ci(yt_15, yp_15, recall_score)

    # 2022
    yt_22, yp_22, n_22 = get_subgroup(df_2022, y_pred_2022, age_code, race_name)
    f1_22 = f1_score(yt_22, yp_22, zero_division=0) if yt_22.sum() > 0 else np.nan
    rec_22 = recall_score(yt_22, yp_22, zero_division=0) if yt_22.sum() > 0 else np.nan
    f1_22_lo, f1_22_hi = bootstrap_ci(yt_22, yp_22, f1_score)
    rec_22_lo, rec_22_hi = bootstrap_ci(yt_22, yp_22, recall_score)

    change = (f1_22 - f1_15) * 100 if not (np.isnan(f1_15) or np.isnan(f1_22)) else np.nan

    row = {
        'Age': age_label,
        'Race': race_name,
        'N_2015': n_15,
        'N_2022': n_22,
        'N_diabetic_2022': int(yt_22.sum()),
        'Diabetes_pct_2022': f"{yt_22.mean()*100:.2f}%",
        'F1_2015': f1_15,
        'F1_2015_CI_low': f1_15_lo,
        'F1_2015_CI_high': f1_15_hi,
        'F1_2022': f1_22,
        'F1_2022_CI_low': f1_22_lo,
        'F1_2022_CI_high': f1_22_hi,
        'F1_Change_pp': change,
        'Recall_2015': rec_15,
        'Recall_2015_CI_low': rec_15_lo,
        'Recall_2015_CI_high': rec_15_hi,
        'Recall_2022': rec_22,
        'Recall_2022_CI_low': rec_22_lo,
        'Recall_2022_CI_high': rec_22_hi,
    }
    results.append(row)

    print(f"    N: {n_22:,} ({int(yt_22.sum())} diabetic)")
    print(f"    2015 F1: {f1_15*100:.1f}%  [{f1_15_lo*100:.1f}, {f1_15_hi*100:.1f}]")
    print(f"    2022 F1: {f1_22*100:.1f}%  [{f1_22_lo*100:.1f}, {f1_22_hi*100:.1f}]")
    print(f"    Change:  {change:+.1f}pp")
    print(f"    2015 Recall: {rec_15*100:.1f}%  [{rec_15_lo*100:.1f}, {rec_15_hi*100:.1f}]")
    print(f"    2022 Recall: {rec_22*100:.1f}%  [{rec_22_lo*100:.1f}, {rec_22_hi*100:.1f}]")

# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, 'table3_with_bootstrap_cis.csv')
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\n  {'Age':<8} {'Race':<10} {'N':>6} {'F1_2015':>20} {'F1_2022':>20} {'Change':>8}")
print("  " + "-" * 78)
for _, r in results_df.iterrows():
    ci15 = f"{r['F1_2015']*100:.1f}% [{r['F1_2015_CI_low']*100:.1f},{r['F1_2015_CI_high']*100:.1f}]"
    ci22 = f"{r['F1_2022']*100:.1f}% [{r['F1_2022_CI_low']*100:.1f},{r['F1_2022_CI_high']*100:.1f}]"
    print(f"  {r['Age']:<8} {r['Race']:<10} {r['N_2022']:>6,} {ci15:>20} {ci22:>20} {r['F1_Change_pp']:>+7.1f}pp")

print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("PART 2 COMPLETE - Run part3_lr_comparison.py next")
print("=" * 70)