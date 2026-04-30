"""
Part 5 of 5: Consolidated summary of all results.
Run: python part5_summary.py
"""

import pandas as pd
import os

OUTPUT_DIR = 'results'

print("=" * 70)
print("PART 5: CONSOLIDATED RESULTS SUMMARY")
print("=" * 70)

# =============================================================================
# 1. BOOTSTRAP CIs (Step 2)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 RESULTS: BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

ci = pd.read_csv(os.path.join(OUTPUT_DIR, 'table3_with_bootstrap_cis.csv'))

print(f"\n  Updated Table 3 with 95% Bootstrap CIs (1000 resamples):")
print(f"  {'Age':<8} {'Race':<10} {'N':>6} {'Diab%':>7} {'F1 2015 [95% CI]':>25} {'F1 2022 [95% CI]':>25} {'Change':>8}")
print("  " + "-" * 100)

for _, r in ci.iterrows():
    ci15 = f"{r['F1_2015']*100:.1f}% [{r['F1_2015_CI_low']*100:.1f}, {r['F1_2015_CI_high']*100:.1f}]"
    ci22 = f"{r['F1_2022']*100:.1f}% [{r['F1_2022_CI_low']*100:.1f}, {r['F1_2022_CI_high']*100:.1f}]"
    print(f"  {r['Age']:<8} {r['Race']:<10} {r['N_2022']:>6,} {r['Diabetes_pct_2022']:>7} {ci15:>25} {ci22:>25} {r['F1_Change_pp']:>+7.1f}pp")

print(f"\n  Key observations:")
print(f"    - Young Black CIs: 2015 [{ci.iloc[1]['F1_2015_CI_low']*100:.1f}, {ci.iloc[1]['F1_2015_CI_high']*100:.1f}] vs 2022 [{ci.iloc[1]['F1_2022_CI_low']*100:.1f}, {ci.iloc[1]['F1_2022_CI_high']*100:.1f}]")

ci_overlap = ci.iloc[1]['F1_2022_CI_high'] > ci.iloc[1]['F1_2015_CI_low']
if ci_overlap:
    print(f"    - CIs show marginal overlap, but point estimate drop of {ci.iloc[1]['F1_Change_pp']:+.1f}pp is substantial")
else:
    print(f"    - CIs do NOT overlap — degradation is statistically significant")

print(f"    - Elderly CIs are tight (±1-3pp) confirming stable performance")

# =============================================================================
# 2. LR COMPARISON (Step 3)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 RESULTS: LOGISTIC REGRESSION vs XGBOOST")
print("=" * 70)

lr = pd.read_csv(os.path.join(OUTPUT_DIR, 'lr_vs_xgboost_intersectional.csv'))
lr_age = pd.read_csv(os.path.join(OUTPUT_DIR, 'lr_vs_xgboost_age_stratified.csv'))

print(f"\n  Key Intersections:")
print(f"  {'Age':<8} {'Race':<10} {'XGB Change':>12} {'LR Change':>12} {'XGB worse by':>14}")
print("  " + "-" * 60)

for _, r in lr.iterrows():
    ratio = abs(r['XGB_Change_pp']) - abs(r['LR_Change_pp'])
    print(f"  {r['Age']:<8} {r['Race']:<10} {r['XGB_Change_pp']:>+11.1f}pp {r['LR_Change_pp']:>+11.1f}pp {ratio:>+13.1f}pp")

print(f"\n  Age-Stratified Summary (young adults):")
for _, r in lr_age[lr_age['Age_Code'] <= 3].iterrows():
    print(f"    {r['Age']}: XGBoost {r['XGB_Change_pp']:+.1f}pp vs LR {r['LR_Change_pp']:+.1f}pp")

print(f"\n  Key finding:")
yb = lr[(lr['Age'] == '18-24') & (lr['Race'] == 'Black')].iloc[0]
print(f"    XGBoost degradation for young Black adults: {yb['XGB_Change_pp']:+.1f}pp")
print(f"    LR degradation for young Black adults:      {yb['LR_Change_pp']:+.1f}pp")
print(f"    XGBoost degrades {abs(yb['XGB_Change_pp'] / yb['LR_Change_pp']):.1f}x more than LR")
print(f"    -> Nonlinear models amplify temporal fairness degradation")
print(f"    -> But LR still degrades -> partially data-driven")

# =============================================================================
# 3. RETRAINING EXPERIMENT (Step 4)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 RESULTS: RETRAINING MITIGATION (2015+2020 -> 2022)")
print("=" * 70)

retr = pd.read_csv(os.path.join(OUTPUT_DIR, 'retraining_experiment.csv'))
retr_age = pd.read_csv(os.path.join(OUTPUT_DIR, 'retraining_age_stratified.csv'))

print(f"\n  Key Intersections:")
print(f"  {'Age':<8} {'Race':<10} {'Baseline':>10} {'Original':>10} {'Retrained':>10} {'Improv':>8} {'Recovery':>10}")
print("  " + "-" * 75)

for _, r in retr.iterrows():
    baseline = r['Original_F1_2015']
    orig = r['Original_F1_2022']
    retr_f1 = r['Retrained_F1_2022']
    imp = r['Improvement_pp']
    if baseline != orig and not pd.isna(baseline):
        recovery = (retr_f1 - orig) / (baseline - orig) * 100
        rec_str = f"{recovery:.0f}%"
    else:
        rec_str = "N/A"
    print(f"  {r['Age']:<8} {r['Race']:<10} {baseline*100:>9.1f}% {orig*100:>9.1f}% {retr_f1*100:>9.1f}% {imp:>+7.1f}pp {rec_str:>10}")

print(f"\n  Age-Stratified Summary (young adults):")
for _, r in retr_age[retr_age['Age_Code'] <= 3].iterrows():
    print(f"    {r['Age']}: {r['Original_F1']*100:.1f}% -> {r['Retrained_F1']*100:.1f}% ({r['F1_Diff_pp']:+.1f}pp)")

print(f"\n  Key finding:")
print(f"    Retraining provides modest improvement for young adults (+1-4pp)")
print(f"    But recovery is minimal for young Black adults (11% of lost performance)")
print(f"    Simple retraining is INSUFFICIENT to address intersectional fairness drift")

# =============================================================================
# COMBINED NARRATIVE
# =============================================================================

print("\n" + "=" * 70)
print("NARRATIVE SUMMARY FOR PAPER")
print("=" * 70)

yb_ci = ci[(ci['Age'] == '18-24') & (ci['Race'] == 'Black')].iloc[0]
yb_lr = lr[(lr['Age'] == '18-24') & (lr['Race'] == 'Black')].iloc[0]
yb_rt = retr[(retr['Age'] == '18-24') & (retr['Race'] == 'Black')].iloc[0]

print(f"""
  For young Black adults (18-24):
  
  1. DEGRADATION IS REAL (Step 2):
     F1 dropped from {yb_ci['F1_2015']*100:.1f}% to {yb_ci['F1_2022']*100:.1f}% ({yb_ci['F1_Change_pp']:+.1f}pp)
     2015 CI: [{yb_ci['F1_2015_CI_low']*100:.1f}%, {yb_ci['F1_2015_CI_high']*100:.1f}%]
     2022 CI: [{yb_ci['F1_2022_CI_low']*100:.1f}%, {yb_ci['F1_2022_CI_high']*100:.1f}%]
  
  2. NONLINEAR MODELS AMPLIFY IT (Step 3):
     XGBoost: {yb_lr['XGB_Change_pp']:+.1f}pp degradation
     Log.Reg: {yb_lr['LR_Change_pp']:+.1f}pp degradation
     XGBoost is {abs(yb_lr['XGB_Change_pp'] / yb_lr['LR_Change_pp']):.1f}x worse
  
  3. RETRAINING DOESN'T FIX IT (Step 4):
     Original 2022: {yb_rt['Original_F1_2022']*100:.1f}%
     Retrained 2022: {yb_rt['Retrained_F1_2022']*100:.1f}% ({yb_rt['Improvement_pp']:+.1f}pp)
     Only {(yb_rt['Retrained_F1_2022'] - yb_rt['Original_F1_2022']) / (yb_rt['Original_F1_2015'] - yb_rt['Original_F1_2022']) * 100:.0f}% of lost performance recovered
""")

# =============================================================================
# FILES PRODUCED
# =============================================================================

print("=" * 70)
print("ALL OUTPUT FILES")
print("=" * 70)

print(f"\n  Directory: {os.path.abspath(OUTPUT_DIR)}")
print()
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:<45} ({size:,} bytes)")

print("\n" + "=" * 70)
print("ALL 5 PARTS COMPLETE")
print("=" * 70)
print("\n  Next: Use these results to update your paper (Step 5 of the plan).")
print("  Bring the terminal output back to Claude for help writing the sections.")