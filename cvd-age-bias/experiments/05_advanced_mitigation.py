import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import os

print("="*60)
print("🚀 ADVANCED BIAS MITIGATION - AGE-STRATIFIED ENSEMBLE")
print("="*60)

# ============================================================
# STEP 1: LOAD DATA AND PREPARE
# ============================================================
print("\n📂 Loading cleaned data...")
df = pd.read_csv('data/kaggle/cardio_cleaned.csv')
print(f"✅ Loaded {len(df):,} records")

# Prepare features
feature_cols = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    'bp_risk', 'cholesterol_risk', 'glucose_risk', 'risk_score'
]

X = df[feature_cols]
y = df['cardio']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Keep demographic info
demo_train = df.loc[X_train.index, ['age_group', 'gender']]
demo_test = df.loc[X_test.index, ['age_group', 'gender']]

print(f"✅ Train/test split complete")

# ============================================================
# BASELINE: For comparison
# ============================================================
print("\n" + "="*60)
print("📊 BASELINE MODEL (for comparison)")
print("="*60)

baseline_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

# Calculate baseline age fairness
baseline_age_results = {}

for age_group in ['young', 'middle_age', 'senior', 'elderly']:
    mask = demo_test['age_group'] == age_group
    if mask.sum() < 50:
        continue
    
    acc = accuracy_score(y_test[mask], baseline_pred[mask])
    rec = recall_score(y_test[mask], baseline_pred[mask])
    baseline_age_results[age_group] = {'accuracy': acc, 'recall': rec}

baseline_range = max([v['accuracy'] for v in baseline_age_results.values()]) - \
                 min([v['accuracy'] for v in baseline_age_results.values()])

print(f"Baseline age accuracy range: {baseline_range:.4f} ({baseline_range*100:.2f}%)")

for age, metrics in baseline_age_results.items():
    print(f"  {age:15s}: Acc={metrics['accuracy']:.4f}, Recall={metrics['recall']:.4f}")

# ============================================================
# TECHNIQUE 4: AGE-STRATIFIED ENSEMBLE
# ============================================================
print("\n" + "="*60)
print("🔧 TECHNIQUE 4: Age-Stratified Ensemble Models")
print("="*60)
print("   Training separate specialized model for EACH age group...")
print("   This allows each model to learn age-specific patterns.")

# Train separate model for each age group
age_models = {}
age_group_sizes = {}

for age_group in ['young', 'middle_age', 'senior', 'elderly']:
    mask_train = demo_train['age_group'] == age_group
    
    if mask_train.sum() < 100:  # Skip if too few samples
        print(f"\n⚠️  {age_group}: Too few training samples ({mask_train.sum()}), using baseline")
        age_models[age_group] = baseline_model
        continue
    
    X_train_age = X_train[mask_train]
    y_train_age = y_train[mask_train]
    
    age_group_sizes[age_group] = len(X_train_age)
    
    print(f"\n🤖 Training {age_group} model on {len(X_train_age):,} samples...")
    
    # Age-specific model with tuned hyperparameters
    if age_group == 'young':
        # Young: fewer samples, prevent overfitting
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif age_group == 'elderly':
        # Elderly: more complex, deeper trees
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    else:
        # Middle age / Senior: standard
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    model.fit(X_train_age, y_train_age)
    age_models[age_group] = model
    print(f"   ✅ {age_group} model trained")

# ============================================================
# MAKE PREDICTIONS USING AGE-SPECIFIC MODELS
# ============================================================
print("\n🔮 Making predictions using age-specific models...")

ensemble_pred = np.zeros(len(X_test), dtype=int)
ensemble_proba = np.zeros(len(X_test))

for age_group in ['young', 'middle_age', 'senior', 'elderly']:
    mask_test = demo_test['age_group'] == age_group
    
    if mask_test.sum() == 0:
        continue
    
    # Use age-specific model
    model = age_models[age_group]
    ensemble_pred[mask_test] = model.predict(X_test[mask_test])
    ensemble_proba[mask_test] = model.predict_proba(X_test[mask_test])[:, 1]
    
    print(f"   {age_group}: {mask_test.sum()} predictions made")

# ============================================================
# EVALUATE ENSEMBLE
# ============================================================
print("\n" + "="*60)
print("📊 ENSEMBLE MODEL RESULTS")
print("="*60)

# Overall performance
overall_acc = accuracy_score(y_test, ensemble_pred)
overall_prec = precision_score(y_test, ensemble_pred)
overall_rec = recall_score(y_test, ensemble_pred)
overall_f1 = f1_score(y_test, ensemble_pred)
overall_auc = roc_auc_score(y_test, ensemble_proba)

print(f"\nOverall Performance:")
print(f"  Accuracy:  {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"  Precision: {overall_prec:.4f}")
print(f"  Recall:    {overall_rec:.4f}")
print(f"  F1-Score:  {overall_f1:.4f}")
print(f"  AUC-ROC:   {overall_auc:.4f}")

# Baseline overall for comparison
baseline_overall_acc = accuracy_score(y_test, baseline_pred)
print(f"\nBaseline overall accuracy: {baseline_overall_acc:.4f}")
print(f"Ensemble overall accuracy: {overall_acc:.4f}")
print(f"Difference: {(overall_acc - baseline_overall_acc):.4f} ({(overall_acc - baseline_overall_acc)*100:.2f}%)")

# Age-specific performance
print("\n" + "="*60)
print("⚖️ AGE-SPECIFIC PERFORMANCE")
print("="*60)

ensemble_age_results = {}

for age_group in ['young', 'middle_age', 'senior', 'elderly']:
    mask = demo_test['age_group'] == age_group
    
    if mask.sum() < 50:
        continue
    
    y_test_age = y_test[mask]
    y_pred_age = ensemble_pred[mask]
    y_proba_age = ensemble_proba[mask]
    
    acc = accuracy_score(y_test_age, y_pred_age)
    prec = precision_score(y_test_age, y_pred_age)
    rec = recall_score(y_test_age, y_pred_age)
    f1 = f1_score(y_test_age, y_pred_age)
    auc = roc_auc_score(y_test_age, y_proba_age)
    
    ensemble_age_results[age_group] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc,
        'n_samples': mask.sum()
    }
    
    print(f"\n{age_group.replace('_', ' ').title()} (n={mask.sum():,}):")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

# Calculate ensemble age fairness
ensemble_range = max([v['accuracy'] for v in ensemble_age_results.values()]) - \
                 min([v['accuracy'] for v in ensemble_age_results.values()])

print("\n" + "="*60)
print("⚖️ AGE FAIRNESS COMPARISON")
print("="*60)

print(f"\nBaseline age accuracy range:  {baseline_range:.4f} ({baseline_range*100:.2f}%)")
print(f"Ensemble age accuracy range:  {ensemble_range:.4f} ({ensemble_range*100:.2f}%)")

improvement = baseline_range - ensemble_range
pct_improvement = (improvement / baseline_range) * 100

if improvement > 0:
    print(f"\n✅ Improvement: {improvement:.4f} ({improvement*100:.2f} percentage points)")
    print(f"   Bias reduction: {pct_improvement:.1f}%")
else:
    print(f"\n⚠️  Change: {improvement:.4f} ({improvement*100:.2f} percentage points)")
    print(f"   (Ensemble did not reduce bias)")

# Detailed comparison
print("\n" + "="*60)
print("📊 DETAILED BASELINE vs ENSEMBLE COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Age_Group': list(baseline_age_results.keys()),
    'Baseline_Acc': [v['accuracy'] for v in baseline_age_results.values()],
    'Ensemble_Acc': [ensemble_age_results[k]['accuracy'] for k in baseline_age_results.keys()],
    'Baseline_Recall': [v['recall'] for v in baseline_age_results.values()],
    'Ensemble_Recall': [ensemble_age_results[k]['recall'] for k in baseline_age_results.keys()]
})

comparison_df['Acc_Change'] = comparison_df['Ensemble_Acc'] - comparison_df['Baseline_Acc']
comparison_df['Recall_Change'] = comparison_df['Ensemble_Recall'] - comparison_df['Baseline_Recall']

print("\n" + comparison_df.to_string(index=False))

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n💾 Saving ensemble results...")

os.makedirs('results', exist_ok=True)

# Save ensemble age results
ensemble_df = pd.DataFrame(ensemble_age_results).T
ensemble_df.to_csv('results/ensemble_age_results.csv')
print("✅ Saved: results/ensemble_age_results.csv")

# Save comparison
comparison_df.to_csv('results/baseline_vs_ensemble.csv', index=False)
print("✅ Saved: results/baseline_vs_ensemble.csv")

# Save summary
summary = {
    'baseline_overall_acc': baseline_overall_acc,
    'ensemble_overall_acc': overall_acc,
    'baseline_age_gap': baseline_range,
    'ensemble_age_gap': ensemble_range,
    'gap_reduction': improvement,
    'gap_reduction_pct': pct_improvement
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('results/ensemble_summary.csv', index=False)
print("✅ Saved: results/ensemble_summary.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ ADVANCED MITIGATION ANALYSIS COMPLETE!")
print("="*60)

print(f"\n📊 Final Results Summary:")
print(f"\n   OVERALL ACCURACY:")
print(f"      Baseline: {baseline_overall_acc*100:.2f}%")
print(f"      Ensemble: {overall_acc*100:.2f}%")

print(f"\n   AGE FAIRNESS GAP:")
print(f"      Baseline: {baseline_range*100:.2f}%")
print(f"      Ensemble: {ensemble_range*100:.2f}%")

if improvement > 0:
    print(f"\n   ✅ SUCCESS: Reduced age bias by {pct_improvement:.1f}%")
    print(f"      ({improvement*100:.2f} percentage points)")
else:
    print(f"\n   ⚠️  Age-stratified ensemble did not improve fairness")
    print(f"      Age bias remains structural, not algorithmic")

print("\n📁 All results saved to results/ folder")
print("\n🎯 Next: Create visualizations for your paper!")