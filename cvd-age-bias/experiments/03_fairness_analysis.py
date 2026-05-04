import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)
import os

print("="*60)
print("⚖️ FAIRNESS ANALYSIS - CARDIOVASCULAR DISEASE PREDICTION")
print("="*60)

# ============================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================
print("\n📂 Loading cleaned data...")
df = pd.read_csv('data/kaggle/cardio_cleaned.csv')
print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features")

# ============================================================
# STEP 2: PREPARE FEATURES FOR MODEL
# ============================================================
print("\n🔧 Preparing features for model training...")

# Select numeric features (model can't use categorical strings)
feature_cols = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo',  # blood pressure
    'cholesterol', 'gluc',  # lab values
    'smoke', 'alco', 'active',  # lifestyle
    'bp_risk', 'cholesterol_risk', 'glucose_risk', 'risk_score'  # engineered features
]

X = df[feature_cols]
y = df['cardio']

print(f"   Features selected: {len(feature_cols)}")
print(f"   Target variable: cardio (0=healthy, 1=disease)")

# ============================================================
# STEP 3: SPLIT DATA INTO TRAIN/TEST
# ============================================================
print("\n✂️  Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,  # reproducible results
    stratify=y  # keep same disease ratio in train/test
)

# Keep demographic info for fairness analysis
demo_test = df.loc[X_test.index, ['age_group', 'gender', 'bmi_category']]

print(f"   Training set: {len(X_train):,} records")
print(f"   Test set: {len(X_test):,} records")
print(f"   Test set disease rate: {y_test.mean()*100:.1f}%")

# ============================================================
# STEP 4: TRAIN BASELINE MODEL
# ============================================================
print("\n🤖 Training Random Forest model...")
print("   This may take 30-60 seconds...")

model = RandomForestClassifier(
    n_estimators=100,  # 100 decision trees
    max_depth=10,  # prevent overfitting
    random_state=42,
    n_jobs=-1,  # use all CPU cores
    class_weight='balanced'  # handle any minor imbalances
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ============================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================
print("\n🔮 Making predictions on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of disease

# ============================================================
# STEP 6: OVERALL MODEL PERFORMANCE
# ============================================================
print("\n" + "="*60)
print("📊 OVERALL MODEL PERFORMANCE")
print("="*60)

overall_acc = accuracy_score(y_test, y_pred)
overall_prec = precision_score(y_test, y_pred)
overall_rec = recall_score(y_test, y_pred)
overall_f1 = f1_score(y_test, y_pred)
overall_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"Precision: {overall_prec:.4f} ({overall_prec*100:.2f}%)")
print(f"Recall:    {overall_rec:.4f} ({overall_rec*100:.2f}%)")
print(f"F1-Score:  {overall_f1:.4f}")
print(f"AUC-ROC:   {overall_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives (correctly predicted healthy):  {tn:,}")
print(f"   False Positives (false alarms):                {fp:,}")
print(f"   False Negatives (missed diseases):             {fn:,}")
print(f"   True Positives (correctly caught disease):     {tp:,}")

# ============================================================
# STEP 7: FAIRNESS ANALYSIS - GENDER
# ============================================================
print("\n" + "="*60)
print("⚖️ FAIRNESS ANALYSIS: GENDER")
print("="*60)

gender_results = {}

for gender_val in [0, 1]:
    gender_name = "Female" if gender_val == 0 else "Male"
    
    # Get indices for this gender
    mask = demo_test['gender'] == gender_val
    
    if mask.sum() == 0:
        continue
    
    # Get predictions for this gender
    y_test_gender = y_test[mask]
    y_pred_gender = y_pred[mask]
    y_proba_gender = y_pred_proba[mask]
    
    # Calculate metrics
    acc = accuracy_score(y_test_gender, y_pred_gender)
    prec = precision_score(y_test_gender, y_pred_gender)
    rec = recall_score(y_test_gender, y_pred_gender)
    f1 = f1_score(y_test_gender, y_pred_gender)
    auc = roc_auc_score(y_test_gender, y_proba_gender)
    
    # Confusion matrix
    cm_gender = confusion_matrix(y_test_gender, y_pred_gender)
    tn_g, fp_g, fn_g, tp_g = cm_gender.ravel()
    
    # Store results
    gender_results[gender_name] = {
        'n_samples': mask.sum(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc,
        'true_negatives': tn_g,
        'false_positives': fp_g,
        'false_negatives': fn_g,
        'true_positives': tp_g
    }
    
    # Print results
    print(f"\n{gender_name} Patients (n={mask.sum():,}):")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

# Calculate fairness gaps
print("\n" + "-"*60)
print("⚖️ GENDER FAIRNESS GAPS:")
print("-"*60)

acc_gap = abs(gender_results['Male']['accuracy'] - gender_results['Female']['accuracy'])
rec_gap = abs(gender_results['Male']['recall'] - gender_results['Female']['recall'])
prec_gap = abs(gender_results['Male']['precision'] - gender_results['Female']['precision'])

print(f"Accuracy gap:  {acc_gap:.4f} ({acc_gap*100:.2f} percentage points)")
print(f"Recall gap:    {rec_gap:.4f} ({rec_gap*100:.2f} percentage points)")
print(f"Precision gap: {prec_gap:.4f} ({prec_gap*100:.2f} percentage points)")

if acc_gap < 0.02:
    print("✅ Fairness assessment: LOW bias (< 2% gap)")
elif acc_gap < 0.05:
    print("⚠️  Fairness assessment: MODERATE bias (2-5% gap)")
else:
    print("🚨 Fairness assessment: HIGH bias (> 5% gap)")

# ============================================================
# STEP 8: FAIRNESS ANALYSIS - AGE GROUP
# ============================================================
print("\n" + "="*60)
print("⚖️ FAIRNESS ANALYSIS: AGE GROUP")
print("="*60)

age_results = {}

for age_group in ['young', 'middle_age', 'senior', 'elderly']:
    mask = demo_test['age_group'] == age_group
    
    if mask.sum() < 50:  # Skip if too few samples
        continue
    
    y_test_age = y_test[mask]
    y_pred_age = y_pred[mask]
    y_proba_age = y_pred_proba[mask]
    
    acc = accuracy_score(y_test_age, y_pred_age)
    prec = precision_score(y_test_age, y_pred_age)
    rec = recall_score(y_test_age, y_pred_age)
    f1 = f1_score(y_test_age, y_pred_age)
    auc = roc_auc_score(y_test_age, y_proba_age)
    
    age_results[age_group] = {
        'n_samples': mask.sum(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc
    }
    
    print(f"\n{age_group.replace('_', ' ').title()} (n={mask.sum():,}):")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")

# Calculate age fairness variance
print("\n" + "-"*60)
print("⚖️ AGE FAIRNESS ANALYSIS:")
print("-"*60)

age_accs = [v['accuracy'] for v in age_results.values()]
age_variance = np.std(age_accs)
age_range = max(age_accs) - min(age_accs)

print(f"Accuracy standard deviation: {age_variance:.4f}")
print(f"Accuracy range: {age_range:.4f} ({age_range*100:.2f} percentage points)")

if age_range < 0.03:
    print("✅ Age fairness: LOW variance across age groups")
elif age_range < 0.06:
    print("⚠️  Age fairness: MODERATE variance across age groups")
else:
    print("🚨 Age fairness: HIGH variance across age groups")

# ============================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("📊 TOP 10 MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================
# STEP 10: SAVE RESULTS
# ============================================================
print("\n💾 Saving results...")

os.makedirs('results', exist_ok=True)

# Save gender fairness results
gender_df = pd.DataFrame(gender_results).T
gender_df.to_csv('results/fairness_gender.csv')
print("✅ Saved: results/fairness_gender.csv")

# Save age fairness results
age_df = pd.DataFrame(age_results).T
age_df.to_csv('results/fairness_age.csv')
print("✅ Saved: results/fairness_age.csv")

# Save feature importance
feature_importance.to_csv('results/feature_importance.csv', index=False)
print("✅ Saved: results/feature_importance.csv")

# Save overall metrics
overall_results = {
    'accuracy': overall_acc,
    'precision': overall_prec,
    'recall': overall_rec,
    'f1_score': overall_f1,
    'auc_roc': overall_auc,
    'gender_accuracy_gap': acc_gap,
    'gender_recall_gap': rec_gap,
    'age_accuracy_variance': age_variance,
    'age_accuracy_range': age_range
}

overall_df = pd.DataFrame([overall_results])
overall_df.to_csv('results/overall_metrics.csv', index=False)
print("✅ Saved: results/overall_metrics.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ FAIRNESS ANALYSIS COMPLETE!")
print("="*60)

print(f"\n📊 Key Findings:")
print(f"   Overall Accuracy: {overall_acc*100:.2f}%")
print(f"   Gender Fairness Gap: {acc_gap*100:.2f} percentage points")
print(f"   Age Fairness Range: {age_range*100:.2f} percentage points")

print(f"\n📁 All results saved to results/ folder")
print(f"   - fairness_gender.csv")
print(f"   - fairness_age.csv")
print(f"   - feature_importance.csv")
print(f"   - overall_metrics.csv")

print("\n🎯 Next step: Review results and create visualizations!")