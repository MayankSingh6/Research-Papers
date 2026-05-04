import pandas as pd
import numpy as np
import os

print("="*60)
print("🧹 DATA CLEANING & FEATURE ENGINEERING")
print("="*60)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/kaggle/cardio_train.csv', delimiter=';')
original_size = len(df)

print(f"✅ Loaded {original_size:,} records")

# Step 1: Convert age from days to years
print("\n🔄 Converting age from days to years...")
df['age_years'] = (df['age'] / 365.25).round().astype(int)
print(f"   Age range: {df['age_years'].min()}-{df['age_years'].max()} years")

# Step 2: Create age groups
print("\n📊 Creating age groups...")
df['age_group'] = pd.cut(df['age_years'], 
                          bins=[0, 40, 50, 60, 100],
                          labels=['young', 'middle_age', 'senior', 'elderly'])
print(f"   Created 4 age groups")

# Step 3: Recode gender (1=female, 2=male → 0=female, 1=male)
print("\n👤 Recoding gender...")
df['gender'] = df['gender'] - 1
print(f"   Gender recoded: 0=Female, 1=Male")

# Step 4: Calculate BMI
print("\n⚖️  Calculating BMI...")
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
print(f"   BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")

# Step 5: Create BMI categories
print("\n📊 Creating BMI categories...")
df['bmi_category'] = pd.cut(df['bmi'],
                             bins=[0, 18.5, 25, 30, 100],
                             labels=['underweight', 'normal', 'overweight', 'obese'])

# Step 6: Remove outliers
print("\n🧹 Removing outliers...")

before_cleaning = len(df)

# Remove impossible blood pressure values
df = df[df['ap_hi'] >= 80]
df = df[df['ap_hi'] <= 200]
df = df[df['ap_lo'] >= 40]
df = df[df['ap_lo'] <= 130]

# Remove impossible height/weight
df = df[df['height'] >= 140]
df = df[df['height'] <= 220]
df = df[df['weight'] >= 30]
df = df[df['weight'] <= 200]

# Remove impossible BMI
df = df[df['bmi'] >= 15]
df = df[df['bmi'] <= 50]

removed = before_cleaning - len(df)
print(f"   Removed {removed:,} outliers")
print(f"   Remaining: {len(df):,} records ({len(df)/original_size*100:.1f}% of original)")

# Step 7: Create risk indicators
print("\n⚠️  Creating risk indicators...")
df['bp_risk'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)
df['cholesterol_risk'] = (df['cholesterol'] >= 2).astype(int)
df['glucose_risk'] = (df['gluc'] >= 2).astype(int)
df['risk_score'] = (df['bp_risk'] + df['cholesterol_risk'] + 
                    df['glucose_risk'] + df['smoke'] + df['alco'])
print(f"   Created 4 risk indicators")

# Step 8: Save cleaned data
print("\n💾 Saving cleaned data...")
output_file = 'data/kaggle/cardio_cleaned.csv'
df.to_csv(output_file, index=False)

file_size_mb = os.path.getsize(output_file) / (1024*1024)
print(f"✅ Saved to: {output_file}")
print(f"   File size: {file_size_mb:.1f} MB")

# Summary
print("\n" + "="*60)
print("📊 CLEANED DATASET SUMMARY")
print("="*60)
print(f"Total records: {len(df):,}")
print(f"Total features: {len(df.columns)}")

print(f"\n👥 Age group distribution:")
print(df['age_group'].value_counts().sort_index())

print(f"\n⚖️  BMI category distribution:")
print(df['bmi_category'].value_counts().sort_index())

print(f"\n❤️  Cardiovascular disease:")
print(f"   Healthy: {(df['cardio'] == 0).sum():,} ({(df['cardio'] == 0).mean()*100:.1f}%)")
print(f"   Disease: {(df['cardio'] == 1).sum():,} ({(df['cardio'] == 1).mean()*100:.1f}%)")

print("\n" + "="*60)
print("✅ CLEANING COMPLETE!")
print("="*60)