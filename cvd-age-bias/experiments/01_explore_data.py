import pandas as pd
import numpy as np
import os

print("="*60)
print("📊 CARDIOVASCULAR DATASET EXPLORATION")
print("="*60)

# Load data
print("\n📂 Loading data from data/kaggle/cardio_train.csv...")

try:
    df = pd.read_csv('data/kaggle/cardio_train.csv', delimiter=';')
    print(f"✅ Successfully loaded {len(df):,} records!")
except FileNotFoundError:
    print("❌ Error: File not found!")
    print("   Make sure cardio_train.csv is in data/kaggle/ folder")
    exit()

print(f"✅ Columns: {len(df.columns)}")

# Show first few rows
print("\n" + "="*60)
print("FIRST 5 ROWS OF DATA")
print("="*60)
print(df.head())

# Show column names
print("\n" + "="*60)
print("COLUMN NAMES")
print("="*60)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# Basic statistics
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Check for missing values
print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print("⚠️ Missing values found:")
    print(missing[missing > 0])

# Target variable distribution
print("\n" + "="*60)
print("CARDIOVASCULAR DISEASE DISTRIBUTION")
print("="*60)
print(df['cardio'].value_counts())
print(f"\nPercentage with disease: {df['cardio'].mean()*100:.1f}%")

# Age statistics
print("\n" + "="*60)
print("AGE STATISTICS")
print("="*60)
df['age_years'] = (df['age'] / 365.25).round()
print(f"Youngest patient: {df['age_years'].min():.0f} years")
print(f"Oldest patient: {df['age_years'].max():.0f} years")
print(f"Average age: {df['age_years'].mean():.1f} years")

# Gender distribution
print("\n" + "="*60)
print("GENDER DISTRIBUTION")
print("="*60)
gender_counts = df['gender'].value_counts()
print(f"Gender 1 (Female): {gender_counts[1]:,}")
print(f"Gender 2 (Male): {gender_counts[2]:,}")

# Save summary
print("\n💾 Saving summary statistics...")
os.makedirs('results', exist_ok=True)
df.describe().to_csv('results/data_summary.csv')
print("✅ Saved to: results/data_summary.csv")

print("\n" + "="*60)
print("✅ EXPLORATION COMPLETE!")
print("="*60)