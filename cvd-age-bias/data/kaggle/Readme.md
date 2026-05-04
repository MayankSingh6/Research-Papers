# 📊 Dataset Information

This project uses the **Cardiovascular Disease Dataset** from Kaggle.

---

## 📥 Download Dataset

Download the dataset from:

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

After downloading, place the file here:

data/kaggle/cardio_train.csv

---

## ⚙️ Data Processing

The raw dataset is not included in this repository.

To generate the cleaned dataset, run:

python experiments/02_clean_data.py

This will:
- Clean invalid records and outliers  
- Convert age into years  
- Create age groups:
  - young (≤40)  
  - middle_age (41–50)  
  - senior (51–60)  
  - elderly (61+)  
- Generate features like BMI and risk scores  

---

## 📁 Output File

The cleaned dataset will be saved as:

data/kaggle/cardio_cleaned.csv

This file is used for all experiments.

---

## 📊 Dataset Summary

- Total records: ~70,000  
- Cleaned records: ~68,334  
- Retention: ~97.6%  

Target variable:
- cardio → 0 (Healthy), 1 (Disease)

---

## ⚠️ Notes

- Do NOT upload dataset files to GitHub  
- Make sure the file path is correct before running scripts  
- All code expects data in:

data/kaggle/