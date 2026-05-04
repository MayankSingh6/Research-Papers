#!/bin/bash

echo "Running CVD Age Bias Research Pipeline..."

echo "Step 1: Exploring dataset"
python experiments/01_explore_data.py

echo "Step 2: Cleaning data and engineering features"
python experiments/02_clean_data.py

echo "Step 3: Running baseline fairness analysis"
python experiments/03_fairness_analysis.py

echo "Step 4: Running bias mitigation experiments"
python experiments/04_bias_mitigation.py

echo "Step 5: Running advanced mitigation experiment"
python experiments/05_advanced_mitigation.py

echo "Step 6: Creating visualizations"
python experiments/06_create_visualizations.py

echo "Step 7: Creating additional heatmaps"
python experiments/cvd_heatmaps_generator.py

echo "Pipeline complete."