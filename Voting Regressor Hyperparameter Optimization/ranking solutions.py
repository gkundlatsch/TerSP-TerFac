# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:07:22 2025

@author: Guilherme Kundlatsch
"""

import pandas as pd

# Load the Excel file
df = pd.read_excel("grid_search_all5params_with_train_test_R2_big.xlsx")

# Calculate the difference
df["difference"] = df["mean_train_score"] - df["mean_test_score"]

# Keep only rows where the difference is less than 0.3
filtered = df[df["difference"] < 0.3]

# Rank by best test score
ranked = filtered.sort_values(by="mean_test_score", ascending=False)

# Save everything (all parameters, scores, and difference)
ranked.to_excel("best_solutions.xlsx", index=False)
