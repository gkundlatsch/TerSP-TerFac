# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:30:47 2025

@author: Guilherme Kundlatsch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD EXCEL FILE
# ────────────────────────────────────────────────────────────────────────────────

# Make sure this path points to your actual file:
excel_path = "grid_search_all5params_with_train_test_R2_small_f.xlsx"
df = pd.read_excel(excel_path)

# ────────────────────────────────────────────────────────────────────────────────
# 2) CREATE PIVOT TABLES FOR TEST AND TRAIN R²
# ────────────────────────────────────────────────────────────────────────────────

pivot_test = df.pivot_table(
    index=["param_max_depth", "param_min_child_weight", "param_subsample"],
    columns=["param_n_estimators", "param_learning_rate"],
    values="mean_test_score"
)

pivot_train = df.pivot_table(
    index=["param_max_depth", "param_min_child_weight", "param_subsample"],
    columns=["param_n_estimators", "param_learning_rate"],
    values="mean_train_score"
)

row_tuples = pivot_test.index.to_list()
col_tuples = pivot_test.columns.to_list()

row_labels = [
    f"d={d}, cw={cw}, sub={sub:.2f}"
    for (d, cw, sub) in row_tuples
]
col_labels = [
    f"n={n}, lr={lr:.4f}"
    for (n, lr) in col_tuples
]

Z_test  = pivot_test.values
Z_train = pivot_train.values

n_rows, n_cols = Z_test.shape

# ────────────────────────────────────────────────────────────────────────────────
# 3) PLOT HEATMAP + HATCH FOR (train R² − test R² > 0.30)
# ────────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5.5))

vmin = df["mean_test_score"].min()
vmax = df["mean_test_score"].max()

im = ax.imshow(
    Z_test,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)

ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(row_labels, fontsize=9)

ax.set_xlabel("N Estimators, Learning Rate", fontsize=10)
ax.set_ylabel("Max Depth, Min Child Weight, Subsample", fontsize=10)
#ax.set_title("CV mean test R² (hatch where train R² − test R² > 0.3)", fontsize=14)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean test R²", rotation=270, labelpad=15, fontsize=10)
cbar.ax.tick_params(labelsize=7)

gap_threshold = 0.3
train_threshold = 0.9

for i in range(n_rows):
    for j in range(n_cols):
        train_val = Z_train[i, j]
        test_val  = Z_test[i, j]
        gap = train_val - test_val

        # overfitting gap hatch
        if gap > gap_threshold:
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False,
                hatch="///",
                edgecolor="black",
                linewidth=0
            ))

        # high train‐R² hatch
        if train_val > train_threshold:
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False,
                hatch="///",
                edgecolor="black",
                linewidth=0
            ))

target_row = (5, 3, 0.65)          # (max_depth, min_child_weight, subsample)
target_col = (4000, 0.001)         # (n_estimators, learning_rate)

try:
    i = row_tuples.index(target_row)
    j = col_tuples.index(target_col)

    ax.add_patch(Rectangle(
        (j - 0.5, i - 0.5), 1, 1,           # position (x,y), width, height
        fill=False,
        edgecolor='black',
        linewidth=2.5
    ))
except ValueError:
    print("Target combination not found in pivot table.")


plt.tight_layout()
plt.show()
fig.savefig("small_heatmap_gap_hatch15.png", dpi=400)
plt.close(fig)
