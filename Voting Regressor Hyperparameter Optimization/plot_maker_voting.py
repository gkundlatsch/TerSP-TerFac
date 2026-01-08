# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:36:58 2025

@author: Guilherme Kundlatsch
"""

# plot_from_excel.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EXCEL_PATH = "voting_weight_sweep_f.xlsx"  # sorted file exported by the training script
PNG_OUT    = "voting_train_vs_test_r2_f4.png"

# --- Load summary ---
summary_df = pd.read_excel(EXCEL_PATH)

# If the file is not sorted (e.g., you point to the _raw file), we enforce the same sorting:
summary_df = summary_df.sort_values('test_r2_mean', ascending=False).reset_index(drop=True)

labels   = summary_df['combination']
x        = np.arange(len(labels))
width    = 0.35
means_tr = summary_df['train_r2_mean']
errs_tr  = summary_df['train_r2_std']
means_te = summary_df['test_r2_mean']
errs_te  = summary_df['test_r2_std']

fig, ax = plt.subplots(figsize=(10, 6))

fig.subplots_adjust(
    left=0.15,   # space for y-label
    right=0.95,  # space for ticks/legend
    top=0.9,     # space for title
    bottom=0.25  # space for x-labels
)

width = 0.32
gap   = 0.05  

rects1 = ax.bar(
    x - (width/2 + gap/2),
    means_tr,
    width,
    yerr=errs_tr,
    capsize=6,
    label='Train R²',
    color='darkorange',
    edgecolor='black',
    alpha=0.8
)

rects2 = ax.bar(
    x + (width/2 + gap/2),
    means_te,
    width,
    yerr=errs_te,
    capsize=6,
    label='Test R²',
    color='steelblue',
    edgecolor='black',
    alpha=0.8
)

ax.set_xlabel("Model Combination", fontsize=12)
ax.set_ylabel("Mean R\u00B2 (5-fold)", fontsize=12)
ax.set_title("Train vs. Test R\u00B2 by Model Combination", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.01, 0.2))

ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.0, -0.3),  # center it horizontally, push it down
    ncol=2,                       # put entries side-by-side (adjust as you like)
    fontsize=10,
    frameon=False
)

ax.grid(axis='y', linestyle='--', alpha=0.5)

def annotate_bars(rects, vals, errs, pad_pts=4):
    # place text at bar_height + error + a few points
    for r, v, e in zip(rects, vals, errs):
        e = 0 if np.isnan(e) else e
        ax.annotate(
            f"{v:.3f}",
            xy=(r.get_x() + r.get_width()/2, r.get_height() + e),
            xytext=(0, pad_pts), textcoords="offset points",
            ha='center', va='bottom', fontsize=8
        )

annotate_bars(rects1, means_tr, errs_tr)
annotate_bars(rects2, means_te, errs_te)


fig.tight_layout()
plt.savefig(PNG_OUT, dpi=300)
plt.show()
print(f"Figure saved to {PNG_OUT}")
