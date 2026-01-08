# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:57:46 2025

@author: Guilherme Kundlatsch
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ——— Load your data ———
df = pd.read_excel('complete dataset.xlsx')
y_test   = df['Average Strength']
test_pred = df['Predicted TS']

# ——— Compute R² ———
test_r2  = r2_score(y_test, test_pred)

# ——— Prepare DataFrame for plotting ———
plot_df = pd.DataFrame({
    'Average Strength Test':  y_test,
    'Average Strength Model': test_pred
}).sort_values(by='Average Strength Test')
plot_df['Number'] = range(1, len(plot_df) + 1)

# ——— Plot ———
fold = 1
fig = plt.figure(figsize=(12, 6))

# Experimental (left)
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(plot_df['Number'], plot_df['Average Strength Test'])
ax1.set_xlabel('Number')
ax1.set_ylabel('Experimental Average Strength')
ax1.set_title('Experimental')
ax1.set_ylim(-10, 300)
ax1.axhline(40, color='red',   linestyle='dashed')

# Predictions (right)
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(plot_df['Number'], plot_df['Average Strength Model'], color='orange')
ax2.set_xlabel('Number')
ax2.set_ylabel('Predicted Average Strength')
ax2.set_title('Predictions')
ax2.text(0.1, 0.9, f'R\u00B2: {test_r2:.3f}', transform=ax2.transAxes, va='top')
ax2.set_ylim(-10, 300)
ax2.axhline(40, color='red',   linestyle='dashed')

plt.tight_layout()
plt.savefig(f'Experimental_vs_Predicted_fold_{fold}.png', dpi=1200)
plt.close(fig)
