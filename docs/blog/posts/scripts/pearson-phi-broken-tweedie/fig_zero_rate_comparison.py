# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

"""Figure 3: Zero rate comparison — MLE vs Pearson vs Observed.

Shows how the Pearson dispersion estimator systematically over-predicts
the zero rate compared to MLE and observed data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = ["dataCar"]
models = ["Observed", "Posterior\nMean", "Pearson\n(statsmodels)"]

zero_rates = np.array([
    [0.932, 0.932, 0.990],      # dataCar
])

colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(datasets))
width = 0.18

for i in range(len(models)):
    vals = zero_rates[:, i]
    mask = ~np.isnan(vals)
    ax.bar(x[mask] + (i - 1.5) * width, vals[mask], width,
           label=models[i], color=colors[i], alpha=0.85, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("Proportion of Zero Claims", fontsize=11)
ax.set_title("Predicted Zero Rate by Model vs Observed Data", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0.90, 1.01)

for i in range(len(datasets)):
    for j in range(len(models)):
        val = zero_rates[i, j]
        if not np.isnan(val):
            ax.annotate(f"{val:.1%}",
                        xy=(x[i] + (j - 1.5) * width, val),
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_zero_rate_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_zero_rate_comparison.png'}")
