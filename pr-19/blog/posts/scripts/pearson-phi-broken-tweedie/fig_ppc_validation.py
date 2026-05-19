# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///

"""Figure 2: PPC validation — observed vs predicted statistics.

Bar charts comparing observed values with posterior predictive means
and 95% intervals for both datasets.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ppc_data = {
    "dataCar (Intercept)": {
        "statistics": ["Zero Rate", "Total Claim", "Max Claim", "N Nonzero"],
        "observed": [0.932, 9_314_604, 55_922, 4624],
        "predicted": [0.932, 9_302_275, 22_149, 4590],
        "ci_low": [0.930, 8_782_116, 17_285, 4422],
        "ci_high": [0.935, 9_844_387, 30_015, 4766],
    },
    "dataCar (GLM)": {
        "statistics": ["Zero Rate", "Total Claim", "Max Claim", "N Nonzero"],
        "observed": [0.932, 9_314_604, 55_922, 4624],
        "predicted": [0.932, 9_383_100, 25_160, 4586],
        "ci_low": [0.930, 8_806_246, 18_595, 4397],
        "ci_high": [0.935, 9_964_676, 37_065, 4761],
    },
    "French TPL (Intercept)": {
        "statistics": ["Zero Rate", "Total Claim", "Max Claim", "N Nonzero"],
        "observed": [0.963, 6_540_679, 1_404_186, 2237],
        "predicted": [0.963, 6_520_860, 35_413, 2216],
        "ci_low": [0.961, 5_908_420, 25_912, 2092],
        "ci_high": [0.965, 7_195_012, 50_532, 2339],
    },
    "French TPL (GLM)": {
        "statistics": ["Zero Rate", "Total Claim", "Max Claim", "N Nonzero"],
        "observed": [0.963, 6_540_679, 1_404_186, 2237],
        "predicted": [0.963, 6_663_320, 103_702, 2220],
        "ci_low": [0.961, 5_989_462, 45_447, 2097],
        "ci_high": [0.965, 7_573_082, 257_210, 2357],
    },
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#4C72B0", "#DD8452"]

for row_idx, (name, data) in enumerate(ppc_data.items()):
    ax = axes[row_idx // 2][row_idx % 2]
    stats = data["statistics"]
    x = np.arange(len(stats))
    width = 0.35

    obs = [v / max(data["observed"]) if max(data["observed"]) > 0 else v
           for v in data["observed"]]
    pred = [v / max(data["observed"]) if max(data["observed"]) > 0 else v
            for v in data["predicted"]]
    ci_l = [v / max(data["observed"]) if max(data["observed"]) > 0 else v
            for v in data["ci_low"]]
    ci_h = [v / max(data["observed"]) if max(data["observed"]) > 0 else v
            for v in data["ci_high"]]

    ax.bar(x - width / 2, obs, width, label="Observed", color=colors[0], alpha=0.8)
    ax.bar(x + width / 2, pred, width, label="Posterior Mean", color=colors[1], alpha=0.8)

    yerr_low = np.array(pred) - np.array(ci_l)
    yerr_high = np.array(ci_h) - np.array(pred)
    ax.errorbar(x + width / 2, pred,
                yerr=[yerr_low, yerr_high],
                fmt="none", color="black", capsize=3, capthick=1)

    ax.set_xticks(x)
    ax.set_xticklabels(stats, fontsize=9)
    ax.set_ylabel("Normalized Value")
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8)

    # Annotate raw values for zero rate
    zero_idx = 0
    ax.annotate(f"{data['observed'][zero_idx]:.1%}",
                xy=(x[zero_idx] - width / 2, obs[zero_idx]),
                ha="center", va="bottom", fontsize=7)
    ax.annotate(f"{data['predicted'][zero_idx]:.1%}",
                xy=(x[zero_idx] + width / 2, pred[zero_idx]),
                ha="center", va="bottom", fontsize=7)

plt.suptitle("Posterior Predictive Check: Observed vs Predicted",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_ppc_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_ppc_validation.png'}")
