# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scipy",
# ]
# ///

"""Single linear-scale panel: zero claims all sit at the bottom,
large claims tower above. The contrast IS the message.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tweedie_utils import tweedie_random

SAVE = True
DPI = 150
SEED = 42

MU = 293.0
PHI = 174.0
P_VAL = 1.574
N = 300


def main():
    rng = np.random.default_rng(SEED)
    y = tweedie_random(MU, PHI, P_VAL, size=N, rng=rng)

    mu_est = y.mean()
    contrib = (y - mu_est) ** 2 / (mu_est ** P_VAL)
    zero = y <= 1e-9

    idx = np.argsort(contrib)
    contrib_s = contrib[idx]
    zero_s = zero[idx]
    n_zero = zero_s.sum()
    n_pos = N - n_zero

    pearson = contrib.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(N)

    ax.scatter(x[zero_s], contrib_s[zero_s],
               s=10, color="#d55e00", alpha=0.6,
               label=f"Zero claims ({n_zero})")

    ax.scatter(x[~zero_s], contrib_s[~zero_s],
               s=18, color="#0072b2", alpha=0.7,
               label=f"Positive claims ({n_pos})")

    ax.axhline(PHI, color="#009e73", ls="-", lw=2, zorder=5,
               label=f"True phi = {PHI}")
    ax.axhline(pearson, color="#cc79a7", ls="--", lw=2, zorder=5,
               label=f"Pearson phi = {pearson:.0f}")

    ax.set_xlabel("Observation index (sorted by contribution)", fontsize=12)
    ax.set_ylabel("Contribution to Pearson phi", fontsize=12)
    ax.set_title("Why Pearson phi inflates for zero-inflated data",
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="upper right", markerscale=2)
    ax.set_xlim(-10, N + 10)

    # annotation below axis
    fig.text(0.5, 0.02,
             "93% of policies (zero claims) contribute nearly nothing. "
             "7% of policies (large claims) dominate the estimate.",
             ha="center", fontsize=12, fontstyle="italic",
             bbox=dict(facecolor="wheat", pad=5, boxstyle="round"))

    if SAVE:
        out_dir = (Path(__file__).resolve().parent.parent.parent.parent
                   / "posts" / "images")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "fig_pearson_mechanism.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"saved -> {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
