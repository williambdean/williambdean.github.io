# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///

"""Figure 1: Profile log-likelihood for phi, showing MLE vs Pearson estimates.

Generates synthetic Tweedie data with known parameters, then computes
the log-likelihood across a grid of phi values. Marks both the MLE
and Pearson dispersion estimates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tweedie_utils import tweedie_logp_series, tweedie_random

# Output path
OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pearson_phi(y, mu, p):
    """Compute Pearson dispersion estimate for Tweedie (unweighted)."""
    resid_sq = (y - mu) ** 2 / (mu**p)
    return resid_sq.sum() / (len(y) - 1)


def profile_likelihood_phi(mu, phi_true, p, y, phi_grid):
    """Compute profile log-likelihood across phi grid at fixed mu, p."""
    ll = np.array([
        tweedie_logp_series(y, mu, phi, p).sum()
        for phi in phi_grid
    ])
    return ll


def generate_data(n, mu, phi, p, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    y = tweedie_random(mu, phi, p, size=n, rng=rng)
    return y


rng = np.random.default_rng(42)

datasets = [
    {"name": "dataCar", "mu": 293.0, "phi": 174.0, "p": 1.574, "n": 15000},
    {"name": "High-Inflation", "mu": 218.0, "phi": 800.0, "p": 1.633, "n": 15000},
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ds in zip(axes, datasets):
    name = ds["name"]
    mu_ds = ds["mu"]
    phi_true = ds["phi"]
    p_ds = ds["p"]

    y = generate_data(ds["n"], mu_ds, phi_true, p_ds, rng=rng)

    phi_grid = np.linspace(phi_true * 0.2, phi_true * 12, 100)
    ll = profile_likelihood_phi(mu_ds, phi_true, p_ds, y, phi_grid)

    phi_pearson = pearson_phi(y, mu_ds, p_ds)
    ll_max = ll.max()
    ll_mle = tweedie_logp_series(y, mu_ds, phi_true, p_ds).sum()
    ll_pearson = tweedie_logp_series(y, mu_ds, phi_pearson, p_ds).sum()

    delta_ll = ll_mle - ll_pearson

    ax.plot(phi_grid, ll, "b-", linewidth=2, label="Log-likelihood")
    ax.axvline(phi_true, color="green", linestyle="--", linewidth=2,
               label=f"MLE φ = {phi_true:.0f}")
    ax.axvline(phi_pearson, color="red", linestyle=":", linewidth=2,
               label=f"Pearson φ = {phi_pearson:.0f}")

    ax.fill_between(phi_grid, ll.min(), ll, alpha=0.1, color="blue")
    ax.set_xlabel("φ (dispersion)")
    ax.set_ylabel("Log-likelihood")
    ax.set_title(f"{name}\nμ={mu_ds}, p={p_ds}", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, phi_grid.max())

    ann_y = ll.min() + 0.05 * (ll_max - ll.min())
    ax.annotate(f"ΔLL = {delta_ll:.0f}\n(MLE is $e^{{{delta_ll:.0f}}}\\times$ more probable)",
                xy=(phi_pearson, ll_pearson),
                xytext=(phi_pearson * 0.5, ann_y),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
                color="red")

plt.suptitle("Profile Log-Likelihood for φ at Fixed μ and p\n"
             "MLE vs Pearson Dispersion Estimate",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_profile_likelihood.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_profile_likelihood.png'}")
