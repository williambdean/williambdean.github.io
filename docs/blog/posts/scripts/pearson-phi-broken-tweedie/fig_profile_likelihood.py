# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///

"""Figure 1: Profile log-likelihood for phi, showing MLE vs Pearson estimates.

Generates synthetic Tweedie data with known parameters, then computes
the log-likelihood across a grid of phi values. Marks both the MLE
and Pearson dispersion estimates.
"""

from pathlib import Path

import sys

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
    {
        "name": "dataCar",
        "mu": 293.0, "phi": 174.0, "p": 1.574, "n": 15000,
        "x_max": 600,
        "phi_pearson_blog": 1227,  # From actual dataCar data (blog post)
    },
    {
        "name": "High-Inflation",
        "mu": 218.0, "phi": 800.0, "p": 1.633, "n": 15000,
        "x_max": 3500,
        "phi_pearson_blog": None,  # Compute from synthetic data
    },
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ds in zip(axes, datasets):
    name = ds["name"]
    mu_ds = ds["mu"]
    phi_true = ds["phi"]
    p_ds = ds["p"]
    x_max = ds["x_max"]

    y = generate_data(ds["n"], mu_ds, phi_true, p_ds, rng=rng)

    # Finer grid over tighter range to show peak curvature
    phi_grid = np.linspace(phi_true * 0.1, x_max, 150)
    ll = profile_likelihood_phi(mu_ds, phi_true, p_ds, y, phi_grid)

    # Use blog's known Pearson φ for dataCar; compute from data otherwise
    phi_pearson = ds.get("phi_pearson_blog") or pearson_phi(y, mu_ds, p_ds)
    ll_pearson = tweedie_logp_series(y, mu_ds, phi_pearson, p_ds).sum()
    ll_mle = tweedie_logp_series(y, mu_ds, phi_true, p_ds).sum()
    delta_ll = ll_mle - ll_pearson

    exponent_10 = delta_ll / np.log(10)
    print(f"[{name}] phi_pearson={phi_pearson:.0f}  phi_mle={phi_true:.0f}  "
          f"delta_ll={delta_ll:.1f}  e^{delta_ll:.0f} = 10^{exponent_10:.0f}",
          file=sys.stderr)

    ax.plot(phi_grid, ll, "b-", linewidth=2, label="Log-likelihood")
    ax.fill_between(phi_grid, ll.min(), ll, alpha=0.1, color="blue")

    # MLE at the peak
    ax.axvline(phi_true, color="green", linestyle="--", linewidth=2,
               label=f"MLE φ = {phi_true:.0f}")

    # Pearson — drawn on-screen or annotated off-screen
    if phi_pearson <= x_max * 0.95:
        ax.axvline(phi_pearson, color="red", linestyle=":", linewidth=2,
                   label=f"Pearson φ = {phi_pearson:.0f}")
        # Small annotation near the Pearson line
        ax.annotate(f"ΔLL = {delta_ll:.0f}",
                    xy=(phi_pearson, ll_pearson),
                    xytext=(phi_pearson * 1.12, ll_pearson + 8),
                    fontsize=8, color="red", ha="left",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    else:
        # Arrow at right edge, y in axes fraction for guaranteed visibility
        ax.annotate(f"Pearson φ = {phi_pearson:.0f} →",
                    xy=(x_max * 0.96, 0.08),
                    xycoords=("data", "axes fraction"),
                    ha="right", fontsize=8, color="red")
        # ΔLL text box in upper-left (axes coords — clear of everything)
        bbox = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85)
        ax.text(0.03, 0.95, f"ΔLL = {delta_ll:.0f}",
                transform=ax.transAxes, fontsize=9, color="red",
                va="top", ha="left", bbox=bbox)

    ax.set_xlabel("φ (dispersion)")
    ax.set_ylabel("Log-likelihood")
    ax.set_title(f"{name}\nμ={mu_ds}, p={p_ds}", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, x_max)

plt.subplots_adjust(top=0.83)
plt.suptitle("Profile Likelihood for φ at Fixed μ and p\n"
             "Tighter view shows peak; Pearson off-screen where likelihood is near-zero",
             fontsize=12)
plt.savefig(OUT_DIR / "fig_profile_likelihood.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR / 'fig_profile_likelihood.png'}")
