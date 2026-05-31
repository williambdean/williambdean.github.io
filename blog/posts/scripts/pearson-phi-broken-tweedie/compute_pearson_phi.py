# /// script
# dependencies = ["numpy", "pandas", "scipy", "requests"]
# ///

"""Compute Pearson φ for Tweedie GLMs on the dataCar dataset.

Verifies the blog post's dataCar φ(Pearson)=1,227 using the
exposure-weighted formula:
  χ² = Σ w_i * (y_i - μ)^2 / μ^p
  φ = χ² / (n - 1)

where y_i = ClaimAmount_i / Exposure_i and μ is exposure-weighted mean.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tweedie_utils import tweedie_logp_series, tweedie_random


def pearson_phi_weighted(y, mu, p, weights):
    """Weighted Pearson dispersion estimate.

    χ² = Σ w_i * (y_i - μ)^2 / μ^p
    φ = χ² / (n - 1)
    """
    resid_sq = (y - mu) ** 2 / (mu**p)
    return (weights * resid_sq).sum() / (len(y) - 1)


def load_datacar(path="/tmp/dataCar.csv"):
    df = pd.read_csv(path)
    claim = df["claimcst0"].values.astype(float)
    exposure = df["exposure"].values.astype(float)
    return claim, exposure


if __name__ == "__main__":
    # --- dataCar verification ---
    claim, exposure = load_datacar()
    y = claim / exposure
    n = len(y)
    mu = np.sum(claim) / np.sum(exposure)
    zero_rate = np.mean(claim == 0)

    print("=" * 60)
    print("  dataCar Dataset (67,856 policies)")
    print("=" * 60)
    print(f"  Weighted mean (pure premium) = ${mu:.2f}")
    print(f"  Zero rate                    = {zero_rate:.4f} ({zero_rate*100:.1f}%)")
    print(f"  Total exposure               = {exposure.sum():,.1f} years")

    # Pearson φ at blog's p=1.574
    p_blog = 1.574
    phi_pearson = pearson_phi_weighted(y, mu, p_blog, exposure)

    print(f"\n  At p={p_blog}:")
    print(f"    φ(Pearson, weighted) = {phi_pearson:.1f}")
    print(f"    Blog claims          = 1,227")
    print(f"    Match?               = {'YES ✓' if abs(phi_pearson - 1227) / 1227 < 0.01 else 'NO ✗'}")

    # Pearson φ at various p
    print(f"\n  φ(Pearson) across power parameter p:")
    for p in [1.1, 1.2, 1.3, 1.4, 1.5, 1.574, 1.6, 1.7, 1.8, 1.9]:
        phi_p = pearson_phi_weighted(y, mu, p, exposure)
        print(f"    p={p:.3f}: φ(Pearson) = {phi_p:,.1f}")
