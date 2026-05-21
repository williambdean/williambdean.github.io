# /// script
# dependencies = ["pymc", "nutpie", "numpy", "pytensor", "matplotlib"]
# ///

"""Time the Tweedie model sampling with nutpie.

Uses pymc.dims with xtensor-based logp and compound dist.
"""

import time

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.xtensor import as_xtensor

from tweedie_utils import tweedie_random


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    """Tweedie log-pdf via series expansion (Dunn & Smyth 2005).

    Uses pytensor.xtensor operations throughout.
    """
    j = as_xtensor(
        pt.arange(1, n_terms + 1, dtype=pytensor.config.floatX),
        dims=("term",),
    )
    alpha = (2 - p) / (p - 1)

    log_v = px.math.log(px.math.where(value > 1e-9, value, 1.0))
    ll_core = (
        (value * mu ** (1 - p) / (1 - p) - mu ** (2 - p) / (2 - p)) / phi
    )

    log_Wj = (
        j * alpha * log_v
        - j * (1 + alpha) * px.math.log(phi)
        - j * px.math.log(abs(2 - p))
        - j * alpha * px.math.log(p - 1)
        - px.math.gammaln(j + 1)
        - px.math.gammaln(px.math.maximum(j * alpha, 1e-10))
    )
    log_a = px.math.log((px.math.exp(log_Wj)).sum(dim="term")) - log_v
    logp_pos = ll_core + log_a
    logp_zero = -(mu ** (2 - p)) / (phi * (2 - p))
    return px.math.where(value <= 1e-9, logp_zero, logp_pos)


def tweedie_dist(mu, phi, p):
    """Tweedie random draws via Poisson-Gamma compound (p ∈ (1, 2))."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha_term = (2 - p) / (p - 1)
    beta = 1.0 / (phi * (p - 1) * mu ** (p - 1))
    N = pmd.Poisson.dist(mu=lam)
    Y = pmd.Gamma.dist(alpha=px.math.maximum(N * alpha_term, 1e-10), beta=beta)
    return px.math.where(N > 0, Y, 0.0)


def build_intercept_only_model(y, p_range=(1.1, 1.9)):
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pmd.Normal("log_mu",
                            mu=np.log(max(y.mean(), 1)), sigma=1)
        mu = pmd.Deterministic("mu", px.math.exp(log_mu))

        log_phi = pmd.Normal("log_phi", mu=0, sigma=1)
        phi = pmd.Deterministic("phi", px.math.exp(log_phi))

        p_logit = pmd.Normal("p_logit", mu=-0.5, sigma=0.5)
        p = pmd.Deterministic("p",
            p_range[0] + (p_range[1] - p_range[0])
            * pmd.math.sigmoid(p_logit))

        pmd.CustomDist(
            "y_obs", mu, phi, p,
            dist=tweedie_dist,
            logp=tweedie_logp_series,
            class_name="Tweedie",
            observed=y, dims="obs",
        )
    return model


# Generate synthetic data matching dataCar parameters
rng = np.random.default_rng(42)
n = 60_000
mu_true = 293.0
phi_true = 174.0
p_true = 1.574

y = tweedie_random(mu_true, phi_true, p_true, size=n, rng=rng)
print(f"Generated {n} obs: zero rate = {np.mean(y == 0):.1%}, mean = {y.mean():.0f}")

model = build_intercept_only_model(y)
print("Sampling with nutpie (4 chains x 1000 draws)...")
start = time.time()
idata = pm.sample(1000, chains=4, nuts_sampler="nutpie", random_seed=42, model=model)
elapsed = time.time() - start

print(f"\nTotal sampling time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
