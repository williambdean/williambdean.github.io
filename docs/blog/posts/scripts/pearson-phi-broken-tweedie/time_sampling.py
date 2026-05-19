# /// script
# dependencies = ["pymc", "nutpie", "numpy", "pytensor", "matplotlib"]
# ///

"""Time the Tweedie model sampling with nutpie.

Uses the same model structure as the blog post (build_intercept_only_model)
with a plain PyTensor logp (functionally identical to the xtensor version).
"""

import time

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from tweedie_utils import tweedie_random


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    j = pt.arange(1, n_terms + 1, dtype=pytensor.config.floatX).reshape((-1, 1))
    alpha = (2 - p) / (p - 1)

    log_v = pt.log(pt.switch(pt.gt(value, 1e-9), value, 1.0))
    ll_core = (
        (value * pt.pow(mu, 1 - p) / (1 - p) - pt.pow(mu, 2 - p) / (2 - p))
        / phi
    )

    log_Wj = (
        j * alpha * log_v
        - j * (1 + alpha) * pt.log(phi)
        - j * pt.log(pt.abs(2 - p))
        - j * alpha * pt.log(p - 1)
        - pt.gammaln(j + 1)
        - pt.gammaln(pt.maximum(j * alpha, 1e-10))
    )
    log_a = pt.log(pt.sum(pt.exp(log_Wj), axis=0)) - log_v
    logp_pos = ll_core + log_a
    logp_zero = -pt.pow(mu, 2 - p) / (phi * (2 - p))
    return pt.switch(pt.le(value, 1e-9), logp_zero, logp_pos)


def build_intercept_only_model(y, p_range=(1.1, 1.9)):
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pm.Normal("log_mu",
                           mu=np.log(max(y.mean(), 1)), sigma=1)
        mu = pm.Deterministic("mu", pt.exp(log_mu))

        log_phi = pm.Normal("log_phi", mu=5.0, sigma=2.0)
        phi = pm.Deterministic("phi", pt.exp(log_phi))

        p_logit = pm.Normal("p_logit", mu=0, sigma=1.5)
        p = pm.Deterministic("p",
            p_range[0] + (p_range[1] - p_range[0])
            * pm.math.sigmoid(p_logit))

        pm.DensityDist(
            "y_obs", mu, phi, p,
            logp=tweedie_logp_series,
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
