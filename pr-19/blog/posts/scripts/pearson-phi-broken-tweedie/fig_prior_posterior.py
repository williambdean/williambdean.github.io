# /// script
# dependencies = ["matplotlib", "numpy", "scipy", "pymc", "arviz", "pytensor"]
# ///

"""Figure: Prior vs Posterior comparison using pm.sample_prior_predictive and pm.sample.

Shows how much the data constrains each parameter (mu, phi, p) by running
actual PyMC sampling on synthetic Tweedie data and comparing prior vs posterior
distributions using az.plot_prior_posterior.
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc import CustomDist

from tweedie_utils import tweedie_random

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    """Tweedie log-pdf via series expansion (Dunn & Smyth 2005)."""
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


class Tweedie:
    def __new__(cls, name, mu, phi, p, **kwargs):
        return CustomDist(
            name, mu, phi, p,
            logp=tweedie_logp_series,
            random=tweedie_random,
            class_name="Tweedie",
            **kwargs,
        )


def build_model(y):
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pm.Normal("log_mu",
                           mu=np.log(max(y.mean(), 1)), sigma=1)
        mu = pm.Deterministic("mu", pt.exp(log_mu))

        log_phi = pm.Normal("log_phi", mu=0, sigma=1)
        phi = pm.Deterministic("phi", pt.exp(log_phi))

        p_logit = pm.Normal("p_logit", mu=-0.5, sigma=0.5)
        p = pm.Deterministic("p", 1.1 + 0.8 * pm.math.sigmoid(p_logit))

        Tweedie("y_obs", mu=mu, phi=phi, p=p, observed=y, dims="obs")
    return model


# --- Generate synthetic data ---
rng = np.random.default_rng(42)
n_obs = 5000
true_mu, true_phi, true_p = 293.0, 174.0, 1.574

y = tweedie_random(true_mu, true_phi, true_p, size=n_obs, rng=rng)
print(f"{n_obs} obs: zero rate = {np.mean(y == 0):.1%}, mean = ${y.mean():.0f}")

# --- Build and sample ---
model = build_model(y)

with model:
    prior = pm.sample_prior_predictive(1000, random_seed=42)
    print("Prior predictive complete.")

    idata = pm.sample(
        500, chains=2, tune=300,
        random_seed=42, nuts_sampler="nutpie",
        progressbar=True,
    )
    print("Posterior sampling complete.")

# --- Combine prior and posterior into a single DataTree ---
combined = az.from_dict(data={
    "prior": {
        "mu": prior.prior["mu"].values,
        "phi": prior.prior["phi"].values,
        "p": prior.prior["p"].values,
    },
    "posterior": {
        "mu": idata.posterior["mu"].values,
        "phi": idata.posterior["phi"].values,
        "p": idata.posterior["p"].values,
    },
})

# --- Plot prior-vs-posterior using arviz ---
pc = az.plot_prior_posterior(
    combined,
    var_names=["mu", "phi", "p"],
    kind="hist",
)

# Add a figure-level title
pc.add_title("Prior vs Posterior: 5000 Observations Tighten All Parameters", fontsize=13)

pc.savefig(OUT_DIR / "fig_prior_posterior.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_prior_posterior.png'}")
