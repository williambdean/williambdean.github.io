# /// script
# dependencies = ["matplotlib", "numpy", "scipy", "pymc", "arviz", "pytensor"]
# ///

"""Figure: Prior vs Posterior comparison using pm.sample_prior_predictive and pm.sample.

Shows how much the data constrains each parameter (mu, phi, p) by running
actual PyMC sampling on synthetic Tweedie data and comparing prior vs posterior
distributions using az.plot_prior_posterior. Uses pymc.dims with xtensor-based
logp and compound dist.
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.xtensor import as_xtensor

from tweedie_utils import tweedie_random

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


class Tweedie:
    def __new__(cls, name, mu, phi, p, **kwargs):
        return pmd.CustomDist(
            name, mu, phi, p,
            dist=tweedie_dist,
            logp=tweedie_logp_series,
            class_name="Tweedie",
            **kwargs,
        )


def build_model(y):
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pmd.Normal("log_mu",
                            mu=np.log(max(y.mean(), 1)), sigma=1)
        mu = pmd.Deterministic("mu", px.math.exp(log_mu))

        log_phi = pmd.Normal("log_phi", mu=0, sigma=1)
        phi = pmd.Deterministic("phi", px.math.exp(log_phi))

        p_logit = pmd.Normal("p_logit", mu=-0.5, sigma=0.5)
        p = pmd.Deterministic("p", 1.1 + 0.8 * pmd.math.sigmoid(p_logit))

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

# --- Plot prior-vs-posterior: (3, 1) vertical layout ---
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=False)
var_names = ["mu", "phi", "p"]
var_labels = {"mu": "μ (pure premium)", "phi": "φ (dispersion)", "p": "p (power)"}

for ax, var in zip(axes, var_names):
    prior_vals = combined["prior"][var].values
    post_vals = combined["posterior"][var].values

    prior_flat = prior_vals.ravel()
    post_flat = post_vals.ravel()

    ax.hist(prior_flat, bins=50, density=True, alpha=0.4, color="C0",
            label="Prior")
    ax.hist(post_flat, bins=50, density=True, alpha=0.6, color="C1",
            label="Posterior")
    ax.set_xlabel(var_labels.get(var, var))
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

fig.suptitle("Prior vs Posterior: 5000 Observations Tighten All Parameters",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_prior_posterior.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig_prior_posterior.png'}")
