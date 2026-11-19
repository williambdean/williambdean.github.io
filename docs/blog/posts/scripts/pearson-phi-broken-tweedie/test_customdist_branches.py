"""Cross-branch compatibility check for the blog's Tweedie CustomDist model.

Runs the Tweedie model from pearson-phi-broken-tweedie.md against both
pymc.dims CustomDist implementations:

- williambdean/pymc branch ``customdist-dims``
  (dist infers/broadcasts output dims automatically)
- RicardoV94/pymc branch ``alt-customdist-dims`` (PR #8346)
  (dist receives extra_dims as last arg; observed must be an xtensor)

The script needs a branch-specific pymc import, so run it from a checkout
of the branch you want to test, e.g. via a git worktree and a venv:

    git worktree add /tmp/rv-customdist alt-customdist-dims
    uv venv /tmp/venv && uv pip install --python /tmp/venv/bin/python \
        -e /tmp/rv-customdist
    /tmp/venv/bin/python test_customdist_branches.py

For Ricardo's branch, set USE_EXTRA_DIMS = True (checked at runtime via
the dist-function behavior; the wrapper is built for the target branch).
"""

USE_EXTRA_DIMS = True  # False for the customdist-dims branch API

import sys

import numpy as np

sys.path.insert(
    0,
    "/Users/will/github/personal/williambdean.github.io/docs/"
    "blog/posts/scripts/pearson-phi-broken-tweedie",
)
from tweedie_utils import tweedie_logp_series as ref_logp  # numpy reference
from tweedie_utils import tweedie_random

import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.xtensor import as_xtensor

rng = np.random.default_rng(42)
N_OBS = 1000
TRUE_MU, TRUE_PHI, TRUE_P = 293.0, 174.0, 1.574

y = tweedie_random(TRUE_MU, TRUE_PHI, TRUE_P, size=N_OBS, rng=rng)
print(f"{N_OBS} obs: zero rate = {np.mean(y == 0):.1%}, mean = ${y.mean():.0f}")


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    """Blog logp, unchanged."""
    j = as_xtensor(
        pt.arange(1, n_terms + 1, dtype=pytensor.config.floatX),
        dims=("term",),
    )
    alpha = (2 - p) / (p - 1)

    log_v = px.math.log(px.math.where(value > 1e-9, value, 1.0))
    ll_core = (value * mu ** (1 - p) / (1 - p) - mu ** (2 - p) / (2 - p)) / phi

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


# --- dist function per branch API ---
if USE_EXTRA_DIMS:

    def tweedie_dist(mu, phi, p, extra_dims):
        """Ricardo's branch: dist attaches the batch dims itself."""
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha_term = (2 - p) / (p - 1)
        beta = 1.0 / (phi * (p - 1) * mu ** (p - 1))
        N = pmd.Poisson.dist(mu=lam, dim_lengths=extra_dims)
        Y = pmd.Gamma.dist(alpha=px.math.maximum(N * alpha_term, 1e-10), beta=beta)
        return px.math.where(N > 0, Y, 0.0)

else:

    def tweedie_dist(mu, phi, p):
        """customdist-dims branch: output dims inferred automatically."""
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha_term = (2 - p) / (p - 1)
        beta = 1.0 / (phi * (p - 1) * mu ** (p - 1))
        N = pmd.Poisson.dist(mu=lam)
        Y = pmd.Gamma.dist(alpha=px.math.maximum(N * alpha_term, 1e-10), beta=beta)
        return px.math.where(N > 0, Y, 0.0)


class Tweedie:
    def __new__(cls, name, mu, phi, p, **kwargs):
        return pmd.CustomDist(
            name,
            mu,
            phi,
            p,
            dist=tweedie_dist,
            logp=tweedie_logp_series,
            class_name="Tweedie",
            **kwargs,
        )


def build_model(Tweedie, y):
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pmd.Normal("log_mu", mu=np.log(max(y.mean(), 1)), sigma=1)
        mu = pmd.Deterministic("mu", px.math.exp(log_mu))

        log_phi = pmd.Normal("log_phi", mu=0, sigma=1)
        phi = pmd.Deterministic("phi", px.math.exp(log_phi))

        p_logit = pmd.Normal("p_logit", mu=-0.5, sigma=0.5)
        p = pmd.Deterministic("p", 1.1 + 0.8 * pmd.math.sigmoid(p_logit))

        observed = as_xtensor(y, dims=("obs",)) if USE_EXTRA_DIMS else y
        Tweedie("y_obs", mu=mu, phi=phi, p=p, observed=observed, dims="obs")
    return model


branch = "alt-customdist-dims (PR #8346)" if USE_EXTRA_DIMS else "customdist-dims"
print(f"\n=== Testing Tweedie model on branch: {branch} ===")
model = build_model(Tweedie, y)

# 1. logp correctness against numpy reference (fixed params, data logp only)
with pm.Model(coords={"obs": np.arange(N_OBS)}) as m_fixed:
    fixed_obs = as_xtensor(y, dims=("obs",)) if USE_EXTRA_DIMS else y
    Tweedie(
        "y_obs",
        mu=TRUE_MU,
        phi=TRUE_PHI,
        p=TRUE_P,
        observed=fixed_obs,
        dims="obs",
    )
logp_val = m_fixed.compile_logp()({})

ref = ref_logp(y, TRUE_MU, TRUE_PHI, TRUE_P)
print(f"graph logp (sum): {logp_val:.4f}")
print(f"reference logp (sum): {ref.sum():.4f}")
assert np.allclose(logp_val, ref.sum(), rtol=1e-3), "logp mismatch!"
print("logp matches reference: OK")

# 2. prior predictive (sampling path: chained RNG through compound)
with model:
    prior = pm.sample_prior_predictive(50, random_seed=42)
y_prior = (prior.prior_predictive["y_obs"].values
           if "y_obs" in getattr(prior.prior_predictive, "data_vars", {})
           else prior.prior["y_obs"].values)
print(f"prior predictive shape: {y_prior.shape}, zeros: {np.mean(y_prior == 0):.1%}")

# 3. short posterior sample
with model:
    idata = pm.sample(draws=100, tune=100, chains=2, random_seed=42, progressbar=False)

summ = pm.summary(idata, var_names=["mu", "phi", "p"])
print(summ)

# 4. posterior predictive
with model:
    ppc = pm.sample_posterior_predictive(idata, random_seed=42)
y_sim = ppc.posterior_predictive["y_obs"].values
print(
    f"PPC: observed mean ${y.mean():.0f}, sim mean ${y_sim.mean():.0f}, "
    f"obs zeros {np.mean(y == 0):.1%}, sim zeros {np.mean(y_sim == 0):.1%}"
)

print(f"\nALL CHECKS PASSED on branch {branch}")
