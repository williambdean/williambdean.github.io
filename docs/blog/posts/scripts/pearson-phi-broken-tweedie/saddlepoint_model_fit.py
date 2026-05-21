# /// script
# dependencies = ["pymc", "nutpie", "numpy", "pytensor", "matplotlib"]
# ///

"""Fit Tweedie model with both Series and Saddlepoint logp, compare posteriors.

Generates synthetic Tweedie data matching dataCar parameters, then fits
two models — one using the series expansion logp, one using the saddlepoint
approximation. Both share the same random sampler (tweedie_random). The
only difference is the logp function.

If the saddlepoint is a valid replacement, the posteriors should overlap.
"""

import time
from pathlib import Path

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

# ── Logp definitions ────────────────────────────────────────────────────


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    """Tweedie log-pdf via series expansion (Dunn & Smyth 2005).

    Returns a regular TensorVariable. Internally uses xtensor for
    named-dim broadcasting across terms and observations.
    """
    value = as_xtensor(value, dims=("obs",))
    mu = pt.squeeze(mu)
    phi = pt.squeeze(phi)
    p = pt.squeeze(p)

    j = as_xtensor(
        pt.arange(1, n_terms + 1, dtype=pytensor.config.floatX),
        dims=("term",),
    )
    alpha = (2 - p) / (p - 1)

    log_v = px.math.log(px.math.maximum(value, 1e-10))
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
    log_a = px.math.log(px.math.exp(log_Wj).sum(dim="term")) - log_v
    logp_pos = ll_core + log_a
    logp_zero = -(mu ** (2 - p)) / (phi * (2 - p))

    return pmd.tensor_from_xtensor(
        px.math.where(value <= 1e-9, logp_zero, logp_pos)
    )


def tweedie_logp_saddlepoint(value, mu, phi, p):
    """Tweedie log-pdf via saddlepoint approximation (Nelder & Pregibon).

    Unit deviance for p in (1, 2):

        d(y, mu, p) = 2 [ y^(2-p)/((1-p)(2-p))
                        - y·mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]

    Saddlepoint log-likelihood:

        log L ≈ -0.5 log(2πφy^p) - d(y,μ,p) / (2φ)

    For y = 0, the exact Poisson mass is used.
    """
    value = as_xtensor(value, dims=("obs",))
    mu = pt.squeeze(mu)
    phi = pt.squeeze(phi)
    p = pt.squeeze(p)

    y_safe = px.math.maximum(value, 1e-10)

    term1 = y_safe ** (2 - p) / ((1 - p) * (2 - p))
    term2 = y_safe * (mu ** (1 - p)) / (1 - p)
    term3 = mu ** (2 - p) / (2 - p)
    deviance = 2 * (term1 - term2 + term3)

    logl_pos = (
        -0.5 * px.math.log(2 * pt.constant(np.pi) * phi * (y_safe ** p))
        - deviance / (2 * phi)
    )
    logl_zero = -(mu ** (2 - p)) / (phi * (2 - p))

    return pmd.tensor_from_xtensor(
        px.math.where(value <= 1e-9, logl_zero, logl_pos)
    )


# ── Model builder ───────────────────────────────────────────────────────


def build_model(y, logp_fn, p_range=(1.1, 1.9)):
    """Build an intercept-only Tweedie model with a swappable logp.

    Uses pm.CustomDist with random=tweedie_random (numpy) for draws
    and a symbolic logp_fn for MCMC inference. The logp_fn is the only
    thing that changes between the Series and Saddlepoint models.
    """
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pm.Normal(
            "log_mu", mu=np.log(max(y.mean(), 1)), sigma=1
        )
        mu = pm.Deterministic("mu", pt.exp(log_mu))

        log_phi = pm.Normal("log_phi", mu=0, sigma=1)
        phi = pm.Deterministic("phi", pt.exp(log_phi))

        p_logit = pm.Normal("p_logit", mu=-0.5, sigma=0.5)
        p = pm.Deterministic(
            "p",
            p_range[0]
            + (p_range[1] - p_range[0]) * pm.math.sigmoid(p_logit),
        )

        pm.CustomDist(
            "y_obs",
            mu, phi, p,
            logp=logp_fn,
            random=tweedie_random,
            observed=y,
            dims="obs",
        )
    return model


# ── Generate data ───────────────────────────────────────────────────────

rng = np.random.default_rng(42)
n = 10_000  # moderate size for reasonable sampling time
MU_TRUE = 293.0
PHI_TRUE = 174.0
P_TRUE = 1.574

y = tweedie_random(MU_TRUE, PHI_TRUE, P_TRUE, size=n, rng=rng)
print(f"Data: n={n}, zero_rate={np.mean(y == 0):.1%}, "
      f"mean={y.mean():.0f}, max={y.max():.0f}")
print()

# ── Fit: Saddlepoint ────────────────────────────────────────────────────

print("=" * 60)
print("SADDLEPOINT MODEL")
print("=" * 60)
model_saddle = build_model(y, tweedie_logp_saddlepoint)
t0 = time.time()
with model_saddle:
    idata_saddle = pm.sample(
        draws=500, tune=500, chains=2,
        nuts_sampler="nutpie", random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )
t_saddle = time.time() - t0
print(f"  Time: {t_saddle:.1f}s ({t_saddle/60:.1f} min)")

# ── Fit: Series ─────────────────────────────────────────────────────────

print()
print("=" * 60)
print("SERIES MODEL")
print("=" * 60)
model_series = build_model(y, tweedie_logp_series)
t0 = time.time()
with model_series:
    idata_series = pm.sample(
        draws=500, tune=500, chains=2,
        nuts_sampler="nutpie", random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )
t_series = time.time() - t0
print(f"  Time: {t_series:.1f}s ({t_series/60:.1f} min)")

# ── Compare posteriors ──────────────────────────────────────────────────

series_post = idata_series.posterior
saddle_post = idata_saddle.posterior

mu_s = series_post["mu"].values.ravel()
phi_s = series_post["phi"].values.ravel()
p_s = series_post["p"].values.ravel()

mu_sp = saddle_post["mu"].values.ravel()
phi_sp = saddle_post["phi"].values.ravel()
p_sp = saddle_post["p"].values.ravel()

print()
print("=" * 60)
print("POSTERIOR COMPARISON")
print("=" * 60)

all_close = True
for name, s_vals, sp_vals, truth in [
    ("mu", mu_s, mu_sp, MU_TRUE),
    ("phi", phi_s, phi_sp, PHI_TRUE),
    ("p", p_s, p_sp, P_TRUE),
]:
    rel_diff = abs(s_vals.mean() - sp_vals.mean()) / max(abs(s_vals.mean()), 1)
    ci_lo_s = np.percentile(s_vals, 2.5)
    ci_hi_s = np.percentile(s_vals, 97.5)
    ci_lo_sp = np.percentile(sp_vals, 2.5)
    ci_hi_sp = np.percentile(sp_vals, 97.5)
    overlap = np.mean(
        (sp_vals >= ci_lo_s) & (sp_vals <= ci_hi_s)
    )

    print(f"\n{name}:")
    print(f"  Truth:          {truth:.2f}")
    print(f"  Series mean:    {s_vals.mean():.2f}  std: {s_vals.std():.2f}")
    print(f"  Saddle mean:    {sp_vals.mean():.2f}  std: {sp_vals.std():.2f}")
    print(f"  Rel diff:       {rel_diff:.4%}")
    print(f"  95% CI series:  [{ci_lo_s:.1f}, {ci_hi_s:.1f}]")
    print(f"  95% CI saddle:  [{ci_lo_sp:.1f}, {ci_hi_sp:.1f}]")
    print(f"  Overlap in series 95% CI: {overlap:.1%}")

    if rel_diff > 0.1:  # more than 10% relative difference
        all_close = False

# Speedup
if t_saddle > 0:
    print(f"\nTiming: series={t_series:.1f}s, saddlepoint={t_saddle:.1f}s, "
          f"speedup={t_series/t_saddle:.1f}x")

# ── Convergence ─────────────────────────────────────────────────────────

print()
print("=" * 60)
print("CONVERGENCE (R-hat)")
print("=" * 60)
for name in ["mu", "phi", "p"]:
    rhat_s = float(idata_series.posterior[name].rhat)
    rhat_sp = float(idata_saddle.posterior[name].rhat)
    print(f"  {name}: series R-hat={rhat_s:.4f}, saddle R-hat={rhat_sp:.4f}")

# ── Figure ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

params = [
    ("\u03bc", mu_s, mu_sp, MU_TRUE, (0, 600)),
    ("\u03c6", phi_s, phi_sp, PHI_TRUE, (0, 500)),
    ("p", p_s, p_sp, P_TRUE, (1.3, 1.85)),
]

for ax, (name, s_vals, sp_vals, truth, xlim) in zip(axes, params):
    ax.hist(s_vals, bins=40, alpha=0.5, density=True,
            color="C0", label=f"Series (mean={s_vals.mean():.1f})")
    ax.hist(sp_vals, bins=40, alpha=0.5, density=True,
            color="C1", label=f"Saddle (mean={sp_vals.mean():.1f})")
    ax.axvline(truth, color="black", ls="--", lw=1.5,
               label=f"True = {truth}")
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.set_title(f"{name} posterior")
    ax.legend(fontsize=8)
    if xlim:
        ax.set_xlim(xlim)

plt.suptitle("Series vs Saddlepoint: Posterior Comparison\n"
             f"(n={n}, true \u03bc={MU_TRUE}, \u03c6={PHI_TRUE}, p={P_TRUE})",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_saddlepoint_posterior.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT_DIR / 'fig_saddlepoint_posterior.png'}")

# ── Summary ─────────────────────────────────────────────────────────────

print()
print("=" * 60)
if all_close:
    print("VERDICT: Saddlepoint posteriors match series posteriors.")
else:
    print("VERDICT: Mismatch detected. Saddlepoint needs further investigation.")
print("=" * 60)
