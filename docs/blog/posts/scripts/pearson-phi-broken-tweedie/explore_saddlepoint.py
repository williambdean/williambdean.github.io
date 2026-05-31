# /// script
# dependencies = ["matplotlib", "numpy", "pytensor"]
# ///

"""Explore Saddlepoint vs Series Tweedie logp accuracy.

Compares the Nelder & Pregibon saddlepoint approximation against the
exact series expansion (Dunn & Smyth 2005). Both logp functions use
pytensor.xtensor and are compiled via pytensor.function.

Key question: is the saddlepoint surface close enough to the series
surface to serve as a drop-in replacement for MCMC inference?

Generates two figures:
  1. Overlay + difference at dataCar params, plus p-sensitivity showing
     where the approximation degrades near the boundaries
  2. Full 2D landscape: Delta(y, p) heatmap
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px
from pytensor.xtensor import as_xtensor

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logp definitions (pytensor.xtensor) ──────────────────────────────────


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    """Tweedie log-pdf via series expansion (Dunn & Smyth 2005)."""
    value = as_xtensor(value, dims=("obs",))
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
        - j * px.math.log(pt.abs(2 - p))
        - j * alpha * px.math.log(p - 1)
        - px.math.gammaln(j + 1)
        - px.math.gammaln(px.math.maximum(j * alpha, 1e-10))
    )
    log_a = px.math.log(px.math.exp(log_Wj).sum(dim="term")) - log_v
    logp_pos = ll_core + log_a
    logp_zero = -(mu ** (2 - p)) / (phi * (2 - p))
    return px.math.where(value <= 1e-9, logp_zero, logp_pos)


def tweedie_logp_saddlepoint(value, mu, phi, p):
    """Tweedie log-pdf via saddlepoint approximation (Nelder & Pregibon).

    The unit deviance for p in (1, 2):

        d(y, mu, p) = 2 * [
            y^(2-p) / ((1-p)(2-p))
            - y * mu^(1-p) / (1-p)
            + mu^(2-p) / (2-p)
        ]

    Saddlepoint log-likelihood:

        log L ≈ -0.5 * log(2*pi*phi*y^p) - d(y,mu,p) / (2*phi)

    For y = 0, the exact Poisson mass is used:
        log P(Y=0) = -mu^(2-p) / (phi*(2-p))
    """
    value = as_xtensor(value, dims=("obs",))
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

    return px.math.where(value <= 1e-9, logl_zero, logl_pos)


# ── Compile for evaluation ───────────────────────────────────────────────

y_sym = pt.dvector("y")
mu_sym = pt.dscalar("mu")
phi_sym = pt.dscalar("phi")
p_sym = pt.dscalar("p")

series_fn = pytensor.function(
    [y_sym, mu_sym, phi_sym, p_sym],
    tweedie_logp_series(y_sym, mu_sym, phi_sym, p_sym),
)

saddle_fn = pytensor.function(
    [y_sym, mu_sym, phi_sym, p_sym],
    tweedie_logp_saddlepoint(y_sym, mu_sym, phi_sym, p_sym),
)

# ── DataCar reference parameters ─────────────────────────────────────────

MU_REF = 293.0
PHI_REF = 174.0
P_REF = 1.574

# ── Figure 1: Comparison at dataCar params + p-sensitivity ───────────────

# Generate y values spanning the distribution
rng = np.random.default_rng(42)
n_y = 500
# Mix of zeros, small positive, and large claims
y_zeros = np.zeros(50)
y_small = np.logspace(-1, 1, 150)
y_mid = np.logspace(1, 3, 200)
y_large = np.logspace(3, 4.5, 100)
y_grid = np.sort(np.concatenate([y_zeros, y_small, y_mid, y_large]))

logp_series = series_fn(y_grid, MU_REF, PHI_REF, P_REF)
logp_saddle = saddle_fn(y_grid, MU_REF, PHI_REF, P_REF)
delta = logp_series - logp_saddle

print(f"Series range:    [{logp_series.min():.2f}, {logp_series.max():.2f}]")
print(f"Saddle range:    [{logp_saddle.min():.2f}, {logp_saddle.max():.2f}]")
print(f"Delta range:     [{delta.min():.4f}, {delta.max():.4f}]")
print(f"Delta std:       {delta.std():.4f}")
print(f"Delta max |curvature|: {np.max(np.abs(np.diff(np.diff(delta)))):.6f}")

# p-sensitivity: delta vs y at several p values
p_values = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.574, 1.6, 1.7, 1.8, 1.9])
y_pos = np.logspace(-1, 4, 200)

# Second metric: delta as function of p at fixed y
y_snapshot = np.array([0.01, 1.0, 10.0, 100.0, 500.0, 2000.0])
p_fine = np.linspace(1.02, 1.98, 100)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (0,0): Logp overlay
ax = axes[0, 0]
pos_mask = y_grid > 1e-9
ax.plot(y_grid[pos_mask], logp_series[pos_mask], "C0-", lw=1.5, alpha=0.8,
        label="Series (exact)")
ax.plot(y_grid[pos_mask], logp_saddle[pos_mask], "C1--", lw=1.5, alpha=0.8,
        label="Saddlepoint")
ax.set_xscale("log")
ax.set_xlabel("y (log scale)")
ax.set_ylabel("log p(y)")
ax.set_title(f"Logp vs y (μ={MU_REF}, φ={PHI_REF}, p={P_REF})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (0,1): Delta vs y
ax = axes[0, 1]
pos_mask = y_grid > 1e-9
ax.plot(y_grid[pos_mask], delta[pos_mask], "C3-", lw=1.5)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.axhline(delta[pos_mask].mean(), color="C3", ls=":", lw=0.8,
           label=f"mean Δ = {delta[pos_mask].mean():.4f}")
ax.set_xscale("log")
ax.set_xlabel("y (log scale)")
ax.set_ylabel("Δ = series − saddlepoint")
ax.set_title("Logp difference vs y")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (0,2): Delta at y=0 (bar chart for p values)
ax = axes[0, 2]
delta_zero = []
for pv in p_values:
    s = float(series_fn(np.array([0.0]), MU_REF, PHI_REF, pv)[0])
    sp = float(saddle_fn(np.array([0.0]), MU_REF, PHI_REF, pv)[0])
    delta_zero.append(s - sp)

ax.bar(np.arange(len(p_values)), delta_zero, color="C3", alpha=0.7)
ax.set_xticks(range(len(p_values)))
ax.set_xticklabels([f"{p:.2f}" for p in p_values], rotation=45, ha="right", fontsize=7)
ax.set_xlabel("p")
ax.set_ylabel("Δ at y=0")
ax.set_title("Delta at y=0 across p values\n(exact match: series = saddlepoint)")
ax.grid(True, alpha=0.2, axis="y")

# (1,0): Delta vs y at several p values (line plot)
ax = axes[1, 0]
cmap = plt.cm.viridis
for i, pv in enumerate(p_values):
    lp_s = series_fn(y_pos, MU_REF, PHI_REF, pv)
    lp_sp = saddle_fn(y_pos, MU_REF, PHI_REF, pv)
    d = lp_s - lp_sp
    color = cmap(i / (len(p_values) - 1))
    ax.plot(y_pos, d, color=color, lw=1.2, alpha=0.8, label=f"p={pv:.3f}")
ax.set_xscale("log")
ax.set_xlabel("y (log scale)")
ax.set_ylabel("Δ = series − saddlepoint")
ax.set_title("Delta vs y at various p values")
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.legend(fontsize=6, ncols=2)
ax.grid(True, alpha=0.2)

# (1,1): Delta vs p at several y snapshots (boundary bias)
ax = axes[1, 1]
cmap = plt.cm.viridis
for i, yv in enumerate(y_snapshot):
    lp_s = np.array([series_fn(np.array([yv]), MU_REF, PHI_REF, pv)[0] for pv in p_fine])
    lp_sp = np.array([saddle_fn(np.array([yv]), MU_REF, PHI_REF, pv)[0] for pv in p_fine])
    d = lp_s - lp_sp
    color = cmap(i / (len(y_snapshot) - 1))
    ax.plot(p_fine, d, color=color, lw=1.5, alpha=0.8, label=f"y={yv:.1f}")
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.3)
ax.axvline(2.0, color="gray", ls=":", lw=0.8, alpha=0.3)
ax.set_xlabel("p")
ax.set_ylabel("Δ = series − saddlepoint")
ax.set_title("Delta vs p at fixed y\n(boundary degradation near p→1 and p→2)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

# (1,2): Summary stats table
ax = axes[1, 2]
ax.axis("off")
text_lines = [
    f"μ = {MU_REF:.0f},  φ = {PHI_REF:.0f},  p = {P_REF:.3f}",
    "",
    "At reference params:",
    f"  mean Δ = {delta.mean():.4f}",
    f"  std Δ  = {delta.std():.4f}",
    f"  max |Δ| = {np.max(np.abs(delta)):.4f}",
    f"  Δ range = [{delta.min():.4f}, {delta.max():.4f}]",
    "",
    "Interpretation:",
    "  Δ roughly constant in y →",
    "  saddlepoint preserves the",
    "  shape of the logp surface.",
    "  Larger variation in Δ →",
    "  gradient distortion.",
    "",
    "p-sensitivity:",
    "  Δ grows near p→1, p→2.",
    "  At p≈1.5, Δ is tight.",
]
for i, line in enumerate(text_lines):
    ax.text(0.05, 0.95 - i * 0.04, line, transform=ax.transAxes,
            fontsize=8, family="monospace", va="top")

plt.suptitle("Saddlepoint vs Series: Logp Surface Comparison",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_saddlepoint_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_saddlepoint_comparison.png'}")


# ── Figure 2: Full landscape heatmap Δ(y, p) ────────────────────────────

y_fine = np.logspace(-0.5, 4, 80)
p_fine = np.linspace(1.02, 1.98, 80)
YY, PP = np.meshgrid(y_fine, p_fine)

delta_grid = np.full(YY.shape, np.nan)
for i in range(YY.shape[0]):
    for j in range(YY.shape[1]):
        yv = YY[i, j]
        pv = PP[i, j]
        s = float(series_fn(np.array([yv]), MU_REF, PHI_REF, pv)[0])
        sp = float(saddle_fn(np.array([yv]), MU_REF, PHI_REF, pv)[0])
        delta_grid[i, j] = s - sp

print(f"Heatmap Δ range: [{delta_grid.min():.4f}, {delta_grid.max():.4f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                          gridspec_kw={"width_ratios": [1, 0.05]})

ax = axes[0]
vlim = max(abs(delta_grid.min()), abs(delta_grid.max()))
im = ax.pcolormesh(YY, PP, delta_grid, cmap="RdBu_r", shading="auto",
                   vmin=-vlim, vmax=vlim, rasterized=True)
ax.contour(YY, PP, delta_grid, levels=[0], colors="black", linewidths=0.8, linestyles="--")
ax.set_xscale("log")
ax.set_xlabel("y (log scale)")
ax.set_ylabel("p")
ax.set_title(f"Δ(y, p) = series − saddlepoint  (μ={MU_REF}, φ={PHI_REF})\n"
             "Blue = saddlepoint over-estimates logp; Red = under-estimates")

# Mark the dataCar reference
ax.axhline(P_REF, color="black", ls=":", lw=1, alpha=0.7)
ax.text(1.5, P_REF + 0.02, f"dataCar p={P_REF}", fontsize=8, va="bottom")

cbar = fig.colorbar(im, cax=axes[1])
cbar.set_label("Δ = series − saddlepoint", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_saddlepoint_landscape.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_saddlepoint_landscape.png'}")
