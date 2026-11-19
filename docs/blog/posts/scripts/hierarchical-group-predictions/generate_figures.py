# /// script
# dependencies = [
#   "pymc>=5.0",
#   "nutpie",
#   "pytensor",
#   "pandas",
#   "matplotlib",
#   "numpy",
#   "arviz",
#   "scipy",
# ]
# ///

"""Figure generation for hierarchical group predictions blog post.

Generates 4 figures:
1. fig_data_overview.png — data with all groups + held-out group highlighted
2. fig_posterior_evolution.png — posterior KDEs for alpha_20, beta_20 at N=0,1,3,5,15
   (reversed: N=0 at top, N=15 at bottom, showing evolution as data accumulates)
3. fig_predictions_comparison.png — faceted predictions comparing hierarchical,
   no-pooling (independent priors), and pooled models for held-out group
4. fig_zero_data_prior_vs_posterior.png — Scenario A focus: prior N(0,5) vs
   posterior at N=0 for both alpha and beta

Usage:
    uv run docs/blog/posts/scripts/hierarchical-group-predictions/generate_figures.py
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

import pymc as pm

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ── Configuration ──────────────────────────────────────────────────────────
N_GROUPS = 20
N_PER_GROUP = 15
HELD_OUT = 19  # 0-indexed, so group 20

# True population parameters
MU_ALPHA_TRUE = 3.0
SIGMA_ALPHA_TRUE = 1.5
MU_BETA_TRUE = 0.8
SIGMA_BETA_TRUE = 0.4
SIGMA_TRUE = 1.5

# ── 1. Generate synthetic data ─────────────────────────────────────────────
alpha_true = rng.normal(MU_ALPHA_TRUE, SIGMA_ALPHA_TRUE, N_GROUPS)
beta_true = rng.normal(MU_BETA_TRUE, SIGMA_BETA_TRUE, N_GROUPS)

X = rng.uniform(0, 10, (N_GROUPS, N_PER_GROUP))
y = np.zeros_like(X)
for g in range(N_GROUPS):
    y[g] = alpha_true[g] + beta_true[g] * X[g] + rng.normal(0, SIGMA_TRUE, N_PER_GROUP)

groups = np.repeat(np.arange(N_GROUPS), N_PER_GROUP)
x_flat = X.ravel()
y_flat = y.ravel()

held_out_mask = groups == HELD_OUT

# ── 2. Build and fit hierarchical model ────────────────────────────────────

def build_hierarchical_model(group_idx, x, y):
    """Build hierarchical model with random intercepts and slopes."""
    with pm.Model(coords={"group": np.arange(N_GROUPS)}) as model:
        mu_a = pm.Normal("mu_alpha", 0, 5)
        sigma_a = pm.HalfNormal("sigma_alpha", 2.5)
        mu_b = pm.Normal("mu_beta", 0, 5)
        sigma_b = pm.HalfNormal("sigma_beta", 2.5)

        alpha = pm.Normal("alpha", mu_a, sigma_a, dims="group")
        beta = pm.Normal("beta", mu_b, sigma_b, dims="group")
        sigma = pm.HalfNormal("sigma", 1)

        mu = alpha[group_idx] + beta[group_idx] * x
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    return model


NS = [0, 1, 3, 5, 15]
idata_by_N = {}
print("Fitting hierarchical models...")

for N in NS:
    if N == 0:
        mask = ~held_out_mask
    else:
        held_out_indices = np.where(held_out_mask)[0][:N]
        mask = ~held_out_mask.copy()
        mask[held_out_indices] = True

    print(f"  N={N}: {mask.sum()} observations ({N} for held-out group)")
    model = build_hierarchical_model(
        groups[mask].astype(int), x_flat[mask], y_flat[mask]
    )
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=rng,
            nuts_sampler="nutpie",
            progressbar=True,
        )
    idata_by_N[N] = idata


# ── 3. Build and fit unpooled model (no pooling) ───────────────────────────

def build_unpooled_model(group_idx, x, y):
    """Build model with independent group priors (no population-level sharing)."""
    with pm.Model(coords={"group": np.arange(N_GROUPS)}) as model:
        alpha = pm.Normal("alpha", 0, 5, dims="group")
        beta = pm.Normal("beta", 0, 5, dims="group")
        sigma = pm.HalfNormal("sigma", 1)

        mu = alpha[group_idx] + beta[group_idx] * x
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    return model


unpooled_idata_by_N = {}
print("Fitting unpooled models...")

for N in NS:
    if N == 0:
        mask = ~held_out_mask
    else:
        held_out_indices = np.where(held_out_mask)[0][:N]
        mask = ~held_out_mask.copy()
        mask[held_out_indices] = True

    print(f"  N={N}: {mask.sum()} observations ({N} for held-out group)")
    model = build_unpooled_model(
        groups[mask].astype(int), x_flat[mask], y_flat[mask]
    )
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            random_seed=rng,
            nuts_sampler="nutpie",
            progressbar=True,
        )
    unpooled_idata_by_N[N] = idata


# ── 4. Fit pooled baseline (ignores groups) ────────────────────────────────
print("Fitting pooled baseline...")
with pm.Model() as pooled_model:
    a = pm.Normal("alpha", 0, 5)
    b = pm.Normal("beta", 0, 5)
    s = pm.HalfNormal("sigma", 1)
    mu = a + b * x_flat[~held_out_mask]
    pm.Normal("y_obs", mu=mu, sigma=s, observed=y_flat[~held_out_mask])
    idata_pooled = pm.sample(
        draws=1000, tune=1000, chains=4, random_seed=rng,
        nuts_sampler="nutpie", progressbar=True,
    )


# ── 5. Figure 1: Data overview ────────────────────────────────────────────
print("Generating Figure 1: Data overview")
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each training group
for g in range(N_GROUPS):
    if g == HELD_OUT:
        continue
    group_mask = groups == g
    ax.scatter(x_flat[group_mask], y_flat[group_mask], alpha=0.3, s=15,
               color="grey")
    x_line = np.linspace(0, 10, 50)
    ax.plot(x_line, alpha_true[g] + beta_true[g] * x_line,
            color="grey", alpha=0.3, linewidth=0.8)

# Plot held-out group
held_idx = groups == HELD_OUT
ax.scatter(x_flat[held_idx], y_flat[held_idx], s=30, color="C3",
           label="Held-out group", zorder=5)
x_line = np.linspace(0, 10, 50)
ax.plot(x_line,
        alpha_true[HELD_OUT] + beta_true[HELD_OUT] * x_line,
        color="C3", linewidth=2, label=f"True (group {HELD_OUT + 1})",
        zorder=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Synthetic Hierarchical Data")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_data_overview.png", dpi=150)
plt.close(fig)


# ── 6. Figure 2: Posterior evolution (ridge plots) ─────────────────────────
print("Generating Figure 2: Posterior evolution")
colors = {0: "#440154", 1: "#3b528b", 3: "#21918c", 5: "#5ec962", 15: "#fde725"}
# viridis-like progression

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

param_configs = [
    ("alpha", r"$\alpha_{20}$", alpha_true[HELD_OUT], 0),
    ("beta", r"$\beta_{20}$", beta_true[HELD_OUT], 1),
]

y_positions = {N: i for i, N in enumerate(reversed(NS))}
for param_name, param_label, true_val, col_idx in param_configs:
    ax = axes[col_idx]

    for N in NS:
        samples = az.extract(idata_by_N[N], var_names=param_name)
        # Get the held-out group's samples
        if "group" in samples.dims:
            hg_samples = samples.sel(group=HELD_OUT).values.flatten()
        else:
            hg_samples = samples.values.flatten()

        xs = np.linspace(hg_samples.min(), hg_samples.max(), 200)
        try:
            kde = gaussian_kde(hg_samples)
            ys = kde(xs)
        except Exception:
            ys = np.histogram(hg_samples, bins=30, density=True)[0]
            xs = np.histogram(hg_samples, bins=30, density=True)[1][:-1]

        # Scale KDE to have consistent height
        ys = ys / ys.max() * 0.8
        ax.fill_between(xs, y_positions[N], y_positions[N] + ys,
                        color=colors[N], alpha=0.7)
        ax.plot(xs, y_positions[N] + ys, color=colors[N], linewidth=0.8)

    ax.axvline(true_val, color="grey", linestyle="--", linewidth=1,
               label="True value")

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"N={N}" for N in reversed(NS)])
    ax.set_xlabel(param_label)
    ax.set_title(f"{param_label} posterior for held-out group", fontsize=11)

axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_posterior_evolution.png", dpi=150)
plt.close(fig)


# ── 7. Figure 3: Prediction comparison (faceted) ───────────────────────────
print("Generating Figure 3: Prediction comparison")
x_grid = np.linspace(0, 10, 100)

pooled_alpha_mean = az.extract(idata_pooled, var_names="alpha").mean().item()
pooled_beta_mean = az.extract(idata_pooled, var_names="beta").mean().item()
pooled_line = pooled_alpha_mean + pooled_beta_mean * x_grid

true_line = alpha_true[HELD_OUT] + beta_true[HELD_OUT] * x_grid

n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9),
                         sharex=True, sharey=True)

# Pre-compute hierarchical predictions
hier_data = {}
for N in NS:
    idata = idata_by_N[N]
    alpha_s = az.extract(idata, var_names="alpha").sel(group=HELD_OUT).values
    beta_s = az.extract(idata, var_names="beta").sel(group=HELD_OUT).values
    mean_line = alpha_s.mean() + beta_s.mean() * x_grid
    draws = alpha_s[:, None] + beta_s[:, None] * x_grid[None, :]
    lo = np.percentile(draws, 3, axis=0)
    hi = np.percentile(draws, 97, axis=0)
    hier_data[N] = (mean_line, lo, hi)

# Pre-compute unpooled predictions
unpooled_data = {}
for N in NS:
    idata = unpooled_idata_by_N[N]
    alpha_s = az.extract(idata, var_names="alpha").sel(group=HELD_OUT).values
    beta_s = az.extract(idata, var_names="beta").sel(group=HELD_OUT).values
    mean_line = alpha_s.mean() + beta_s.mean() * x_grid
    draws = alpha_s[:, None] + beta_s[:, None] * x_grid[None, :]
    lo = np.percentile(draws, 3, axis=0)
    hi = np.percentile(draws, 97, axis=0)
    unpooled_data[N] = (mean_line, lo, hi)

# Panel labels
panel_ns = [0, 1, 3, 5, 15]
panel_data_counts = [0, 1, 3, 5, 15]
panel_titles = [
    "No data",
    "1 observation",
    "3 observations",
    "5 observations",
    "15 observations",
]

held_out_indices = np.where(held_out_mask)[0]

for i, (N, title) in enumerate(zip(panel_ns, panel_titles)):
    row, col = divmod(i, n_cols)
    ax = axes[row, col]

    # Show data points: colored if seen in this panel, grey otherwise
    n_seen = panel_data_counts[i] if i > 0 else 0
    for j, pt_idx in enumerate(held_out_indices):
        seen = j < n_seen
        c = "C3" if seen else "lightgrey"
        ax.scatter(x_flat[pt_idx], y_flat[pt_idx], color=c, s=20,
                   alpha=0.8 if seen else 0.4, zorder=3)

    # True line
    ax.plot(x_grid, true_line, color="black", linewidth=1.8,
            linestyle="--", label="True")

    # Pooled baseline
    ax.plot(x_grid, pooled_line, color="grey", linewidth=1.5,
            linestyle=":", label="Pooled")

    # Hierarchical prediction
    mean_line, lo, hi = hier_data[N]
    ax.plot(x_grid, mean_line, color=colors[N], linewidth=2,
            label="Hierarchical", zorder=5)
    ax.fill_between(x_grid, lo, hi, color=colors[N], alpha=0.15, zorder=4)

    # Unpooled (no pooling) prediction
    up_mean, up_lo, up_hi = unpooled_data[N]
    ax.plot(x_grid, up_mean, color="#D55E00", linewidth=1.5,
            linestyle="--", label="No pooling", zorder=3)
    ax.fill_between(x_grid, up_lo, up_hi, color="#D55E00", alpha=0.06,
                    zorder=2)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(y_flat[held_out_mask].min() - 3,
                y_flat[held_out_mask].max() + 3)
    if col == 0:
        ax.set_ylabel("y")
    if row == n_rows - 1 or (row == n_rows - 1 and col < n_cols - 1):
        ax.set_xlabel("x")

# Sixth panel: legend only
ax_leg = axes[1, 2]
ax_leg.axis("off")
legend_elements = [
    plt.Line2D([], [], color="black", linestyle="--", linewidth=1.8, label="True"),
    plt.Line2D([], [], color="grey", linestyle=":", linewidth=1.5, label="Pooled"),
    plt.Line2D([], [], color="#D55E00", linestyle="--", linewidth=1.5,
               label="No pooling"),
    plt.Line2D([], [], color="C0", linewidth=2, label="Hierarchical"),
    plt.Line2D([], [], color="C3", marker="o", linestyle="", markersize=8,
               label="Observed data"),
    plt.Line2D([], [], color="lightgrey", marker="o", linestyle="",
               markersize=8, label="Unobserved"),
]
ax_leg.legend(handles=legend_elements, loc="center", fontsize=10,
              frameon=False)

fig.suptitle("Out-of-sample Predictions for Held-out Group",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_predictions_comparison.png", dpi=150)
plt.close(fig)

# ── 8. Figure 4: Scenario A — Prior vs Posterior at N=0 ────────────────────
print("Generating Figure 4: Prior vs posterior at N=0")

alpha_n0 = az.extract(
    idata_by_N[0], var_names="alpha"
).sel(group=HELD_OUT).values.flatten()
beta_n0 = az.extract(
    idata_by_N[0], var_names="beta"
).sel(group=HELD_OUT).values.flatten()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

param_panels = [
    (axes[0], alpha_n0, alpha_true[HELD_OUT], r"$\alpha_{20}$", (-8, 8)),
    (axes[1], beta_n0, beta_true[HELD_OUT], r"$\beta_{20}$", (-5, 5)),
]

for ax, samples, true_val, param_label, xlim in param_panels:
    xs = np.linspace(xlim[0], xlim[1], 500)

    # Prior N(0, 5)
    prior = norm.pdf(xs, 0, 5)
    prior = prior / prior.max()

    # Posterior KDE at N=0
    kde = gaussian_kde(samples)
    posterior = kde(xs)
    posterior = posterior / posterior.max()

    ax.fill_between(xs, prior, alpha=0.15, color="grey", zorder=1)
    ax.plot(xs, prior, color="grey", linewidth=1.5,
            linestyle="--", label="Prior N(0,5)", zorder=2)

    ax.fill_between(xs, posterior, alpha=0.35, color="#440154", zorder=3)
    ax.plot(xs, posterior, color="#440154", linewidth=2,
            label="Posterior (N=0)", zorder=4)

    ax.axvline(true_val, color="black", linestyle=":",
               linewidth=1.2, label=f"True {param_label}", zorder=5)

    ax.set_xlabel(param_label)
    ax.set_xlim(xlim)
    ax.set_ylabel("Scaled density")
    ax.set_title(param_label, fontsize=12)

axes[0].legend(fontsize=8)
fig.suptitle("Scenario A: Zero Data — Prior vs Posterior",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_zero_data_prior_vs_posterior.png", dpi=150)
plt.close(fig)

print(f"All figures saved to {OUT_DIR}")
print("Done!")
