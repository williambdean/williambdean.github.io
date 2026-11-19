"""Regenerate the three sensitivity figures for the compiled-markov-credit-risk post.

Matches the verified values in verify_snippets.py: same seed, same call order
per compiled function, so figure and text agree exactly.

Usage: uv run python docs/blog/posts/scripts/compiled-markov-credit-risk/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from pytensor.graph.traversal import explicit_graph_inputs

from credit_model import HORIZON, RATINGS, build_loss_graph, deterministic_trace

IMAGES = Path(__file__).resolve().parents[2] / "images"
SEED = 42
BLUE = "#1f77b4"
RED = "#e63946"
ORANGE = "#ff9f1c"

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "savefig.dpi": 150,
    }
)


def main() -> None:
    # Same call order as verify_snippets.py so the numbers match the text.
    loss_s, _ = build_loss_graph(sweep=True)
    inputs = sorted(explicit_graph_inputs([loss_s]), key=lambda v: v.name)
    fn_sens = pm.compile(inputs, loss_s, random_seed=SEED)

    mults = np.linspace(0.6, 1.2, 7)
    lgd_mean, lgd_lo, lgd_hi = [], [], []
    for mult in mults:
        vals = np.array([fn_sens(mult, 1.0) for _ in range(2_000)])
        lgd_mean.append(vals.mean())
        lgd_lo.append(np.percentile(vals, 10))
        lgd_hi.append(np.percentile(vals, 90))
    lgd_mean = np.array(lgd_mean) / 1e6
    lgd_lo = np.array(lgd_lo) / 1e6
    lgd_hi = np.array(lgd_hi) / 1e6

    stresses = [1.0, 1.5, 2.0, 2.5, 3.0]
    pd_mean, pd_lo, pd_hi = [], [], []
    for stress in stresses:
        vals = np.array([fn_sens(1.0, stress) for _ in range(2_000)])
        pd_mean.append(vals.mean())
        pd_lo.append(np.percentile(vals, 10))
        pd_hi.append(np.percentile(vals, 90))
    pd_mean = np.array(pd_mean) / 1e6
    pd_lo = np.array(pd_lo) / 1e6
    pd_hi = np.array(pd_hi) / 1e6

    total_loss, _ = build_loss_graph(sweep=False)
    fn_psa = pm.compile([], total_loss, random_seed=SEED)
    results = np.array([fn_psa() for _ in range(10_000)]) / 1e6
    p95 = np.percentile(results, 95)
    p999 = np.percentile(results, 99.9)

    baseline = lgd_mean[list(mults).index(1.0)]  # LGD 1.0x

    # ── Figure 1: standalone LGD sweep ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mults, lgd_mean, "o-", color=BLUE, lw=2.5, ms=9, zorder=3)
    ax.fill_between(mults, lgd_lo, lgd_hi, color=BLUE, alpha=0.15, label="80% interval")
    ax.axhline(baseline, color=ORANGE, ls="--", alpha=0.8)
    ax.axvline(1.0, color=ORANGE, ls="--", alpha=0.8)
    ax.annotate(
        f"${baseline:.1f}M",
        (1.0, baseline),
        xytext=(8, 8),
        textcoords="offset points",
        color="tab:orange",
        fontweight="bold",
    )
    ax.annotate(
        f"${lgd_mean[0]:.1f}M",
        (mults[0], lgd_mean[0]),
        xytext=(10, 8),
        textcoords="offset points",
    )
    ax.annotate(
        f"${lgd_mean[-1]:.1f}M",
        (mults[-1], lgd_mean[-1]),
        xytext=(-12, 12),
        textcoords="offset points",
        ha="right",
    )
    ax.set_xlabel("LGD Multiplier")
    ax.set_ylabel("Expected Loss ($M)")
    ax.set_ylim(0, None)
    ax.set_title("Sensitivity to Loss Given Default (LGD)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES / "credit_migration_lgd_sweep.png")
    plt.close(fig)

    # ── Figure 2: portfolio evolution (stacked area) ──
    trace_pct = deterministic_trace() * 100  # (HORIZON + 1, 7)
    years = np.arange(HORIZON + 1)
    colors = [
        "#4caf50",  # AAA
        "#8bc34a",  # AA
        "#ffc107",  # A
        "#ff9800",  # BBB
        "#f4511e",  # BB
        "#e53935",  # B
        "#9c27b0",  # D
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(years, trace_pct.T, labels=RATINGS, colors=colors)
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 100)
    ax.set_xticks(years)
    ax.set_xticklabels([f"Year {y}" for y in years])
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in np.arange(0, 101, 20)])
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Fraction")
    ax.set_title(
        "Credit Migration: Portfolio Evolution Over 5 Years",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "credit_migration_trace.png")
    plt.close(fig)

    # ── Figure 3: static vs ramped default path ──
    _, dr_static = build_loss_graph(sweep=False)
    fn_dr_static = pm.compile([], dr_static, random_seed=SEED)
    draws_static = np.array([fn_dr_static() for _ in range(1_000)])

    _, dr_ramp = build_loss_graph(pd_ramp=[0.9, 1.0, 1.1, 1.2, 1.3])
    fn_dr_ramp = pm.compile([], dr_ramp, random_seed=SEED)
    draws_ramp = np.array([fn_dr_ramp() for _ in range(1_000)])

    med_static = np.median(draws_static, axis=0) * 100
    med_ramp = np.median(draws_ramp, axis=0) * 100
    yrs = np.arange(1, HORIZON + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yrs, med_static, "o-", color=BLUE, lw=2.5, ms=9, label="Static")
    ax.plot(yrs, med_ramp, "s-", color=RED, lw=2.5, ms=9, label="Ramped")
    ax.set_xlim(0.8, HORIZON + 0.2)
    ax.set_ylim(0, None)
    ax.set_xticks(yrs)
    ax.set_xticklabels([f"Year {y}" for y in yrs])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Path: Static vs Ramped Transition Tensor", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES / "credit_migration_default_path.png")
    plt.close(fig)

    # ── Figure 3: PSA histogram + both sweeps ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(results, bins=50, color="#3d9be9", edgecolor="white", linewidth=0.3)
    ax.axvline(p95, color=ORANGE, ls="--", lw=2.5, label=f"95th: ${p95:.1f}M")
    ax.axvline(p999, color=RED, ls="--", lw=2.5, label=f"99.9th: ${p999:.1f}M")
    ax.set_xlabel("Loss ($M)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Loss Distribution (10,000 iterations)", fontsize=13)
    ax.legend()

    ax = axes[1]
    ax.plot(mults, lgd_mean, "o-", color=BLUE, lw=2.5, ms=8)
    ax.fill_between(mults, lgd_lo, lgd_hi, color=BLUE, alpha=0.15)
    ax.set_xlabel("LGD Multiplier")
    ax.set_ylabel("Expected Loss ($M)")
    ax.set_title("Sensitivity to LGD", fontsize=13)

    ax = axes[2]
    ax.plot(stresses, pd_mean, "s-", color=RED, lw=2.5, ms=8)
    ax.fill_between(stresses, pd_lo, pd_hi, color=RED, alpha=0.15)
    ax.set_xlabel("PD Stress Multiplier")
    ax.set_ylabel("Expected Loss ($M)")
    ax.set_title("Sensitivity to PD Stress", fontsize=13)

    # Shared y-limits so the two panels are directly comparable
    ymax = 1.05 * max(lgd_hi.max(), pd_hi.max())
    axes[1].set_ylim(0, ymax)
    axes[2].set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(IMAGES / "credit_migration_results.png")
    plt.close(fig)

    # ── Figure 4: two-way heatmap ──
    lgd_grid = np.linspace(0.6, 1.2, 13)
    pd_grid = np.linspace(1.0, 3.0, 21)
    Z = np.zeros((len(pd_grid), len(lgd_grid)))
    for i, stress in enumerate(pd_grid):
        for j, mult in enumerate(lgd_grid):
            vals = np.array([fn_sens(mult, stress) for _ in range(100)])
            Z[i, j] = vals.mean() / 1e6

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(False)
    mesh = ax.pcolormesh(lgd_grid, pd_grid, Z, cmap="YlOrRd", shading="auto", vmin=0)
    ax.set_xticks(np.linspace(0.6, 1.2, 7))
    ax.set_yticks(np.linspace(1.0, 3.0, 9))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Loss ($M)")
    ax.set_xlabel("LGD Multiplier")
    ax.set_ylabel("PD Stress Multiplier")
    ax.set_title("Expected Loss: LGD × PD Stress Interaction", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGES / "credit_migration_heatmap.png")
    plt.close(fig)

    print(f"Wrote 5 figures to {IMAGES}")
    print(f"Monte Carlo mean ${results.mean():.2f}M, 95th ${p95:.2f}M, 99.9th ${p999:.2f}M")
    print(f"LGD sweep: ${lgd_mean[0]:.2f}M → ${lgd_mean[-1]:.2f}M (baseline ${baseline:.2f}M)")
    print(f"PD sweep:  ${pd_mean[0]:.2f}M → ${pd_mean[-1]:.2f}M")


if __name__ == "__main__":
    main()
