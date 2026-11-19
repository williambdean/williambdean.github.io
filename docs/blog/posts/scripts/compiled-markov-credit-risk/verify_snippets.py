"""Run every snippet/output block from the compiled-markov-credit-risk post.

Prints verified values so the post's output blocks match a real run.
Usage: uv run python docs/blog/posts/scripts/compiled-markov-credit-risk/verify_snippets.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.traversal import explicit_graph_inputs, get_var_by_name
from pymc.pytensorf import rvs_in_graph

from credit_model import (
    DEFAULT_IDX,
    HORIZON,
    PORTFOLIO_VALUE,
    RATINGS,
    build_loss_graph,
    deterministic_trace,
)

SECTION = "=" * 72
SEED = 42


def section(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


def main() -> None:
    # ── 1. pm.compile creates seeded random functions ──
    section("pm.compile: seeded random draws")
    cost = pm.Gamma.dist(mu=410, sigma=100)
    recovery = pm.Beta.dist(mu=0.45, sigma=0.15)
    fn = pm.compile([], [cost, recovery], random_seed=SEED)
    r1, r2 = fn(), fn()
    print(f"Draw 1: cost={r1[0]:.0f}, recovery={r1[1]:.3f}")
    print(f"Draw 2: cost={r2[0]:.0f}, recovery={r2[1]:.3f}")

    # ── 2. dprint of the simple scan + per-call timing ──
    section("Compiled graph (dprint) for the simple scan")
    x = pt.vector("x")
    T = pt.matrix("T")

    def step(x_t, T):
        return x_t @ T

    scan_out = pytensor.scan(
        fn=step,
        outputs_info=[x],
        non_sequences=[T],
        n_steps=5,
        return_updates=False,
    )
    fn_scan = pytensor.function([x, T], scan_out)
    fn_scan.dprint()

    x_val = np.full(7, 1.0 / 7)
    T_val = np.full((7, 7), 0.02)
    np.fill_diagonal(T_val, 0.86)
    fn_scan(x_val, T_val)
    n = 100_000
    t0 = time.perf_counter()
    for _ in range(n):
        fn_scan(x_val, T_val)
    per_call = (time.perf_counter() - t0) / n * 1000
    print(f"\nSimple scan timing: {per_call:.3f} ms per call over {n:,} calls")

    # ── 3. Deterministic Markov trace ──
    section("Markov trace (deterministic, mean transition matrix)")
    trace_det = deterministic_trace()
    print("Year  " + "  ".join(f"{r:>7}" for r in RATINGS))
    for year, row in enumerate(trace_det):
        print(f"{year:>4}  " + "  ".join(f"{v * 100:6.1f}%" for v in row))

    # ── 4. rvs_in_graph ──
    section("rvs_in_graph: detect uncertainty")
    total_loss, dr_path = build_loss_graph(sweep=False)
    rvs = rvs_in_graph([total_loss])
    print(f"Random variables ({len(rvs)}): {sorted(str(r) for r in rvs)}")
    default_rate_det = float(deterministic_trace()[-1, DEFAULT_IDX])
    total_loss_det = default_rate_det * PORTFOLIO_VALUE * 0.55
    rvs_det = rvs_in_graph([pt.as_tensor_variable(total_loss_det)])
    print(f"Deterministic model: {rvs_det}")

    # ── 5. explicit_graph_inputs ──
    section("explicit_graph_inputs: auto-discovery")
    loss_s, _ = build_loss_graph(sweep=True)
    recovery_senior = pt.scalar("recovery_senior")  # defined but unused
    inputs = list(explicit_graph_inputs([loss_s]))
    print(f"Raw discovery: {[inp.name for inp in inputs]}")
    inputs = sorted(inputs, key=lambda v: v.name)
    print(f"Sorted inputs: {[inp.name for inp in inputs]}")

    # ── 6. Sensitivity sweeps with the compiled function ──
    section("LGD sweep (mean of 2,000 draws per point)")
    fn_sens = pm.compile(inputs, loss_s, random_seed=SEED)
    for mult in np.linspace(0.6, 1.2, 7):
        vals = np.array([fn_sens(mult, 1.0) for _ in range(2_000)])
        print(f"LGD {mult:.1f}x → Loss ${vals.mean():>12,.0f}")

    section("PD stress sweep (mean of 2,000 draws per point)")
    for stress in [1.0, 1.5, 2.0, 2.5, 3.0]:
        vals = np.array([fn_sens(1.0, stress) for _ in range(2_000)])
        print(f"PD {stress:.1f}x → Loss ${vals.mean():>12,.0f}")

    # ── 7. Monte Carlo ──
    section("Monte Carlo: 10,000 draws")
    t0 = time.perf_counter()
    fn_psa = pm.compile([], total_loss, random_seed=SEED)
    compile_s = time.perf_counter() - t0
    print(f"(compile time: {compile_s:.2f}s)")
    t0 = time.perf_counter()
    results = np.array([fn_psa() for _ in range(10_000)])
    elapsed = time.perf_counter() - t0
    print(f"Mean loss:         ${results.mean():>12,.0f}")
    print(f"95th percentile:   ${np.percentile(results, 95):>12,.0f}")
    print(f"99.9th percentile: ${np.percentile(results, 99.9):>12,.0f}")
    print(f"{elapsed / len(results) * 1000:.3f} ms per call ({len(results):,} draws)")

    # ── 8. Default rate uncertainty ──
    section("Default rate uncertainty (1,000 draws)")
    _, dr_path2 = build_loss_graph(sweep=False)
    fn_dr = pm.compile([], dr_path2, random_seed=SEED)
    draws = np.array([fn_dr() for _ in range(1_000)])  # (1000, HORIZON)
    for year in range(1, HORIZON + 1):
        col = draws[:, year - 1]
        lo, med, hi = np.percentile(col, [10, 50, 90])
        print(
            f"Year {year}: median {med * 100:.1f}%  "
            f"80% interval [{lo * 100:.1f}%, {hi * 100:.1f}%]"
        )

    # ── 9. Time-varying transition tensor ──
    section("Time-varying transitions (per-year PD ramp [0.9, 1.0, 1.1, 1.2, 1.3])")
    pd_ramp = [0.9, 1.0, 1.1, 1.2, 1.3]
    loss_tv, dr_tv = build_loss_graph(pd_ramp=pd_ramp)

    # Named variables make the internals inspectable
    T_stack_node = get_var_by_name([dr_tv], "T_stack")[0]
    trace_node = get_var_by_name([dr_tv], "trace")[0]
    stack_val = np.asarray(T_stack_node.eval())
    trace_val = np.asarray(trace_node.eval())
    print(f"T_stack shape: {stack_val.shape}")
    print(f"T_stack rows sum to 1 in every year: "
          f"{np.allclose(stack_val.sum(axis=2), 1.0)}")
    print(f"Cohort conserved (trace rows sum to 1): "
          f"{np.allclose(trace_val.sum(axis=1), 1.0)}")

    fn_tv_dr = pm.compile([], dr_tv, random_seed=SEED)
    draws_tv = np.array([fn_tv_dr() for _ in range(1_000)])
    for year in range(1, HORIZON + 1):
        static = np.median(draws[:, year - 1]) * 100
        ramped = np.median(draws_tv[:, year - 1]) * 100
        print(f"Year {year}: static {static:.1f}%  ramped {ramped:.1f}%")

    fn_tv_loss = pm.compile([], loss_tv, random_seed=SEED)
    res_tv = np.array([fn_tv_loss() for _ in range(10_000)])
    print(f"Ramped mean loss: ${res_tv.mean():>12,.0f}")

    # ── 10. get_var_by_name with named landmarks ──
    section("get_var_by_name: named landmarks")
    lgd_multiplier = pt.scalar("lgd_multiplier")
    recovery = 0.45 * lgd_multiplier
    lgd = 1 - recovery
    lgd.name = "lgd"
    toy_loss = 1_000 * lgd
    found = get_var_by_name([toy_loss], "lgd")[0]
    print(f"Found: {found.name}")
    print(f"Value at multiplier 0.8: {found.eval({lgd_multiplier: 0.8})}")

    # ── 11. Posterior draws as an embedded distribution ──
    section("Posterior integration: Categorical index over draws")
    rng = np.random.default_rng(0)
    posterior = rng.normal(loc=[0.10, 0.55], scale=[0.02, 0.15], size=(1_000, 2))
    n = posterior.shape[0]
    idx = pm.Categorical.dist(p=np.full(n, 1 / n))
    stacked = pt.as_tensor_variable(posterior)
    alpha = stacked[:, 0][idx]
    beta = stacked[:, 1][idx]
    fn_post = pm.compile([], [alpha, beta], random_seed=SEED)
    out = np.array([fn_post() for _ in range(5)])
    print(np.round(out, 4))
    rows = {tuple(np.round(r, 6)) for r in posterior}
    print(f"Rows are exact posterior samples: {all(tuple(np.round(r, 6)) in rows for r in out)}")


if __name__ == "__main__":
    main()
