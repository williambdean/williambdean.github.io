# /// script
# dependencies = ["numpy", "pymc", "pytensor"]
# ///

"""Verify the Tweedie dist function against theoretical values.

Compares the symbolic PyMC tweedie_dist (used by sample_prior_predictive
and sample_posterior_predictive) against:
  - theoretical values (E[Y]=mu, Var[Y]=phi*mu^p, P(Y=0)=exp(-lambda))
  - the numpy tweedie_random reference (known correct)

Tests two versions: the blog post version (suspected bug) and the corrected
version (beta as rate = 1/(phi*(p-1)*mu^(p-1))).
"""

import sys

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as px

# ---- Import the known-correct numpy reference ----
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from tweedie_utils import tweedie_random


def tweedie_dist_buggy(mu, phi, p):
    """Original blog post version (suspected wrong)."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha_term = (2 - p) / (p - 1)
    beta = phi * (p - 1) * mu ** (p - 1)
    N = pmd.Poisson.dist(mu=lam)
    Y = pmd.Gamma.dist(alpha=px.math.maximum(N * alpha_term, 1e-10), beta=beta)
    return px.math.where(N > 0, Y, 0.0)


def tweedie_dist_correct(mu, phi, p):
    """Corrected version: beta is rate = 1/scale."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha_term = (2 - p) / (p - 1)
    beta = 1.0 / (phi * (p - 1) * mu ** (p - 1))
    N = pmd.Poisson.dist(mu=lam)
    Y = pmd.Gamma.dist(alpha=px.math.maximum(N * alpha_term, 1e-10), beta=beta)
    return px.math.where(N > 0, Y, 0.0)


def theoretical_values(mu, phi, p):
    """Compute theoretical Tweedie moments."""
    lam = mu ** (2 - p) / (phi * (2 - p))
    zero_rate = np.exp(-lam)
    return {
        "mean": float(mu),
        "std": float(np.sqrt(phi * mu**p)),
        "zero_rate": float(zero_rate),
    }


def draw_and_stats(dist, draws=50_000, rng=None):
    """Draw from a symbolic dist and return mean, std, zero_rate."""
    samples = pm.draw(dist, draws=draws, random_seed=rng)
    samples = np.asarray(samples).ravel()
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "zero_rate": float(np.mean(samples == 0)),
    }


def compare(name, stats, theo, tol_mean=0.05, tol_zero=0.01):
    """Compare sampled stats against theoretical values."""
    results = []
    passed = True

    for key in ("mean", "std", "zero_rate"):
        s, t = stats[key], theo[key]
        rel_err = abs(s - t) / max(abs(t), 1e-10)
        tol = tol_zero if key == "zero_rate" else tol_mean
        ok = rel_err < tol
        if not ok:
            passed = False
        status = "✓" if ok else "✗"
        results.append(f"  {status} {key:<10s}: {s:>14.4f} (theory={t:>14.4f}, rel_err={rel_err:>8.4%})")

    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"{'='*75}")
    for r in results:
        print(r)
    print(f"  {'ALL PASS' if passed else 'FAILURES DETECTED'}")
    return passed


def compare_vs_numpy(name, pymc_stats, numpy_stats):
    """Compare PyMC dist stats against numpy reference."""
    print(f"\n{'='*75}")
    print(f"  {name} vs numpy reference")
    print(f"{'='*75}")
    all_ok = True
    for key in ("mean", "std", "zero_rate"):
        s_pymc, s_np = pymc_stats[key], numpy_stats[key]
        rel_diff = abs(s_pymc - s_np) / max(abs(s_np), 1e-10)
        ok = rel_diff < 0.05
        if not ok:
            all_ok = False
        status = "✓" if ok else "✗"
        print(f"  {status} {key:<10s}: pymc={s_pymc:>14.4f}  numpy={s_np:>14.4f}  diff={rel_diff:>8.4%}")
    print(f"  {'MATCHES NUMPY' if all_ok else 'DIVERGES FROM NUMPY'}")
    return all_ok


def main():
    test_cases = [
        {"mu": 10.0, "phi": 2.0, "p": 1.5, "label": "μ=10, φ=2.0, p=1.5"},
        {"mu": 50.0, "phi": 1.5, "p": 1.3, "label": "μ=50, φ=1.5, p=1.3"},
        {"mu": 293.0, "phi": 174.0, "p": 1.574, "label": "μ=293, φ=174, p=1.574 (dataCar-like)"},
    ]

    draws = 50_000
    all_passed = True
    all_match_numpy = True

    for tc in test_cases:
        mu = tc["mu"]
        phi = tc["phi"]
        p = tc["p"]
        label = tc["label"]

        print(f"\n{'#'*75}")
        print(f"# {label}")
        print(f"{'#'*75}")

        # Seed for reproducibility
        rng = np.random.default_rng(42)
        seed = 42

        theo = theoretical_values(mu, phi, p)
        print(f"\n  Theoretical: mean={theo['mean']:.4f}, std={theo['std']:.4f}, zero_rate={theo['zero_rate']:.4%}")

        # ---- 1. Numpy reference ----
        np_samples = tweedie_random(mu, phi, p, size=draws, rng=rng)
        np_stats = {
            "mean": float(np.mean(np_samples)),
            "std": float(np.std(np_samples)),
            "zero_rate": float(np.mean(np_samples == 0)),
        }
        compare("numpy reference (tweedie_random)", np_stats, theo)

        # ---- 2. Buggy PyMC dist ----
        dist_buggy = tweedie_dist_buggy(mu, phi, p)
        buggy_stats = draw_and_stats(dist_buggy, draws=draws, rng=seed)
        passed_buggy = compare("BUGGY tweedie_dist (blog post)", buggy_stats, theo)
        match_buggy = compare_vs_numpy("BUGGY tweedie_dist", buggy_stats, np_stats)
        if not passed_buggy:
            all_passed = False
        if not match_buggy:
            all_match_numpy = False

        # ---- 3. Corrected PyMC dist ----
        dist_correct = tweedie_dist_correct(mu, phi, p)
        correct_stats = draw_and_stats(dist_correct, draws=draws, rng=seed)
        passed_correct = compare("CORRECT tweedie_dist", correct_stats, theo)
        match_correct = compare_vs_numpy("CORRECT tweedie_dist", correct_stats, np_stats)
        if not passed_correct:
            all_passed = False
        if not match_correct:
            all_match_numpy = False

    # ---- Final verdict ----
    print(f"\n{'='*75}")
    print(f"  SUMMARY")
    print(f"{'='*75}")
    if all_passed:
        print(f"  Buggy version FAILED theory check — CONFIRMED BUG")
    else:
        print(f"  Buggy version MAY have passed — UNEXPECTED, investigate")

    print(f"  Corrected version matches numpy reference: {'YES' if all_match_numpy else 'NO'}")

    if all_passed:
        print(f"\n  >>> Bug confirmed. Proceed with fix in 3 locations. <<<")
    else:
        print(f"\n  >>> Unexpected results. Investigate before proceeding. <<<")


if __name__ == "__main__":
    main()
