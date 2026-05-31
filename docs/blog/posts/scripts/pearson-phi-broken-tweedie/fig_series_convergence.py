# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Figure: Series convergence — log-pdf error vs number of terms.

Shows how rapidly the Tweedie series expansion converges for typical
insurance parameters. Even 5 terms match 100 terms to machine precision.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tweedie_utils import tweedie_logp_series

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n_test = 1000
mu, phi, p = 293.0, 174.0, 1.574

y_pos = rng.gamma(shape=2, scale=mu / 2, size=n_test)
y_zero = np.zeros(n_test // 2)
y_test = np.concatenate([y_pos, y_zero])
rng.shuffle(y_test)

ref_terms = 200
ref_logp = tweedie_logp_series(y_test, mu, phi, p, n_terms=ref_terms)

term_grid = np.arange(1, 51)
max_abs_errors = []
for n in term_grid:
    logp_n = tweedie_logp_series(y_test, mu, phi, p, n_terms=n)
    max_abs_errors.append(np.max(np.abs(logp_n - ref_logp)))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(term_grid, max_abs_errors, "b.-", linewidth=1.5, markersize=4)
ax.axhline(1e-15, color="gray", linestyle=":", alpha=0.5, label="Machine epsilon")
ax.set_xlabel("Number of series terms")
ax.set_ylabel("Max |log-pdf error| vs 200-term reference")
ax.set_title("Series Convergence: Maximum Log-PDF Error by Number of Terms\n"
             f"(μ={mu}, φ={phi}, p={p}, {n_test} test points)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Annotate the rapid convergence
ax.annotate(f"{max_abs_errors[4]:.1e} at 5 terms",
            xy=(5, max_abs_errors[4]),
            xytext=(15, max_abs_errors[4] * 10),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="C0", lw=1),
            color="C0")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_series_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_series_convergence.png'}")
