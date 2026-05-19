# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///

"""Figure 5: Pricing profiles — predictive distributions by risk profile.

Shows the full posterior predictive distribution for different driver
age profiles, demonstrating the richer information the Bayesian model
provides compared to point estimates alone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tweedie_utils import tweedie_random

OUT_DIR = Path(__file__).parents[2] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


rng = np.random.default_rng(42)
n_sim = 10000  # policies per profile
n_ppc = 2000   # draws from posterior

# Three age profiles using GLM coefficients from dataCar model.
# agecat coefficient = -0.144: higher agecat → lower pure premium.
profiles = [
    {"name": "Younger\n(agecat 1\u20132)", "mu": 360, "phi": 174, "p": 1.574, "color": "#E24A33"},
    {"name": "Middle\n(agecat 3\u20134)", "mu": 290, "phi": 174, "p": 1.574, "color": "#348ABD"},
    {"name": "Older\n(agecat 5\u20136)", "mu": 220, "phi": 174, "p": 1.574, "color": "#188487"},
]

# Approximate posterior uncertainty
mu_se = 20
phi_se = 5
p_se = 0.005

fig, ax = plt.subplots(figsize=(10, 5))

rows = []
hist_data = {}

for prof in profiles:
    name = prof["name"]
    mu = prof["mu"]
    phi = prof["phi"]
    p = prof["p"]
    color = prof["color"]

    # Generate draws from posterior predictive
    all_pps = []
    zeros_total = 0
    n_total = 0
    for i in range(n_ppc):
        mu_draw = rng.normal(mu, mu_se)
        phi_draw = rng.normal(phi, phi_se)
        p_draw = np.clip(rng.normal(p, p_se), 1.05, 1.95)
        y_sim = tweedie_random(
            max(mu_draw, 50), max(phi_draw, 10), p_draw, size=n_sim, rng=rng
        )
        all_pps.extend(y_sim)
        zeros_total += np.sum(y_sim == 0)
        n_total += n_sim

    all_pps = np.array(all_pps)
    zero_rate = zeros_total / n_total
    hist_data[name] = all_pps[all_pps > 0]

    mean_pp = np.mean(all_pps)
    p50 = np.percentile(all_pps, 50)
    p95 = np.percentile(all_pps, 95)
    p99 = np.percentile(all_pps, 99)
    p_loss_5k = np.mean(all_pps > 5000)

    rows.append({
        "Profile": name.replace("\n", " "),
        "Expected PP": f"${mean_pp:.0f}",
        "Median PP": f"${p50:.0f}",
        "95th Pctl": f"${p95:,.0f}",
        "P(PP > $5K)": f"{p_loss_5k:.1%}",
        "Zero Rate": f"{zero_rate:.1%}",
    })

# Histogram of per-policy pure premium distributions
bins = np.linspace(0, 3000, 61)
for prof in profiles:
    name = prof["name"]
    color = prof["color"]
    data = hist_data[name]
    mu_val = prof["mu"]
    if len(data) > 5000:
        data = rng.choice(data, 5000, replace=False)
    ax.hist(data, bins=bins, density=True, alpha=0.4, color=color,
            histtype="stepfilled", label=f"{name} (μ=${mu_val:.0f})")

ax.set_xlabel("Pure Premium ($)")
ax.set_ylabel("Density")
ax.set_title("Predictive Distribution by Age Profile\n"
             "(per-policy)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0, 3000)

plt.suptitle("Bayesian Tweedie Pricing: Age-Based Risk Profiles",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_pricing_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR / 'fig_pricing_profiles.png'}")

# Print table values for blog post alignment
print("\nTABLE VALUES FOR BLOG POST:")
print(f"{'Profile':<30s} {'Exp PP':>8s} {'Med PP':>8s} {'95th':>8s} {'P(>5K)':>8s} {'Zero':>8s}")
print("-" * 70)
for r in rows:
    print(f"{r['Profile']:<30s} {r['Expected PP']:>8s} {r['Median PP']:>8s} "
          f"{r['95th Pctl']:>8s} {r['P(PP > $5K)']:>8s} {r['Zero Rate']:>8s}")
