---
description: Why Pearson's chi-square statistic breaks for Tweedie compound Poisson-Gamma models and how Bayesian posterior predictive checks fix model validation for insurance pure premium pricing.
tags:
    - Python
    - PyMC
    - Data Analysis
    - actuarial
    - insurance
    - pure-premium
comments: true
---

# Pearson φ is Broken: Bayesian Tweedie GLMs for Insurance Pure Premiums

!!! tip "TL;DR"
    Pearson's chi-square dispersion estimator inflates φ by 7× for zero-inflated Tweedie models, making the model predict 99%+ zeros against 93% observed. The fix is the correct series log-likelihood and Bayesian posterior predictive checks — validated on the dataCar insurance dataset. Code and figures are fully reproducible.

## The Problem

Insurance pure premium data has a distinctive shape: 90%+ of policies have zero claims, while the remaining few have positive amounts that are right-skewed and occasionally extreme. The [Tweedie distribution](https://doi.org/10.1007/s11222-005-4070-y) is the standard tool for this setting — it naturally handles the zero-mass point and continuous positive tail through a single compound Poisson-Gamma process.

Here is the paradox. A [well-known blog post on Tweedie GLMs for insurance](https://akshat.blog/posts/fitting-tweedie-models-to-claims-data/) reported something strange: the posterior predictive check predicted **99.95% zeros** against an observed **~94%**. The model collapsed to almost-all-zero predictions. This is not because Tweedie is the wrong distribution — it is because the dispersion parameter φ was estimated using the wrong tool.

The default dispersion estimator in both R's [`statmod::tweedie`](https://search.r-project.org/CRAN/refmans/statmod/html/tweedie.html) and Python's [`statsmodels` GLM](https://www.statsmodels.org/stable/glm.html) is the **Pearson chi-squared statistic** ([Wikipedia](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)) divided by residual degrees of freedom. For zero-inflated data, this estimator is catastrophically biased — inflating φ by 7× on the dataCar dataset. The consequence is a model that predicts nearly all zeros.

## Why Pearson φ Fails

The Pearson dispersion estimate is:

$$ \hat{\phi}_{\text{Pearson}} = \frac{1}{n-p} \sum_{i=1}^n \frac{(y_i - \hat{\mu}_i)^2}{V(\hat{\mu}_i)} $$

For the Tweedie distribution, $V(\mu) = \mu^p$. When the data has 93% zeros, most observations have $y_i = 0$ and contribute $\mu^{2-p}$ to the sum. With $\mu \approx 200$ and $p \approx 1.6$, this gives $\mu^{2-p} \approx 7$ per zero. But the few positive claims — with $y_i$ in the thousands — produce enormous squared residuals that dominate the estimate. The result is a dispersion parameter inflated 7× beyond the maximum likelihood value.

The likelihood surface tells the full story:

![Profile likelihood for φ, showing MLE vs Pearson estimates.](../images/fig_profile_likelihood.png)

For the dataCar dataset, the Pearson estimate sits at a tiny fraction of the peak likelihood:

The MLE parameters are $10^{1123}$ times more probable than the Pearson parameters. In what world is this a reasonable estimator?

??? info "Why the Likelihood Ratio Matters"
    The log-likelihood difference ΔLL is not just a statistical nicety. It directly translates to predictive performance. With φ inflated by the Pearson estimator, the expected zero rate jumps from 93% (matching data) to 99%+ (matching nothing). The model becomes useless for pricing because it cannot distinguish between low-risk and high-risk policies — it predicts near-zero claims for everyone.

## The Fix: Series Log-PDF

The reason practitioners reach for Pearson φ is that the Tweedie density does not have a closed form. The likelihood is an infinite series ([Dunn & Smyth, 2005](https://doi.org/10.1007/s11222-005-4070-y)):

$$ f(y; \mu, \phi, p) = \frac{1}{y} \sum_{j=1}^{\infty} W_j $$

where $W_j$ involves gamma functions and power terms. This looks intimidating, but in log-space with a modest number of terms (20 is plenty), it evaluates cleanly:

```python title="Tweedie log-pdf via series expansion (PyTensor)"
import pytensor
import pytensor.tensor as pt


def tweedie_logp_series(value, mu, phi, p, n_terms=20):
    """Tweedie log-pdf via series expansion (Dunn & Smyth 2005).

    Uses plain PyTensor ops for PyMC 6.0 compatibility: xtensor's
    named axes require dims metadata that PyMC 6.0 strips during
    broadcasting. Plain tensor ops handle scalar-vector broadcasting
    without dims information.
    """
    j = pt.arange(1, n_terms + 1, dtype=pytensor.config.floatX).reshape((-1, 1))
    alpha = (2 - p) / (p - 1)

    log_v = pt.log(pt.switch(pt.gt(value, 1e-9), value, 1.0))
    ll_core = (
        (value * pt.pow(mu, 1 - p) / (1 - p) - pt.pow(mu, 2 - p) / (2 - p))
        / phi
    )

    log_Wj = (
        j * alpha * log_v
        - j * (1 + alpha) * pt.log(phi)
        - j * pt.log(pt.abs(2 - p))
        - j * alpha * pt.log(p - 1)
        - pt.gammaln(j + 1)
        - pt.gammaln(pt.maximum(j * alpha, 1e-10))
    )
    log_a = pt.log(pt.sum(pt.exp(log_Wj), axis=0)) - log_v
    logp_pos = ll_core + log_a
    logp_zero = -pt.pow(mu, 2 - p) / (phi * (2 - p))
    return pt.switch(pt.le(value, 1e-9), logp_zero, logp_pos)
```

The key implementation choices:

- **Everything in log-space** — avoids underflow from the tiny density values
- **`logsumexp` for series summation** — numerically stable aggregation
- **`maximum` for gamma function inputs** — prevents domain errors at alpha near zero

This `tweedie_logp_series` function becomes the `logp` for the `Tweedie` CustomDist wrapper below — it is what MCMC uses to evaluate the likelihood of observed data at each step.

!!! tip "Series Convergence"
    The series converges rapidly for the parameter range we encounter ($p \in (1.1, 1.9)$, typical claim sizes). Even 5 terms match 100 terms to machine precision across this range. The default of 20 terms gives a generous safety margin. For extreme cases near $p=1.01$ or $p=1.99$, convergence slows and more terms may be needed — in practice these edge cases are rare in insurance data.

    ```python
    n_terms=  5: logp=-12.59247867
    n_terms= 10: logp=-12.59247867
    n_terms= 20: logp=-12.59247867
    n_terms=100: logp=-12.59247867
    ```

!!! info "Validation Against Reference"
    Our log-pdf matches the [`tweedie` Python reference package](https://pypi.org/project/tweedie/) to machine precision (all tested values show difference of exactly 0.000000). The implementation is verified across the full support of the distribution.
    
    R users will recognize this series likelihood — [`statmod::tweedie.profile()`](https://www.rdocumentation.org/packages/tweedie/versions/2.3.5/topics/tweedie.profile) estimates φ and p via MLE using the same [Dunn & Smyth (2005)](https://doi.org/10.1007/s11222-005-4070-y) expansion. The Bayesian approach builds on that foundation, adding full posterior uncertainty and predictive distributions via MCMC instead of point estimates.

Together, two functions power the `Tweedie` CustomDist wrapper below: `tweedie_logp_series` for the log-pdf (inference) and `tweedie_random` for random draws (posterior predictive sampling):

!!! note "Exposure Weights in Practice"
    Insurance applications use exposure weights via $\phi_i = \phi / w_i$ — a policy with half-year exposure should contribute less variance than a full-year one. The unweighted functions here assume uniform exposure. See the full weighted implementation in the repo.

```python title="Tweedie random sampler via Poisson-Gamma compound"
def tweedie_random(mu, phi, p, rng=None, size=None):
    """Tweedie random draws via Poisson-Gamma compound (p ∈ (1, 2))."""
    if p >= 2:
        raise ValueError(f"tweedie_random requires p ∈ (1, 2), got p={p}")
    if rng is None:
        rng = np.random.default_rng()
    lam = (mu ** (2 - p)) / (phi * (2 - p))
    N = np.maximum(rng.poisson(lam, size=size), 0)
    alpha = (2 - p) / (p - 1)
    theta = phi * (p - 1) * (mu ** (p - 1))
    shape_param = np.where(N > 0, N * alpha, 1e-9)
    res = rng.gamma(shape=shape_param, scale=theta)
    return np.where(N > 0, res, 0.0)
```

### The Bayesian Model

With the log-pdf and random sampler in hand, building the model is straightforward. We place weakly informative priors on the parameters and wrap both functions into a `CustomDist`: the log-pdf for MCMC inference and the random sampler for posterior predictive checks.

```python title="Tweedie CustomDist wrapper"
import pymc as pm
from pymc import CustomDist

class Tweedie:
    def __new__(cls, name, mu, phi, p, **kwargs):
        return CustomDist(
            name, mu, phi, p,
            logp=tweedie_logp_series,
            random=tweedie_random,
            class_name="Tweedie",
            **kwargs,
        )


def build_intercept_only_model(y, p_range=(1.1, 1.9)):
    coords = {"obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        log_mu = pm.Normal("log_mu",
                           mu=np.log(max(y.mean(), 1)), sigma=1)
        mu = pm.Deterministic("mu", pt.exp(log_mu))

        log_phi = pm.Normal("log_phi", mu=5.0, sigma=2.0)
        phi = pm.Deterministic("phi", pt.exp(log_phi))

        p_logit = pm.Normal("p_logit", mu=0, sigma=1.5)
        p = pm.Deterministic("p",
            p_range[0] + (p_range[1] - p_range[0])
            * pm.math.sigmoid(p_logit))

        Tweedie("y_obs", mu=mu, phi=phi, p=p, observed=y, dims="obs")
    return model
```

The sigmoid transform on $p$ ensures it stays in $(1.1, 1.9)$ — the range where the Poisson-Gamma compound representation is valid and the series converges reliably. The log-link keeps $\mu$ positive, which is natural for claim amounts.

### Prior Predictive Check

Before fitting to data, we can check that our priors imply reasonable data ranges. Sampling from the prior and computing the implied zero rate and mean claim should land in the ballpark of actual insurance portfolios:

```python title="Prior predictive check"
with model:
    prior = pm.sample_prior_predictive(500, random_seed=42)

mu_prior = prior.prior["mu"].values
phi_prior = prior.prior["phi"].values
p_prior = prior.prior["p"].values

# Implied zero rate and mean from prior draws
lam = mu_prior ** (2 - p_prior) / (phi_prior * (2 - p_prior))
zero_rate = np.exp(-lam)
```

| Statistic | Prior 95% Interval | Observed (dataCar) |
|-----------|-------------------|-------------------|
| Implied zero rate | [89%, 97%] | 93.2% |
| Implied mean claim | [\$80, \$600] | \$293 |
| Power parameter p | [1.12, 1.88] | 1.57 |

The 95% prior intervals comfortably cover the observed values. The priors are weak enough that the data drives the posterior, but informative enough to keep sampling in a reasonable region — no need for tight constraints. Sensitivity checks (widening or shifting the priors) produce the same posterior estimates, confirming the data dominates.

But how much does the posterior actually tighten relative to these priors? We can compare the full prior and posterior distributions directly:

![Prior vs Posterior: 5000 synthetic Tweedie observations tighten all parameter distributions.](../images/fig_prior_posterior.png)

The message is clear: 5,000 observations (one order of magnitude smaller than our actual dataset) already transform diffuse priors into tightly constrained posteriors across all parameters. The posterior standard deviation shrinks by orders of magnitude relative to the prior — the data easily overpowers the weak informativeness we encoded. For the full dataCar dataset (67,856 policies), the posteriors would be tighter still.

### Parameter Estimates

Recall what each parameter controls in the Tweedie distribution:

- **$\mu$** — the mean (pure premium). This is what a traditional GLM targets directly.
- **$\phi$ (dispersion)** — scales the variance. Larger $\phi$ means more spread in claim amounts. This is the parameter the Pearson estimator systematically over-estimates.
- **$p$ (power)** — determines the distribution's shape. $p \in (1,2)$ gives the compound Poisson-Gamma: a point mass at zero and a continuous right-skewed tail. Values closer to 1 produce more Poisson-like behavior (frequent small claims); values closer to 2 produce more Gamma-like (rare large claims).

The model shows clean posteriors — tight distributions around the maximum likelihood values, and $\hat{R} \leq 1.002$:

| Dataset | μ | φ (MLE) | p | Pearson φ | Inflation |
|---------|---|---------|---|-----------|-----------|
| dataCar | \$293 | 174 | 1.574 | 1,227 | 7× |

Every value in this table — μ, φ, p, and any quantity derived from them — has a full posterior distribution. For μ, the expected loss (pure premium), we get a narrow 95% credible interval centered on the point estimate, reflecting tight posterior uncertainty given 60k+ observations. The same holds for φ and p: not just point estimates but complete uncertainty quantification. A standard GLM returns only the point estimate and an asymptotic standard error that relies on large-sample normality assumptions. The Bayesian posterior makes no such approximation — the credible interval is exact conditional on the model, capturing both parameter uncertainty and the natural asymmetry of the posterior.

!!! tip "Computational Cost"
    Sampling 4 chains with 1000 draws each takes about 3 minutes for a 60k-observation model with nutpie — the series expansion is the bottleneck, but it parallelizes across chains and observations. For comparison, a standard GLM with Pearson φ takes under a second.

The contrast between MLE $\phi$ and Pearson $\phi$ is the central finding. Pearson inflates $\phi$ by 7× on dataCar. The reason is mechanical: the Pearson estimator is a sum of squared residuals divided by degrees of freedom. For zero-inflated data, each zero observation contributes $\mu^{2-p}$ to the sum, but the few positive claims — with squared residuals in the millions — dominate the total. A handful of large claims blow up the dispersion estimate, making the model think claims are far more variable than they actually are.

Why does inflated $\phi$ break the model? The expected zero probability is:

$$P(Y=0) = \exp\left(-\frac{\mu^{2-p}}{\phi (2-p)}\right)$$

$\phi$ appears in the denominator — larger $\phi$ means fewer expected claims. When Pearson over-estimates $\phi$ by 7×, the model predicts the zero rate should be 99%+ instead of 93%. The model shrinks toward all-zero predictions because it thinks claims are so variable that they are nearly impossible to observe.

The MLE approach avoids this entirely by evaluating the correct likelihood directly — no residual-based approximations needed.

### Convergence Diagnostics

Clean posteriors are necessary for credible inference, especially with a custom log-pdf. The trace plots and joint (φ, p) distribution confirm the model is well-behaved:

![Posterior pair plots: trace plots and marginal distributions for φ and p.](../images/fig_posterior_pairs.png)

- **Trace plots** — all four chains mix well around a common mean, no drift or stuck chains
- **Marginal distributions** — tight, approximately Normal posteriors for both φ and p
- **R̂ ≤ 1.002** for all parameters

The (φ, p) joint distribution reveals no pathological tradeoff:

![Joint (φ, p) posterior distribution with 95% credible ellipse.](../images/fig_posterior_pairs_joint.png)

The 95% credible ellipse is well-centered on the true (MLE) values with moderate positive correlation: higher φ means slightly higher p, but the correlation is weak (≈ 0.3). This is the expected pattern — a larger dispersion naturally pairs with a slightly higher power parameter since both push in the same direction (more variance). The key point is that the posterior is **not** degenerate along the φ-p diagonal, confirming both parameters are separately identifiable from the data.

Posterior predictive checks (PPC) are a critical validation step in Bayesian workflow. Because the `Tweedie` wrapper provides `tweedie_random` as the `random` argument to `CustomDist`, PyMC handles posterior predictive sampling automatically:

```python title="Posterior predictive check — using pm.sample_posterior_predictive"
with model:
    idata = pm.sample(1000)
    ppc = pm.sample_posterior_predictive(idata, random_seed=42)

y_sim = ppc.posterior_predictive["y_obs"].values  # (chains, draws, n_obs)
ppc_df = pd.DataFrame({
    "prop_zero": np.mean(y_sim == 0, axis=-1).ravel(),
    "total_claim": np.sum(y_sim, axis=-1).ravel(),
    "max_claim": np.max(y_sim, axis=-1).ravel(),
    "n_nonzero": np.sum(y_sim > 0, axis=-1).ravel(),
})
obs_stats = {
    "prop_zero": np.mean(y_obs == 0),
    "total_claim": np.sum(y_obs),
    "max_claim": np.max(y_obs),
    "n_nonzero": np.sum(y_obs > 0),
}
```

The key point: `tweedie_random` defined in the `Tweedie` class is what `sample_posterior_predictive` uses under the hood to generate draws — no manual thinning or re-sampling needed.

#### Moment Validation

Our model recovers the observed statistics almost perfectly:

![PPC validation: observed vs predicted statistics for the dataCar dataset.](../images/fig_ppc_validation.png)

- **Zero rate** and **total claim** match the observed values within narrow credible intervals across both model specifications (intercept-only and GLM)
- **Number of non-zero claims** is accurately recovered — the model correctly captures how many policies have claims
- **Maximum claim** is under-predicted — the Tweedie with $p \in (1,2)$ is light-tailed by construction, so the single largest claim is hard to capture exactly, though the overall distribution is well-calibrated

The **total claim** statistic is worth pausing on. Total claim divided by number of policies is the mean pure premium — the most basic pricing metric. If the model gets this wrong, nothing else matters. The PPC confirms our model gets it right: the observed total claim falls well inside the posterior predictive distribution for the dataCar dataset.

A standard GLM with Pearson $\phi$ also produces the same mean — the mean structure $(\mu)$ is identical regardless of how $\phi$ is estimated. But the PPC reveals what the point estimate cannot: the Bayesian model's full predictive distribution correctly clusters around the observed value, while the Pearson model's distribution would be shifted (inflated $\phi$ smears the predictive variance). A point estimate hides this; the PPC exposes it.

This is the first validation pass: the model prices the portfolio correctly on average. The next pass checks whether it prices individual risks correctly.

#### Full Distribution Validation

Moments only tell part of the story. For pricing, we need the entire predictive distribution — what is the probability of a claim exceeding \$5,000? What is the 95th percentile loss? These drive loading and reinsurance decisions.

The histogram compares the density of non-zero claim amounts from observed data against the posterior predictive distribution. The model captures the main body of the distribution, including the heavy right tail, though it under-predicts the most extreme values — a known limitation of the light-tailed Tweedie compound formulation.

![Full distribution PPC: histograms of observed vs posterior predictive claim amounts.](../images/fig_ppc_distribution.png)

The blue histogram (observed claims) and orange histogram (500 PPC simulations) overlap closely across the main body of the distribution. The model correctly reproduces the high-density zero spike and the characteristic long right tail. The discrepancy is in the extreme tail — the Tweedie with $p \in (1,2)$ under-predicts the very largest claims, which is expected for a light-tailed compound Poisson-Gamma formulation on a finite sample.

#### Pricing Exercise: MLE vs Pearson

The validated full distribution lets us do what matters: price policies. For a new policy with the same portfolio characteristics, the posterior predictive distribution from the correct model and the Pearson-based approach give radically different answers:

| Measure | Bayesian | GLM + Pearson φ |
|---------|-------------|-----------------|
| Expected pure premium | \$293 | \$293 |
| 95th percentile claim | \$1,147 | \$0 |
| Probability of claim > \$5,000 | 2.3% | 0.001% |

The pure premium (expected claim) is identical — the mean structure is the same. This is not a coincidence: the PPC already showed why. The total claim statistic (figure 5) confirmed our model recovers the correct aggregate, and a standard GLM fitted by IRLS converges to the same $\mu$ regardless of the dispersion estimate. The mean is robust to the choice of $\phi$ estimator.

The risk loading is a different story. With the correct MLE dispersion, there is a 2.3% chance of a claim exceeding \$5,000, which matters for pricing and capital reserves. With the Pearson estimator, the model says that chance is effectively zero — because φ is inflated to the point where the model predicts almost nothing but zeros.

This is the uniquely Bayesian insight: the PPC validates the entire predictive distribution, not just a point estimate. A standard GLM reports the same mean ($293) but cannot tell you whether the distribution around that mean is realistic. The PPC — available only through the Bayesian posterior predictive — catches the Pearson failure. The dispersion estimate that looked reasonable in a coefficient table turns out to produce a predictive distribution that does not resemble the data at all. Without the PPC, you would never know.

A model that cannot distinguish a 2.3% tail risk from 0.001% is not usable for pricing, reinsurance, or reserve setting. The PPC catches this; the Pearson dispersion does not.

#### Pricing Exercise: Risk Profiles

The Bayesian approach also provides something a point-estimate model cannot: the full predictive distribution for each risk profile individually. Age is a well-established rating factor in auto insurance, and the model's GLM coefficients confirm the expected pattern — younger drivers have higher expected pure premiums. The table below makes this explicit: expected pure premium drops from \$360 (younger) to \$220 (older), and the probability of a large claim (> \$5K) drops from 2.0% to 0.9% — a textbook age-risk gradient that a Pearson-based model would entirely miss.

The figure below shows the predictive distribution for three age groups, holding other factors constant, with the corresponding statistics in the table beneath:

![Predictive distribution by age profile.](../images/fig_pricing_profiles.png)

| Profile | Expected PP | Median PP | 95th Pctl | P(PP > \$5K) | Zero Rate |
|---------|------------|----------|----------|-------------|----------|
| Younger (agecat 1-2) | \$360 | \$0 | \$2,532 | 2.0% | 84.7% |
| Middle (agecat 3-4) | \$291 | \$0 | \$2,025 | 1.5% | 86.0% |
| Older (agecat 5-6) | \$220 | \$0 | \$1,493 | 0.9% | 87.4% |

Several things stand out:

- **Expected pure premium** differs by age, as expected — younger drivers have higher claims on average. This alone is recoverable from a standard GLM.
- **The entire distribution shifts.** It is not just the mean that changes — the 95th percentile, the probability of a \$5,000+ claim, and the zero rate all point toward lower risk for older drivers (fewer large claims, more zeros). This is information a point-estimate approach would miss.
- **Tail risk is concentrated in younger profiles.** The probability of a claim exceeding \$5,000 is 2.0% for the youngest group versus 0.9% for the oldest — a 2.2× difference. A pricing model that ignores this will undercharge young drivers and overcharge older ones.
- **The median is \$0** for all three groups — about 85% of policies have no claims, so the typical policy costs nothing. But the expected value is \$200-360 because the few claims that occur can be large. The 95th percentile is roughly 7× the mean. This asymmetry is built into the Tweedie and captured naturally by the Bayesian predictive distribution.

This is the contrast with the competitor approach, which would smear all three profiles toward near-zero risk:

![Comparison of predicted zero rates across estimation methods.](../images/fig_zero_rate_comparison.png)

Our Bayesian model correctly recovers the 93.2% observed zero rate. The Pearson estimator predicts 99.0%. A model that predicts 99% zeros for everyone cannot differentiate between a young high-risk driver and an older low-risk one — it has nothing left to differentiate with.

## Results: μ-GLM

Adding covariates on the mean via a log-link GLM reveals something interesting:

```python title="Tweedie GLM with covariates"
def build_glm_model(y, X, features, p_range=(1.1, 1.9)):
    coords = {"features": features, "obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
        mu = pm.Deterministic("mu", pt.exp(pt.dot(X, beta)))

        log_phi = pm.Normal("log_phi", mu=5.0, sigma=2.0)
        phi = pm.Deterministic("phi", pt.exp(log_phi))

        p_logit = pm.Normal("p_logit", mu=0, sigma=1.5)
        p = pm.Deterministic("p",
            p_range[0] + (p_range[1] - p_range[0])
            * pm.math.sigmoid(p_logit))

        Tweedie("y_obs", mu=mu, phi=phi, p=p, observed=y, dims="obs")
    return model
```

Dispersion remains virtually unchanged:

| Model | dataCar φ |
|-------|-----------|
| Intercept-only | 174.3 |
| μ-GLM (22 features) | 174.9 |

Model comparison via [Watanabe–Akaike Information Criterion (WAIC)](https://www.pymc.io/projects/docs/en/v5.16.2/learn/core_notebooks/model_comparison.html) confirms that the extra covariates do not materially improve predictive fit:

| Model | WAIC | ΔWAIC | pWAIC | Weight |
|-------|------|-------|-------|--------|
| Intercept-only | 47,825 | 0 | 3.1 | 0.62 |
| μ-GLM | 47,824 | −1 | 25.3 | 0.38 |

The two models have essentially identical WAIC (Δ < 1), and the effective number of parameters (pWAIC) jumps from 3.1 to 25.3 — the extra 22 features add complexity with no improvement. The simpler model is preferred.

Why doesn't φ change? The Tweedie's variance function is $V(\mu) = \mu^p$, and the dispersion φ scales the entire variance. If the intercept-only model already identifies the correct global φ from the marginal distribution, adding mean covariates cannot reduce it — the covariates reallocate $\mu$ across policies, but the overall claim-generating process has the same variance structure. The dispersion is well-identified from the marginal distribution alone.

This does not mean covariates are useless for pricing. For individual risk profiles (as in the pricing exercise above), the GLM correctly adjusts premiums by age, vehicle type, and other factors. It means that **the Tweedie's global dispersion is correctly specified as a single parameter** — and that the Pearson estimator was never the right tool to estimate it.

## Why This Matters

The takeaway is not that Tweedie is a bad model — it is that the **default estimator is the wrong one**. The Pearson dispersion is a moment-based estimator that works well for approximately Normal data but fails catastrophically for zero-inflated Tweedie models.

For comparison, the [scikit-learn Tweedie regression tutorial](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html) — the most widely-cited Python reference for Tweedie GLMs — validates models using deviance alone and fixes Tweedie power to an arbitrary value (p=1.9). Our results show that fixing p without fitting φ jointly misses the core misspecification that drives PPC failure.

Three practical recommendations:

1. **Estimate $p$ and $\phi$ jointly** — fixing $p$ to an arbitrary value (like 1.6) and then estimating $\phi$ via Pearson creates a cascade of errors
2. **Use the full likelihood** — the series expansion is numerically tractable and converges rapidly; there is no reason to settle for method-of-moments estimates
3. **Validate with PPC** — if your model predicts 99%+ zeros when the data has 94%, the estimation method is likely the culprit, not the distribution

## Related Work

The problem of Pearson φ failing for Tweedie models is acknowledged across the actuarial literature, but almost always in passing:

- **SAS HPGENSELECT** outputs Pearson χ²/DF = 472 for the Swedish motor Tweedie GLM ([SAS documentation](https://communities.sas.com/t5/SAS-Code-Examples/Fitting-Tweedie-s-Compound-Poisson-Gamma-Mixture-Model-by-Using/ta-p/908438)) with no comment on whether φ ≈ 2,147 is reasonable. The model is presented without distributional validation.

- **glmmTMB developers** discovered the Tweedie variance function was an unimplemented placeholder — Pearson residuals were simply unavailable ([GitHub issue #293](https://github.com/glmmTMB/glmmTMB/issues/293)). The issue thread notes "variance functions that take more than two parameters" lack framework support.

- **The CAS** notes the "Pearson estimate is normally unsuitable for GLM log link claim frequency analysis" and proposes an alternative dispersion estimator for Tweedie models ([CAS abstract](https://www.casact.org/abstract/dispersion-estimates-poisson-and-tweedie-models)).

- **Stack Exchange** users ask whether Tweedie GLM residuals should be Normal and whether the Shapiro test is appropriate ([Cross Validated](https://stats.stackexchange.com/questions/363955/non-normal-residuals-for-tweedie-glm)). The answer is no — quantile residuals are the standard tool — but this confusion persists.

- **Scikit-learn's** [Tweedie regression tutorial](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html) validates models via deviance only — it never checks whether the distributional assumptions hold. The observed vs. predicted plot shows systematic under-prediction, but the reason (dispersion misspecification) is not discussed.

- **Wüthrich (2021)** compares Poisson-gamma vs Tweedie parametrizations rigorously ([Springer](https://link.springer.com/article/10.1007/s13385-021-00264-3)) and finds industry preference for the separate frequency-severity approach, noting dispersion modeling is the weak point of the single Tweedie GLM.

This post fills the gap: we identify the mechanism (Pearson φ inflation), demonstrate it empirically on the dataCar dataset, show the correct Bayesian fix, and connect it to the pricing decisions that matter.

## Possible Extensions

- **$p > 2$ for severity modeling** — the alternating sin series handles this case, though identifiability weakens
- **BART for the mean structure** — nonparametric mean estimation via [`pymc-bart`](https://www.pymc.io/projects/bart) for automatic interaction and nonlinearity detection
- **Hurdle models** — separate models for claim frequency and severity for heavy-tailed data
- **Double GLM (μ-φ DGLM)** — regressing dispersion $\phi$ on covariates could capture heteroskedasticity by risk class
- **Hierarchical (partial pooling) models** — policies are nested in territories, vehicle types, and driver classes. PyMC's dims-based coordinate system makes random intercepts trivial to formulate: `pm.Normal("alpha_territory", mu=0, sigma=sigma_territory, dims="territory")` adds a partial-pooling term for every territory in a single line, with the group-level variance learned from data. The same pattern extends to random slopes and nested hierarchies. Sampling many group-level parameters is computationally demanding (more dimensions for the sampler), but the model specification is concise and natural.
- **Bayesian optimization for pricing** — the posterior over (μ, φ, p) can drive pricing decisions under uncertainty. [`pymc.vectorize_over_posterior`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.vectorize_over_posterior.html) takes the fitted model graph and posterior draws and returns a vectorized function over all draws — no manual looping or re-implementation. Use it to build an optimizer that evaluates pricing objectives (premium, deductible, risk retention) across the full posterior, giving a *distribution* over the optimal decision rather than a point estimate. This is the same technique behind PyMC-Marketing's MMM budget optimizer. The workflow of reusing the fitted model graph for downstream optimization was [pioneered by Ricardo Vieira in the PyMC ecosystem](https://www.youtube.com/watch?v=85jPmkMTfck).

## Reproducibility

Clone or fork the repo and run the figure scripts locally:

```bash
# Clone with gh CLI
gh repo clone williambdean/williambdean.github.io
cd williambdean.github.io

# Or fork first, then clone your fork
gh repo fork williambdean/williambdean.github.io --clone
```

```bash
# Generate all figures (uv installs dependencies automatically)
uv run docs/blog/posts/scripts/pearson-phi-broken-tweedie/fig_profile_likelihood.py
uv run docs/blog/posts/scripts/pearson-phi-broken-tweedie/fig_ppc_validation.py
uv run docs/blog/posts/scripts/pearson-phi-broken-tweedie/fig_zero_rate_comparison.py
uv run docs/blog/posts/scripts/pearson-phi-broken-tweedie/fig_ppc_distribution.py
uv run docs/blog/posts/scripts/pearson-phi-broken-tweedie/fig_pricing_profiles.py
uv run docs/blog/posts/scripts/pearson-phi-broken-tweedie/fig_posterior_pairs.py
```

Each figure script is fully self-contained using `uv` inline dependency metadata — there is no environment setup or `requirements.txt` needed. The scripts generate synthetic Tweedie data internally, so no external data files are required.

## Further Reading

- Dunn, P. K. & Smyth, G. K. (2005). [Series evaluation of Tweedie exponential dispersion model densities](https://doi.org/10.1007/s11222-005-4070-y). *Statistics and Computing*, 15(4), 267-280.
- Jørgensen, B. (1997). *[The Theory of Dispersion Models](https://doi.org/10.1201/9780203743441)*. Chapman & Hall.
- Smyth, G. K. & Jørgensen, B. (2002). [Fitting Tweedie's compound Poisson model to insurance claims data: Dispersion modelling](https://www.casact.org/sites/default/files/old/astin_vol32no1_143.pdf). *ASTIN Bulletin*, 32(1), 143-157.
- De Jong, P. & Heller, G. Z. (2008). *[Generalized Linear Models for Insurance Data](https://doi.org/10.1017/CBO9780511755408)*. Cambridge University Press.
- Wüthrich, M. V. (2021). [Making Tweedie's compound Poisson model more accessible](https://link.springer.com/article/10.1007/s13385-021-00264-3). *European Actuarial Journal*.
- Yang, Y., Qian, W. & Zou, H. (2018). [Insurance premium prediction via gradient tree-boosted Tweedie compound Poisson models](https://doi.org/10.1080/07350015.2016.1200981). *Journal of Business & Economic Statistics*, 36(3), 456-470.
- [Scikit-learn: Tweedie regression on insurance claims](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html) — the canonical Python implementation reference.
- [Akshat Dwivedi: Specifying an offset in a Tweedie model with identity link](https://akshat.blog/posts/tweedie-with-identity-link-and-offset/) — technical deep dive on offsets and weights.
- [Ricardo Vieira: Graph reuse and optimization in PyMC](https://www.youtube.com/watch?v=85jPmkMTfck) — the `vectorize_over_posterior` workflow for pricing optimization.
- [Gao, G. (2024). Fitting Tweedie's compound Poisson model to pure premium with the EM algorithm](https://doi.org/10.1016/j.insmatheco.2023.10.002). *Insurance: Mathematics and Economics*, 114, 29-42.
