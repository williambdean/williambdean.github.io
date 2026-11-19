---
description: When PyMC's automatic logp derivation works for CustomDist and when it doesn't, with examples from pymc-extras including Skellam, BetaNegativeBinomial, Maxwell, and Chi.
tags:
    - Python
    - PyMC
    - Development
comments: true
---

# CustomDist: Why Provide a logp?

[PyMC 6.0](https://www.pymc.io/blog/pymc_v6_ecosystem_updates.html#automatic-probability-customdist) introduced *automatic* log-density derivation for `pm.CustomDist`. You provide a `dist` function that builds the sampling graph, and PyMC traces through it to compute the logp. For distributions built from standard transforms of primitives, this works. For many real distributions, it doesn't. You supply `logp=` explicitly.

All examples come from [pymc-extras](https://github.com/pymc-devs/pymc-extras) distributions I contributed: Chi, Maxwell, Skellam, and BetaNegativeBinomial.

## The `dist` Parameter

`pm.CustomDist` with a `dist` function handles random sampling. PyMC compiles the graph and uses it for posterior predictive checks, prior predictive sampling, and any task that generates draws.

The entire Chi implementation:

```python title="Chi distribution (pymc-extras)"
import pytensor.tensor as pt
from pymc import ChiSquared, CustomDist


class Chi:
    @staticmethod
    def chi_dist(nu, size):
        return pt.math.sqrt(ChiSquared.dist(nu=nu, size=size))

    def __new__(cls, name, nu, **kwargs):
        return CustomDist(name, nu, dist=cls.chi_dist, class_name="Chi", **kwargs)
```

Chi is the square root of a ChiSquared. The sampling graph is one line. No `logp` is needed because `sqrt` is a differentiable, monotonic transform. PyMC derives the logp by applying the change-of-variables formula to the ChiSquared logp.

The [Maxwell distribution](https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.distributions.Maxwell.html) ([PR #261](https://github.com/pymc-devs/pymc-extras/pull/261)) extends the same pattern: `Chi(nu=3) * a`. Again, a simple scale transform. Automatic logp works.

## When Automatic logp Is Not Enough

Three situations where automatic logp fails:

### No Closed Form

The [Tweedie distribution](https://williambdean.github.io/blog/posts/2026/pearson-phi-broken-tweedie/) has no closed-form density. Its `dist` function builds a compound Poisson-Gamma graph with a `where` branch on a discrete latent. PyMC cannot derive the true log-density from that graph. The [Dunn & Smyth (2005)](https://doi.org/10.1007/s11222-005-4070-y) series expansion must be supplied as `logp`.

### Convolution of Independent Variables

The [Skellam distribution](https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.distributions.Skellam.html) ([PR #260](https://github.com/pymc-devs/pymc-extras/pull/260)) models the difference of two independent Poisson random variables. The `dist` function is trivial: subtract two Poissons. But the logp of the subtraction is not the Skellam PMF, which requires the modified Bessel function of the first kind.

```python title="Skellam distribution (pymc-extras)"
import pymc as pm
import pytensor.tensor as pt


class Skellam:
    @staticmethod
    def skellam_dist(mu1, mu2, size):
        return pm.Poisson.dist(mu=mu1, size=size) - pm.Poisson.dist(mu=mu2, size=size)

    @staticmethod
    def skellam_logp(value, mu1, mu2):
        return (
            -mu1 - mu2
            + 0.5 * value * (pt.log(mu1) - pt.log(mu2))
            + pt.log(pt.iv(value, 2 * pt.sqrt(mu1 * mu2)))
        )

    def __new__(cls, name, mu1, mu2, **kwargs):
        return pm.CustomDist(
            name, mu1, mu2,
            dist=cls.skellam_dist,
            logp=cls.skellam_logp,
            class_name="Skellam",
            **kwargs,
        )

    @classmethod
    def dist(cls, mu1, mu2, **kwargs):
        return pm.CustomDist.dist(
            mu1, mu2,
            dist=cls.skellam_dist,
            logp=cls.skellam_logp,
            class_name="Skellam",
            **kwargs,
        )
```

The `dist` function generates draws by subtracting two independent Poisson samples. The `logp` function evaluates the true PMF using the Bessel function. Both are needed: `dist` for sampling (no Bessel evaluation required) and `logp` for inference (MCMC needs the correct density).

### Compound Marginal

The [BetaNegativeBinomial distribution](https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.distributions.BetaNegativeBinomial.html) ([PR #258](https://github.com/pymc-devs/pymc-extras/pull/258)) is a compound where a Beta prior on the success probability is marginalized out of a NegativeBinomial. The `dist` function composes `Beta → NegativeBinomial`, which samples correctly. But the logp through that composition is conditional on the Beta draw, not the marginal PMF. The actual logp involves Beta and Gamma functions.

```python title="BetaNegativeBinomial logp (pymc-extras)"
from pymc.distributions.dist_math import betaln, factln


def beta_negative_binomial_logp(value, alpha, beta, r):
    return (
        betaln(r + value, alpha + beta)
        - betaln(r, alpha)
        + pt.gammaln(value + beta)
        - factln(value)
        - pt.gammaln(beta)
    )
```

Same wrapper pattern as Skellam: `__new__` passes both `dist` and `logp` to `CustomDist`, with a `dist` classmethod for `pm.Distribution`-style usage.

## The Pattern

Every distribution in pymc-extras that needs an explicit logp uses the same structure:

1. A static `dist` method that builds the sampling graph
2. A static `logp` method that computes the density
3. A `__new__` method that wires both into `CustomDist`
4. A `dist` classmethod for standalone use outside a model

Chi and Maxwell show automatic logp works for simple differentiable transforms. Skellam and BetaNegativeBinomial show convolutions, compounds, and distributions without closed forms need an explicit `logp`. Both approaches live in the same package.

## The dims-aware CustomDist in pymc.dims

PyMC now also ships a dims-aware `CustomDist` in `pymc.dims`, the xtensor-native API for models with named dimensions ([PR #8346](https://github.com/pymc-devs/pymc/pull/8346), merged into main and shipping in the next release). It follows the same `dist` + `logp` pattern with a different contract:

- The `dist` function receives the parameters as `XTensorVariable`, plus an `extra_dims` dict as its last argument: the dims requested via `dims=`/`observed=` that no parameter implies, the counterpart of the `size` argument of `pm.CustomDist`. The function must attach those dims itself (for example `pmd.Poisson.dist(mu=lam, dim_lengths=extra_dims)`) and return an `XTensorVariable`
- `observed` data must carry its dims, via `pmd.as_xtensor(y, dims=("obs",))`
- A user-provided `logp` receives the value and parameters as `XTensorVariable` and may return one, so `pytensor.xtensor` operations work throughout

The [Tweedie post](https://williambdean.github.io/blog/posts/2026/pearson-phi-broken-tweedie/) is a worked example: a compound Poisson-Gamma `dist` for sampling, the Dunn & Smyth series as `logp` for inference, built on this API.
