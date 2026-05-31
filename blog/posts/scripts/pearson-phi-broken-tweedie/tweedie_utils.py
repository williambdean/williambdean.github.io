"""Core Tweedie distribution functions (numpy version) for figure generation.

Unweighted variants matching the blog post code blocks. For exposure-weighted
applications, see the full implementation in the repo.
"""

import numpy as np
from scipy.special import gammaln, logsumexp


def tweedie_logp_series(y, mu, phi, p, n_terms=30):
    """Tweedie log-pdf via series expansion (Dunn & Smyth 2005).

    Supports p in (1, 2). Returns log f(y | mu, phi, p).
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    alpha = (2 - p) / (p - 1)
    theta = mu ** (1 - p) / (1 - p)
    kappa = mu ** (2 - p) / (2 - p)
    j = np.arange(1, n_terms + 1)

    logp = np.zeros(len(y))
    for i, yi in enumerate(y):
        if yi <= 1e-9:
            logp[i] = -mu ** (2 - p) / (phi * (2 - p))
        else:
            ll_core = (yi * theta - kappa) / phi
            log_Wj = (
                j * alpha * np.log(yi)
                - j * (1 + alpha) * np.log(phi)
                - j * np.log(2 - p)
                - j * alpha * np.log(p - 1)
                - gammaln(j + 1)
                - gammaln(np.maximum(j * alpha, 1e-10))
            )
            log_a = logsumexp(log_Wj) - np.log(yi)
            logp[i] = ll_core + log_a
    return logp


def tweedie_random(mu, phi, p, size=None, rng=None):
    """Generate Tweedie(p in (1,2)) variates via Poisson-Gamma compound."""
    if p >= 2:
        raise ValueError(f"p must be in (1, 2), got {p}")
    if rng is None:
        rng = np.random.default_rng()
    lam = (mu ** (2 - p)) / (phi * (2 - p))
    N = rng.poisson(lam, size=size)
    N = np.maximum(N, 0)
    alpha = (2 - p) / (p - 1)
    theta = phi * (p - 1) * (mu ** (p - 1))
    shape_param = np.where(N > 0, N * alpha, 1e-9)
    res = rng.gamma(shape=shape_param, scale=theta)
    return np.where(N > 0, res, 0.0)


def expected_zero_rate(mu, phi, p):
    """Expected zero rate: P(Y=0) = exp(-lambda)."""
    lam = (mu ** (2 - p)) / (phi * (2 - p))
    return np.exp(-lam)
