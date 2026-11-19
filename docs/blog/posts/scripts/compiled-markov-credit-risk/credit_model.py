"""Shared credit migration model for the compiled-markov-credit-risk blog post.

Standalone by design: only pymc, pytensor, and numpy. No imports from any
other project.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "D"]
DEFAULT_IDX = 6
PORTFOLIO_VALUE = 100_000_000
HORIZON = 5

# Historical one-year transition counts, rows = FROM rating, cols = TO rating.
# Default is absorbing and pinned to the identity below, so only the six
# living ratings enter the Dirichlet.
COUNTS = np.array(
    [
        [8744, 889, 336, 41, 7, 3, 0],  # AAA
        [6, 8749, 959, 181, 46, 24, 3],  # AA
        [2, 128, 8676, 961, 150, 52, 5],  # A
        [1, 5, 244, 8516, 900, 236, 33],  # BBB
        [0, 2, 8, 335, 7910, 1123, 498],  # BB
        [0, 0, 3, 18, 345, 7634, 2000],  # B
    ],
    dtype="int64",
)

COHORT0 = np.array([0.05, 0.15, 0.40, 0.25, 0.10, 0.05, 0.00])


def mean_transition_matrix() -> np.ndarray:
    """The Dirichlet mean: row-normalized counts with default absorbing.

    Returns:
        (7, 7) stochastic matrix.
    """
    full = np.zeros((7, 7))
    full[:6] = COUNTS / COUNTS.sum(axis=1, keepdims=True)
    full[DEFAULT_IDX, DEFAULT_IDX] = 1.0
    return full


def deterministic_trace() -> np.ndarray:
    """Cohort trace under the mean transition matrix.

    Returns:
        Array of shape (HORIZON + 1, 7), rows = years 0 through HORIZON.
    """
    T = mean_transition_matrix()
    rows = [COHORT0]
    for _ in range(HORIZON):
        rows.append(rows[-1] @ T)
    return np.asarray(rows)


def build_loss_graph(sweep: bool = False, pd_ramp=None):
    """Build the credit migration loss graph.

    Args:
        sweep: If True, weave in the ``lgd_multiplier`` and ``pd_stress``
            scalars so they become inputs of the compiled function.
        pd_ramp: Optional per-year PD stress profile of length HORIZON.
            When given, the transition matrix is stacked into a
            ``(HORIZON, 7, 7)`` tensor, the default column of year ``t``
            is scaled by ``pd_ramp[t]``, and each year is renormalized.
            The scan then iterates the horizon axis via ``sequences``.

    Returns:
        Tuple ``(total_loss, default_rate_path)`` where ``total_loss`` is
        the loss for one Monte Carlo draw and ``default_rate_path`` is the
        per-year default rate, shape ``(HORIZON,)``.
    """
    T_nondefault = pm.Dirichlet.dist(COUNTS)  # (6, 7): one row per living rating
    T = pt.zeros((7, 7))
    T = pt.set_subtensor(T[:6], T_nondefault)
    T = pt.set_subtensor(T[DEFAULT_IDX], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    if sweep:
        pd_stress = pt.scalar("pd_stress")
        col = T[:, DEFAULT_IDX] * pd_stress
        T = pt.set_subtensor(T[:, DEFAULT_IDX], col)
        T = T / T.sum(axis=1, keepdims=True)

    cohort0 = pt.as_tensor_variable(COHORT0)

    def step(cohort_t, T_arg):
        return cohort_t @ T_arg

    if pd_ramp is None:
        trace = pytensor.scan(
            fn=step,
            outputs_info=[cohort0],
            non_sequences=[T],
            n_steps=HORIZON,
            return_updates=False,
        )  # (HORIZON, 7)
    else:
        ramp_t = pt.as_tensor_variable(pd_ramp)
        T_stack = pt.stack([T] * HORIZON)  # (HORIZON, 7, 7)
        col = T_stack[:, :, DEFAULT_IDX] * ramp_t[:, None]  # broadcasts over ratings
        T_stack = pt.set_subtensor(T_stack[:, :, DEFAULT_IDX], col)
        T_stack = T_stack / T_stack.sum(axis=2, keepdims=True)
        T_stack.name = "T_stack"

        # Scan passes sequence items BEFORE output taps, so the step
        # signature is (T_t, cohort_t) here, unlike the static scan above.
        def step_tv(T_t, cohort_t):
            return cohort_t @ T_t

        trace = pytensor.scan(
            fn=step_tv,
            outputs_info=[cohort0],
            sequences=[T_stack],
            n_steps=HORIZON,
            return_updates=False,
        )  # (HORIZON, 7)

    trace.name = "trace"

    senior_secured = pm.Beta.dist(mu=0.60, sigma=0.15)
    senior_unsecured = pm.Beta.dist(mu=0.45, sigma=0.20)
    subordinated = pm.Beta.dist(mu=0.25, sigma=0.15)
    recovery = pm.Mixture.dist(
        w=[0.30, 0.50, 0.20],
        comp_dists=[senior_secured, senior_unsecured, subordinated],
    )

    lgd = 1 - recovery
    if sweep:
        lgd_multiplier = pt.scalar("lgd_multiplier")
        lgd = lgd * lgd_multiplier

    defaulted_exposure = trace[-1, DEFAULT_IDX] * PORTFOLIO_VALUE
    total_loss = defaulted_exposure * lgd
    default_rate_path = trace[:, DEFAULT_IDX]
    return total_loss, default_rate_path
