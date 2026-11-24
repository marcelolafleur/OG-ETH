"""
The initial wealth shape ``initial_hhwealth_dist`` (S x J), imputed by a
damped burn-in to the shape households choose in period 0; the level is
``initial_wealth_ratio`` (macro.md, Initial household wealth).
"""

import numpy as np


def per_person(b, omega):
    """Wealth per person by age and group, relative to the population mean."""
    b = np.asarray(b, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mean = (b * omega).sum() / omega.sum()
    return b / mean


# Damping of one burn-in step (a full step overshoots)
BURN_IN_STEP = 0.3
# From this age the period-0 savings are mostly bequests, so the oldest
# ages keep the starting shape
BURN_IN_MAX_AGE = 80


def burn_in_dist(
    tpi, omega0, step=BURN_IN_STEP, E=20, max_age=BURN_IN_MAX_AGE
):
    """
    One damped burn-in step: the starting shape moved ``step`` of the way
    toward period-0 savings, as shares of wealth.

    Args:
        tpi (dict): a solved ``TPI_vars.pkl`` (needs ``b_sp1_preTP`` and
            ``b_sp1``)
        omega0 (array_like): first-period population, (S, J)
        step (float): share of the way toward the period-0 choice
        E (int): entry age (row s is age E + s + 1)
        max_age (int): ages from this one keep the starting shape

    Returns:
        Numpy array: (S, J) shares summing to one
    """
    omega0 = np.asarray(omega0, dtype=float)

    def shares(b):
        w = np.clip(np.asarray(b, dtype=float), 0.0, None) * omega0
        return w / w.sum()

    start = shares(tpi["b_sp1_preTP"])
    chosen = shares(np.asarray(tpi["b_sp1"])[0])
    ages = E + 1 + np.arange(omega0.shape[0])
    young = (ages < max_age).reshape(-1, 1)
    # move only the life-cycle cells
    dist = np.where(young, (1 - step) * start + step * chosen, start)
    # keep their total share, so the oldest ages' shares are unchanged
    life = young.ravel()
    dist[life] *= start[life].sum() / dist[life].sum()
    return dist / dist.sum()


def shape_gap(tpi, omega0, min_share=1e-3):
    """
    Largest relative gap in wealth per person between the starting and
    period-0 shapes, over cells holding at least ``min_share`` of wealth.

    Args:
        tpi (dict): a solved ``TPI_vars.pkl`` (needs ``b_sp1_preTP`` and
            ``b_sp1``)
        omega0 (array_like): first-period population, (S, J)

    Returns:
        float: max |shape_1 / shape_0 - 1| over cells holding wealth
    """
    omega0 = np.asarray(omega0, dtype=float)
    start = per_person(tpi["b_sp1_preTP"], omega0)
    chosen = per_person(np.asarray(tpi["b_sp1"])[0], omega0)
    share = np.clip(start, 0, None) * omega0
    share = share / share.sum()
    cells = share >= min_share
    return float(np.max(np.abs(chosen[cells] / start[cells] - 1)))
