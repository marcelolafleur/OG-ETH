"""
The initial distribution of household wealth across ages and lifetime-income
groups, OG-Core's ``initial_hhwealth_dist``.

OG-Core sets the *level* of initial household wealth from
``initial_wealth_ratio`` (a ratio to first-period GDP) by scaling a fixed
*shape* uniformly inside the time-path solution. The shape is
``initial_hhwealth_dist``, an S x J matrix of shares of aggregate wealth;
OG-Core divides it by the first-period population to get wealth per person
and then rescales. Left unset, the shape is the steady state's wealth per
person, which describes households that have lived under the long-run
demographics and policy, not today's.

Ethiopia has no published wealth-by-age profile (its National Transfer
Accounts have labor income only; the ESS asset modules are not yet used),
so the shape is imputed from the model itself by a burn-in: the shape that
households choose in the first period of the transition is fed back as the
starting shape, and the step is repeated until the two agree. At the fixed
point the starting distribution is one the model's own households would
hold, so the first years carry no jump from an inconsistent start. The
shape is passed as wealth per person times the first-period population, so
OG-Core's division by that population returns exactly the per-person shape
(OG-Core aggregates with the pre-period population, which differs from the
first-period one by a few percent by age; the subsequent rescaling absorbs
that). The level stays with ``initial_wealth_ratio``.
"""

import numpy as np


def per_person(b, omega):
    """Wealth per person by age and group, relative to the population mean."""
    b = np.asarray(b, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mean = (b * omega).sum() / omega.sum()
    return b / mean


# Share of the way the starting shape moves toward the period-0 choice in
# one burn-in step. A full step overshoots: the oldest cohorts' period-0
# savings (mostly bequests) are several times their starting wealth, and
# feeding them back whole left the poorest group's oldest households so rich
# that the transition stalled (75 iterations, against 17).
BURN_IN_STEP = 0.3
# The burn-in applies to life-cycle wealth only. From this age on, the
# savings chosen in period 0 are mostly bequests (several times the
# starting wealth at ages 98-100), and feeding them back raised the period-0
# bequests the same group's young receive enough that a start cohort of the
# poorest group failed to solve; the oldest ages keep the starting shape.
BURN_IN_MAX_AGE = 80


def burn_in_dist(
    tpi, omega0, step=BURN_IN_STEP, E=20, max_age=BURN_IN_MAX_AGE
):
    """
    One damped burn-in step for the starting shape: the shape that started
    the solved transition moved ``step`` of the way toward the savings
    households chose in period 0, as shares of aggregate wealth weighted by
    the first-period population.

    Args:
        tpi (dict): a solved ``TPI_vars.pkl`` (needs ``b_sp1_preTP`` and
            ``b_sp1``)
        omega0 (array_like): first-period population, (S, J)
        step (float): share of the way toward the period-0 choice
        E (int): age at which households enter the model (row s is age
            E + s + 1)
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
    # move the life-cycle cells and keep the oldest ages' starting wealth
    # per person: the rescaling below then only spreads the change in the
    # young cells' total
    dist = np.where(young, (1 - step) * start + step * chosen, start)
    # hold the life-cycle cells' total share at its start value, so the
    # oldest ages' shares are unchanged
    life = young.ravel()
    dist[life] *= start[life].sum() / dist[life].sum()
    return dist / dist.sum()


def shape_gap(tpi, omega0, min_share=1e-3):
    """
    How far the starting shape is from the shape chosen in the first
    period: the largest relative difference in wealth per person across
    the (age, group) cells that hold at least ``min_share`` of wealth.

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
