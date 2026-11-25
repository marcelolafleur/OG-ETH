"""
Tests of the initial wealth burn-in.
"""

import numpy as np

from ogeth import initial_wealth


def test_burn_in_joins_the_oldest_ages():
    """The oldest ages keep their shape but continue from the last
    life-cycle age, so wealth per person has no step at the cutoff."""
    S, J, E = 80, 3, 20
    rng = np.random.default_rng(0)
    omega0 = rng.uniform(0.5, 1.5, (S, J))
    omega0 /= omega0.sum()
    start = rng.uniform(0.5, 1.5, (S, J))
    start[59:] *= 3.0  # a step at age 80
    chosen = rng.uniform(0.5, 1.5, (S, J))
    tpi = {"b_sp1_preTP": start, "b_sp1": chosen[np.newaxis]}
    dist = initial_wealth.burn_in_dist(tpi, omega0, E=E)
    assert np.isclose(dist.sum(), 1.0)
    pp = dist / omega0
    last = initial_wealth.BURN_IN_MAX_AGE - E - 2  # age 79
    assert np.allclose(pp[last + 1], pp[last])
    # shape within the oldest ages is kept
    old = start[last + 1 :] / start[last + 1]
    assert np.allclose(pp[last + 1 :] / pp[last + 1], old)
