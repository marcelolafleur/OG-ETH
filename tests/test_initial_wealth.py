"""Tests of the burn-in initial wealth distribution."""

import numpy as np
import pytest

from ogeth import initial_wealth


def _tpi(start, chosen):
    return {"b_sp1_preTP": start, "b_sp1": np.stack([chosen, chosen])}


def test_burn_in_dist_is_shares_of_wealth():
    rng = np.random.default_rng(1)
    omega0 = rng.uniform(0.5, 1.5, (80, 7))
    b = rng.uniform(0.0, 2.0, (80, 7))
    dist = initial_wealth.burn_in_dist(_tpi(b, b), omega0)
    assert dist.shape == (80, 7)
    assert dist.sum() == pytest.approx(1.0)
    # OG-Core divides by the first-period population: that returns the
    # per-person shape the period-0 choice had
    per_person = dist / omega0
    assert np.allclose(per_person / per_person.mean(), b / b.mean())


def test_shape_gap_is_zero_at_the_fixed_point_and_scale_free():
    rng = np.random.default_rng(2)
    omega0 = rng.uniform(0.5, 1.5, (80, 7))
    b = rng.uniform(0.1, 2.0, (80, 7))
    assert initial_wealth.shape_gap(_tpi(b, 3.0 * b), omega0) == pytest.approx(
        0.0, abs=1e-12
    )
    moved = b.copy()
    moved[40] *= 1.2
    assert initial_wealth.shape_gap(_tpi(b, moved), omega0) > 0.1
