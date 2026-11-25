"""
Tests of the net family transfers (phi_FT) and their fit to the NTA
consumption target.
"""

import json
from importlib.resources import files

import numpy as np

from ogeth import macro_params as mp
from ogeth import targets


def test_family_transfers_net_to_zero_within_groups():
    """The steady-state matrix and every packaged year move resources
    only within each lifetime-income group."""
    assert np.abs(mp.family_transfers_ss().sum(axis=0)).max() < 1e-12
    phi = np.asarray(
        json.loads(
            files("ogeth")
            .joinpath("ogeth_default_parameters.json")
            .read_text(encoding="utf-8")
        )["phi_FT"]
    )
    assert np.abs(phi.sum(axis=1)).max() < 1e-12
    assert np.allclose(phi[-1], mp.family_transfers_ss())


def test_consumption_target_is_relative_to_prime_age():
    ages = np.arange(30, 50)
    assert np.isclose(targets.consumption_target(ages).mean(), 1.0, atol=1e-3)
