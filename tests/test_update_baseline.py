"""
Tests of update_baseline.py, which writes the packaged parameters.
"""

import json
from importlib.resources import files

import numpy as np
import pytest
from ogcore.parameters import Specifications

from ogeth import macro_params as mp


@pytest.fixture(scope="module")
def packaged():
    return json.loads(
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )


def test_packaged_parameters_are_the_regeneration(packaged):
    """The shipped values are what update_baseline.py derives from the
    sources, and the inputs the code also holds as constants agree."""
    p = Specifications()
    p.update_specifications(packaged)
    assert not p.errors
    for key, value in mp.baseline_params(p).items():
        if isinstance(value, str):
            assert packaged[key] == value, key
        else:
            assert np.allclose(packaged[key], value, rtol=1e-8, atol=1e-12), (
                key
            )
    assert packaged["adjustment_factor_for_cit_receipts"][0] == pytest.approx(
        mp.CIT_COLLECTIONS_FACTOR, abs=1e-4
    )
    # inputs the code reads as constants (gamma, informality) that the JSON
    # also holds
    assert packaged["cit_rate"][0][0] == mp.CIT_RATE
    assert packaged["c_corp_share_of_assets"] == mp.C_CORP_SHARE_OF_ASSETS
    assert np.allclose(packaged["delta_tau_annual"], mp.DELTA_TAU_ANNUAL)
    assert packaged["lambdas"] == mp.LAMBDAS
