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
    sources, and the CIT factor gamma is derived with is the tuned one."""
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
