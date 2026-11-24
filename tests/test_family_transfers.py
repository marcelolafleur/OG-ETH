"""
Net family transfers (OG-Core's phi_FT): the packaged path is the one the
rule in macro_params builds, it only moves resources between households,
and the consumption profile it is sized to is NTA's.
"""

import json
from importlib.resources import files

import numpy as np
import pytest
from ogcore.parameters import Specifications

from ogeth import macro_params as mp
from ogeth import targets


@pytest.fixture(scope="module")
def packaged():
    return json.loads(
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def params(packaged):
    p = Specifications()
    p.update_specifications(packaged)
    return p


def test_packaged_phi_ft_is_the_rule(packaged, params):
    """A regeneration must rebuild phi_FT from the same rule and inputs."""
    expected = mp.scenario_params(params)["phi_FT"]
    assert np.allclose(packaged["phi_FT"], expected, rtol=1e-8, atol=1e-12)


def test_phi_ft_nets_to_zero_every_year(params):
    phi = np.asarray(params.phi_FT)
    assert phi.shape == (params.T + params.S, params.S, params.J)
    assert np.allclose(phi.sum(axis=(1, 2)), 0.0, atol=1e-12)


def test_givers_and_receivers_by_age(params):
    """Ages 25-64 give, no one younger than 25 gives or receives, and in
    the steady state the elders receive FAMILY_TRANSFER_SHARE_SS of output
    before the government transfers they already get are netted."""
    phi = np.asarray(params.phi_FT)[-1]
    age = np.arange(params.S) + int(params.E)
    lo, hi = mp.FAMILY_GIVER_AGES
    assert np.all(phi[age < lo] == 0.0)
    assert np.all(phi[(age >= lo) & (age < hi)] <= 0.0)
    received = phi[age >= mp.FAMILY_RECEIVER_AGE].sum()
    assert 0.0 < received < mp.FAMILY_TRANSFER_SHARE_SS


def test_nta_old_age_consumption():
    """NTA 2005 private consumption relative to ages 30-49."""
    ratios = targets.nta_old_age_consumption()
    assert ratios[(65, 70)] == pytest.approx(0.870, abs=1e-3)
    assert ratios[(70, 80)] == pytest.approx(0.836, abs=1e-3)
    assert ratios[(80, 86)] == pytest.approx(0.804, abs=1e-3)
