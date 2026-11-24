"""
Tests of income.py: the WID x NTA reshaping of the OG-USA ability matrix.

The OG-USA download is replaced by a synthetic matrix so the tests run
offline; the Ethiopian and US inputs are the packaged data files.
"""

import json
from importlib.resources import files

import numpy as np
import pytest

from ogeth import income

LAMBDAS = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
LAMBDAS_USA = np.array(
    [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]
)


@pytest.fixture
def synthetic_usa_e(monkeypatch):
    """A smooth (80, 10) matrix standing in for the OG-USA download."""
    ages = income.model_ages(income.OGUSA_E, income.OGUSA_S)
    hump = 1 + 0.02 * (ages - 20) - 0.0003 * (ages - 20) ** 2
    groups = np.array([0.4, 0.55, 0.8, 1.05, 1.35, 2.5, 5.0, 7.0, 15.0, 40.0])
    e_usa = hump.reshape(-1, 1) * groups.reshape(1, -1)
    monkeypatch.setattr(
        income, "load_ogusa_e", lambda: (e_usa.copy(), LAMBDAS_USA.copy())
    )
    return e_usa


@pytest.fixture(scope="module")
def packaged():
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(content)


@pytest.mark.parametrize("S", [80, 40])
def test_get_e_interp_shape_and_normalization(synthetic_usa_e, S):
    rng = np.random.default_rng(0)
    omega = rng.uniform(0.5, 1.5, size=(S, 7)) * LAMBDAS.reshape(1, 7)
    omega /= omega.sum()
    e = income.get_e_interp(20, S, 7, LAMBDAS, omega)
    assert e.shape == (S, 7)
    assert (e * omega).sum() == pytest.approx(1.0)
    assert (e > 0).all()
    # a one-dimensional age distribution is spread with lambdas
    e1 = income.get_e_interp(20, S, 7, LAMBDAS, omega.sum(axis=1))
    assert (e1 * omega.sum(axis=1).reshape(S, 1) * LAMBDAS).sum() == (
        pytest.approx(1.0)
    )


def test_get_e_interp_applies_both_factors(synthetic_usa_e):
    """Relative to the interpolated US matrix, the age pattern of every
    group follows the age factor and the group pattern at every age
    follows the group factor."""
    omega = np.full((80, 7), 1 / 80) * LAMBDAS.reshape(1, 7)
    e = income.get_e_interp(20, 80, 7, LAMBDAS, omega)
    base = income.interpolate_usa_e(
        synthetic_usa_e, LAMBDAS_USA, 20, 80, LAMBDAS
    )
    ratio = e / base
    age_factor = income.earnings_age_factor(income.model_ages(20, 80))
    for j in range(7):
        np.testing.assert_allclose(
            ratio[:, j] / ratio[0, j], age_factor / age_factor[0]
        )
    np.testing.assert_allclose(
        income.implied_group_shares(e, omega),
        income.wid_group_shares("ETH", LAMBDAS),
        atol=1e-4,
    )


def test_packaged_e_is_the_reshaped_matrix(packaged):
    """The shipped e carries the WID group pattern and the NTA age pattern
    and averages to one over the shipped population, times the unit
    normalization EARNINGS_SCALE."""
    from ogeth import macro_params as mp

    e = np.asarray(packaged["e"]) / mp.EARNINGS_SCALE
    omega = np.asarray(packaged["omega_SS"])
    lambdas = np.asarray(packaged["lambdas"])
    assert e.shape == (80, 7)
    assert (e * omega).sum() == pytest.approx(1.0, abs=1e-6)
    shares = income.implied_group_shares(e, omega)
    np.testing.assert_allclose(
        shares, income.wid_group_shares("ETH", lambdas), atol=1e-4
    )
