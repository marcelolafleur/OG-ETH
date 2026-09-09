"""
Tests of the fiscal program path in the packaged single-industry JSON.

The transition's fiscal block follows the IMF fifth-ECF-review program
(Country Report 26/174) for FY2024/25-FY2030/31 and holds the last value
thereafter: spending ratios mapped onto the model's instruments, revenue
gains spread over the consumption and corporate taxes, and a real effective
interest rate on debt that reproduces the program's debt path. These tests
pin the packaged paths to the derivation in ``ogeth.macro_params`` and check
the mapping reproduces the IMF accounts. No model solve is involved.
"""

import json
from importlib.resources import files

import numpy as np
import pytest
from ogcore.parameters import Specifications

from ogeth import macro_params as mp

N = len(mp.PROGRAM_YEARS)


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


def test_spending_paths_reproduce_imf_primary_expenditure():
    """G + TR + I_g equals IMF expenditure net of interest, within 0.1 pp."""
    sp = mp.program_spending_paths()
    primary = (
        np.array(sp["alpha_G"][:N])
        + np.array(sp["alpha_T"][:N])
        + np.array(sp["alpha_I"][:N])
    )
    imf = (np.array(mp.IMF_EXPENDITURE) - np.array(mp.IMF_INTEREST)) / 100
    assert np.allclose(primary, imf, atol=1e-3)


def test_program_paths_start_at_fy2024_25_values(packaged):
    """Period 0 is FY2024/25: 50.5 percent debt, 5.0 percent capital
    spending, 1.7 percent grants (IMF CR 26/174, Tables 1 and 2b)."""
    assert packaged["initial_debt_ratio"] == pytest.approx(0.505)
    assert packaged["alpha_I"][0] == pytest.approx(0.050)
    assert packaged["alpha_FA"][0] == pytest.approx(0.017)
    assert packaged["alpha_G"][0] == pytest.approx(0.043)
    assert packaged["alpha_T"][0] == pytest.approx(0.020)


def test_closure_starts_after_the_program_horizon(packaged):
    """Spending follows the program for its seven fiscal years; OG-Core's
    debt-closure rule takes over from FY2031/32."""
    assert packaged["tG1"] == N


def test_revenue_paths_add_the_imf_revenue_gain(params):
    """Consumption-tax and CIT paths carry the program's revenue increase
    (9.2 -> 12.3 percent of GDP) in the stated two-thirds / one-third split,
    evaluated at the FY2024/25 collections they scale."""
    tau_c = np.asarray(params.tau_c)[:N, 0]
    cit = np.asarray(params.adjustment_factor_for_cit_receipts)[:N]
    gain_cons = (tau_c / tau_c[0] - 1) * mp.BASE_CONS_TAX_REVENUE
    gain_cit = (cit / cit[0] - 1) * mp.BASE_CIT_REVENUE
    imf_gain = (np.array(mp.IMF_REVENUE) - mp.IMF_REVENUE[0]) / 100
    assert np.allclose(gain_cons + gain_cit, imf_gain, atol=1e-12)
    assert np.allclose(gain_cons, mp.INDIRECT_SHARE_OF_REVENUE_GAIN * imf_gain)


def test_implied_real_rate_on_debt_is_negative_through_the_program(params):
    """The program's debt decline (50.5 -> 28.6 percent of GDP with primary
    balances near zero) is only possible with a negative real effective rate
    on the legacy debt: inflation and concessional terms."""
    r = mp.implied_real_rate_on_debt(params.g_y, params.g_n)
    assert r.shape == (N - 1,)
    assert (r < 0).all() and (r > -0.06).all()


def test_r_gov_path_sits_on_the_floor_through_the_program(params):
    """OG-Core floors r_gov at zero, so the shipped shift path targets the
    floor (not the negative program-implied rate) in every program year."""
    shift = np.asarray(params.r_gov_shift)[:N]
    scale = float(np.asarray(params.r_gov_scale).flatten()[0])
    d = np.array(mp.IMF_PUBLIC_DEBT) / 100
    # ogcore: r_gov = scale r - shift + r_gov_DY d + r_gov_DY2 d^2
    r_gov = (
        scale * mp.R_SS_FOR_R_GOV
        - shift
        + params.r_gov_DY * d
        + params.r_gov_DY2 * d**2
    )
    assert np.allclose(r_gov, mp.R_GOV_FLOOR, atol=1e-9)


def test_r_gov_shift_path_converges_to_the_long_run_rate(params):
    """After the program and the convergence window, r_gov_scale r - shift
    (premium zero at the debt target) equals LONG_RUN_R_GOV."""
    shift = np.asarray(params.r_gov_shift)
    scale = float(np.asarray(params.r_gov_scale).flatten()[0])
    r_ss = mp.R_SS_FOR_R_GOV
    D = params.debt_ratio_ss
    long_run = (
        scale * r_ss
        - shift[-1]
        + params.r_gov_DY * D
        + params.r_gov_DY2 * D**2
    )
    assert long_run == pytest.approx(mp.LONG_RUN_R_GOV, abs=1e-9)
    assert shift[-1] == pytest.approx(shift[N + mp.R_GOV_CONVERGENCE_PERIODS])


def test_packaged_fiscal_paths_match_the_derivation(packaged, params):
    """A regeneration must rewrite the fiscal paths from the same tables."""
    expected = mp.fiscal_program_params(params)
    for key in ("alpha_G", "alpha_T", "alpha_I", "alpha_FA"):
        assert np.allclose(
            np.array(packaged[key])[: len(expected[key])], expected[key]
        )
    assert np.allclose(
        np.array(packaged["adjustment_factor_for_cit_receipts"])[
            : len(expected["adjustment_factor_for_cit_receipts"])
        ],
        expected["adjustment_factor_for_cit_receipts"],
    )
    assert np.allclose(
        np.array(packaged["tau_c"])[: len(expected["tau_c"])],
        expected["tau_c"],
    )
    assert np.allclose(
        np.array(packaged["r_gov_shift"])[: len(expected["r_gov_shift"])],
        expected["r_gov_shift"],
    )


def test_initial_wealth_ratio_is_built_from_its_components():
    """B/Y = K/Y (PWT) - foreign-owned capital (FDI stock) + domestic debt."""
    assert mp.INITIAL_WEALTH_RATIO == pytest.approx(
        mp.PWT_CAPITAL_OUTPUT_RATIO
        - mp.FDI_STOCK_TO_GDP
        + mp.DOMESTIC_DEBT_TO_GDP,
        abs=1e-3,
    )
    assert 1.5 < mp.INITIAL_WEALTH_RATIO < 3.0
