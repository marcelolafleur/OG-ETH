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
    if mp.PENSIONS_ON:
        # pension benefits are paid by the pension system, not as transfers
        primary = primary + mp.PENSIONS_SHARE_OF_GDP / 100
    imf = (np.array(mp.IMF_EXPENDITURE) - np.array(mp.IMF_INTEREST)) / 100
    assert np.allclose(primary, imf, atol=1e-3)


def test_program_paths_start_at_fy2024_25_values(packaged):
    """Period 0 is FY2024/25: 50.5 percent debt, 5.0 percent capital
    spending, 1.7 percent grants (IMF CR 26/174, Tables 1 and 2b)."""
    assert packaged["initial_debt_ratio"] == pytest.approx(0.505)
    assert packaged["alpha_I"][0] == pytest.approx(0.050)
    assert packaged["alpha_FA"][0] == pytest.approx(0.017)
    assert packaged["alpha_G"][0] == pytest.approx(0.043)
    cash = 0.020 - (mp.PENSIONS_SHARE_OF_GDP / 100 if mp.PENSIONS_ON else 0)
    assert packaged["alpha_T"][0] == pytest.approx(cash)


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
    # the formalization path phases in PIT_FORMALIZATION_GAIN over the program
    formal = mp.PIT_FORMALIZATION_GAIN * np.arange(N) / (N - 1)
    assert np.allclose(gain_cons + gain_cit + formal, imf_gain, atol=1e-12)
    assert np.allclose(
        gain_cons, mp.INDIRECT_SHARE_OF_REVENUE_GAIN * (imf_gain - formal)
    )


def test_formalization_path_broadens_the_base_over_the_program(packaged):
    """Group 6 goes from half to full compliance and group 5 from none to a
    fifth, linearly over the seven program years, then holds; labor and
    capital rates move together (the SS diagnostic tiles one from the
    other)."""
    lab = np.array(packaged["labor_income_tax_noncompliance_rate"])
    cap = np.array(packaged["capital_income_tax_noncompliance_rate"])
    assert lab.shape[0] == N + 1 and np.allclose(lab, cap)
    # the shape vectors give compliance relative to the top group; the
    # calibrated scale sets the level (see taxes.md)
    start = 1 - mp.COMPLIANCE_SCALE * (1 - np.array(mp.NONCOMPLIANCE_START))
    end = 1 - mp.COMPLIANCE_SCALE * (1 - np.array(mp.NONCOMPLIANCE_END))
    assert lab[0].tolist() == pytest.approx(start.tolist())
    assert lab[N - 1].tolist() == pytest.approx(end.tolist())
    assert lab[N].tolist() == pytest.approx(end.tolist())
    assert 0 < mp.COMPLIANCE_SCALE < 1
    assert (np.diff(lab, axis=0) <= 1e-12).all()  # compliance only improves


def test_pensions_go_to_the_formal_groups_only(packaged):
    """replacement_rate_adjust mirrors the compliance structure: no public
    pension for the five informal groups, half for group 6, full for group
    7."""
    adj = np.array(packaged["replacement_rate_adjust"])
    assert adj.shape[-1] == 7
    expected = np.array(mp.PENSION_COVERAGE) * mp.PENSION_COVERAGE_SCALE
    assert adj[-1].tolist() == pytest.approx(expected.tolist())
    assert adj[-1][:5].tolist() == pytest.approx([0.0] * 5)
    assert adj[-1][5] == pytest.approx(0.5 * adj[-1][6])
    assert 0 < mp.PENSION_COVERAGE_SCALE < 1


def test_implied_real_rate_on_debt_is_negative_through_the_program(params):
    """The program's debt decline (50.5 -> 28.6 percent of GDP with primary
    balances near zero) is only possible with a negative real effective rate
    on the legacy debt: inflation and concessional terms."""
    r = mp.implied_real_rate_on_debt(params.g_y, params.g_n)
    assert r.shape == (N - 1,)
    assert (r < 0).all() and (r > -0.06).all()


def test_r_gov_path_targets_the_program_implied_rates(params):
    """Through the program years the shift path puts r_gov (before OG-Core's
    floor) on the negative real rates that reproduce the IMF debt path; the
    example lowers r_gov_floor so they bind where the installed ogcore has
    the parameter (PSLmodels/OG-Core#1203)."""
    shift = np.asarray(params.r_gov_shift)[1:N]
    scale = float(np.asarray(params.r_gov_scale).flatten()[0])
    d = np.array(mp.IMF_PUBLIC_DEBT[1:]) / 100
    r_gov = (
        scale * mp.R_SS_FOR_R_GOV
        - shift
        + params.r_gov_DY * d
        + params.r_gov_DY2 * d**2
    )
    implied = mp.implied_real_rate_on_debt(params.g_y, params.g_n)
    assert np.allclose(r_gov, implied, atol=1e-9)
    assert (implied > mp.R_GOV_FLOOR).all()


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


def test_zeta_d_path_follows_the_external_share_of_the_debt_decline(packaged):
    """During the program external creditors absorb the share of the debt
    decline the IMF external/domestic split implies (0.37 -> 1.0 -> 0.77);
    afterwards zeta_D returns to the calibrated 0.15."""
    z = np.asarray(packaged["zeta_D"])
    assert np.allclose(z[: N + 1], mp.program_zeta_D_path())
    tot = np.array(mp.IMF_PUBLIC_DEBT)
    ext = tot - np.array(mp.IMF_DOMESTIC_DEBT)
    assert np.allclose(z[1:N], np.clip(np.diff(ext) / np.diff(tot), 0, 1))
    assert z[N] == pytest.approx(mp.LONG_RUN_ZETA_D)
    assert (z >= 0).all() and (z <= 1).all()


def test_transfers_are_targeted_by_programme(packaged, params):
    """eta gives the safety net to the poorest quarter, the fertilizer
    subsidy to the bottom 70 percent, and pensions to the retired in the
    formal groups; it is rebuilt from the packaged demographics."""
    eta = np.array(packaged["eta"])
    assert eta.shape == (80, 7) and eta.sum() == pytest.approx(1.0)
    retire_idx = int(params.retirement_age[0]) - int(params.E)
    expected = mp.transfer_eta(
        packaged["omega_SS"], packaged["lambdas"], retire_idx
    )
    assert np.allclose(eta, expected, atol=1e-12)
    pensions = 0.0 if mp.PENSIONS_ON else mp.PENSIONS_SHARE_OF_GDP
    total = (
        mp.PSNP_SHARE_OF_GDP + mp.FERTILIZER_SUBSIDY_SHARE_OF_GDP + pensions
    )
    share = eta.sum(axis=0)
    assert share[3:5].sum() == pytest.approx(0.0)  # groups 4-5 get nothing
    # with the pension system on, the retired formal groups are paid
    # benefits instead of transfers
    assert share[5:].sum() == pytest.approx(pensions / total)
    assert eta[:retire_idx, 5:].sum() == pytest.approx(
        0.0
    )  # pensions: retired
    assert share[0] > share[1] > share[2]
