"""
Tests of the baseline's fiscal paths.

Every fiscal input takes its FY2024/25 outturn or enacted FY2025/26 budget
value (IMF Country Report 26/174) and is held there, and debt stabilizes at
its measured ratio. These tests pin the packaged paths to their derivation
and check that the mapping onto the model's instruments reproduces the
published accounts. No model solve is involved.
"""

import json
from importlib.resources import files

import numpy as np
import pytest
from ogcore.parameters import Specifications

from ogeth import macro_params as mp

N = len(mp.FISCAL_YEARS)


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
    sp = mp.spending_paths()
    primary = (
        np.array(sp["alpha_G"][:N])
        + np.array(sp["alpha_T"][:N])
        + np.array(sp["alpha_I"][:N])
    )
    # the paths are shares of the model's Y; the IMF's of market-price GDP
    primary = primary / mp.MODEL_PER_MARKET_GDP
    if mp.PENSIONS_ON:
        # pension benefits are paid by the pension system, not as transfers
        primary = primary + mp.PENSIONS_SHARE_OF_GDP / 100
    imf = (np.array(mp.IMF_EXPENDITURE) - np.array(mp.IMF_INTEREST)) / 100
    assert np.allclose(primary, imf, atol=1e-3)


def test_paths_start_at_fy2024_25_values(packaged):
    """Period 0 is FY2024/25: general-government debt 43.3, capital spending
    5.0, grants 1.7 and recurrent spending net of interest and transfers 4.6
    percent of old-base GDP (IMF WEO Oct 2026; CR 26/174 Tables 1 and 2b),
    times old over rebased GDP (15,094 / 24,031), in model units."""
    f = 15094.0 / 24031.07
    assert mp.rebase_factors()[0] == pytest.approx(f)
    # general-government debt, split into external debt (alpha_D_ext) and
    # market debt (initial_debt_ratio), which is domestic: non-residents
    # hold no birr government securities
    total = packaged["initial_debt_ratio"] + packaged["alpha_D_ext"][0]
    assert total == pytest.approx(mp.to_model(0.433 * f), abs=1e-4)
    external = mp.to_model(0.433 * f) * (31.8 - 7.2) / 43.3
    assert packaged["alpha_D_ext"][0] == pytest.approx(external, abs=1e-4)
    assert packaged["initial_foreign_debt_ratio"] == 0.0
    assert packaged["alpha_I"][0] == pytest.approx(
        mp.to_model(0.050 * f), abs=1e-5
    )
    assert packaged["alpha_FA"][0] == pytest.approx(
        mp.to_model(0.017 * f), abs=1e-5
    )
    # cash transfers: the published 2.0 percent with its assumed 0.5 of
    # pensions replaced by the schemes' measured benefits (0.20)
    cash_total = 0.020 - 0.005 + mp.PENSION_BENEFITS_PUBLISHED / 100
    assert packaged["alpha_G"][0] == pytest.approx(
        mp.to_model((0.071 - 0.008 - cash_total) * f), abs=1e-5
    )
    cash = cash_total * f - (
        mp.PENSIONS_SHARE_OF_GDP / 100 if mp.PENSIONS_ON else 0
    )
    assert packaged["alpha_T"][0] == pytest.approx(mp.to_model(cash), abs=1e-5)


def test_closure_starts_in_fy2031_32(packaged):
    """OG-Core's debt-closure rule takes over from model period 7
    (FY2031/32); the fiscal paths are written out to it."""
    assert packaged["tG1"] == mp.TG1 == 7
    assert len(packaged["alpha_G"]) == mp.FISCAL_PATH_LENGTH


def test_baseline_holds_fy2025_26_inputs(packaged, params):
    """The packaged baseline: FY2024/25 outturn and FY2025/26 budget, then
    held; effective revenue rates, compliance and the foreign borrowing
    share flat; debt stabilizes at its measured ratio; remittances flat."""
    expected = mp.fiscal_params(params)
    for key in ("alpha_G", "alpha_T", "alpha_I", "alpha_FA"):
        path = np.array(packaged[key])
        assert np.allclose(path[: len(expected[key])], expected[key])
        # nothing moves after the enacted budget year
        assert np.allclose(path[N - 1 :], path[-1])
    tau_c = np.asarray(packaged["tau_c"])
    assert np.allclose(tau_c, tau_c[0])
    cit = np.asarray(packaged["adjustment_factor_for_cit_receipts"])
    assert np.allclose(cit, cit[0])
    lab = np.array(packaged["labor_income_tax_noncompliance_rate"])
    assert np.allclose(lab, lab[0])
    assert np.allclose(packaged["zeta_D"], mp.MARKET_FOREIGN_DEBT_SHARE)
    assert packaged["debt_ratio_ss"] == pytest.approx(mp.DEBT_RATIO_SS)
    # the long-run anchor is the FY2024/25 ratio of all debt, which is all
    # market debt once external debt has been repaid
    assert packaged["debt_ratio_ss"] == pytest.approx(
        packaged["initial_debt_ratio"] + packaged["alpha_D_ext"][0], abs=1e-4
    )
    assert packaged["alpha_D_ext"][-1] == 0.0
    assert packaged["alpha_RM_T"] == packaged["alpha_RM_1"]
    assert packaged["r_gov_DY"] == pytest.approx(
        -2 * packaged["r_gov_DY2"] * mp.DEBT_RATIO_SS
    )


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


def test_domestic_market_debt_matches_the_imf_domestic_debt():
    """Market debt less its foreign-held part is domestically held debt, which
    the IMF reports separately (18.7 percent of old-base GDP in FY2024/25):
    the external-debt split reproduces it to within 0.1 points of GDP."""
    domestic = mp.INITIAL_MARKET_DEBT_RATIO * (
        1 - mp.MARKET_FOREIGN_DEBT_SHARE
    )
    assert domestic == pytest.approx(
        mp.to_model(mp.IMF_DOMESTIC_DEBT[0] / 100), abs=1e-3
    )


def test_r_gov_targets_the_dsa_market_rate(params):
    """At the debt anchor (premium zero) r_gov follows the
    DSA's real rate on domestic debt year by year to 2036 (the legacy
    stock rolling over at market rates), then the long-run rate."""
    targets = mp.market_rate_targets()
    # OG-Core extends the path by holding its last value
    shift = np.asarray(params.r_gov_shift)[: len(targets)]
    scale = float(np.asarray(params.r_gov_scale).flatten()[0])
    D = params.debt_ratio_ss
    r_gov = (
        scale * mp.R_SS_FOR_R_GOV
        - shift
        + params.r_gov_DY * D
        + params.r_gov_DY2 * D**2
    )
    assert np.allclose(r_gov, targets, atol=1e-9)
    assert r_gov[0] == pytest.approx(-0.110)
    assert r_gov[6] == pytest.approx(0.033)  # FY2030/31
    assert r_gov[11] == pytest.approx(0.046)  # 2036
    assert (np.diff(r_gov) > 0).all()


def test_long_run_r_gov_is_the_dsa_rate_on_new_domestic_debt(params):
    """The long-run real rate on market debt is the mean of the DSA's terms
    for new domestic borrowing (4.9 percent medium and long term, 4.5 short
    term); OG-Core holds the last shift value."""
    assert mp.LONG_RUN_R_GOV == pytest.approx(0.047)
    shift = np.asarray(params.r_gov_shift)
    scale = float(np.asarray(params.r_gov_scale).flatten()[0])
    D = params.debt_ratio_ss
    long_run = (
        scale * mp.R_SS_FOR_R_GOV
        - shift[-1]
        + params.r_gov_DY * D
        + params.r_gov_DY2 * D**2
    )
    # R_SS_FOR_R_GOV is stored to six digits
    assert long_run == pytest.approx(mp.LONG_RUN_R_GOV, abs=1e-6)


def test_packaged_fiscal_paths_match_the_derivation(packaged, params):
    """A regeneration must rewrite the fiscal paths from the same tables."""
    expected = mp.baseline_params(params)
    assert np.allclose(packaged["g_RM"], expected["g_RM"])
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
    assert packaged["frisch"] == expected["frisch"] == mp.FRISCH
    assert packaged["sigma"] == expected["sigma"] == mp.SIGMA
    assert packaged["c_min"] == expected["c_min"] == [mp.C_MIN]
    assert packaged["Z"] == [[1.0]]
    # e carries the unit normalization: population-weighted mean
    e = np.asarray(packaged["e"])
    omega = np.asarray(packaged["omega_SS"])
    assert (e * omega).sum() / omega.sum() == pytest.approx(
        mp.EARNINGS_SCALE, rel=1e-6
    )
    for key in ("alpha_D_ext", "r_D_ext_annual"):
        assert len(packaged[key]) >= len(expected[key]), key
        assert np.allclose(
            np.array(packaged[key])[: len(expected[key])], expected[key]
        ), key


def test_initial_wealth_ratio_is_built_from_its_components():
    """B/Y = private K/Y (rebased total less rebased IMF ICSD public
    capital) - foreign-owned capital (UNCTAD FDI stock, mid-FY2024/25, over
    rebased USD GDP) + domestically held general-government debt, in model
    units."""
    assert mp.PRIVATE_CAPITAL_OUTPUT_RATIO == pytest.approx(
        mp.TOTAL_CAPITAL_OUTPUT_RATIO - mp.PUBLIC_CAPITAL_OUTPUT_RATIO
    )
    # net replacement-cost FDI stock from IMF equity inflows; cumulative
    # inflows match UNCTAD's 42.5 billion (end-2024) closely
    assert mp.FDI_STOCK_TO_GDP == pytest.approx(
        mp.fdi_stock_usd_bn() / 201.42, abs=1e-4
    )
    assert mp.INITIAL_WEALTH_RATIO == pytest.approx(
        mp.to_model(
            mp.PRIVATE_CAPITAL_OUTPUT_RATIO
            - mp.FDI_STOCK_TO_GDP
            + mp.DOMESTIC_DEBT_TO_GDP
        ),
        abs=1e-3,
    )
    # OG-Core's valid range for initial_wealth_ratio
    assert 0.8 < mp.INITIAL_WEALTH_RATIO < 7.0


def test_external_debt_follows_the_dsa_then_ends():
    """External debt moves with the DSA's public external debt ratio
    (31.8 percent of old-base GDP in FY2024/25, 16.8 in FY2030/31, 9.2 in
    2036, 4.1 in 2045), then falls to zero at the 2036-2045 pace: no
    concessional subsidy in the long run."""
    a = np.array(mp.external_debt_path())
    assert a[0] == pytest.approx(mp.EXTERNAL_DEBT_RATIO)
    assert a[6] / a[0] == pytest.approx(16.8 / 31.8, abs=1e-4)
    assert a[11] / a[0] == pytest.approx(9.2 / 31.8, abs=1e-4)
    assert a[20] / a[0] == pytest.approx(4.1 / 31.8, abs=1e-4)
    assert (np.diff(a) <= 0).all()
    assert a[-1] == 0.0 and a[-2] > 0.0


def test_external_rate_is_the_dsa_external_rate_in_real_terms():
    """r_D_ext is the DSA's average nominal rate on external debt (USD) less
    US inflation: -1.3 percent real in FY2024/25, 0.6 at the FY2026/27 peak,
    held after 2045."""
    r = np.array(mp.external_rate_path())
    assert r[0] == pytest.approx(0.007 - mp.US_INFLATION)
    assert r[2] == pytest.approx(0.026 - mp.US_INFLATION)
    assert r[-1] == pytest.approx(0.009 - mp.US_INFLATION)
    assert len(r) == 2045 - 2025 + 1


def test_market_debt_is_held_at_home():
    """Market debt is birr government securities, held by the central bank,
    banks and pension funds, so its foreign share and the foreign share of
    new market debt are zero; borrowing abroad enters through alpha_D_ext."""
    z = np.asarray(mp.zeta_D_path())
    assert np.allclose(z, 0.0)
    assert mp.MARKET_FOREIGN_DEBT_SHARE == 0.0


def test_transfers_are_targeted_by_programme(packaged, params):
    """eta gives the safety net to the poorest quarter, the fertilizer
    subsidy to the bottom 70 percent, and pensions to the retired in the
    formal groups; it is rebuilt from the packaged demographics, for each
    year from that year's population, ending at the steady-state matrix."""
    path = np.array(packaged["eta"])
    n = path.shape[0] - 1
    assert path.shape[1:] == (80, 7) and 0 < n < params.T
    assert np.allclose(path.sum(axis=(1, 2)), 1.0, atol=1e-12)
    retire_idx = int(params.retirement_age[0]) - int(params.E)
    omega = np.array(packaged["omega"])
    for t in range(n):
        expected = mp.transfer_eta(omega[t], packaged["lambdas"], retire_idx)
        assert np.allclose(path[t], expected, atol=1e-12)
    eta = path[-1]
    expected = mp.transfer_eta(
        packaged["omega_SS"], packaged["lambdas"], retire_idx
    )
    assert np.array_equal(eta, expected)
    # the steady-state matrix holds for the remaining years and in the
    # steady state
    assert np.array_equal(
        params.eta[n:], np.tile(eta, (params.T + params.S - n, 1, 1))
    )
    # PSNP by its design, in every year: per person, the poorest group gets
    # the fertilizer subsidy (as group 1 does) plus PSNP, and PSNP pays
    # twice as much from PSNP_DIRECT_SUPPORT_AGE (12 months of direct
    # support) as below it (6 months of public works)
    age = np.arange(params.S) + int(params.E)
    old = age >= mp.PSNP_DIRECT_SUPPORT_AGE
    pops = list(omega[:n]) + [np.array(packaged["omega_SS"])]
    for eta_t, om in zip(path, pops):
        per_person = eta_t / om
        psnp = per_person[:, 0] - per_person[:, 1]
        assert np.allclose(psnp[old], 2 * psnp[~old].mean(), rtol=1e-9)
        assert np.allclose(psnp[~old], psnp[~old].mean(), rtol=1e-9)
    # with the pension system on, old-age pensions are paid as benefits and
    # only survivor and invalidity benefits reach the covered groups as
    # transfers, at all ages
    covered = mp.SURVIVOR_INVALIDITY_SHARE_OF_GDP
    if not mp.PENSIONS_ON:
        covered += mp.PENSIONS_SHARE_OF_GDP
    total = mp.PSNP_SHARE_OF_GDP + mp.FERTILIZER_SUBSIDY_SHARE_OF_GDP + covered
    share = eta.sum(axis=0)
    assert share[3:5].sum() == pytest.approx(0.0)  # groups 4-5 get nothing
    assert share[5:].sum() == pytest.approx(covered / total)
    assert share[0] > share[1] > share[2]
