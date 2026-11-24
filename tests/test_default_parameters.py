"""
Tests that the packaged default parameters carry the tax and informality
calibration.

The income-tax functions are Ethiopia's statutory schedules in
Gouveia-Strauss form (Proclamation 979/2016 in FY2024/25, 1395/2025 from
FY2025/26). Informality is graded income-tax non-compliance by
lifetime-income group (OG-Core's by-(t, j) noncompliance parameters, PR
#816), with lifetime income as the proxy for formality: the formal 22.35
percent of the employed (ILOSTAT, 2021 Labour Force and Migration Survey)
are the richest households. The compliance vector gives the shape of the
formal tax boundary and COMPLIANCE_SCALE its level, tuned so personal income
tax collects 0.81 percent of rebased GDP in FY2024/25; the CIT collections
factor is tuned so corporate income tax and the other capital-income lines
collect 1.26 percent (examples/tune.py). income_tax_filer stays all-ones on
purpose: informality is carried by non-compliance, not by non-filing.

These are fast, structural pins on the shipped JSON. The revenue the
calibration collects requires a model solve and is therefore not asserted
here.
"""

import json
from importlib.resources import files

import numpy as np
import pytest
from ogcore.parameters import Specifications

from ogeth import macro_params as mp

# Expected income-tax non-compliance by lifetime-income group (low -> high),
# shared by the labor and capital rows.
# the shape of compliance across groups; the level is the calibrated
# COMPLIANCE_SCALE in macro_params (see taxes.md)
# formality by group from ILOSTAT's 77.65 percent informal employment (the
# formal 22.35 percent at the top of the lifetime-income distribution), times
# the calibrated compliance scale
NONCOMPLIANCE_BY_GROUP = (
    1
    - mp.COMPLIANCE_SCALE
    * np.array(mp.formal_share_by_group(mp.INFORMAL_EMPLOYMENT_SHARE))
).tolist()


@pytest.fixture(scope="module")
def defaults():
    """Raw packaged default-parameters JSON as a dict."""
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(content)


@pytest.fixture(scope="module")
def p(defaults):
    """Packaged defaults loaded into ogcore Specifications.

    update_specifications validates and broadcasts to the full T dimension, so
    tests index the steady-state row (-1) to stay robust to that broadcast.
    """
    spec = Specifications(baseline=True)
    spec.update_specifications(defaults)
    return spec


def test_packaged_defaults_validate(p):
    """Shipped defaults load into Specifications with no validation errors."""
    assert not p.errors


def test_statutory_tax_functions(p):
    """The income-tax functions are the statutory schedules: Proclamation
    979/2016 in FY2024/25, 1395/2025 from FY2025/26, in Gouveia-Strauss
    form, with a flat dividend rate on capital income and the birr anchor
    at GDP per adult."""
    from ogeth import macro_params as mp

    assert p.tax_func_type == "GS"
    etr = np.asarray(p.etr_params)
    assert etr.shape[-1] == 3
    fitted = mp.statutory_tax_params()
    assert etr[0, 0].tolist() == pytest.approx(fitted["etr_params"][0][0])
    assert etr[1, 0].tolist() == pytest.approx(fitted["etr_params"][1][0])
    assert etr[-1, 0].tolist() == pytest.approx(fitted["etr_params"][1][0])
    assert np.asarray(p.mtrx_params)[-1, 0].tolist() == pytest.approx(
        fitted["etr_params"][1][0]
    )
    assert np.asarray(p.mtry_params)[-1, 0, 0] == pytest.approx(
        mp.CAPITAL_INCOME_TAX_RATE
    )
    assert p.mean_income_data == pytest.approx(mp.mean_income_data())
    # the 2025 schedule, in FY2024/25 birr, at the top decile's 55,000 birr
    # a month is close to its 35 percent top rate; at the exemption
    # threshold (2,000 birr a month in FY2025/26 birr, about 1,550 in
    # FY2024/25 birr) it is near zero
    top = mp.gs_rates(55_000 * 12, etr[-1, 0], "etr")
    assert 0.28 < top < 0.35
    growth = (31934 / 24031.07) / (1 + mp.POPULATION_GROWTH)
    assert mp.gs_rates(12 * 2000 / growth, etr[-1, 0], "etr") < 0.04


def test_informality_noncompliance_by_group(p):
    """Compliance rises with lifetime income: the bottom three groups are
    fully informal, groups 5-7 formal, scaled by COMPLIANCE_SCALE."""
    # period 0 (FY2024/25) carries the informality calibration, held
    labor = np.asarray(p.labor_income_tax_noncompliance_rate)
    capital = np.asarray(p.capital_income_tax_noncompliance_rate)
    assert labor[0].tolist() == pytest.approx(NONCOMPLIANCE_BY_GROUP)
    assert capital[0].tolist() == pytest.approx(NONCOMPLIANCE_BY_GROUP)
    assert np.allclose(labor, capital)


def test_cit_collections_factor(p):
    """CIT collections factor re-anchored to hit FY2024/25 corporate-tax
    revenue and held there."""
    factor = np.asarray(p.adjustment_factor_for_cit_receipts)
    # tuned so that corporate income tax plus the other capital-income lines
    # collect 1.26 percent of rebased GDP in FY2024/25 (examples/tune.py)
    # the factor that the gamma derivation uses must be the tuned one
    from ogeth import macro_params as mp

    assert float(factor[0]) == pytest.approx(
        mp.CIT_COLLECTIONS_FACTOR, abs=1e-4
    )
    assert np.allclose(factor, factor[0])


def test_all_households_are_filers(p):
    """income_tax_filer stays all-ones: non-compliance, not non-filing."""
    assert np.asarray(p.income_tax_filer)[-1].tolist() == pytest.approx(
        [1.0] * 7
    )


def test_informality_dimensions_consistent(defaults, p):
    """Group vectors match the 7 income groups; ~90% weight in the tail."""
    lambdas = np.asarray(p.lambdas).flatten()
    assert lambdas.shape[0] == 7
    assert len(defaults["labor_income_tax_noncompliance_rate"][0]) == 7
    assert len(defaults["capital_income_tax_noncompliance_rate"][0]) == 7
    # Bottom 5 groups (fully non-compliant) carry ~90% of households.
    assert lambdas[:5].sum() == pytest.approx(0.90)
