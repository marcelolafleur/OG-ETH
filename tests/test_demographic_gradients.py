"""
Tests of the income gradients in fertility and mortality that Calibration
passes to OG-Core's demographics, built from the Demographic-Gradients
library's values for Ethiopia (commit ccf53a0, 2026-11-16).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ogeth import calibrate

# library values for Ethiopia
TFR, IMR, CMR, G45 = -0.73607, -0.65661, -0.82433, 0.0745
OFFSETS = {
    15: -0.77754,
    20: -0.09955,
    25: 0.20895,
    30: 0.32450,
    35: 0.09223,
    40: -0.39232,
    45: -0.98271,
}
BAND = {"15-29": -0.4063, "30-44": -0.18, "45-59": 0.2069}
RULE_SLOPE = -0.4324


def test_gradients_by_age_in_library_units():
    g = calibrate.demographic_gradients(20, 80)
    fert, mort = g["fert_gradient"], g["mort_gradient"]
    assert fert.shape == mort.shape == (100,)
    # TFR plus the pooled offset of each age group, divided by 100
    for age, off in OFFSETS.items():
        assert fert[age] == pytest.approx((TFR + off) / 100, abs=1e-6)
    assert fert[10] == pytest.approx(fert[15])  # below 15: 15-19 offset
    assert fert[60] == pytest.approx(fert[47])  # 50 and over: 45-49 offset
    assert g["infmort_gradient"] == pytest.approx(IMR / 100, abs=1e-6)
    assert mort[0] == 0.0  # age 0 is infmort_gradient
    assert np.allclose(mort[1:5], CMR / 100, atol=1e-6)
    assert np.allclose(mort[5:30], (G45 + BAND["15-29"]) / 100, atol=1e-6)
    assert np.allclose(mort[30:45], (G45 + BAND["30-44"]) / 100, atol=1e-6)
    top = (G45 + BAND["45-59"]) / 100
    assert np.allclose(mort[45:60], top, atol=1e-6)
    # straight line from the 45-59 value at 60 to 0 at 75, then 0
    assert mort[60] == pytest.approx(top)
    assert mort[67] == pytest.approx(top * 8 / 15)
    assert np.all(mort[75:] == 0.0)


def _rates(p=0.1, n=400):
    mort = np.full((n, 100), p)
    mort[:, -1] = 1.0
    return {
        "fert": np.full((n, 100), p),
        "mort": mort,
        "infmort": np.full(n, p),
        "first_year": 2023,
    }


def test_conversion_to_ogcore_units():
    g = calibrate.demographic_gradients(20, 80, rates=_rates(0.1))
    base = calibrate.demographic_gradients(20, 80)
    # gradient / (100 (1 - p)) at every age and year
    assert np.allclose(g["fert_gradient"][0], base["fert_gradient"] / 0.9)
    assert np.allclose(g["infmort_gradient"], base["infmort_gradient"] / 0.9)
    assert np.allclose(
        g["mort_gradient"][0, :-1], base["mort_gradient"][:-1] / 0.9
    )
    assert np.all(g["mort_gradient"][:, -1] == 0.0)  # final age
    assert calibrate.to_ogcore_units(-0.5, 1.0) == 0.0


def test_adult_mortality_steepens_with_income():
    g = calibrate.demographic_gradients(
        20, 80, T=320, start_year=2025, g_y_annual=0.047, rates=_rates(0.0)
    )
    mort = g["mort_gradient"]
    assert mort.shape == (400, 100)
    # row 2 is 2025, the latest GDP year: the rule's change since 2007
    shift = RULE_SLOPE * np.log(3094.87 / 1109.14)
    assert mort[2, 20] == pytest.approx(
        (G45 + shift + BAND["15-29"]) / 100, abs=1e-6
    )
    assert mort[2, 3] == pytest.approx(CMR / 100)  # children: no rule
    assert mort[2, 80] == 0.0  # 75 and over: 0
    # steeper over time, held once GDP per head reaches the rule's range
    assert mort[40, 30] < mort[2, 30]
    capped = RULE_SLOPE * np.log(18000 / 1109.14)
    assert mort[-1, 20] == pytest.approx(
        (G45 + capped + BAND["15-29"]) / 100, abs=1e-6
    )


def test_gradients_missing_indicator_raises(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text(
        "indicator,year,sex,age_lo,age_hi,measure,slope\n"
        "TFR,2024,all,,,,-0.7\n"
    )
    with pytest.raises(ValueError):
        calibrate.demographic_gradients(20, 80, path=str(path))


def test_calibration_passes_gradients_to_demographics():
    p = MagicMock()
    p.I, p.M, p.E, p.S, p.T, p.J = 1, 1, 20, 80, 320, 7
    p.start_year = 2025
    p.g_y_annual = 0.047
    p.lambdas = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
    p.g_y = 0.04
    p.retirement_age = np.array([65])
    omega = np.full((80, 7), 1 / 80) * p.lambdas.reshape(1, 7)
    demog = {"omega_SS": omega, "g_n": np.full(400, 0.02)}
    with patch("ogeth.calibrate.macro_params") as mock_macro:
        mock_macro.get_macro_params.return_value = {}
        mock_macro.derive_remittance_params.return_value = {}
        mock_macro.transfer_eta.return_value = np.zeros((80, 7))
        with patch("ogeth.calibrate.demographic_rates") as mock_rates:
            mock_rates.return_value = _rates(0.05)
            with patch("ogcore.demographics.get_pop_objs") as mock_demog:
                mock_demog.return_value = demog
                with patch("ogeth.calibrate.income.get_e_interp") as mock_e:
                    mock_e.return_value = np.ones((80, 7))
                    c = calibrate.Calibration(p, update_from_api=True)
    kwargs = mock_demog.call_args.kwargs
    assert kwargs["country_id"] == calibrate.UN_COUNTRY_CODE
    np.testing.assert_array_equal(kwargs["income_percentiles"], p.lambdas)
    assert kwargs["fert_gradient"].shape == (400, 100)
    assert kwargs["mort_gradient"].shape == (400, 100)
    assert kwargs["infmort_gradient"].shape == (400,)
    assert kwargs["infmort_gradient"][0] == pytest.approx(
        IMR / (100 * 0.95), abs=1e-6
    )
    assert "e" in c.get_dict()
