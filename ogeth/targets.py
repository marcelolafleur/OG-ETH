"""
Data targets for the OG-ETH calibration dashboard, and the functions that
compute the matching model moments from a solved ``SS_vars.pkl`` or
``TPI_vars.pkl``.

Every ratio is to Ethiopia's rebased GDP (base year FY2024/25, Ministry of
National Strategy, November 2026) at market prices. OG-Core's output Y is
GDP before consumption taxes, so model moments are divided by Y plus the
consumption tax (see ``ogeth.macro_params.MODEL_PER_MARKET_GDP``).

Two sets of targets:

* ``TRANSITION_TARGETS`` are moments of the first model period, FY2024/25,
  the year the data describe. The dials tuned to them are the tax rates and
  scales (revenue by instrument), the pension and payroll coverage scales
  and the foreign-capital fill share zeta_K (first-year foreign share of
  private capital). Ethiopia's population is far from stationary, so these
  are scored where the data are, not in the steady state. First-year
  investment is scored, not tuned (see ``SS_TARGETS``).
* ``SS_TARGETS`` are long-run anchors and the survey distributions that the
  steady state is matched to (hours by age and wealth shares are scored by
  ``examples/tune.py``), plus moments reported as outcomes. The level of
  the discount factor is set so that the long-run return equals the world
  rate: a small open economy's domestic return converges to the return
  foreign investors earn as capital controls and financial repression are
  unwound (IMF CR 26/174). Tuning patience to first-year investment
  instead would make households finance all of the 35.3 percent of GDP the
  rebased accounts record, including state-enterprise investment financed
  through state banks; that requires a long-run return of about 2 percent,
  below the world rate, with foreign-owned capital leaving. The measurable
  state channels (state-enterprise retained earnings about 0.7 percent of
  GDP; repression beyond the below-market government borrowing rate the
  model already carries about 0.2 percent) close less than a point of the
  gap, so first-year investment is left as a scored miss (macro.md).
"""

import os

import numpy as np
import pandas as pd

from ogeth import macro_params as mp

# moment: (target, dial, source)
TRANSITION_TARGETS = {
    "PIT/Y": (
        mp.PIT_REVENUE_SHARE,
        "compliance scale",
        "personal income tax 194.3 bn birr, FY2024/25 outturn (NBE Annual "
        "Report 2024/25, Table 6.3)",
    ),
    "CIT/Y": (
        mp.CIT_REVENUE_SHARE,
        "adjustment_factor_for_cit_receipts",
        "business profit tax 229.6 + other income taxes 52.9 + government "
        "investment income 20.0 bn birr (NBE AR 2024/25)",
    ),
    "cons tax/Y": (
        mp.CONS_TAX_SHARE,
        "tau_c",
        "indirect taxes 650.9 + charges and fees 39.4 + sales of goods and "
        "services 32.8 bn birr (NBE AR 2024/25)",
    ),
    "wealth tax/Y": (
        mp.WEALTH_TAX_SHARE,
        "p_wealth",
        "rural and urban land-use fees 44.1 bn birr (NBE AR 2024/25)",
    ),
    "payroll/Y": (
        mp.PAYROLL_REVENUE_SHARE,
        "payroll coverage scale",
        "pension contributions about 69 bn birr (POESSA ~29, PSSSA ~40, "
        "estimates)",
    ),
    "pensions/Y": (
        mp.PENSIONS_SHARE_OF_GDP / 100,
        "pension coverage scale",
        "old-age pensions: PSSSA and POESSA benefits (ILO Social Security "
        "Inquiry, 2024), 60 percent old-age, rebased",
    ),
    "(I+I_g)/Y": (
        mp.GROSS_CAPITAL_FORMATION_REBASED / 100,
        "check",
        "gross capital formation 35.3 percent of rebased GDP (rebased "
        "national accounts, FY2024/25); scored, not tuned: the household-"
        "financed share is not published",
    ),
    "K_f/K": (
        mp.FOREIGN_CAPITAL_SHARE,
        "zeta_K",
        f"inward FDI stock {mp.FDI_STOCK_TO_GDP:.3f} of GDP (UNCTAD WIR "
        f"2026) over private K/Y {mp.PRIVATE_CAPITAL_OUTPUT_RATIO:.2f} "
        "(PWT 11.0 rebuilt on the rebased accounts, less IMF ICSD public "
        "capital)",
    ),
    "B/Y": (
        mp.INITIAL_WEALTH_RATIO / mp.MODEL_PER_MARKET_GDP,
        "initial_wealth_ratio (imposed)",
        "private K/Y less FDI stock plus domestic debt",
    ),
    "D/Y": (
        mp.IMF_PUBLIC_DEBT[0] / 100,
        "initial_debt_ratio + alpha_D_ext (imposed)",
        "general-government debt 43.3 percent of old-base GDP (IMF WEO "
        "October 2026), rebased; market (domestic) plus external debt",
    ),
    # scored, not tuned
    "C/Y purch.": (
        mp.PRIVATE_CONSUMPTION_REBASED / 100,
        "check",
        "private consumption 64.4 percent of rebased GDP (MoNS rebasing "
        "report, Table 4), of which non-profit institutions serving "
        "households 2.8",
    ),
    "G/Y": (
        mp.GOVERNMENT_CONSUMPTION_SHARE,
        "check",
        "government final consumption 1,136.8 bn birr over rebased GDP "
        "(MoNS rebasing report, Table 4); model G plus depreciation of "
        "recorded public capital, as in the national accounts",
    ),
    "NX/Y": (
        mp.NET_EXPORTS_SHARE,
        "check",
        "resource balance -1,065.6 bn birr over rebased GDP (MoNS rebasing "
        "report, Table 4); the IMF goods and services balance (CR 26/174 "
        "Table 4a) gives -4.5",
    ),
    "RM/Y": (
        mp.IMF_PRIVATE_TRANSFERS[0] / 100,
        "alpha_RM (imposed)",
        "net private transfers USD 7.04 bn (IMF CR 26/174)",
    ),
    "factor payments/Y": (
        752.0 / 201420.0,
        "check",
        "net primary income paid abroad USD 0.75 bn (IMF CR 26/174 Table 4a); "
        "the model counts payments on foreign-owned capital and market debt "
        "at r_p and interest on external debt at r_D_ext",
    ),
    "total tax/Y": (
        (1172 + 39.4 + 32.8 + 20.0)
        / mp.NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"]
        + mp.PAYROLL_REVENUE_SHARE,
        "check",
        "sum of the mapped instruments",
    ),
}

# Long-run anchors and outcomes, scored on the steady state
SS_TARGETS = {
    "D/Y": (
        mp.INITIAL_DEBT_RATIO / mp.MODEL_PER_MARKET_GDP,
        "debt_ratio_ss (all market debt by then)",
    ),
    "r_gov": (mp.LONG_RUN_R_GOV, "r_gov_shift (long run)"),
    "r": (mp.WORLD_INT_RATE_ANNUAL, "discount factor level"),
    "K/Y": (mp.PRIVATE_CAPITAL_OUTPUT_RATIO, "outcome (provisional anchor)"),
}


# Consumption of older people relative to ages 30-49, the profile net family
# transfers (phi_FT) are sized to: Ethiopia's National Transfer Accounts,
# 2005, private consumption per person (data/nta_consumption_by_age.csv),
# simple means over single ages; the model moment is the steady state's
# consumption per person, population weighted over lifetime-income groups.
NTA_CONSUMPTION_BASE_AGES = (30, 50)
NTA_CONSUMPTION_OLD_AGE_BANDS = ((65, 70), (70, 80), (80, 86))


def nta_old_age_consumption():
    """
    NTA private consumption per person in each old-age band relative to ages
    30-49.

    Returns:
        dict: {(first age, last age + 1): ratio}
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "nta_consumption_by_age.csv",
    )
    df = pd.read_csv(path, comment="#").set_index("age")
    c = df["private_consumption"]
    lo, hi = NTA_CONSUMPTION_BASE_AGES
    base = c.loc[lo : hi - 1].mean()
    return {
        (a, b): float(c.loc[a : b - 1].mean() / base)
        for a, b in NTA_CONSUMPTION_OLD_AGE_BANDS
    }


def _cfc_g(k_g):
    """
    Consumption of fixed capital on public capital as the national accounts
    record it: depreciation on the stock at cost. The model's K_g is in
    efficiency units (netted by the PIMA efficiency gap), so the recorded
    stock is K_g over the efficiency.
    """
    return mp.DELTA_G_ANNUAL * k_g / (1 - mp.PUBLIC_INVESTMENT_EFFICIENCY_GAP)


def _market_y(y, cons_tax):
    return y + cons_tax


def tpi_moments(tpi, t=0):
    """
    Transition moments in period t as shares of GDP at market prices.

    Args:
        tpi (dict): a solved ``TPI_vars.pkl``
        t (int): model period, 0 for FY2024/25

    Returns:
        m (dict): moments keyed like ``TRANSITION_TARGETS``, plus extras
    """

    def at(key):
        if key not in tpi:  # an ogcore without external debt at its own rate
            return 0.0
        x = np.asarray(tpi[key])
        x = x.reshape(x.shape[0], -1).sum(axis=1) if x.ndim > 1 else x
        return float(x[t])

    Ym = at("Y")
    Y = _market_y(Ym, at("cons_tax_revenue"))
    r_p = at("r_p")
    nx = Ym - at("C") - at("I_total") - at("I_g") - at("G")
    return {
        "PIT/Y": at("iit_revenue") / Y,
        "CIT/Y": at("business_tax_revenue") / Y,
        "cons tax/Y": at("cons_tax_revenue") / Y,
        "wealth tax/Y": at("wealth_tax_revenue") / Y,
        "payroll/Y": at("payroll_tax_revenue") / Y,
        "pensions/Y": at("agg_pension_outlays") / Y,
        "(I+I_g)/Y": (at("I_total") + at("I_g")) / Y,
        "K_f/K": at("K_f") / at("K"),
        "B/Y": at("B") / Y,
        "D/Y": (at("D") + at("D_ext")) / Y,
        "C/Y purch.": (at("C") + at("cons_tax_revenue")) / Y,
        "G/Y": (at("G") + _cfc_g(at("K_g"))) / Y,
        "NX/Y": nx / Y,
        "RM/Y": at("RM") / Y,
        "factor payments/Y": (
            r_p * (at("K_f") + at("D_f")) + at("debt_service_ext")
        )
        / Y,
        "total tax/Y": at("total_tax_revenue") / Y,
        "D market/Y": at("D") / Y,
        "D_ext/Y": at("D_ext") / Y,
        "K/Y": at("K") / Y,
        "r": at("r"),
        "r_gov": at("r_gov"),
    }


def ss_moments(ss):
    """
    Steady-state moments as shares of GDP at market prices.

    Args:
        ss (dict): a solved ``SS_vars.pkl``

    Returns:
        m (dict): moments
    """

    def sc(key):
        return float(np.sum(ss[key])) if key in ss else 0.0

    Ym = float(np.squeeze(ss["Y"]))
    Y = _market_y(Ym, sc("cons_tax_revenue"))
    nx = Ym - sc("C") - sc("I_total") - sc("I_g") - sc("G")
    return {
        "PIT/Y": sc("iit_revenue") / Y,
        "CIT/Y": sc("business_tax_revenue") / Y,
        "cons tax/Y": sc("cons_tax_revenue") / Y,
        "wealth tax/Y": sc("wealth_tax_revenue") / Y,
        "payroll/Y": sc("payroll_tax_revenue") / Y,
        "pensions/Y": sc("agg_pension_outlays") / Y,
        "total tax/Y": sc("total_tax_revenue") / Y,
        "D/Y": (sc("D") + sc("D_ext")) / Y,
        "D market/Y": sc("D") / Y,
        "D_f/D": sc("D_f") / sc("D") if sc("D") else np.nan,
        "G/Y": (sc("G") + _cfc_g(sc("K_g"))) / Y,
        "TR/Y": sc("TR") / Y,
        "I_g/Y": sc("I_g") / Y,
        "(I+I_g)/Y": (sc("I_total") + sc("I_g")) / Y,
        "K/Y": sc("K") / Y,
        "K_f/K": sc("K_f") / sc("K"),
        "B/Y": sc("B") / Y,
        "C/Y purch.": (sc("C") + sc("cons_tax_revenue")) / Y,
        "NX/Y": nx / Y,
        "RM/Y": sc("RM") / Y,
        "r": sc("r"),
        "r_gov": sc("r_gov"),
        "factor": sc("factor"),
    }
