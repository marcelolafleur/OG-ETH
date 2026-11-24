"""
Data targets for the OG-ETH calibration, and the model moments computed
from a solved ``SS_vars.pkl`` or ``TPI_vars.pkl``. Ratios are to rebased GDP
(FY2024/25) at market prices: model Y plus the consumption tax (macro.md).
"""

import os

import numpy as np
import pandas as pd

from ogeth import macro_params as mp

# First-period (FY2024/25) moments; moment: (target, dial, source)
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
        f"inward FDI stock {mp.FDI_STOCK_TO_GDP:.3f} of GDP (perpetual "
        "inventory of IMF balance-of-payments equity inflows) over private "
        f"K/Y {mp.PRIVATE_CAPITAL_OUTPUT_RATIO:.2f} "
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
        752.0 / (mp.REBASED_GDP_USD_BN * 1000),
        "check",
        "net primary income paid abroad USD 0.75 bn (IMF CR 26/174 Table 4a); "
        "the model counts payments on foreign-owned capital and market debt "
        "at r_p and interest on external debt at r_D_ext",
    ),
    "total tax/Y": (
        mp.PIT_REVENUE_SHARE
        + mp.CIT_REVENUE_SHARE
        + mp.CONS_TAX_SHARE
        + mp.WEALTH_TAX_SHARE
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


# Old-age consumption relative to ages 30-49, the target for family
# transfers (phi_FT): NTA 2005 private consumption per person, simple means
# over single ages (households.md, Net family transfers)
NTA_CONSUMPTION_BASE_AGES = (30, 50)
NTA_CONSUMPTION_OLD_AGE_BANDS = ((65, 70), (70, 80), (80, 86))


def nta_old_age_consumption():
    """
    NTA consumption per person in each old-age band relative to ages 30-49.

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
    Depreciation of public capital at cost (K_g over PIMA efficiency).
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
        return float(np.sum(ss[key]))

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
