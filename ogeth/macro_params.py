"""
Macroeconomic and fiscal parameters for OG-ETH, from the national accounts,
fiscal tables and other sources documented in the calibration chapters.
To look up the latest World Bank values when updating the calibration, see
scripts/world_bank_values.py.
"""

# imports
import numpy as np

# Solver settings for the example scripts: outer-loop damping (nu) and
# Anderson acceleration of the time-path loop.
NU = 0.2
TPI_ANDERSON_M = 5
TPI_ANDERSON_BETA = 1.0

# Frisch elasticity of labour supply; range 0.15-0.4 (Elminejad et al.
# 2023, Review of Economic Dynamics 51; households.md).
FRISCH = 0.25

# Coefficient of relative risk aversion (1/EIS above subsistence);
# OG-Core's default (households.md).
SIGMA = 1.5
# Subsistence floor (c_min), consumption per adult in model units,
# growth-adjusted. 80 percent of the cost of an energy-sufficient diet
# (World Bank/FAO Food Prices for Nutrition, 2021 PPP USD 1.17 a day),
# so the poorest group clears it at every age (households.md).
C_MIN = 0.1088
# Units of effective labor, a normalization: scales the earnings matrix e
# (mean one) so chi_n stays inside OG-Core's bounds; neutral for every
# other equilibrium condition (earnings.md).
EARNINGS_SCALE = 10.0

# ---------------------------------------------------------------------------
# Remittances: a share of GDP (alpha_RM_1, alpha_RM_T) allocated by eta_RM.
# g_RM and eta_RM derive from the demographics and are rebuilt with them
# (update_baseline.py; macro.md, Remittances).
# ---------------------------------------------------------------------------

# Share of total remittance value received by each income quintile,
# poorest to richest: World Bank / Bendixen & Amandi, "Remittances to
# Ethiopia" (2010 survey; 9, 47 and 37 percent of recipients in the
# bottom-two, middle-two and top income brackets; macro.md, Remittances).
RM_QUINTILE_VALUE_SHARES = [
    9 / 93 / 2,
    9 / 93 / 2,
    47 / 93 / 2,
    47 / 93 / 2,
    37 / 93,
]


def flat_share_g_RM(g_y, g_n):
    """
    Remittance growth path that keeps remittances a constant share of GDP.

    Args:
        g_y (scalar): model-period productivity growth rate
        g_n (array_like): population growth path, length T + S

    Returns:
        g_RM (Numpy array): length T + S; get_RM's growth factor is one
    """
    g_n = np.asarray(g_n, dtype=float)
    g_RM = np.exp(g_y) * (1.0 + np.roll(g_n, 1)) - 1.0
    # get_RM never reads g_RM[0]; keep it consistent with period 0 anyway
    g_RM[0] = np.exp(g_y) * (1.0 + g_n[0]) - 1.0
    return g_RM


def remittance_level_params():
    """
    Remittance share of GDP, FY2024/25 (IMF CR 26/174, Table 4a), held.

    Returns:
        dict: ``{"alpha_RM_1", "alpha_RM_T"}``
    """
    share = round(to_model(float(IMF_PRIVATE_TRANSFERS[0]) / 100), 6)
    return {"alpha_RM_1": share, "alpha_RM_T": share}


def remittance_eta(omega_SS, lambdas, quintile_value_shares=None):
    """
    Allocation matrix distributing aggregate remittances across households.

    Args:
        omega_SS (array_like): steady-state population, (S, J)
        lambdas (array_like): group population shares, poorest first
        quintile_value_shares (array_like): defaults to
            RM_QUINTILE_VALUE_SHARES

    Returns:
        eta_RM (Numpy array): (S, J), sums to one; per capita within groups
    """
    omega_SS = np.asarray(omega_SS, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float).flatten()
    if quintile_value_shares is None:
        quintile_value_shares = RM_QUINTILE_VALUE_SHARES
    q = np.asarray(quintile_value_shares, dtype=float)
    q = q / q.sum()
    # cumulative remittance value at each population percentile, evaluated
    # at the lambdas group boundaries
    cum_pop = np.concatenate([[0.0], np.cumsum(np.full(q.size, 1 / q.size))])
    cum_val = np.concatenate([[0.0], np.cumsum(q)])
    bounds = np.concatenate([[0.0], np.cumsum(lambdas)])
    group_share = np.diff(np.interp(bounds, cum_pop, cum_val))
    group_share = group_share / group_share.sum()
    within = omega_SS / omega_SS.sum(axis=0, keepdims=True)
    return within * group_share.reshape(1, -1)


def derive_remittance_params(g_y, g_n, omega_SS, lambdas, omega):
    """
    The remittance parameters that depend on a given set of demographics.

    Args:
        g_y (scalar): model-period productivity growth rate
        g_n (array_like): population growth path, length T + S
        omega_SS (array_like): steady-state population distribution, (S, J)
        lambdas (array_like): population shares of the J groups
        omega (array_like): population by year, (T+S, S, J); eta_RM is a
            path built from each year's population (allocation_path)

    Returns:
        dict: g_RM and eta_RM, as lists
    """
    g_RM = flat_share_g_RM(g_y, g_n)
    eta_RM = allocation_path(
        lambda om: remittance_eta(om, lambdas), omega, omega_SS
    )
    return {"g_RM": g_RM.tolist(), "eta_RM": eta_RM.tolist()}


def derived_remittance_params(p):
    """
    The remittance parameters that depend on the demographics in ``p``.

    Args:
        p (Specifications): model parameters

    Returns:
        dict: see ``derive_remittance_params``
    """
    g_n = np.asarray(p.g_n, dtype=float)[: p.T + p.S]
    return derive_remittance_params(
        p.g_y, g_n, p.omega_SS, p.lambdas, omega=p.omega
    )


# ---------------------------------------------------------------------------
# GDP basis: the rebased national accounts (base FY2024/25; Ministry of
# National Strategy, "High Level Report: 2017 EFY GDP Rebasing", Tables 2
# and 5; ogeth/data/mons_rebased_gdp_fy2024_25.csv). Nominal GDP, billions
# of birr. Old base: IMF CR 26/174, Table 1. Every ratio to GDP is to the
# rebased series. Published fiscal and BoP ratios convert by old over new
# GDP (rebase_factors); national-accounts components come from the rebased
# accounts directly (macro.md, GDP basis).
# ---------------------------------------------------------------------------
NOMINAL_GDP_OLD_BASE_BIRR_BN = {
    "FY2023/24": 11752.0,
    "FY2024/25": 15094.0,
    "FY2025/26": 19268.0,
}
NOMINAL_GDP_REBASED_BIRR_BN = {"FY2024/25": 24031.07, "FY2025/26": 31934.0}

# Fiscal years used: FY2024/25 outturn (model period 0) and FY2025/26
# budget (period 1), the latter held for the long run.
FISCAL_YEARS = ["FY2024/25", "FY2025/26"]
# Debt-closure rule starts in model period 7 (FY2031/32); fiscal paths run
# to it plus the long run.
TG1 = 7
FISCAL_PATH_LENGTH = TG1 + 1


def rebase_factors(n=len(FISCAL_YEARS)):
    """
    Old-base over rebased nominal GDP for each fiscal year.

    Args:
        n (int): number of years from FY2024/25

    Returns:
        Numpy array: length n; years past FY2025/26 hold its factor
    """
    known = [
        NOMINAL_GDP_OLD_BASE_BIRR_BN[y] / NOMINAL_GDP_REBASED_BIRR_BN[y]
        for y in ("FY2024/25", "FY2025/26")
    ]
    return np.array(known + [known[-1]] * (n - len(known)))[:n]


def rebased(series):
    """
    Convert old-base percent of GDP (from FY2024/25) to rebased GDP.

    Args:
        series (array_like): published values

    Returns:
        list: rebased values, rounded to 4 decimals
    """
    x = np.asarray(series, dtype=float)
    return [round(float(v), 4) for v in x * rebase_factors(x.size)]


# Published, percent of old-base GDP, general government (IMF CR 26/174,
# Tables 1, 2b and 4a): FY2024/25 outturn, FY2025/26 budget. Public debt:
# WEO October 2026, GGXWDG_NGDP. The wide debt (program perimeter) adds NBE
# and Ethio Telecom external debt the budget does not service.
IMF_PUBLIC_DEBT_PUBLISHED = [43.3, 40.7]
IMF_PUBLIC_DEBT_WIDE_PUBLISHED = [50.5, 45.3]
IMF_GRANTS_PUBLISHED = [1.7, 1.3]
IMF_RECURRENT_PUBLISHED = [7.1, 8.2]
IMF_INTEREST_PUBLISHED = [0.8, 1.3]
IMF_CAPITAL_PUBLISHED = [5.0, 5.8]
IMF_PRIVATE_TRANSFERS_PUBLISHED = [5.6, 6.0]
IMF_DOMESTIC_DEBT_PUBLISHED = [18.7, 15.4]

# The same series in percent of rebased GDP: what the calibration uses
IMF_PUBLIC_DEBT = rebased(IMF_PUBLIC_DEBT_PUBLISHED)
IMF_GRANTS = rebased(IMF_GRANTS_PUBLISHED)
IMF_RECURRENT = rebased(IMF_RECURRENT_PUBLISHED)
IMF_INTEREST = rebased(IMF_INTEREST_PUBLISHED)
IMF_CAPITAL = rebased(IMF_CAPITAL_PUBLISHED)
IMF_PRIVATE_TRANSFERS = rebased(IMF_PRIVATE_TRANSFERS_PUBLISHED)
IMF_DOMESTIC_DEBT = rebased(IMF_DOMESTIC_DEBT_PUBLISHED)
# Foreign-held share of general-government debt, FY2024/25: external debt
# (31.8, wide perimeter) less the part outside general government, / 43.3
GG_FOREIGN_DEBT_SHARE = round(
    (31.8 - (IMF_PUBLIC_DEBT_WIDE_PUBLISHED[0] - IMF_PUBLIC_DEBT_PUBLISHED[0]))
    / IMF_PUBLIC_DEBT_PUBLISHED[0],
    4,
)

# Government debt contracted abroad (alpha_D_ext, r_D_ext_annual): all
# general-government external debt, mostly concessional (IMF CR 26/174,
# DSA; macro.md, Debt contracted abroad).
# Public DSA years (IMF CR 26/174, Table 2), fiscal years ending June.
DSA_YEARS = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2036, 2045]
# Public external debt, percent of old-base GDP; after 2045 it falls to
# zero at its 2036-2045 pace (external_debt_path).
DSA_EXTERNAL_DEBT = [31.8, 29.9, 27.2, 24.7, 21.5, 18.8, 16.8, 9.2, 4.1]
# Average nominal interest rate on external debt, percent, USD terms;
# concessionality enters as this low rate, not as a grant.
DSA_EXTERNAL_RATE = [0.7, 1.8, 2.6, 2.1, 1.9, 1.8, 1.6, 1.3, 0.9]
# US inflation (Fed target), deflates the USD rate; real exchange rate held.
US_INFLATION = 0.02
# Average real interest rate on domestic (market) debt, percent; negative
# under financial repression (IMF CR 26/174, DSA).
DSA_DOMESTIC_REAL_RATE = [-11.0, -9.8, -5.1, -0.6, 1.0, 2.6, 3.3, 4.6, 6.6]
# DSA real rates on new domestic borrowing, percent: medium/long-term and
# short-term (borrowing assumptions table).
DSA_MARGINAL_DOMESTIC_REAL_RATES = (4.9, 4.5)

# Rebased expenditure-side national accounts, FY2024/25 (rebasing report,
# Table 4): percent of GDP or billions of birr. The model's G excludes
# public capital's consumption of fixed capital; targets.py adds
# delta_g K_g to the moment.
PRIVATE_CONSUMPTION_REBASED = 64.4
GROSS_CAPITAL_FORMATION_REBASED = 35.3
GOVERNMENT_CONSUMPTION_BIRR_BN = 1136.844
GOVERNMENT_CONSUMPTION_SHARE = round(
    GOVERNMENT_CONSUMPTION_BIRR_BN / NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"],
    5,
)
RESOURCE_BALANCE_BIRR_BN = -1065.627
NET_EXPORTS_SHARE = round(
    RESOURCE_BALANCE_BIRR_BN / NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"], 5
)

# Price basis: OG-Core's Y excludes the consumption tax; market-price GDP
# is Y plus it. Shares OG-Core applies to Y are data shares times
# MODEL_PER_MARKET_GDP (macro.md, GDP basis).
# General-government revenue outturn, FY2024/25, billions of birr (NBE
# Annual Report 2024/25, Table 6.3). Mapping to model instruments
# (taxes.md):
#   personal income tax                         -> income tax (compliance)
#   business profit tax, other income taxes
#     (rental, withholding, interest, capital
#     gains) and government investment income
#     (NBE residual surplus, SOE dividends)     -> CIT collections factor
#   indirect taxes, charges and fees, sales of
#     goods and services                        -> consumption tax tau_c
#   rural and urban land-use fees               -> flat wealth tax
#   "other" non-tax (extraordinary items and
#     pension contributions) and property sales -> not mapped: contributions
#     are the payroll tax, the rest is one-off
REVENUE_FY2425_BIRR_BN = {
    "personal income tax": 194.3,
    "business profit tax": 229.6,
    "other income taxes": 52.9,
    "land-use fees": 44.1,
    "indirect taxes": 650.9,
    "charges and fees": 39.4,
    "government investment income": 20.0,
    "sales of goods and services": 32.8,
    "reimbursements and property sales": 4.7,
    "other non-tax incl. pension contributions": 117.8,
    "grants": 185.0,
}


def revenue_share(*lines, year="FY2024/25"):
    """
    Revenue lines as a share of rebased GDP at market prices.

    Args:
        *lines (str): keys of REVENUE_FY2425_BIRR_BN to add up
        year (str): fiscal year of the GDP denominator

    Returns:
        float: share of GDP
    """
    total = sum(REVENUE_FY2425_BIRR_BN[k] for k in lines)
    return round(total / NOMINAL_GDP_REBASED_BIRR_BN[year], 5)


PIT_REVENUE_SHARE = revenue_share("personal income tax")
CIT_REVENUE_SHARE = revenue_share(
    "business profit tax", "other income taxes", "government investment income"
)
CONS_TAX_SHARE = revenue_share(
    "indirect taxes", "charges and fees", "sales of goods and services"
)
WEALTH_TAX_SHARE = revenue_share("land-use fees")
MODEL_PER_MARKET_GDP = 1 / (1 - CONS_TAX_SHARE)


def to_model(share):
    """
    Convert a share of GDP at market prices into the share of OG-Core's Y.

    Args:
        share (scalar or array_like): share of market-price GDP

    Returns:
        same type: share of the model's Y
    """
    if np.ndim(share):
        return [float(x) * MODEL_PER_MARKET_GDP for x in share]
    return float(share) * MODEL_PER_MARKET_GDP


# Initial general-government debt, FY2024/25, model units
INITIAL_DEBT_RATIO = round(to_model(IMF_PUBLIC_DEBT[0] / 100), 4)
# Its split, model units: external debt (D_ext) and market (domestic) debt
# (D). Non-residents hold no birr securities, so the foreign share of
# market debt is zero (IMF CR 26/174).
EXTERNAL_DEBT_RATIO = round(INITIAL_DEBT_RATIO * GG_FOREIGN_DEBT_SHARE, 4)
INITIAL_MARKET_DEBT_RATIO = round(INITIAL_DEBT_RATIO - EXTERNAL_DEBT_RATIO, 4)
MARKET_FOREIGN_DEBT_SHARE = 0.0

# Cash transfers to households, percent of old-base GDP: subsidies, PSNP
# and pensions (0.5 assumed) inside IMF recurrent spending (IMF CR 26/174,
# paras. 9, 20-22 and safety-net box).
CASH_TRANSFERS_PUBLISHED = [2.0, 1.8]
# Public pension benefits replace the 0.5 assumption: PSSSA paid 22.2
# billion birr to 750,812 recipients; POESSA's 43,347 at PSSSA's average
# (ILO Social Security Inquiry, 2024 data); over FY2023/24 old-base GDP.
# Old-age share about 60 percent; survivors' and invalidity pensions are
# the rest (taxes.md, Public pensions).
PENSION_BENEFITS_BIRR_BN = 22.2 + 43_347 * 22.2 / 750_812
PENSION_BENEFITS_PUBLISHED = round(
    100 * PENSION_BENEFITS_BIRR_BN / NOMINAL_GDP_OLD_BASE_BIRR_BN["FY2023/24"],
    3,
)
OLD_AGE_PENSION_SHARE = 0.60
PUBLISHED_PENSION_ASSUMPTION = 0.5
CASH_TRANSFERS = rebased(
    [
        x - PUBLISHED_PENSION_ASSUMPTION + PENSION_BENEFITS_PUBLISHED
        for x in CASH_TRANSFERS_PUBLISHED
    ]
)

# Real return on foreign direct investment (world_int_rate_annual): pooled
# FDI income over FDI liabilities for 14 African peers, 2015-2024, 6.1
# percent nominal USD, less US inflation (IMF BOP and IIP;
# ogeth/data/imf_fdi_income_africa_peers.csv; macro.md, World interest rate).
WORLD_INT_RATE_ANNUAL = 0.04

# Long-run real rate on market debt: mean of the DSA's rates on new
# domestic borrowing, 4.7 percent (macro.md, Interest rates).
LONG_RUN_R_GOV = round(sum(DSA_MARGINAL_DOMESTIC_REAL_RATES) / 200, 4)
# Floor on r_gov (r_gov_floor), below the -11.0 percent FY2024/25 rate.
R_GOV_FLOOR = -0.12
# Steady-state return at which r_gov_shift is evaluated: the solved
# baseline of examples/run_og_eth.py.
R_SS_FOR_R_GOV = 0.039962

# Informal share of employment (ILOSTAT SDG 8.3.1, LFMS 2021). The formal
# share is the richest lifetime-income groups (taxes.md, Informality).
INFORMAL_EMPLOYMENT_SHARE = 0.7765
LAMBDAS = [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]


def formal_share_by_group(informal_share, lambdas=LAMBDAS):
    """
    Formal share of each group, with formality at the top of the ranking.

    Args:
        informal_share (scalar): informal share of the employed
        lambdas (array_like): group population shares, poorest first

    Returns:
        list: formal share of each group, between 0 and 1
    """
    lam = np.asarray(lambdas, dtype=float)
    top = np.cumsum(lam[::-1])[::-1]  # share at or above each group's floor
    formal = 1.0 - float(informal_share)
    above = top - lam  # share strictly above each group
    share = np.clip((formal - above) / lam, 0.0, 1.0)
    return [round(float(x), 4) for x in share]


# Non-compliance by group, held: one minus the formal share.
NONCOMPLIANCE = [
    round(1 - x, 4) for x in formal_share_by_group(INFORMAL_EMPLOYMENT_SHARE)
]
# Scale on compliance so PIT collects PIT_REVENUE_SHARE of GDP in
# FY2024/25 (examples/tune.py; taxes.md).
COMPLIANCE_SCALE = 0.056434

# Pension coverage by group: PSSSA and POESSA cover formal employment only.
PENSION_COVERAGE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
# Defined-benefit scheme (Proclamations 1267/2022 and 1268/2022): pension
# at 60 after 10 years, 30 percent of the last 3 years' salary plus 1.25
# percent per year beyond 10, capped at 70. alpha_db is the replacement
# rate per year of contribution.
LEGAL_RETIREMENT_AGE = 60
# OG-Core counts years of life, so legal age 60 is entered as 61.
PENSION_RETIREMENT_AGE = LEGAL_RETIREMENT_AGE + 1
PENSION_MIN_YEARS = 10
PENSION_BASE_REPLACEMENT = 0.30
PENSION_REPLACEMENT_PER_YEAR = 0.0125
PENSION_MAX_REPLACEMENT = 0.70
PENSION_AVERAGING_YEARS = 3
PENSION_CAREER_YEARS = 40
# Scale on coverage so pension outlays match PENSIONS_SHARE_OF_GDP
# (examples/tune.py).
PENSION_COVERAGE_SCALE = 0.097098
# Contributions: 7 percent employee plus 11 employer (Proclamations
# 1267/1268 of 2022), on the covered groups times a scale tuned to about
# 69 billion birr in FY2024/25 (POESSA ~29, Reporter 2025; PSSSA ~40,
# Addis Fortune). Least certain fiscal anchor (taxes.md, Public pensions).
PAYROLL_STATUTORY_RATE = 0.18
PAYROLL_REVENUE_SHARE = round(
    69.0 / NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"], 5
)
PAYROLL_COVERAGE_SCALE = 0.097527

# PIT Schedule A, birr per month, (upper bound, marginal rate): Proclamation
# 979/2016 in FY2024/25, Proclamation 1395/2025 from FY2025/26.
PIT_SCHEDULE_2016 = [
    (600, 0.0),
    (1650, 0.10),
    (3200, 0.15),
    (5250, 0.20),
    (7800, 0.25),
    (10900, 0.30),
    (None, 0.35),
]
PIT_SCHEDULE_2025 = [
    (2000, 0.0),
    (4000, 0.15),
    (7000, 0.20),
    (10000, 0.25),
    (14000, 0.30),
    (None, 0.35),
]
# Schedule D dividend rate, applied to all capital income.
CAPITAL_INCOME_TAX_RATE = 0.10
# Model birr are FY2024/25 birr. Population (UN, as the demographics) and
# adult share, 2024, give GDP per adult; the 2025 schedule is deflated by
# nominal GDP growth per head (taxes.md, Personal income taxes).
POPULATION_2024 = 132.06e6
ADULT_SHARE_2024 = (
    0.4986  # aged 20 and over: 1 - 0.3906 (0-14) - 0.1108 (15-19)
)
POPULATION_GROWTH = 0.026
# Income grid for the Gouveia-Strauss fit, birr per year
TAX_FIT_INCOME_RANGE = (12_000.0, 5_000_000.0)
# Large third GS parameter (within OG-Core's cap) for a flat rate
GS_FLAT_PHI2 = 2.0e4


def _held(values):
    """The FY2024/25 and FY2025/26 values, the second held to the long run."""
    values = list(values)
    return values + [values[-1]] * (FISCAL_PATH_LENGTH - len(values))


def spending_paths():
    """
    Spending ratios, FY2024/25 outturn and FY2025/26 budget, held (macro.md).

    Returns:
        dict: alpha_G, alpha_T, alpha_I, alpha_FA, length FISCAL_PATH_LENGTH
    """
    rec = np.array(IMF_RECURRENT) / 100
    interest = np.array(IMF_INTEREST) / 100
    tr = np.array(CASH_TRANSFERS) / 100
    g = rec - interest - tr
    # old-age pensions are paid by the pension system, not as transfers;
    # survivor and invalidity benefits stay in the cash transfers
    tr = tr - PENSIONS_SHARE_OF_GDP / 100
    ig = np.array(IMF_CAPITAL) / 100
    fa = np.array(IMF_GRANTS) / 100
    # shares of market-price GDP -> shares of the model's Y
    return {
        "alpha_G": to_model(_held(g)),
        "alpha_T": to_model(_held(tr)),
        "alpha_I": to_model(_held(ig)),
        "alpha_FA": to_model(_held(fa)),
    }


def revenue_paths(tau_c_0, cit_factor_0):
    """
    Consumption-tax rate and CIT collections factor, held at FY2024/25.

    Args:
        tau_c_0 (scalar): FY2024/25 effective consumption-tax rate
        cit_factor_0 (scalar): FY2024/25 CIT collections factor

    Returns:
        dict: tau_c and adjustment_factor_for_cit_receipts paths
    """
    n = FISCAL_PATH_LENGTH
    return {
        "tau_c": [[float(tau_c_0)]] * n,
        "adjustment_factor_for_cit_receipts": [float(cit_factor_0)] * n,
    }


def compliance_paths(compliance_scale=None, pension_coverage_scale=None):
    """
    Income-tax non-compliance paths (held) and pension coverage.

    Args:
        compliance_scale (scalar): defaults to COMPLIANCE_SCALE
        pension_coverage_scale (scalar): defaults to PENSION_COVERAGE_SCALE

    Returns:
        dict: labor/capital non-compliance paths, replacement_rate_adjust
    """
    if compliance_scale is None:
        compliance_scale = COMPLIANCE_SCALE
    if pension_coverage_scale is None:
        pension_coverage_scale = PENSION_COVERAGE_SCALE
    rate = list(
        np.round(1 - compliance_scale * (1 - np.array(NONCOMPLIANCE)), 6)
    )
    coverage = [float(c * pension_coverage_scale) for c in PENSION_COVERAGE]
    return {
        "labor_income_tax_noncompliance_rate": [
            list(rate) for _ in range(FISCAL_PATH_LENGTH)
        ],
        "capital_income_tax_noncompliance_rate": [
            list(rate) for _ in range(FISCAL_PATH_LENGTH)
        ],
        "replacement_rate_adjust": [coverage],
    }


def schedule_tax_rates(annual_income, schedule, scale=1.0):
    """
    Effective and marginal rates of a monthly schedule at annual incomes.

    Args:
        annual_income (array_like): income in birr per year
        schedule (list): (monthly upper bound or None, marginal rate) pairs
        scale (scalar): multiplies the bracket thresholds

    Returns:
        etr (Numpy array): tax paid over income
        mtr (Numpy array): marginal rate of the bracket the income falls in
    """
    annual_income = np.asarray(annual_income, dtype=float)
    monthly = annual_income / 12.0
    tax = np.zeros_like(monthly)
    mtr = np.zeros_like(monthly)
    lower = 0.0
    for upper, rate in schedule:
        hi = np.inf if upper is None else upper * scale
        tax += rate * np.clip(monthly - lower, 0.0, hi - lower)
        mtr = np.where((monthly > lower) & (monthly <= hi), rate, mtr)
        lower = hi
    with np.errstate(divide="ignore", invalid="ignore"):
        etr = np.where(annual_income > 0, 12.0 * tax / annual_income, 0.0)
    return etr, mtr


def gs_rates(income, phi, rate_type):
    """
    OG-Core's Gouveia-Strauss tax function.

    Args:
        income (array_like): income in birr per year
        phi (array_like): the three parameters
        rate_type (str): "etr" or "mtr"

    Returns:
        rates (Numpy array): effective or marginal rates
    """
    x = np.asarray(income, dtype=float)
    phi0, phi1, phi2 = phi
    if rate_type == "etr":
        return phi0 * (x - (x**-phi1 + phi2) ** (-1 / phi1)) / x
    return phi0 * (
        1 - x ** (-phi1 - 1) * (x**-phi1 + phi2) ** ((-1 - phi1) / phi1)
    )


def fit_gs_parameters(schedule, scale=1.0):
    """
    Least-squares fit of the Gouveia-Strauss form to a bracket schedule.

    Args:
        schedule (list): (monthly upper bound or None, marginal rate) pairs
        scale (scalar): multiplies the bracket thresholds

    Returns:
        phi (list): the three parameters, floats
    """
    from scipy import optimize

    incomes = np.logspace(
        np.log10(TAX_FIT_INCOME_RANGE[0]),
        np.log10(TAX_FIT_INCOME_RANGE[1]),
        400,
    )
    etr, mtr = schedule_tax_rates(incomes, schedule, scale)
    top_rate = schedule[-1][1]

    def loss(theta):
        if np.any(np.asarray(theta) <= 0):
            return 1e6
        return np.mean((gs_rates(incomes, theta, "etr") - etr) ** 2) + np.mean(
            (gs_rates(incomes, theta, "mtr") - mtr) ** 2
        )

    best = None
    for phi1 in (0.5, 1.0, 2.0, 4.0):
        for phi2 in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
            res = optimize.minimize(
                loss,
                [top_rate, phi1, phi2],
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000},
            )
            if best is None or res.fun < best.fun:
                best = res
    return [float(v) for v in best.x]


def flat_gs_parameters(rate):
    """
    Gouveia-Strauss parameters that reproduce a flat rate.

    Args:
        rate (scalar): the flat tax rate

    Returns:
        phi (list): the three parameters
    """
    return [float(rate), 1.0, GS_FLAT_PHI2]


def mean_income_per_adult():
    """
    Rebased GDP per adult, FY2024/25 birr.

    Returns:
        float: birr per year
    """
    adults = POPULATION_2024 * (1 + POPULATION_GROWTH / 2) * ADULT_SHARE_2024
    return NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"] * 1e9 / adults


# Model household income (r_p b + w e n) over market-price GDP in the
# solved steady state (examples/tune.py); scales GDP per adult into
# mean_income_data.
MODEL_INCOME_SHARE_OF_GDP = 0.861073


def model_income_share(ss, p):
    """
    Mean household income over market-price GDP in a solved steady state.

    Args:
        ss (dict): steady-state output (SS_vars.pkl)
        p (Specifications): its parameters

    Returns:
        float: household income over market-price GDP
    """
    e = np.asarray(p.e)[-1, :, :]
    income = (
        (
            float(np.squeeze(ss["r_p"])) * np.asarray(ss["b_s"])
            + float(np.squeeze(ss["w"])) * e * np.asarray(ss["n"])
        )
        * np.asarray(p.omega_SS)
    ).sum()
    gdp = float(np.sum(ss["Y"])) + float(np.sum(ss["cons_tax_revenue"]))
    return float(income / gdp)


def mean_income_data(share=None):
    """
    OG-Core's mean_income_data: household income per adult, FY2024/25 birr.

    Args:
        share (float): defaults to MODEL_INCOME_SHARE_OF_GDP

    Returns:
        float: birr per year
    """
    if share is None:
        share = MODEL_INCOME_SHARE_OF_GDP
    return float(mean_income_per_adult() * share)


def statutory_tax_params():
    """
    Income-tax function parameters from the statutory schedules.

    Returns:
        dict: tax_func_type, etr_params, mtrx_params, mtry_params,
            mean_income_data
    """
    nominal_growth_per_head = (
        NOMINAL_GDP_REBASED_BIRR_BN["FY2025/26"]
        / NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"]
    ) / (1 + POPULATION_GROWTH)
    phi_2016 = fit_gs_parameters(PIT_SCHEDULE_2016)
    phi_2025 = fit_gs_parameters(
        PIT_SCHEDULE_2025, 1 / nominal_growth_per_head
    )
    labor = [[phi_2016], [phi_2025]]
    return {
        "tax_func_type": "GS",
        "etr_params": labor,
        "mtrx_params": [[list(p) for p in row] for row in labor],
        "mtry_params": [[flat_gs_parameters(CAPITAL_INCOME_TAX_RATE)]],
        "mean_income_data": mean_income_data(),
    }


def defined_benefit_params():
    """
    OG-Core defined-benefit pension parameters for Ethiopia's schemes.

    Returns:
        dict: pension_system, retirement_age, yr_contrib,
            avg_earn_num_years, alpha_db
    """
    replacement = min(
        PENSION_BASE_REPLACEMENT
        + PENSION_REPLACEMENT_PER_YEAR
        * (PENSION_CAREER_YEARS - PENSION_MIN_YEARS),
        PENSION_MAX_REPLACEMENT,
    )
    return {
        "pension_system": "Defined Benefits",
        "retirement_age": [PENSION_RETIREMENT_AGE],
        "yr_contrib": PENSION_CAREER_YEARS,
        "avg_earn_num_years": PENSION_AVERAGING_YEARS,
        # OG-Core >= 0.21 takes alpha_db by year and lifetime-income group
        "alpha_db": [[replacement / PENSION_CAREER_YEARS]],
    }


def payroll_tax_params(payroll_coverage_scale=None):
    """
    Effective payroll tax by lifetime-income group.

    Args:
        payroll_coverage_scale (scalar): defaults to PAYROLL_COVERAGE_SCALE

    Returns:
        dict: tau_payroll as a (1, J) nested list
    """
    if payroll_coverage_scale is None:
        payroll_coverage_scale = PAYROLL_COVERAGE_SCALE
    return {
        "tau_payroll": [
            [
                round(PAYROLL_STATUTORY_RATE * c * payroll_coverage_scale, 6)
                for c in PENSION_COVERAGE
            ]
        ]
    }


# Cash transfers by programme, percent of rebased GDP, allocated by eta
# (macro.md, Aggregate transfers). PSNP: the whole programme incl. the urban
# net, 0.4 percent of old-base GDP (World Bank PAD P172479, 2020).
PSNP_SHARE_OF_GDP = rebased([0.4])[0]
# PSNP direct support (12 months a year) vs public works (6), same monthly
# rate (PSNP IV PIM, 2014, s. 3.5); direct support proxied by age 60+.
PSNP_DIRECT_SUPPORT_AGE = 60
PSNP_DIRECT_SUPPORT_MONTHS = 12
PSNP_PUBLIC_WORKS_MONTHS = 6
# Old-age pensions (pension system) and survivor and invalidity benefits
# (cash transfers to the covered groups)
PENSIONS_SHARE_OF_GDP = rebased(
    [PENSION_BENEFITS_PUBLISHED * OLD_AGE_PENSION_SHARE]
)[0]
SURVIVOR_INVALIDITY_SHARE_OF_GDP = rebased(
    [PENSION_BENEFITS_PUBLISHED * (1 - OLD_AGE_PENSION_SHARE)]
)[0]
FERTILIZER_SUBSIDY_SHARE_OF_GDP = rebased([0.6])[0]
PSNP_GROUPS = [0]  # poorest quarter of households
FARM_GROUPS = [0, 1, 2]  # bottom 70 percent: smallholder agriculture
PENSION_GROUPS = [5, 6]  # the formal groups, as in PENSION_COVERAGE


def transfer_eta(omega_SS, lambdas, first_age=20):
    """
    Allocation matrix eta of aggregate transfers, by programme, (S, J).

    Args:
        omega_SS (array_like): population distribution, (S, J)
        lambdas (array_like): population shares of the J groups
        first_age (int): age of the first model age (OG-Core's E)

    Returns:
        Numpy array: (S, J), sums to one
    """
    omega_SS = np.asarray(omega_SS, dtype=float)
    S, J = omega_SS.shape
    eta = np.zeros((S, J))

    def spread(groups, ages, weight, per_person=None):
        # per capita within the cells, or in proportion to per_person (by
        # age) times the population
        pp = np.ones(S) if per_person is None else np.asarray(per_person)
        mask = np.zeros((S, J))
        mask[np.ix_(ages, groups)] = (omega_SS * pp[:, np.newaxis])[
            np.ix_(ages, groups)
        ]
        return weight * mask / mask.sum()

    # PSNP: direct support (from PSNP_DIRECT_SUPPORT_AGE) pays twice the
    # public works amount per person a year
    age = np.arange(S) + first_age
    psnp_per_person = np.where(
        age >= PSNP_DIRECT_SUPPORT_AGE,
        PSNP_DIRECT_SUPPORT_MONTHS / PSNP_PUBLIC_WORKS_MONTHS,
        1.0,
    )
    eta += spread(PSNP_GROUPS, range(S), PSNP_SHARE_OF_GDP, psnp_per_person)
    eta += spread(FARM_GROUPS, range(S), FERTILIZER_SUBSIDY_SHARE_OF_GDP)
    # survivor and invalidity benefits to the households of the covered
    # groups, at all ages (widows, orphans and invalidity pensioners)
    eta += spread(PENSION_GROUPS, range(S), SURVIVOR_INVALIDITY_SHARE_OF_GDP)
    return eta / eta.sum()


def allocation_path(allocation, omega, omega_SS, tol=1e-12):
    """
    Allocation matrix for each transition year, from that year's population.

    Args:
        allocation (callable): population (S, J) -> matrix summing to one
        omega (array_like): population by year, (T+S, S, J)
        omega_SS (array_like): steady-state population, (S, J)
        tol (scalar): trailing years within tol of steady state are dropped

    Returns:
        Numpy array: (n + 1, S, J), ending with the steady-state matrix
    """
    omega = np.asarray(omega, dtype=float)
    eta_SS = allocation(np.asarray(omega_SS, dtype=float))
    path = np.array([allocation(omega_t) for omega_t in omega])
    n = path.shape[0]
    while n > 0 and np.abs(path[n - 1] - eta_SS).max() < tol:
        n -= 1
    return np.concatenate([path[:n], eta_SS[np.newaxis]])


def transfer_eta_path(omega, omega_SS, lambdas, first_age=20):
    """
    ``transfer_eta`` for each transition year (``allocation_path``).

    Returns:
        Numpy array: (n + 1, S, J), each year summing to one
    """
    return allocation_path(
        lambda om: transfer_eta(om, lambdas, first_age),
        omega,
        omega_SS,
    )


def derived_transfer_eta(p):
    """The transfer allocation path for the demographics in ``p``."""
    return {
        "eta": transfer_eta_path(
            p.omega, p.omega_SS, p.lambdas, int(p.E)
        ).tolist()
    }


# Net family transfers (phi_FT): working-age members of each group support
# its elders, net of government transfers to them; size set so old-age
# consumption matches NTA 2005 (data/nta_consumption_by_age.csv). Steady-
# state size is a share of the model's Y, not market-price GDP
# (households.md, Net family transfers).
FAMILY_GIVER_AGES = (25, 65)
FAMILY_RECEIVER_AGE = 65
FAMILY_TRANSFER_SHARE_SS = 0.17


def family_transfers(omega, e, eta, alpha_T, total_share, first_age=20):
    """
    Net family transfer shares of GDP for one period.

    Args:
        omega (array_like): population distribution, (S, J)
        e (array_like): earnings matrix, (S, J)
        eta (array_like): transfer allocation matrix, (S, J)
        alpha_T (scalar): government transfers, share of GDP
        total_share (scalar): family transfers before netting, share of GDP
        first_age (int): age of the first model age (OG-Core's E)

    Returns:
        Numpy array: (S, J), positive for receivers, summing to zero
    """
    omega = np.asarray(omega, dtype=float)
    e = np.asarray(e, dtype=float)
    S, J = omega.shape
    age = np.arange(S) + first_age
    give = (age >= FAMILY_GIVER_AGES[0]) & (age < FAMILY_GIVER_AGES[1])
    recv = age >= FAMILY_RECEIVER_AGE
    capacity = omega * e * give[:, np.newaxis]
    group_share = capacity.sum(axis=0) / capacity.sum()
    public = alpha_T * np.asarray(eta, dtype=float)[recv].sum(axis=0)
    pot = np.maximum(total_share * group_share - public, 0.0)
    phi = np.zeros((S, J))
    for j in range(J):
        if pot[j] == 0.0:
            continue
        phi[give, j] = -pot[j] * capacity[give, j] / capacity[give, j].sum()
        phi[recv, j] = pot[j] * omega[recv, j] / omega[recv, j].sum()
    return phi


def family_transfer_path(
    omega, omega_SS, e, eta, alpha_T, first_age=20, tol=1e-12
):
    """
    Net family transfer shares for each transition year (allocation_path).

    Args:
        omega (array_like): population by year, (T+S, S, J)
        omega_SS (array_like): steady-state population, (S, J)
        e (array_like): earnings matrix, (S, J)
        eta (array_like): transfer matrices by year, (T+S, S, J)
        alpha_T (array_like): government transfers, share of GDP, length T+S
        first_age (int): age of the first model age (OG-Core's E)
        tol (scalar): trailing years within tol of steady state are dropped

    Returns:
        Numpy array: (n + 1, S, J)
    """
    omega = np.asarray(omega, dtype=float)
    omega_SS = np.asarray(omega_SS, dtype=float)
    eta = np.asarray(eta, dtype=float)
    alpha_T = np.asarray(alpha_T, dtype=float).flatten()
    S = omega_SS.shape[0]
    recv = np.arange(S) + first_age >= FAMILY_RECEIVER_AGE

    def elder_share(om):
        return om[recv].sum() / om.sum()

    share_SS = elder_share(omega_SS)
    phi_SS = family_transfers(
        omega_SS, e, eta[-1], alpha_T[-1], FAMILY_TRANSFER_SHARE_SS, first_age
    )
    path = np.array(
        [
            family_transfers(
                omega[t],
                e,
                eta[t],
                alpha_T[min(t, alpha_T.size - 1)],
                FAMILY_TRANSFER_SHARE_SS * elder_share(omega[t]) / share_SS,
                first_age,
            )
            for t in range(omega.shape[0])
        ]
    )
    n = path.shape[0]
    while n > 0 and np.abs(path[n - 1] - phi_SS).max() < tol:
        n -= 1
    return np.concatenate([path[:n], phi_SS[np.newaxis]])


def _time_path(value, length):
    """A path of ``length`` periods from a shorter one, holding its last."""
    value = np.asarray(value, dtype=float)
    if value.shape[0] >= length:
        return value[:length]
    pad = np.repeat(value[-1:], length - value.shape[0], axis=0)
    return np.concatenate([value, pad])


def derived_family_transfers(p, eta, alpha_T):
    """The net family transfer path for ``p``, ``eta`` and ``alpha_T``."""
    TS = int(p.T) + int(p.S)
    e = np.asarray(p.e, dtype=float)
    e = e[-1] if e.ndim == 3 else e
    return {
        "phi_FT": family_transfer_path(
            p.omega,
            p.omega_SS,
            e,
            _time_path(eta, TS),
            _time_path(np.asarray(alpha_T, dtype=float).flatten(), TS),
            int(p.E),
        ).tolist()
    }


def dsa_annual(values, last_year=None):
    """
    Annual path from FY2024/25 of a DSA series, linearly interpolated.

    Args:
        values (array_like): the series at DSA_YEARS
        last_year (int): last fiscal year; defaults to the last DSA year

    Returns:
        Numpy array: one value per year from FY2024/25 to last_year
    """
    last_year = DSA_YEARS[-1] if last_year is None else last_year
    years = np.arange(DSA_YEARS[0], last_year + 1)
    return np.interp(years, DSA_YEARS, values)


def external_debt_path():
    """
    External debt over Y (alpha_D_ext) by year: DSA path, then to zero.

    Returns:
        list: alpha_D_ext path
    """
    ext = dsa_annual(DSA_EXTERNAL_DEBT)
    pace = (DSA_EXTERNAL_DEBT[-2] - DSA_EXTERNAL_DEBT[-1]) / (
        DSA_YEARS[-1] - DSA_YEARS[-2]
    )
    tail = []
    level = ext[-1] - pace
    while level > 0:
        tail.append(level)
        level -= pace
    path = np.concatenate([ext, tail, [0.0]]) / DSA_EXTERNAL_DEBT[0]
    return [round(float(x) * EXTERNAL_DEBT_RATIO, 6) for x in path]


def external_rate_path():
    """
    Real rate on external debt (r_D_ext_annual): DSA rate less US inflation.

    Returns:
        list: r_D_ext_annual path
    """
    return [
        round(float(x) / 100 - US_INFLATION, 6)
        for x in dsa_annual(DSA_EXTERNAL_RATE)
    ]


def external_debt_params():
    """
    External-debt parameters and the initial market debt.

    Returns:
        dict: alpha_D_ext, r_D_ext_annual, initial_debt_ratio,
            initial_foreign_debt_ratio
    """
    return {
        "alpha_D_ext": external_debt_path(),
        "r_D_ext_annual": external_rate_path(),
        "initial_debt_ratio": INITIAL_MARKET_DEBT_RATIO,
        "initial_foreign_debt_ratio": MARKET_FOREIGN_DEBT_SHARE,
    }


def market_rate_targets():
    """
    Target real rate on market debt: DSA to 2036, then LONG_RUN_R_GOV.

    Returns:
        Numpy array: targets, FY2024/25 to 2036 plus the long-run value
    """
    path = dsa_annual(DSA_DOMESTIC_REAL_RATE, last_year=2036) / 100
    return np.append(path, LONG_RUN_R_GOV)


def market_r_gov_shift_path(r_gov_scale, r_gov_DY2, debt_ratio_ss, r_ss):
    """
    r_gov_shift path putting r_gov on market_rate_targets (macro.md).

    The debt premium is centred on debt_ratio_ss; r is held at r_ss.

    Args:
        r_gov_scale (scalar): pass-through of the market return
        r_gov_DY2 (scalar): curvature of the debt-elastic premium
        debt_ratio_ss (scalar): long-run debt anchor (model units)
        r_ss (scalar): steady-state market return

    Returns:
        list: r_gov_shift path, one value per year to 2036 plus the long run
    """
    targets = market_rate_targets()
    centering = r_gov_DY2 * debt_ratio_ss**2
    shift = r_gov_scale * r_ss - centering - targets
    return [float(x) for x in shift]


def zeta_D_path():
    """
    Foreign share of new market debt (zero; borrowing abroad: alpha_D_ext).

    Returns:
        list: length FISCAL_PATH_LENGTH
    """
    return [MARKET_FOREIGN_DEBT_SHARE] * FISCAL_PATH_LENGTH


# ---------------------------------------------------------------------------
# Capital stocks, depreciation and initial household wealth
#
# Total capital over rebased GDP: PWT 11.0 net stock over GDP at current
# national prices, 2023 (ogeth/data/pwt110_eth_capital_detail.csv), scaled
# by a perpetual inventory that adds the investment the old accounts missed
# (36.1 points of old-base GDP in FY2024/25, phased in from FY2015/16;
# ogeth/data/pwt110_eth_capital.csv; scripts/derive_sourced_values.py).
# Range 1.39-2.94 (macro.md, GDP basis). Replace with official stocks.
PWT_CAPITAL_OUTPUT_RATIO = 2.3655
REBASING_INVESTMENT_GAP = 0.361
TOTAL_CAPITAL_OUTPUT_RATIO = 2.186
# Public capital: IMF ICSD general-government stock over GDP, current
# prices, 2019 (CAPSTCK_S13_V_XDC / B1GQ_V_XDC, billions of birr), held;
# a birr stock, so rebased like fiscal stocks.
PUBLIC_CAPITAL_OUTPUT_RATIO_OLD_BASE = round(2265.27 / 2696.22, 3)
PUBLIC_CAPITAL_OUTPUT_RATIO = round(
    PUBLIC_CAPITAL_OUTPUT_RATIO_OLD_BASE * rebase_factors()[0], 3
)
# The private stock is the total less public capital.
PRIVATE_CAPITAL_OUTPUT_RATIO = round(
    TOTAL_CAPITAL_OUTPUT_RATIO - PUBLIC_CAPITAL_OUTPUT_RATIO, 3
)
# Private depreciation: PWT 11.0 total rate (2023) net of public capital
# at delta_g (OG-Core default), so delta_p K_p + delta_g K_g = 0.0414 K.
PWT_DEPRECIATION_RATE = 0.0414
DELTA_G_ANNUAL = 0.02
DELTA_ANNUAL = round(
    (
        PWT_DEPRECIATION_RATE
        - PUBLIC_CAPITAL_OUTPUT_RATIO
        / TOTAL_CAPITAL_OUTPUT_RATIO
        * DELTA_G_ANNUAL
    )
    / (1 - PUBLIC_CAPITAL_OUTPUT_RATIO / TOTAL_CAPITAL_OUTPUT_RATIO),
    4,
)
# Foreign-owned capital, USD billions, mid-FY2024/25: IMF BoP FDI inflows
# 1990-2024 by perpetual inventory at the private depreciation rate,
# revalued 2.5 percent a year (ogeth/data/imf_fdi_inflows_eth.csv;
# scripts/derive_sourced_values.py; macro.md). Biased down (no reinvested
# earnings recorded).
FDI_STOCK_USD_BN = 33.98
REBASED_GDP_USD_BN = 201.42
FDI_STOCK_TO_GDP = round(FDI_STOCK_USD_BN / REBASED_GDP_USD_BN, 4)
# foreign-owned share of private capital in FY2024/25, the zeta_K target
FOREIGN_CAPITAL_SHARE = round(
    FDI_STOCK_TO_GDP / PRIVATE_CAPITAL_OUTPUT_RATIO, 4
)
# Domestically held general-government debt, FY2024/25
DOMESTIC_DEBT_TO_GDP = round(IMF_DOMESTIC_DEBT[0] / 100, 4)
# Initial household wealth over Y: domestically owned private capital plus
# domestic debt, model units (OG-Core >= 0.22).
INITIAL_WEALTH_RATIO = round(
    to_model(
        PRIVATE_CAPITAL_OUTPUT_RATIO - FDI_STOCK_TO_GDP + DOMESTIC_DEBT_TO_GDP
    ),
    3,
)
# Public investment efficiency gap (IMF-World Bank PIMA, June 2025, s. II):
# the leakage of new public investment and the unproductive share of the
# initial stock.
PUBLIC_INVESTMENT_EFFICIENCY_GAP = 0.53
INITIAL_KG_RATIO = round(
    to_model(
        PUBLIC_CAPITAL_OUTPUT_RATIO * (1 - PUBLIC_INVESTMENT_EFFICIENCY_GAP)
    ),
    3,
)

# Output elasticity of public capital, gamma_g: low end of Bom and
# Ligthart (2014, Journal of Economic Surveys 28(5)), 0.083-0.122 (firms.md).
GAMMA_G = 0.08
# Private capital share gamma, from the firms' first-order condition at
# the measured capital stock and a 5 percent target return, r = (1 - tau_b)
# gamma Y/K - delta + tau_b delta_tau (Caselli and Feyrer 2007; firms.md).
FIRM_RETURN_TARGET = 0.05
# Effective business tax rate, FY2024/25: CIT rate x corporate share of
# assets x the tuned CIT collections factor (examples/tune.py); fold the
# factor here after each retune (tests check it matches the JSON).
CIT_RATE = 0.30
C_CORP_SHARE_OF_ASSETS = 0.55
CIT_COLLECTIONS_FACTOR = 0.3755
BUSINESS_TAX_WEDGE = round(
    CIT_RATE * C_CORP_SHARE_OF_ASSETS * CIT_COLLECTIONS_FACTOR, 4
)
DELTA_TAU_ANNUAL = 0.027
GAMMA = round(
    (FIRM_RETURN_TARGET + DELTA_ANNUAL - BUSINESS_TAX_WEDGE * DELTA_TAU_ANNUAL)
    / (1 - BUSINESS_TAX_WEDGE)
    * to_model(PRIVATE_CAPITAL_OUTPUT_RATIO),
    3,
)


# ---------------------------------------------------------------------------
# The baseline: fiscal and external inputs at FY2024/25 values or the
# FY2025/26 budget, held. Long-run debt ratio: the FY2024/25 ratio of all
# general-government debt (all market debt once external debt is repaid).
# ---------------------------------------------------------------------------
DEBT_RATIO_SS = INITIAL_DEBT_RATIO


def fiscal_params(p, r_ss=None):
    """
    The baseline's fiscal and external parameters.

    Args:
        p (Specifications): model parameters
        r_ss (scalar): return for the shift path; default R_SS_FOR_R_GOV

    Returns:
        dict: ready for Specifications.update_specifications
    """
    if r_ss is None:
        r_ss = R_SS_FOR_R_GOV
    out = {}
    out.update(spending_paths())
    out.update(compliance_paths())
    out["zeta_D"] = zeta_D_path()
    out.update(derived_transfer_eta(p))
    out.update(derived_family_transfers(p, out["eta"], out["alpha_T"]))
    out.update(
        revenue_paths(
            float(np.asarray(p.tau_c).flatten()[0]),
            float(
                np.asarray(p.adjustment_factor_for_cit_receipts).flatten()[0]
            ),
        )
    )
    out["r_gov_shift"] = market_r_gov_shift_path(
        float(np.asarray(p.r_gov_scale).flatten()[0]),
        float(p.r_gov_DY2),
        DEBT_RATIO_SS,
        r_ss,
    )
    out.update(external_debt_params())
    out["debt_ratio_ss"] = DEBT_RATIO_SS
    out["r_gov_DY"] = -2 * float(p.r_gov_DY2) * DEBT_RATIO_SS
    out["tG1"] = TG1
    out.update(remittance_level_params())
    out.update(common_params())
    return out


def common_params():
    """
    Taxes, pensions, capital, preferences, initial conditions and solver.

    Returns:
        dict: ready for Specifications.update_specifications
    """
    out = {}
    out.update(statutory_tax_params())
    out.update(defined_benefit_params())
    out.update(payroll_tax_params())
    out.update(
        {
            "world_int_rate_annual": [WORLD_INT_RATE_ANNUAL],
            "gamma": [float(GAMMA)],
            "gamma_g": [GAMMA_G],
            "infra_investment_leakage_rate": PUBLIC_INVESTMENT_EFFICIENCY_GAP,
            "delta_annual": DELTA_ANNUAL,
            "delta_g_annual": DELTA_G_ANNUAL,
            "initial_Kg_ratio": INITIAL_KG_RATIO,
            "initial_wealth_ratio": INITIAL_WEALTH_RATIO,
            "r_gov_floor": R_GOV_FLOOR,
            "frisch": FRISCH,
            "sigma": SIGMA,
            "c_min": [C_MIN],
            "nu": NU,
            "TPI_outer_method": "anderson",
            "TPI_anderson_m": TPI_ANDERSON_M,
            "TPI_anderson_beta": TPI_ANDERSON_BETA,
        }
    )
    return out


def baseline_params(p, r_ss=None):
    """
    Baseline parameters that update_baseline.py writes into the JSON.

    Args:
        p (Specifications): see fiscal_params
        r_ss (scalar): see fiscal_params

    Returns:
        dict: ready for Specifications.update_specifications
    """
    out = fiscal_params(p, r_ss)
    out.update(derived_remittance_params(p))
    return out
