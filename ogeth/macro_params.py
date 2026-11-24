"""
This module uses data from World Bank WDI, World Bank Quarterly Public
Sector Debt (QPSD) database, the IMF, and UN ILO to find values for
parameters for the OG-ETH model that rely on macro data for calibration.
"""

# imports
import pandas as pd
import numpy as np
import requests
import datetime
from pathlib import Path

# Solver settings used by the example scripts: the damping of OG-Core's
# outer loops (nu, the weight on the new iterate) and Anderson acceleration
# of the time-path outer loop (PSLmodels/OG-Core master; the example applies
# it only where the installed OG-Core has the parameters). With discount
# factors that differ by lifetime-income group the plain damped iteration
# converges slowly, and a smaller step with acceleration is both faster and
# more robust.
NU = 0.2
TPI_ANDERSON_M = 5
TPI_ANDERSON_BETA = 1.0

# Frisch elasticity of labour supply. OG-USA's 0.4 sits at the top of the
# range for Ethiopia. Corrected for publication bias, the micro evidence
# puts the intensive margin at about 0.2 and total hours at about 0.25
# (Elminejad, Havranek, Horvath and Irsova 2023, Review of Economic
# Dynamics 51); Chetty, Guren, Manoli and Weber (2013, NBER Macroeconomics
# Annual) bound the intensive margin below 0.47. In a setting like
# Ethiopia's, randomly offered daily wages in rural Malawi give 0.15-0.16
# (Goldberg 2016, AEJ: Applied), and off-farm hours in western Ethiopia
# 0.13-0.22. Participation is near universal, so the extensive margin adds
# little. Range 0.15-0.4.
FRISCH = 0.25

# Coefficient of relative risk aversion, the inverse of the intertemporal
# elasticity of substitution (EIS) above subsistence. OG-Core's households
# face no income risk, so sigma acts as 1/EIS. OG-Core's default, 1.5 (EIS
# 0.67): below the 0.8 above which Havranek (2015, JEEA 13(6)) finds
# calibrations inconsistent with the evidence, and above his reported mean
# of 0.5 and his bias-corrected 0.3-0.4 for asset holders. The low
# EIS of poor households comes from the subsistence floor below rather than
# from sigma: with Stone-Geary preferences the EIS of total consumption is
# (1/sigma)(1 - c_min/c) (Ogaki, Ostry and Reinhart 1996, IMF Staff Papers),
# so in the solved steady state the effective sigma is 1.63 at mean
# consumption and 2.2 for the poorest group (the floor is 32 percent of its
# consumption), against 1.5 for the richest. A sigma of 3 for everyone
# (the earlier working value) put that curvature on the richer households
# too, where with 4.7 percent growth it made them want to consume against
# higher future income, which pushed patience to its bound; the ESS 2021/22
# shows credit is informal and saving is precautionary (80 percent of
# savers save for emergencies, 0.9 percent for old age).
SIGMA = 1.5
# Subsistence floor (OG-Core's c_min), consumption per adult in model units
# that households buy before the rest of consumption enters utility. Its
# source is the cost of an energy-sufficient diet in Ethiopia (World
# Bank/FAO Food Prices for Nutrition, 2021 PPP USD 1.17 a day), 12.2
# percent of mean consumption per person on the rebased accounts (analysis
# in ~/Projects/og-issue-drafts/sigma-evidence). Applied as that share times
# private consumption over GDP (0.644) times GDP per model adult, the floor
# would be 0.136 in the FY2024/25 solution; the value used is 0.1088, 80
# percent of it, because the floor applies to every age of every group and
# the poorest group (a quarter of the population) has to clear it at its
# lowest-consumption age, which it does with 0.02 to spare; in the data
# about 6 percent of people consume less than the diet costs. OG-Core
# expresses c_min in growth-adjusted units, so the floor grows with labor
# productivity (a constant share of average income).
C_MIN = 0.1088
# Units of effective labor, a normalization. The earnings matrix e is
# built with a population-weighted mean of one; multiplying it by k scales
# every income in the model by k, which leaves every equilibrium condition
# unchanged except the labor condition, where the chi_n it needs scales by
# k^(1 - sigma). At sigma 3 with k = 1, model consumption is about
# 0.05-0.3 and chi_n would have to exceed OG-Core's bound of 10,000 at
# every age. Production has constant returns in K, K_g and L, and OG-Core's
# capital demand at the world rate is proportional to L, so the open-economy
# block scales exactly too: a steady-state probe with e x 4 (and chi_n x
# 4^(1-sigma)) reproduced r, K/Y, B/Y, C/Y, K_f/K and the wealth shares to
# every printed digit. (Scaling productivity Z is not neutral: OG-Core's
# capital demand at the world rate omits public capital, so Z moves the
# foreign capital share.)
EARNINGS_SCALE = 10.0

# Window for the g_y_annual average-growth calculation: the mean of the
# year-over-year GDP-per-capita growth rates for G_Y_START_YEAR through
# G_Y_END_YEAR (inclusive). We use the post-2015 decade rather than the
# full available history: the 2004-2015 state-led investment boom (which
# lifted the 2006-2024 average to ~6.0%) is explicitly not expected to
# repeat (IMF Country Report 26/20 DSA assumes long-run growth "slower
# than historical rates of around 10 percent"), so this window gives a
# balanced-growth-path productivity rate that averages over the post-boom
# normalization, the 2020-2022 conflict/COVID dip, and the recent
# gold-driven recovery. Average annual GDP-per-capita growth for the ten
# years 2016-2025 = 4.7% (see macro.md). Computing the growth rate for
# G_Y_START_YEAR requires the previous year's level, so the level filter
# below keeps data back to G_Y_START_YEAR - 1.
G_Y_START_YEAR = 2016
G_Y_END_YEAR = 2025


def _fetch_wb_data(indicators, country_iso, start_year, end_year, source):
    """
    Fetch a set of World Bank indicators and return a single DataFrame.

    Args:
        indicators (dict): mapping of human-readable labels to indicator codes
        country_iso (str): ISO country code
        start_year (int): first year to request
        end_year (int): last year to request
        source (int): World Bank source ID

    Returns:
        pandas.DataFrame: DataFrame indexed by year/quarter label
    """
    if source == 2:
        date_range = f"{start_year}:{end_year}"
    elif source == 20:
        date_range = f"{start_year}Q1:{end_year}Q4"
    else:
        raise ValueError(f"Unsupported World Bank source: {source}")

    data_frames = []
    for label, indicator_code in indicators.items():
        response = requests.get(
            (
                "https://api.worldbank.org/v2/country/"
                f"{country_iso}/indicator/{indicator_code}"
            ),
            params={
                "date": date_range,
                "source": source,
                "format": "json",
                "per_page": 10000,
            },
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Malformed World Bank response for {indicator_code}"
            ) from exc

        if (
            not isinstance(payload, list)
            or len(payload) < 2
            or not isinstance(payload[1], list)
            or not payload[1]
        ):
            raise ValueError(
                f"Empty or malformed World Bank response for {indicator_code}"
            )

        series_data = {}
        for row in payload[1]:
            date = row.get("date")
            if date is None:
                continue
            series_data[date] = row.get("value")

        if not series_data:
            raise ValueError(
                "No dated observations in World Bank response "
                f"for {indicator_code}"
            )

        series = pd.Series(series_data, name=label)
        series = pd.to_numeric(series, errors="coerce")
        data_frames.append(series.to_frame())

    data = pd.concat(data_frames, axis=1)
    data.index.name = "year"
    # Preserve descending time order used by the existing pct_change(-1) logic.
    data = data.sort_index(ascending=False)
    return data


def _get_imf_macro_params(
    country_iso,
    target_year,
    data_path=None,
):
    """
    Fetch IMF GFS data and compute alpha_T and alpha_G.

    Args:
        country_iso (str): ISO alpha-3 country code
        target_year (int): preferred calibration year
        data_path (str | Path | None): optional path to save IMF CSV data

    Returns:
        dict: IMF-derived macro parameters
    """
    required_indicators = {"G2_T", "G24_T", "G27_T", "G271_T"}
    data_path = Path(data_path) if data_path is not None else None
    response = requests.get(
        (
            "https://api.imf.org/external/sdmx/3.0/data/dataflow/"
            f"IMF.STA/GFS_SOO/12.0.0/"
            f"{country_iso}.S1311.G2M.*.POGDP_PT.A"
        ),
        timeout=30,
    )
    response.raise_for_status()
    try:
        payload = response.json()
        data = payload["data"]
        structure = data["structures"][0]
        data_set = data["dataSets"][0]
        series_dimensions = structure["dimensions"]["series"]
        observation_years = [
            value.get("id", value.get("value"))
            for value in structure["dimensions"]["observation"][0]["values"]
        ]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Empty or malformed IMF response for GFS_SOO"
        ) from exc

    records = []
    for series_key, series in data_set["series"].items():
        dimension_indexes = [int(idx) for idx in series_key.split(":")]
        labels = {
            dim["id"]: dim["values"][idx]["id"]
            for dim, idx in zip(series_dimensions, dimension_indexes)
        }
        indicator = labels.get("INDICATOR")
        if indicator not in required_indicators:
            continue
        for observation_key, observation in series.get(
            "observations", {}
        ).items():
            value = observation[0]
            if value is None:
                continue
            records.append(
                {
                    "year": observation_years[int(observation_key)],
                    "indicator": indicator,
                    "value": float(value),
                    "country_iso": country_iso,
                    "sector": "S1311",
                    "dataset": "IMF.STA:GFS_SOO(12.0.0)",
                }
            )

    imf_data = pd.DataFrame(records)
    if imf_data.empty:
        raise ValueError("Empty or malformed IMF response for GFS_SOO")

    if data_path is not None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        imf_data.sort_values(["indicator", "year"]).to_csv(
            data_path, index=False
        )
        print(f"IMF data saved to {data_path}")

    imf_data["year"] = pd.to_numeric(imf_data["year"], errors="coerce")
    imf_data["value"] = pd.to_numeric(imf_data["value"], errors="coerce")
    imf_data = imf_data.dropna(subset=["year", "value"])

    available = (
        imf_data.pivot_table(
            index="year", columns="indicator", values="value", aggfunc="first"
        )
        .sort_index()
        .dropna(subset=sorted(required_indicators))
    )
    available = available.loc[available.index <= int(target_year)]

    if available.empty:
        raise ValueError(
            f"No complete IMF data available for {country_iso} "
            f"up to {target_year}"
        )

    selected_year = (
        int(target_year)
        if int(target_year) in available.index
        else int(available.index.max())
    )
    if selected_year != int(target_year):
        print(
            f"Warning: No IMF data for {target_year}. "
            f"Using last available year: {selected_year}"
        )

    values = available.loc[selected_year]
    return {
        "alpha_T": [(values["G27_T"] - values["G271_T"]) / 100],
        "alpha_G": [
            (values["G2_T"] - values["G24_T"] - values["G27_T"]) / 100
        ],
    }


def get_macro_params(
    data_start_date=datetime.datetime(1947, 1, 1),
    data_end_date=datetime.datetime(2025, 12, 31),
    country_iso="ETH",
    update_from_api=False,
    imf_data_year=None,
    imf_data_path=None,
):
    """
    Compute values of parameters that are derived from macro data

    Args:
        data_start_date (datetime): start date for data
        data_end_date (datetime): end date for data
        country_iso (str): ISO code for country
        imf_data_year (int | None): IMF target year override. Defaults to
            data_end_date.year when None.
        imf_data_path (str | Path | None): optional path to save IMF CSV data

    Returns:
        macro_parameters (dict): dictionary of parameter values
    """
    # initialize a dictionary of parameters
    macro_parameters = {}

    """
    Retrieve data from the World Bank World Development Indicators.
    """
    # Dictionaries of variables and their corresponding World Bank codes
    # Annual data
    wb_a_variable_dict = {
        "GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD",
        "Real GDP (constant 2015 US$)": "NY.GDP.MKTP.KD",
        "Nominal GDP (current US$)": "NY.GDP.MKTP.CD",
        (
            "General government final consumption expenditure (current US$)"
        ): "NE.CON.GOVT.CD",
    }
    # Quarterly public-sector-debt (QPSD) indicators are intentionally
    # not fetched for OG-ETH: the World Bank QPSD database has no Ethiopia
    # data. The debt parameters (initial_debt_ratio,
    # initial_foreign_debt_ratio, zeta_D) are hand-calibrated from the
    # Ethiopia MoF Public Sector Debt Bulletin No. 51 (see macro.md).
    if update_from_api:
        try:
            wb_data_a = _fetch_wb_data(
                wb_a_variable_dict,
                country_iso,
                data_start_date.year,
                data_end_date.year,
                source=2,
            )
            # Compute annual GDP-per-capita growth (g_y_annual) from the
            # World Bank WDI series, averaging the year-over-year growth
            # rates over the [G_Y_START_YEAR, G_Y_END_YEAR] window (see
            # macro.md). Debt parameters are not derived here (see note
            # above).
            if "GDP per capita (constant 2015 US$)" in wb_data_a.columns:
                gdp_pc = wb_data_a["GDP per capita (constant 2015 US$)"]
                years = gdp_pc.index.astype(int)
                # Keep one year before G_Y_START_YEAR so the growth rate for
                # G_Y_START_YEAR itself can be formed; pct_change(-1) on the
                # descending series then yields the growth rates for
                # G_Y_START_YEAR..G_Y_END_YEAR (the anchor year becomes NaN
                # and is dropped by mean()).
                gdp_pc = gdp_pc[
                    (years >= G_Y_START_YEAR - 1) & (years <= G_Y_END_YEAR)
                ]
                g_y_series = gdp_pc.pct_change(-1)

                # If all values are NaN, return None
                macro_parameters["g_y_annual"] = (
                    g_y_series.mean() if not g_y_series.isna().all() else None
                )
                print(
                    "g_y_annual updated from World Bank API: "
                    f"{macro_parameters['g_y_annual']}"
                )
            else:
                print(
                    "Warning: Missing GDP per capita data in World "
                    "Bank data. Skipping update for g_y_annual."
                )
        except Exception:
            print("Failed to retrieve data from World Bank")
            print("Will not update g_y_annual")
    else:
        print("Not updating from World Bank API")

    # gamma is deliberately NOT refreshed from the ILOSTAT labour share: the
    # naive 1 - labour share is biased by Ethiopia's self-employment, and
    # gamma is set from the firms' first-order condition at the measured
    # capital stock (GAMMA, firms.md). A live refresh would silently overwrite
    # it, so it is off the live path entirely.

    # alpha_T and alpha_G are NOT pulled from the IMF API for OG-ETH: the
    # IMF SDMX endpoint returns only 2002-vintage data for Ethiopia, and
    # the documented sources differ (alpha_G -> World Bank NE.CON.GOVT.ZS;
    # alpha_T -> IMF GFS + IMF Country Report 26/20, hand-combined). Both
    # stay at the committed values in macro.md. _get_imf_macro_params is
    # retained (tested independently) for reuse if wired to the right data.

    # The government-debt interest-rate parameters (r_gov_scale, r_gov_shift,
    # r_gov_DY, r_gov_DY2) are NOT returned from the live path. r_gov_scale
    # and the base r_gov_shift come from inverting the Li, Magud, Werner,
    # Witte (2021) sovereign-vs-corporate yield relationship (a deterministic
    # calculation, scripts/derive_sourced_values.py), but the committed
    # r_gov_shift is then re-centered for the debt-elastic premium
    # (r_gov_DY, r_gov_DY2) so the premium is exactly zero at debt_ratio_ss
    # (see macro.md). Returning the raw LMW shift here would un-center that
    # premium and silently move the steady state, so all four stay frozen at
    # the documented values in ogeth_default_parameters.json.
    if update_from_api:
        print(
            "Not updating r_gov_* (frozen, debt-elastic premium re-centered; "
            "see macro.md)"
        )
    else:
        print("Not computing r_gov_shift, r_gov_scale")

    return macro_parameters


# ---------------------------------------------------------------------------
# Remittances
#
# Remittances enter OG-Core as an exogenous share of GDP (alpha_RM_1 in the
# first period, alpha_RM_T in the long run) distributed across households by
# eta_RM. Two of the shipped values are derived from the packaged
# demographics rather than hand-set, so they are rebuilt whenever the
# demographics are (update_baseline.py):
#
# * g_RM -- aggregates.get_RM advances detrended remittances by the factor
#   (1 + g_RM[t]) / (exp(g_y) * (1 + g_n[t-1])) between the first period and
#   tG1, independently of that period's output, so remittances keep pace with
#   trend GDP (a constant share on the balanced growth path) only when that
#   factor is one. g_n moves along the transition, so the growth rate that
#   does this is a path, not a scalar (flat_share_g_RM), and alpha_RM_1 and
#   alpha_RM_T are both the measured FY2024/25 share
#   (remittance_level_params). Along the transition RM/Y then moves
#   inversely with output's deviation from trend.
# * eta_RM -- the share of aggregate remittances each household receives,
#   shaped (S, J). OG-Core's default hands every lifetime-income group
#   exactly its population share; remittance_eta instead maps a data
#   distribution of remittance value across income quintiles onto the
#   model's J groups and spreads it per capita across ages within a group.
# ---------------------------------------------------------------------------

# Share of total remittance VALUE received by each household income
# quintile, poorest to richest. World Bank / Bendixen & Amandi, "Remittances
# to Ethiopia" (Future of African Remittances national survey, 2010; 2,412
# adults): recipients' monthly household income was 9% below ETB 1,000, 47%
# at ETB 1,000-3,500 and 37% above ETB 3,500 (7% no answer). Against 2010
# GDP per capita of ETB 4,262 (about ETB 1,600 a month for a 4.6-person
# household) those brackets map to the bottom two, middle two and top
# quintile; renormalized over respondents and split evenly within each
# pair. See docs/book/content/calibration/macro.md (Remittances).
RM_QUINTILE_VALUE_SHARES = [
    9 / 93 / 2,
    9 / 93 / 2,
    47 / 93 / 2,
    47 / 93 / 2,
    37 / 93,
]


def flat_share_g_RM(g_y, g_n):
    """
    Remittance growth path that keeps aggregate remittances growing with
    trend GDP, so their share is constant on the balanced growth path.

    Args:
        g_y (scalar): model-period productivity growth rate
        g_n (array_like): population growth path, length T + S

    Returns:
        g_RM (Numpy array): growth rate of remittances, length T + S, such
            that ``(1 + g_RM[t]) / (exp(g_y) * (1 + g_n[t - 1])) == 1`` for
            every t >= 1 (the factor ``get_RM`` applies)
    """
    g_n = np.asarray(g_n, dtype=float)
    g_RM = np.exp(g_y) * (1.0 + np.roll(g_n, 1)) - 1.0
    # get_RM never reads g_RM[0]; keep it consistent with period 0 anyway
    g_RM[0] = np.exp(g_y) * (1.0 + g_n[0]) - 1.0
    return g_RM


def remittance_level_params():
    """
    The remittance share of GDP in the first period and in the long run:
    the measured FY2024/25 share (IMF CR 26/174, Table 4a, rebased), held.

    Returns:
        dict: ``{"alpha_RM_1", "alpha_RM_T"}``
    """
    share = round(to_model(float(IMF_PRIVATE_TRANSFERS[0]) / 100), 6)
    return {"alpha_RM_1": share, "alpha_RM_T": share}


def remittance_eta(omega_SS, lambdas, quintile_value_shares=None):
    """
    Allocation matrix distributing aggregate remittances across households.

    Args:
        omega_SS (array_like): steady-state population distribution, (S, J),
            with ``omega_SS.sum(axis=0) == lambdas``
        lambdas (array_like): population shares of the J lifetime-income
            groups, poorest to richest
        quintile_value_shares (array_like): share of total remittance value
            received by each population quintile, poorest to richest;
            defaults to ``RM_QUINTILE_VALUE_SHARES``

    Returns:
        eta_RM (Numpy array): (S, J) matrix summing to one. Each group's
            column sums to its data share of remittance value (the quintile
            shares interpolated onto the group boundaries, assuming a
            uniform density of remittance value within a quintile); within a
            group, remittances are spread per capita across ages, so every
            household in the group receives the same amount.
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


def derive_remittance_params(g_y, g_n, omega_SS, lambdas, omega=None):
    """
    The remittance parameters that depend on a given set of demographics.

    Args:
        g_y (scalar): model-period productivity growth rate
        g_n (array_like): population growth path, length T + S
        omega_SS (array_like): steady-state population distribution, (S, J)
        lambdas (array_like): population shares of the J groups
        omega (array_like): population distribution by year, (T+S, S, J);
            if given, eta_RM is a path built from each year's population
            (see ``allocation_path``), otherwise the steady-state matrix

    Returns:
        dict: ``{"g_RM": list, "eta_RM": nested list}`` ready for
            ``Specifications.update_specifications``
    """
    g_RM = flat_share_g_RM(g_y, g_n)
    if omega is None:
        eta_RM = remittance_eta(omega_SS, lambdas)
    else:
        eta_RM = allocation_path(
            lambda om: remittance_eta(om, lambdas), omega, omega_SS
        )
    return {"g_RM": g_RM.tolist(), "eta_RM": eta_RM.tolist()}


def derived_remittance_params(p):
    """
    The remittance parameters that depend on the demographics in ``p``.

    Args:
        p (Specifications): parameters carrying ``g_y``, ``g_n``,
            ``omega``, ``omega_SS`` and ``lambdas``

    Returns:
        dict: see ``derive_remittance_params``
    """
    g_n = np.asarray(p.g_n, dtype=float)[: p.T + p.S]
    return derive_remittance_params(
        p.g_y, g_n, p.omega_SS, p.lambdas, omega=p.omega
    )


# ---------------------------------------------------------------------------
# GDP basis: the rebased national accounts
#
# Ethiopia rebased its national accounts in November 2026 (Ministry of
# National Strategy, "High Level Report: 2017 EFY GDP Rebasing", with IMF
# technical assistance; figures in ogeth/data/mons_rebased_gdp_fy2024_25.csv):
# base year FY2024/25, 2008 SNA, wider coverage of informal mining and
# manufacturing, transport and digital services and regional state-owned
# enterprises. FY2024/25 nominal GDP went from 15,094 to 24,031 billion birr
# (USD 126.5 to 201.4 billion at the NBE average rate of 119.3; report
# Table 2) and FY2025/26 (preliminary) is 31,934 billion birr (Table 5; the
# report's text also gives 31.86 and 31.97 trillion, but Table 5's figure is
# the sum of its components and the denominator of its sector shares).
# Every ratio to GDP in this calibration is to the rebased series.
#
# The IMF tables below are kept exactly as published, on the 2015/16 base
# (IMF CR 26/174, June 2026, Table 1: FY2024/25 actual 15,094 billion birr;
# FY2025/26 19,268; WEO October 2026 has the same figures). A fiscal flow
# or stock in birr, or a balance-of-payments flow in dollars, has the same
# numerator on either basis, so its rebased ratio is the published ratio
# times old GDP / new GDP of that year.
# National-accounts components (consumption, investment) are NOT converted
# this way: the rebasing revised their numerators too, so they come from the
# rebased accounts directly (GROSS_CAPITAL_FORMATION etc. below).
# ---------------------------------------------------------------------------
NOMINAL_GDP_OLD_BASE_BIRR_BN = {
    "FY2023/24": 11752.0,
    "FY2024/25": 15094.0,
    "FY2025/26": 19268.0,
}
NOMINAL_GDP_REBASED_BIRR_BN = {"FY2024/25": 24031.07, "FY2025/26": 31934.0}

# IMF fiscal and external tables (IMF Country Report 26/174, fifth ECF
# review, Tables 1, 2b and 4a): the FY2024/25 outturn (model period 0, start
# year 2025) and the enacted FY2025/26 budget (period 1), the two fiscal
# years the calibration uses; the budget-year value is held for the long
# run. Published percent of OLD-base GDP, general government; converted
# below.
# ---------------------------------------------------------------------------
FISCAL_YEARS = ["FY2024/25", "FY2025/26"]
# OG-Core's debt-closure rule starts in model period 7 (FY2031/32): until
# then debt follows the budget identity. The fiscal paths are written out to
# that period plus the long run.
TG1 = 7
FISCAL_PATH_LENGTH = TG1 + 1


def rebase_factors(n=len(FISCAL_YEARS)):
    """
    Old-base over rebased nominal GDP for each fiscal year: the factor that
    turns a published ratio to old-base GDP into a ratio to rebased GDP when
    the numerator is unaffected by the rebasing.

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
    Convert a published old-base percent-of-GDP series (one value per
    fiscal year from FY2024/25) to percent of rebased GDP.

    Args:
        series (array_like): published values

    Returns:
        list: rebased values, rounded to 4 decimals
    """
    x = np.asarray(series, dtype=float)
    return [round(float(v), 4) for v in x * rebase_factors(x.size)]


# Published, percent of old-base GDP (IMF CR 26/174 Tables 1, 2b and 4a):
# FY2024/25 outturn, FY2025/26 budget. Public debt is the general-government
# gross debt of the IMF's World Economic Outlook (October 2026,
# GGXWDG_NGDP, "2025" = FY2024/25), which matches the general-government
# revenue, spending and interest the model's budget uses. The program
# report's wider public debt (50.5 percent in FY2024/25) adds about 7.2
# points of NBE long-term foreign liabilities and Ethio Telecom external
# debt that the budget does not service.
IMF_PUBLIC_DEBT_PUBLISHED = [43.3, 40.7]
IMF_PUBLIC_DEBT_WIDE_PUBLISHED = [50.5, 45.3]
IMF_REVENUE_PUBLISHED = [9.2, 10.8]
IMF_GRANTS_PUBLISHED = [1.7, 1.3]
IMF_EXPENDITURE_PUBLISHED = [12.0, 14.1]
IMF_RECURRENT_PUBLISHED = [7.1, 8.2]
IMF_INTEREST_PUBLISHED = [0.8, 1.3]
IMF_CAPITAL_PUBLISHED = [5.0, 5.8]
IMF_PRIVATE_TRANSFERS_PUBLISHED = [5.6, 6.0]
IMF_DOMESTIC_DEBT_PUBLISHED = [18.7, 15.4]

# The same series in percent of rebased GDP: what the calibration uses
IMF_PUBLIC_DEBT = rebased(IMF_PUBLIC_DEBT_PUBLISHED)
IMF_REVENUE = rebased(IMF_REVENUE_PUBLISHED)  # excl. grants
IMF_GRANTS = rebased(IMF_GRANTS_PUBLISHED)
IMF_EXPENDITURE = rebased(IMF_EXPENDITURE_PUBLISHED)
IMF_RECURRENT = rebased(IMF_RECURRENT_PUBLISHED)
IMF_INTEREST = rebased(IMF_INTEREST_PUBLISHED)
IMF_CAPITAL = rebased(IMF_CAPITAL_PUBLISHED)
IMF_PRIVATE_TRANSFERS = rebased(IMF_PRIVATE_TRANSFERS_PUBLISHED)
IMF_DOMESTIC_DEBT = rebased(IMF_DOMESTIC_DEBT_PUBLISHED)
# foreign-held share of general-government debt in FY2024/25: external debt
# (31.8 percent of old GDP in the program's wider perimeter) less the 7.2
# points outside general government, over 43.3
GG_FOREIGN_DEBT_SHARE = round(
    (31.8 - (IMF_PUBLIC_DEBT_WIDE_PUBLISHED[0] - IMF_PUBLIC_DEBT_PUBLISHED[0]))
    / IMF_PUBLIC_DEBT_PUBLISHED[0],
    4,
)

# Government debt contracted abroad (OG-Core's alpha_D_ext and
# r_D_ext_annual, PSLmodels/OG-Core#1125): debt borrowed directly from
# foreign lenders at its own interest rate, outside the domestic market and
# households' portfolio. All of general-government external debt is of this
# kind. Most of it is owed to official lenders on concessional terms (IDA
# about 39 percent, official non-Paris Club creditors 31 percent), and
# non-official creditors hold 10 percent (IMF CR 26/174, DSA paragraph on
# debt composition), chiefly the defaulted USD 1 billion Eurobond. The World
# Bank's International Debt Statistics put private creditors at 18 percent
# of public and publicly guaranteed external debt (USD 6.0 of 34.0 billion,
# end-2024, of which bonds 1.0); the gap to the IMF's 10 percent is mostly
# state-enterprise loans backed by export credit agencies, outside general
# government. The rate below is the DSA's average across all of it.
# The public DSA of IMF CR 26/174 (Table 2, public sector debt), fiscal years
# ending June: FY2024/25 to FY2030/31, then 2036 and 2045.
DSA_YEARS = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2036, 2045]
# Public external debt, percent of old-base GDP. The external debt falls with
# it: amortization outpaces new concessional loans as the program's
# financing surge ends and growth erodes the ratio. Ethiopia's income was 74
# percent of IDA's operational cut-off when the program was approved (CR
# 26/174), so concessional terms continue over the DSA horizon (the grant
# element of new external borrowing stays at about 47 percent to FY2044/45,
# Text Table 3) and end with IDA graduation after it: beyond 2045 the ratio
# falls to zero at its 2036-2045 pace (external_debt_path).
DSA_EXTERNAL_DEBT = [31.8, 29.9, 27.2, 24.7, 21.5, 18.8, 16.8, 9.2, 4.1]
# Average nominal interest rate on external debt, percent (USD terms). The
# rise from 0.7 to 2.6 percent is the end of the restructuring's interest
# relief and the program's IMF and blended loans; it falls back as those are
# repaid and IDA credits dominate. Its path is how the falling grant element
# of the stock enters the model: concessionality shows up as a low interest
# rate on the stock, not as a separate grant (alpha_FA holds cash grants
# only, IMF Table 2b).
DSA_EXTERNAL_RATE = [0.7, 1.8, 2.6, 2.1, 1.9, 1.8, 1.6, 1.3, 0.9]
# Real rate on external debt in the model's units of output: the USD
# nominal rate less US inflation (2 percent, the Federal Reserve's target),
# with the real exchange rate held constant beyond the FY2024/25 float.
US_INFLATION = 0.02
# Average real interest rate on domestic debt, percent: deeply negative under
# financial repression (sub-market mandatory bond purchases, -11.0 percent in
# FY2024/25) and rising as legacy instruments roll over at market rates (the
# policy and Treasury-bill rates turned positive in real terms by 2026, CR
# 26/174 paragraph 12). This is the measured rate on market debt.
DSA_DOMESTIC_REAL_RATE = [-11.0, -9.8, -5.1, -0.6, 1.0, 2.6, 3.3, 4.6, 6.6]
# The DSA's terms for new domestic borrowing, its baseline 10-year
# projections: 4.9 percent real on medium- and long-term debt and 4.5 percent
# on short-term debt (Table: borrowing assumptions for the stress tests).
DSA_MARGINAL_DOMESTIC_REAL_RATES = (4.9, 4.5)

# Rebased expenditure-side national accounts, FY2024/25 (Ministry of National
# Strategy rebasing report, Table 4, billions of birr): private consumption
# 15,473.1 (64.4 percent of GDP; households 14,807.3 or 61.6 percent and
# non-profit institutions serving households 665.8 or 2.8 percent), gross
# capital formation 8,486.8 (35.3 percent, fixed capital plus inventories;
# old base 20.1), government consumption 1,136.8 (4.7 percent) and the
# resource balance -1,065.6 (-4.4 percent; the IMF's goods and services
# balance over rebased GDP gives -4.5). National-accounts government
# consumption includes the consumption of fixed capital on public capital,
# an imputed cost the model does not count in G, so the model moment adds
# delta_g times K_g (ogeth/targets.py).
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

# Price basis. OG-Core's output Y is GDP at producer prices: the consumption
# tax is collected on top of it, so GDP at market prices, the basis of every
# data ratio, is Y plus the consumption tax. Shares OG-Core applies to its
# own Y (alpha_G, alpha_T, alpha_I, alpha_FA, alpha_RM, the debt ratios,
# initial wealth and public capital) are the data share times
# MODEL_PER_MARKET_GDP, and model moments are scored over Y plus the
# consumption tax (ogeth/targets.py). The consumption-tax share is the
# FY2024/25 outturn of the revenue mapped to tau_c below, over rebased GDP.
# General-government revenue outturn, FY2024/25, billions of birr (NBE Annual
# Report 2024/25, Table 6.3, from Ministry of Finance data; IMF CR 26/174
# Table 2a has the same totals). Each line is mapped to the model instrument
# that taxes the same margin (taxes.md):
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


# Initial general-government debt and its foreign-held share, FY2024/25, in
# model units (end of the year before the first model period)
INITIAL_DEBT_RATIO = round(to_model(IMF_PUBLIC_DEBT[0] / 100), 4)
# Its split, FY2024/25, model units: external debt (OG-Core's D_ext), and
# market debt (OG-Core's D), which is domestic debt, 18.7 percent of
# old-base GDP. OG-Core's initial_foreign_debt_ratio and zeta_D are shares
# of market debt held by non-residents: birr Treasury bills and bonds are
# held by the central bank, banks and pension funds (CR 26/174), and the
# securities market opened to non-residents only in 2025, so the share is
# zero.
EXTERNAL_DEBT_RATIO = round(INITIAL_DEBT_RATIO * GG_FOREIGN_DEBT_SHARE, 4)
INITIAL_MARKET_DEBT_RATIO = round(INITIAL_DEBT_RATIO - EXTERNAL_DEBT_RATIO, 4)
MARKET_FOREIGN_DEBT_SHARE = 0.0

# Cash transfers to households as a share of GDP, published percent of
# old-base GDP. The IMF recurrent line bundles the wage bill, goods and
# services, interest, and transfers; the household-transfer part is the fuel
# and fertilizer subsidies (about 1 percent of GDP in FY2024/25; fuel
# subsidies eliminated by March 2026, fertilizer about 1 percent of GDP),
# the Productive Safety Net Program
# (budget contribution about 0.4 percent of GDP), and public pension payouts
# (about 0.5 percent of GDP). IMF CR 26/174 paragraphs 9, 20-22 and Box on
# social safety nets.
CASH_TRANSFERS_PUBLISHED = [2.0, 1.8]
# The path above counts public pension payouts at half a percent of GDP, a
# round figure with no table behind it. The schemes' own accounts give much
# less (ILO Social Security Inquiry, Ethiopia, 2024 data): the public
# servants' scheme (PSSSA) paid 22.2 billion birr to 750,812 recipients, and
# the private scheme (POESSA) 43,347 recipients, here at PSSSA's average
# benefit (its reported 11.58 billion birr of expenditure includes items
# other than pensions, about nine times PSSSA's spending per recipient):
# about 23.5 billion birr, 0.20 percent of FY2023/24 old-base GDP (11,752
# billion birr). Old-age pensions are about 60 percent of it: survivors are
# the largest group (356,003) but receive fractions of the deceased member's
# pension (Proclamation 1267/2022: spouse 50, child 20 percent), and
# invalidity pensioners are few (10,870); range 55-65 percent.
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

# Return foreign investors earn on capital in Ethiopia (OG-Core's
# world_int_rate_annual, a real rate). In OG-Core it sets K_open, the capital
# firms would demand at that return, and foreign capital fills a share zeta_K
# of the gap K_f = zeta_K (K_open - K_d), so it is the return on foreign
# direct investment, not a bond yield. Ethiopia's own statistics cannot
# measure it: the NBE balance of payments records investment income as
# interest and dividends only, with dividends near zero (NBE Quarterly
# Bulletin 2024/25 Q4, Table 5.1) and no reinvested earnings, and the IMF
# series stops splitting direct-investment income after 2005. We therefore
# measure it on Ethiopia's peers: direct-investment income paid abroad over
# direct-investment liabilities (IMF Balance of Payments and International
# Investment Position statistics, BOP DB_T.D_F_D4P over IIP L_P.D, pooled
# over 2015-2024) for fourteen low- and lower-middle-income African
# countries that report both, reinvested earnings included
# (ogeth/data/imf_fdi_income_africa_peers.csv). The pooled nominal dollar
# return is 6.1 percent, between 4.0 and 8.8 in every year (Kenya 7.1,
# Tanzania 5.5, Uganda 3.9, Rwanda 4.6); less US inflation, as for the
# external-debt rate, that is 4 percent real
# (scripts/derive_sourced_values.py).
WORLD_INT_RATE_ANNUAL = 0.04

# Long-run real interest rate on market debt (r_gov in the steady state).
# Concessional debt is modelled at its own rate, so r_gov prices market debt
# only, and its long-run value is the DSA's terms for new domestic borrowing:
# 4.9 percent real on medium- and long-term debt and 4.5 on short-term, the
# mean 4.7, which the DSA's average real rate on domestic debt reaches by
# 2036 (4.6). Its 2045 value (6.6) is not used: by then the DSA has public
# debt down to 13.6 percent of GDP and a thin domestic market, while the
# model holds debt at its FY2024/25 ratio. With external debt in the
# FY2024/25 stock near -1 percent real, the blended rate is near the 2
# percent long-run effective rate on all public debt this calibration used
# before external debt was modelled at its own rate.
LONG_RUN_R_GOV = round(sum(DSA_MARGINAL_DOMESTIC_REAL_RATES) / 200, 4)
# OG-Core clips the sovereign rate at ``r_gov_floor`` (a parameter since
# PSLmodels/OG-Core#1203). The shift path targets the measured real rate on
# market debt, -11.0 percent in FY2024/25, so the floor sits below it.
R_GOV_FLOOR = -0.12
# Steady-state return on capital the r_gov_shift path is evaluated at: the
# solved baseline steady state of examples/run_og_eth.py.
R_SS_FOR_R_GOV = 0.039962

# Informality. 77.65 percent of the employed are in informal employment
# (ILOSTAT harmonised series, SDG 8.3.1, from the 2021 Labour Force and
# Migration Survey; 84.4 percent in agriculture, 65.9 outside it). The
# informality calibration (taxes.md) grades income-tax compliance by
# lifetime-income group, with lifetime income as the proxy for formality:
# the formal 22.35 percent are the richest households, so groups 5-7 (the
# top 20 percent) are formal and group 4 is 23.5 percent formal.
INFORMAL_EMPLOYMENT_SHARE = 0.7765
LAMBDAS = [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]


def formal_share_by_group(informal_share, lambdas=LAMBDAS):
    """
    Share of each lifetime-income group that is formal when the formal
    share of the population sits at the top of the lifetime-income
    distribution.

    Args:
        informal_share (scalar): informal share of the employed
        lambdas (array_like): population shares of the groups, poorest
            first

    Returns:
        list: formal share of each group, between 0 and 1
    """
    lam = np.asarray(lambdas, dtype=float)
    top = np.cumsum(lam[::-1])[::-1]  # share at or above each group's floor
    formal = 1.0 - float(informal_share)
    above = top - lam  # share strictly above each group
    share = np.clip((formal - above) / lam, 0.0, 1.0)
    return [round(float(x), 4) for x in share]


# The FY2024/25 compliance shape, held: the formal share of each group from
# INFORMAL_EMPLOYMENT_SHARE.
NONCOMPLIANCE = [
    round(1 - x, 4) for x in formal_share_by_group(INFORMAL_EMPLOYMENT_SHARE)
]
# With the statutory schedule as the tax function, the compliance vectors
# above give the *shape* of the formal tax boundary and this scale its
# level: even the top percent remits only part of the tax the schedule
# implies, because much of its income is business and self-employment income
# outside Schedule A. The scale is calibrated so that personal income tax
# collects PIT_REVENUE_TARGET of GDP in FY2024/25
# (examples/tune.py, scored on the first transition year).
COMPLIANCE_SCALE = 0.056434
PIT_REVENUE_TARGET = PIT_REVENUE_SHARE

# Pension coverage by lifetime-income group. Ethiopia's two schemes (PSSSA for
# public servants, POESSA for private formal employees) cover only formal
# employment, a small share of the labour force; the same formality proxy is
# used, so the informal groups draw no public pension.
PENSION_COVERAGE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
# The public pension is a defined-benefit scheme (Proclamations 1267/2022 for
# public servants and 1268/2022 for private employees): retirement at 60
# after at least ten years of service, a benefit of 30 percent of the average
# salary of the last three years plus 1.25 percent for every year of service
# beyond ten, capped at 70 percent. A full career from 20 to 60 gives 67.5
# percent. OG-Core's defined-benefit formula is yr_contrib * alpha_db times
# average earnings over avg_earn_num_years, so alpha_db is the replacement
# rate per year of contribution.
PENSIONS_ON = True
LEGAL_RETIREMENT_AGE = 60
# OG-Core counts ages as years of life (households are in the economy at
# "ages 21 to 100", so the first model age is someone aged 20), and its
# retirement index is round((retirement_age - starting_age) S / 80) - 1:
# retirement_age R makes the R-th year of life, age R - 1, the first with a
# pension. A legal retirement age of 60 is therefore entered as 61.
PENSION_RETIREMENT_AGE = LEGAL_RETIREMENT_AGE + 1
PENSION_MIN_YEARS = 10
PENSION_BASE_REPLACEMENT = 0.30
PENSION_REPLACEMENT_PER_YEAR = 0.0125
PENSION_MAX_REPLACEMENT = 0.70
PENSION_AVERAGING_YEARS = 3
PENSION_CAREER_YEARS = 40
# The coverage vector above says which groups draw a pension; the scale
# multiplies it so that pension outlays match PENSIONS_SHARE_OF_GDP. The
# formal groups hold 10 percent of households but 40 percent of labor income,
# far more than the scheme's 800,000 pensioners and two percent of the
# working-age population, so the scale is well below one. Calibrated by
# examples/tune.py, scored on the first transition year.
PENSION_COVERAGE_SCALE = 0.097098
# Contributions: 7 percent from the employee and 11 from the employer
# (Proclamations 1267/1268 of 2022). OG-Core's payroll tax can differ by
# lifetime-income group, so the effective rate is the statutory rate on the
# pension-covered groups (the coverage shape above) times a contribution
# scale, zero for the informal groups. The scale is calibrated to
# contributions: about 69 billion birr in FY2024/25, 0.29 percent of rebased
# GDP (POESSA about 29 billion, Reporter 2025; PSSSA about 40 billion,
# extrapolated from 30.5 billion collected in 2023, Addis Fortune). The
# contributions sit inside the 117.8 billion "other" non-tax line of the
# budget outturn. The model's formal groups earn about 40 percent of labor
# income, far more than the real covered wage bill, so the scale is well
# below the pension coverage scale. Contribution data are the least certain
# fiscal anchor here (no official FY2024/25 scheme accounts were found).
PAYROLL_STATUTORY_RATE = 0.18
PAYROLL_REVENUE_SHARE = round(
    69.0 / NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"], 5
)
PAYROLL_COVERAGE_SCALE = 0.097527

# Personal income tax: the statutory employment-income schedule (Schedule A)
# in birr per month, as (upper bound, marginal rate) with an open last
# bracket. Proclamation 979/2016 applies in FY2024/25 (model period 0) and
# Proclamation 1395/2025, in force from July 2025, from FY2025/26 on.
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
# Capital income is taxed under Schedule D at flat rates, 10 percent on
# dividends and 5 percent on interest; the model uses the dividend rate.
CAPITAL_INCOME_TAX_RATE = 0.10
# The model's birr are FY2024/25 birr, the start year and the rebased base
# year: mean_income_data is household factor income per adult of that year,
# rebased GDP per adult (24,031 billion birr over the adult population in
# the middle of the fiscal year, World Bank population and age structure,
# 2024, grown half a year) times the model's household income share of GDP
# (MODEL_INCOME_SHARE_OF_GDP).
# Proclamation 979/2016 applies as written in FY2024/25; the thresholds of
# Proclamation 1395/2025, written in FY2025/26 birr, are deflated by the
# rebased growth of nominal GDP per head between the two years
# (31,934 / 24,031, less population growth) so that they bite at the same
# real incomes. The population is the UN figure the model's demographics use
# (the Ministry of Finance uses 110.6 million).
POPULATION_2024 = 132.06e6
ADULT_SHARE_2024 = (
    0.4986  # aged 20 and over: 1 - 0.3906 (0-14) - 0.1108 (15-19)
)
POPULATION_GROWTH = 0.026
# income grid, birr per year, on which the Gouveia-Strauss form is fitted to
# the schedule: from the old exemption threshold to well past the top decile
# of wage earners (55,000 birr a month)
TAX_FIT_INCOME_RANGE = (12_000.0, 5_000_000.0)
# OG-Core caps the third Gouveia-Strauss parameter; a flat rate is the limit
# of a very large one
GS_FLAT_PHI2 = 2.0e4


def _held(values):
    """The FY2024/25 and FY2025/26 values, the second held to the long run."""
    values = list(values)
    return values + [values[-1]] * (FISCAL_PATH_LENGTH - len(values))


def spending_paths():
    """
    Government spending ratios mapped onto the model's three spending
    instruments: the FY2024/25 outturn and the enacted FY2025/26 budget,
    held thereafter.

    Government consumption is recurrent spending net of interest (which the
    model pays through r_gov) and of cash transfers; transfers are the cash
    items in CASH_TRANSFERS; public investment is capital expenditure. The
    sum reproduces the IMF's primary expenditure to within a tenth of a
    percent of GDP in both years.

    Returns:
        dict: alpha_G, alpha_T, alpha_I, alpha_FA as lists of length
            FISCAL_PATH_LENGTH (OG-Core holds the last value thereafter)
    """
    rec = np.array(IMF_RECURRENT) / 100
    interest = np.array(IMF_INTEREST) / 100
    tr = np.array(CASH_TRANSFERS) / 100
    g = rec - interest - tr
    if PENSIONS_ON:
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
    Effective consumption-tax rate and CIT collections factor, held at their
    FY2024/25 values.

    Args:
        tau_c_0 (scalar): FY2024/25 effective consumption-tax rate
        cit_factor_0 (scalar): FY2024/25 CIT collections adjustment factor

    Returns:
        dict: tau_c (list of [rate] per period, I = 1) and
            adjustment_factor_for_cit_receipts, each of length
            FISCAL_PATH_LENGTH
    """
    n = FISCAL_PATH_LENGTH
    return {
        "tau_c": [[float(tau_c_0)]] * n,
        "adjustment_factor_for_cit_receipts": [float(cit_factor_0)] * n,
    }


def compliance_paths(compliance_scale=None, pension_coverage_scale=None):
    """
    Income-tax non-compliance (labor and capital, identical), held at its
    FY2024/25 calibration, and the pension-coverage matrix.

    Args:
        compliance_scale (scalar): multiplies the compliance (one minus
            non-compliance) of every group; defaults to COMPLIANCE_SCALE
        pension_coverage_scale (scalar): multiplies PENSION_COVERAGE;
            defaults to PENSION_COVERAGE_SCALE

    Returns:
        dict: labor_income_tax_noncompliance_rate,
            capital_income_tax_noncompliance_rate (lists of J-vectors of
            length FISCAL_PATH_LENGTH) and replacement_rate_adjust
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
    Effective and marginal rates of a monthly bracket schedule at given
    annual incomes.

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
    Fit OG-Core's Gouveia-Strauss form to a statutory bracket schedule by
    least squares on the effective and marginal rates over a log-spaced
    income grid.

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
    Rebased GDP per adult in FY2024/25 birr, the mean_income_data that
    converts the
    model's income units into the birr the tax schedule is written in.

    Returns:
        float: birr per year
    """
    adults = POPULATION_2024 * (1 + POPULATION_GROWTH / 2) * ADULT_SHARE_2024
    return NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"] * 1e9 / adults


# OG-Core's factor converts model income into birr as mean_income_data over
# the model's mean household income, r_p b + w e n per person. The data
# counterpart is therefore household factor income per adult, not GDP per
# adult: GDP per adult times the model's household income over GDP at market
# prices, from the solved steady state (examples/tune.py computes it each
# round; the converged value is folded here like the other scales).
MODEL_INCOME_SHARE_OF_GDP = 0.861073


def model_income_share(ss, p):
    """
    The model's mean household income, r_p b + w e n per person, over GDP
    at market prices (output plus the consumption tax), in a solved steady
    state.

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
    OG-Core's mean_income_data: household factor income per adult in
    FY2024/25 birr, GDP per adult times the model's household income share.

    Args:
        share (float): household income over GDP; defaults to
            MODEL_INCOME_SHARE_OF_GDP

    Returns:
        float: birr per year
    """
    if share is None:
        share = MODEL_INCOME_SHARE_OF_GDP
    return float(mean_income_per_adult() * share)


def statutory_tax_params():
    """
    Income-tax function parameters from the statutory schedules: the
    FY2024/25 schedule in the first period and the 2025 schedule (deflated
    to FY2024/25 birr) thereafter for effective and labor marginal rates,
    a flat
    dividend rate on capital income, and the birr-per-model-unit anchor.

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
    Effective payroll tax by lifetime-income group: the statutory pension
    contribution rate on the covered groups times the contribution scale.

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


# Cash transfers by programme, percent of GDP, long run (see CASH_TRANSFERS):
# the Productive Safety Net Program goes to the poorest households, public
# pensions to retired formal-sector workers, the fertilizer subsidy to farming
# households. OG-Core's eta matrix allocates aggregate transfers across ages
# and lifetime-income groups; the default is population-proportional.
# published percent of old-base GDP, rebased with the FY2024/25 factor.
# The figure matches the whole programme rather than the government's share
# of it: PSNP5 costs US$2.28 billion over 2021-2025 (World Bank PAD P172479,
# 2020), about US$460 million a year, 0.35-0.4 percent of old-base GDP at the
# floated exchange rate, of which the government finances about 30 percent.
# The urban safety net (Urban Productive Safety Net and Jobs Project, 1.7
# million people in 88 cities, about 0.05 percent of GDP) has the same
# design and is counted within this envelope.
PSNP_SHARE_OF_GDP = rebased([0.4])[0]
# PSNP's two components (PSNP IV Programme Implementation Manual, 2014,
# section 3.5; PSNP5 PAD P172479, 2020, para. 38 and footnote 59): a
# household with at least one able-bodied adult (over 16, not chronically
# ill or disabled) is a public works client and receives transfers for 6
# months a year in exchange for work; a household with none receives
# Permanent Direct Support, unconditional transfers for 12 months a year.
# The monthly rate per person is the same in both (15 kg of cereal and 4 kg
# of pulses, or its cash value), so a direct-support client receives twice
# as much a year. The model has no households or disability, so direct
# support is proxied by age: the poorest group's members from 60, the usual
# definition of an older person in Ethiopian social protection. That gives
# direct support 19 percent of PSNP's value in 2025, against about 26
# percent in the programme (15 percent of 8 million clients at 12 months
# against 85 percent at 6); the difference is disabled and chronically ill
# clients under 60. The cap of five transfers per household is not
# represented.
PSNP_DIRECT_SUPPORT_AGE = 60
PSNP_DIRECT_SUPPORT_MONTHS = 12
PSNP_PUBLIC_WORKS_MONTHS = 6
# old-age pensions, paid by OG-Core's pension system (the pensions/Y target),
# and survivor and invalidity benefits, paid as cash transfers to the
# households of the covered groups (see CASH_TRANSFERS)
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


def transfer_eta(omega_SS, lambdas, retire_age_index, first_age=20):
    """
    Allocation matrix eta distributing aggregate transfers across households,
    shaped (S, J): the safety net within PSNP_GROUPS, twice as much per
    person from PSNP_DIRECT_SUPPORT_AGE (direct support) as below it
    (public works), the
    fertilizer subsidy per capita within FARM_GROUPS, and pensions per
    capita among the retired (age index >= retire_age_index) in
    PENSION_GROUPS, each programme weighted by its share of GDP.

    Args:
        omega_SS (array_like): population distribution, (S, J)
        lambdas (array_like): population shares of the J groups
        retire_age_index (int): model age index of retirement
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
    if not PENSIONS_ON:
        # pensions inside the cash-transfer ratio, to the retired formal
        # groups; with the pension system on they are paid as benefits
        eta += spread(
            PENSION_GROUPS, range(retire_age_index, S), PENSIONS_SHARE_OF_GDP
        )
    return eta / eta.sum()


def allocation_path(allocation, omega, omega_SS, tol=1e-12):
    """
    An allocation matrix for each year of the transition. OG-Core pays
    household (s, j) in year t the share eta[t, s, j] of an aggregate,
    divided by its population omega[t, s, j]; held at its steady-state
    value while the population ages, eta pays each person in a group
    that is larger in the steady state than today more than the
    programme's rules give (about 11 times the safety net per person for
    the poorest group at 85 in 2025). Each year's matrix is instead built
    from that year's population, so per-person amounts follow the
    programme's rules every year.

    Args:
        allocation (callable): maps a population distribution, (S, J), to
            an allocation matrix, (S, J), summing to one
        omega (array_like): population distribution by year, (T+S, S, J)
        omega_SS (array_like): steady-state population distribution,
            (S, J)
        tol (scalar): years at the end of the path whose matrix is within
            tol of the steady-state one are dropped

    Returns:
        Numpy array: (n + 1, S, J), the matrices of the first n years and
            then the steady-state matrix exactly, which OG-Core holds for
            the remaining years and uses in the steady state
    """
    omega = np.asarray(omega, dtype=float)
    eta_SS = allocation(np.asarray(omega_SS, dtype=float))
    path = np.array([allocation(omega_t) for omega_t in omega])
    n = path.shape[0]
    while n > 0 and np.abs(path[n - 1] - eta_SS).max() < tol:
        n -= 1
    return np.concatenate([path[:n], eta_SS[np.newaxis]])


def transfer_eta_path(
    omega, omega_SS, lambdas, retire_age_index, first_age=20
):
    """
    ``transfer_eta`` for each year of the transition (see
    ``allocation_path``).

    Returns:
        Numpy array: (n + 1, S, J), each year summing to one
    """
    return allocation_path(
        lambda om: transfer_eta(om, lambdas, retire_age_index, first_age),
        omega,
        omega_SS,
    )


def derived_transfer_eta(p):
    """The transfer allocation path for the demographics in ``p``."""
    retire_idx = int(np.asarray(p.retirement_age).flatten()[0]) - int(p.E)
    return {
        "eta": transfer_eta_path(
            p.omega, p.omega_SS, p.lambdas, retire_idx, int(p.E)
        ).tolist()
    }


# Net family transfers (OG-Core's phi_FT): working-age members of each
# lifetime-income group support the group's elders. Ethiopia's National
# Transfer Accounts (2005) have consumption, labor income and the lifecycle
# deficit but no transfer profiles (no African account in the NTA database
# has a private-transfer age profile), so the transfers follow a rule with
# one size, set to the NTA consumption profile:
# * givers are ages FAMILY_GIVER_AGES, each in proportion to their earnings
#   capacity (population times the earnings matrix e); receivers are ages
#   FAMILY_RECEIVER_AGE and above, equal per person;
# * each group's transfers net to zero within the group, and the group's
#   share of the total is its share of givers' earnings capacity;
# * a group's elders get, from the family, the group's share less the
#   government cash transfers they already receive (PSNP direct support
#   and the subsidies in eta); old-age pensions, about 0.075 percent of GDP,
#   are not netted;
# * per elder, the transfer is held fixed relative to GDP per person, so
#   the total grows with the elderly share of the population and is
#   FAMILY_TRANSFER_SHARE_SS of the model's output in the steady state.
# Like OG-Core's other shares (alpha_T, alpha_RM), phi_FT multiplies the
# model's output Y, which excludes the consumption tax: 0.17 of Y is 16.5
# percent of GDP at market prices (MODEL_PER_MARKET_GDP). At that size the
# steady state's consumption of people aged 65 and over, relative to ages
# 30-49, is close to NTA's private consumption
# (data/nta_consumption_by_age.csv, targets.nta_old_age_consumption): 0.88
# against 0.87 at 65-69, 0.90 against 0.84 at 70-79 and 0.86 against 0.80
# at 80-85 in the calibrated baseline (0.45 at 70-79 without family
# transfers). People aged 65 and over are 29.5 percent of adults in the
# steady state, 6.5 percent in 2025.
FAMILY_GIVER_AGES = (25, 65)
FAMILY_RECEIVER_AGE = 65
FAMILY_TRANSFER_SHARE_SS = 0.17


def family_transfers(omega, e, eta, alpha_T, total_share, first_age=20):
    """
    Net family transfer shares of GDP for one period (see the rule above).

    Args:
        omega (array_like): population distribution, (S, J)
        e (array_like): earnings matrix, (S, J)
        eta (array_like): transfer allocation matrix, (S, J)
        alpha_T (scalar): government transfers as a share of GDP
        total_share (scalar): family transfers received, share of GDP,
            before netting the government transfers to elders
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
    Net family transfer shares of GDP for each year of the transition, from
    that year's population, transfer matrix and transfer share, ending with
    the steady-state matrix (as ``allocation_path``).

    Args:
        omega (array_like): population distribution by year, (T+S, S, J)
        omega_SS (array_like): steady-state population distribution, (S, J)
        e (array_like): earnings matrix, (S, J)
        eta (array_like): transfer matrices by year, (T+S, S, J)
        alpha_T (array_like): government transfers, share of GDP, length T+S
        first_age (int): age of the first model age (OG-Core's E)
        tol (scalar): trailing years within tol of the steady state are
            dropped

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
    """
    The net family transfer path for the demographics and earnings in
    ``p``, the transfer matrices ``eta`` and transfer shares ``alpha_T``.
    """
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
    Annual path, FY2024/25 onward, of a series published at the DSA's years
    (DSA_YEARS), linearly interpolated between them.

    Args:
        values (array_like): the series at DSA_YEARS
        last_year (int): last fiscal year (ending) of the path; defaults to
            the last DSA year

    Returns:
        Numpy array: one value per year from FY2024/25 to last_year
    """
    last_year = DSA_YEARS[-1] if last_year is None else last_year
    years = np.arange(DSA_YEARS[0], last_year + 1)
    return np.interp(years, DSA_YEARS, values)


def external_debt_path():
    """
    Government debt contracted abroad as a share of OG-Core's Y
    (alpha_D_ext), one value per year from FY2024/25: the FY2024/25 stock
    (EXTERNAL_DEBT_RATIO) moved with the DSA's public external debt ratio to
    2045, then down to zero at the DSA's 2036-2045 pace, after which OG-Core
    holds the last value (zero).

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
    Real interest rate on external debt (r_D_ext_annual), one value per year
    from FY2024/25 to 2045, held after: the DSA's average nominal rate on
    external debt less US inflation.

    Returns:
        list: r_D_ext_annual path
    """
    return [
        round(float(x) / 100 - US_INFLATION, 6)
        for x in dsa_annual(DSA_EXTERNAL_RATE)
    ]


def external_debt_params():
    """
    OG-Core's external-debt parameters and the market-debt part of the
    initial debt.

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
    Real interest rate on market debt that r_gov targets, one value per year:
    the DSA's average real rate on domestic debt from FY2024/25 to 2036, then
    LONG_RUN_R_GOV.

    Returns:
        Numpy array: targets, FY2024/25 to 2036 plus the long-run value
    """
    path = dsa_annual(DSA_DOMESTIC_REAL_RATE, last_year=2036) / 100
    return np.append(path, LONG_RUN_R_GOV)


def market_r_gov_shift_path(r_gov_scale, r_gov_DY2, debt_ratio_ss, r_ss):
    """
    Level-shift path for the sovereign rate on market debt, r_gov =
    r_gov_scale r - shift + premium, that puts r_gov on market_rate_targets
    with the debt-elastic premium centred on the long-run debt anchor
    debt_ratio_ss. OG-Core evaluates the premium at all government debt,
    market plus external; external debt is repaid by the late 2040s, so the
    long-run anchor is all market debt. OG-Core adds r_gov_DY d +
    r_gov_DY2 d^2 with r_gov_DY = -2 r_gov_DY2 D, which equals
    r_gov_DY2 (d - D)^2 - r_gov_DY2 D^2; the shift absorbs the constant,
    so the premium prices only deviations of debt from the anchor. The
    market return r is approximated by its steady-state value.

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
    Foreign share of new market debt: non-residents' share of birr
    government securities (MARKET_FOREIGN_DEBT_SHARE, zero). New borrowing
    abroad is modelled through alpha_D_ext.

    Returns:
        list: same length as the spending paths (FISCAL_PATH_LENGTH)
    """
    return [MARKET_FOREIGN_DEBT_SHARE] * FISCAL_PATH_LENGTH


# ---------------------------------------------------------------------------
# Capital stocks, depreciation and initial household wealth
#
# Total capital. Penn World Table 11.0 (October 2025) puts the capital stock
# at current PPPs over output, cn / cgdpo, at 2.167 in 2023, its latest year
# (2.17 to 2.26 over 2020-2023), measured against the old national accounts.
# The rebased accounts have not produced capital stocks or back series yet,
# and they revise gross capital formation up far more than GDP: 35.3 against
# 20.1 percent of GDP in FY2024/25 (179 percent in birr), which the Ministry
# attributes to investment by regional state-owned enterprises and non-profit
# institutions that the old accounts missed. The old accounts themselves had
# investment at 35-38 percent of GDP in 2013-2017, falling to 20 percent by
# FY2024/25, so the undercount built up after the old base year rather than
# running through the whole history.
#
# We therefore rebuild the stock by perpetual inventory from PWT's investment
# and GDP at constant national prices (ogeth/data/pwt110_eth_capital.csv),
# extended to FY2024/25 with the old-base investment share and real growth,
# and add the investment the old accounts missed, phased in linearly from
# zero in FY2015/16, the old base year. The calculation runs in units of
# old-base GDP: rebased investment, 35.3 percent of rebased GDP, is 56.2
# percent of old-base GDP, so the gap is 36.1 points of old-base GDP in
# FY2024/25. The ratio of the revised to the unrevised stock-output ratio
# scales PWT's 2023 net capital stock over GDP at current national prices
# (2.37, ogeth/data/pwt110_eth_capital_detail.csv), and the result is
# converted to rebased GDP: 2.19, in a range of 1.39 (no revision) to 2.94
# (the gap in every year since 1970). scripts/derive_sourced_values.py
# reproduces these numbers. They should be replaced by the Ministry's capital
# stocks or back series when published.
PWT_CAPITAL_OUTPUT_RATIO = 2.3655
REBASING_INVESTMENT_GAP = 0.361
TOTAL_CAPITAL_OUTPUT_RATIO = 2.186
# Public capital: IMF Investment and Capital Stock Dataset (api.imf.org,
# IMF.FAD ICSD), general-government capital stock at current prices over GDP
# at current prices, CAPSTCK_S13_V_XDC / B1GQ_V_XDC = 2,265.27 / 2,696.22
# billion birr = 0.840 in 2019, its latest year (the IMF-World Bank PIMA of
# June 2025 quotes the same 84 percent), held at that ratio to FY2024/25.
# The ratio at domestic current prices matches the total stock above; the
# real ratio at 2017 prices (CAPSTCK_S13_Q_POGDP_PT, 0.667) was used before.
# The stock is accumulated budget capital spending, in birr, which rebasing
# does not change, so its ratio converts like every other fiscal stock
# (times 0.628).
PUBLIC_CAPITAL_OUTPUT_RATIO_OLD_BASE = round(2265.27 / 2696.22, 3)
PUBLIC_CAPITAL_OUTPUT_RATIO = round(
    PUBLIC_CAPITAL_OUTPUT_RATIO_OLD_BASE * rebase_factors()[0], 3
)
# The private stock is the total less public capital.
PRIVATE_CAPITAL_OUTPUT_RATIO = round(
    TOTAL_CAPITAL_OUTPUT_RATIO - PUBLIC_CAPITAL_OUTPUT_RATIO, 3
)
# Depreciation of private capital: PWT 11.0's depreciation rate of the total
# stock, 0.0414 in 2023, less public capital's share depreciating at
# delta_g = 0.02 (OG-Core's default public-capital rate, kept), so that
# delta_p (K - K_g) + delta_g K_g = 0.0414 K.
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
# Foreign-owned capital. Ethiopia publishes no international investment
# position. UNCTAD's inward FDI stock (World Investment Report 2026: USD 42.5
# billion at end-2024) is essentially the cumulative inflows at historical
# cost (IMF balance of payments, direct-investment equity liabilities,
# 1990-2024: USD 40.7 billion with the unreported 1995-2002 filled at 0.25
# billion a year), with no depreciation. The model's foreign-owned capital is
# a net stock at replacement cost like its K, so we accumulate the same
# inflows by perpetual inventory at the private depreciation rate, revalued
# by 2.5 percent a year for the dollar price of capital goods
# (ogeth/data/imf_fdi_inflows_eth.csv; scripts/derive_sourced_values.py):
# USD 34.0 billion in mid-FY2024/25, 0.169 of rebased GDP of USD 201.4
# billion. Unrecorded reinvested earnings (Ethiopia reports none) bias both
# measures down.
FDI_STOCK_USD_BN = 33.98
REBASED_GDP_USD_BN = 201.42
FDI_STOCK_TO_GDP = round(FDI_STOCK_USD_BN / REBASED_GDP_USD_BN, 4)
# foreign-owned share of private capital in FY2024/25, the zeta_K target
FOREIGN_CAPITAL_SHARE = round(
    FDI_STOCK_TO_GDP / PRIVATE_CAPITAL_OUTPUT_RATIO, 4
)
# Domestically held general-government debt, FY2024/25
DOMESTIC_DEBT_TO_GDP = round(IMF_DOMESTIC_DEBT[0] / 100, 4)
# OG-Core (0.22 and later, PSLmodels/OG-Core#1254) anchors aggregate
# household wealth in the first period of the transition to a multiple of
# that period's GDP. Household wealth in the model is domestically owned
# private capital plus domestically held government debt, so the data
# counterpart is private K/Y less the foreign-owned stock plus domestic debt,
# converted to the model's output basis.
INITIAL_WEALTH_RATIO = round(
    to_model(
        PRIVATE_CAPITAL_OUTPUT_RATIO - FDI_STOCK_TO_GDP + DOMESTIC_DEBT_TO_GDP
    ),
    3,
)
# Efficiency of public investment: the IMF-World Bank Public Investment
# Management Assessment for Ethiopia (June 2025, section II) estimates a
# hybrid efficiency gap of 53 percent (access and quality of infrastructure
# for the capital stock per head, against the efficiency frontier). It enters
# once: as OG-Core's leakage of new public investment
# (infra_investment_leakage_rate) and as the share of the measured stock
# that is productive at the start. It does not scale the elasticity gamma_g,
# which multiplies the productive stock.
PUBLIC_INVESTMENT_EFFICIENCY_GAP = 0.53
INITIAL_KG_RATIO = round(
    to_model(
        PUBLIC_CAPITAL_OUTPUT_RATIO * (1 - PUBLIC_INVESTMENT_EFFICIENCY_GAP)
    ),
    3,
)

# The output elasticity of public capital, gamma_g. Bom and Ligthart (2014,
# Journal of Economic Surveys 28(5), a meta-analysis of 578 estimates from 68
# studies) find 0.083 in the short run and 0.122 in the long run for capital
# supplied by central government, 0.106 on average. We take the low end,
# 0.08, because OG-Core passes public capital's rent to the owners of private
# capital (aggregates.get_r_p), so their return exceeds the firms' marginal
# return by about (1 - tau_b) gamma_g Y/K: at Ethiopia's capital-output ratio
# a larger gamma_g would push the firms' return below the 4 percent world
# rate (WORLD_INT_RATE_ANNUAL), where foreign capital would leave.
GAMMA_G = 0.08
# The share of reproducible private capital, gamma. Ethiopia publishes no
# income-side GDP, so the capital share cannot be measured directly: the
# IFPRI 2022 SAM books the self-employed's mixed income as capital (58
# percent of value added), and the ILO labour share (0.385) is an imputation
# that values self-employed work at employee earnings, which in Ethiopia
# fall far below the self-employed's value added per worker. We therefore set
# gamma from the firms' first-order condition at a target marginal return,
# r = (1 - tau_b) gamma Y/K - delta + tau_b delta_tau, at the measured
# capital stock (the fallback when profits cannot be measured; Caselli and
# Feyrer 2007). The target firms' return is 5 percent, just above the world
# rate; with public capital's rent, owners of capital earn about 9.5 percent,
# inside the 6-10 percent range of the evidence (realized FDI returns in
# African peers of about 4 percent real after tax, plus the 2-4 point
# marginal premium; Caselli and Feyrer's natural-capital-corrected returns).
# OG-Core's labour share, 1 - gamma - gamma_g, is then household income from
# work, land and enterprise counted as labour, inside Gollin's (2002) range
# of 0.65-0.80 for labour shares that count the self-employed's income.
FIRM_RETURN_TARGET = 0.05
# effective business tax rate on capital income, FY2024/25: CIT rate 0.30 x
# corporate share of assets 0.55 x the CIT collections factor, the tuned
# value in the packaged parameters (examples/tune.py matches CIT/Y). gamma
# and the factor depend on each other (profits set collections), so the
# factor is folded here after each retune; tests check the two agree.
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
# The baseline
#
# Every fiscal and external input takes its measured FY2024/25 value or the
# enacted FY2025/26 budget and is held there, legislated tax changes
# included; debt, growth and the external balance are model outcomes.
#
# Long-run debt ratio: the measured FY2024/25 ratio of all general-government
# debt, so the closure rule stabilizes debt where it starts. OG-Core's
# debt_ratio_ss anchors market debt; external debt is repaid by the late
# 2040s (external_debt_path), so in the long run all debt is market debt and
# the two anchors coincide. Along the way the closure rule refinances the
# falling external stock with market debt.
# ---------------------------------------------------------------------------
DEBT_RATIO_SS = INITIAL_DEBT_RATIO


def fiscal_params(p, r_ss=None):
    """
    The fiscal and external parameters of the baseline, derived from the
    Specifications object's growth and premium settings.

    Args:
        p (Specifications): parameters carrying g_y, g_n, r_gov_scale,
            r_gov_DY2, tau_c, adjustment_factor_for_cit_receipts
        r_ss (scalar): steady-state market return used in the shift path;
            defaults to R_SS_FOR_R_GOV

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
    The statutory income tax, the pension scheme and its contributions,
    capital, depreciation, preferences, the initial conditions and the
    solver settings.

    Returns:
        dict: ready for Specifications.update_specifications
    """
    out = {}
    out.update(statutory_tax_params())
    if PENSIONS_ON:
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
    The fiscal, external and remittance parameters of the baseline: what
    update_baseline.py writes into the packaged JSON.

    Args:
        p (Specifications): see fiscal_params
        r_ss (scalar): see fiscal_params

    Returns:
        dict: ready for Specifications.update_specifications
    """
    out = fiscal_params(p, r_ss)
    out.update(derived_remittance_params(p))
    return out
