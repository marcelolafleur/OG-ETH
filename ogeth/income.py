"""
Lifetime earnings profiles (the ability matrix e) for OG-ETH: the OG-USA
profiles reshaped over age with NTA labor income per hour and across groups
to WID pre-tax income shares (earnings.md).
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.interpolate as si
from ogcore import parameter_plots as pp

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_PATH, "data")
OGUSA_PARAMS_URL = (
    "https://raw.githubusercontent.com/PSLmodels/OG-USA/master/"
    "ogusa/ogusa_default_parameters.json"
)
# the OG-USA profiles cover ages 20 to 99 in one-year steps
OGUSA_E = 20
OGUSA_S = 80

# WID pre-tax national income (equal-split adults), latest Ethiopian year
WID_YEAR = 2021
WID_SHARE_VARIABLE = "sptincj992"
WID_GINI_VARIABLE = "gptincj992"
# WID net personal wealth shares (equal-split adults); floored because the
# bottom quarter's share is negative and households cannot borrow
WID_WEALTH_VARIABLE = "shwealj992"
WEALTH_SHARE_FLOOR = 0.005
# NTA labor income per capita: Ethiopia's only profile (2005), nearest US
NTA_YEAR = {"ETH": 2005, "US": 2006}
# ILOSTAT five-year age bands at their midpoints; open 65+ band at 70
EMPLOYMENT_BAND_MIDPOINTS = {
    "15-19": 17,
    "20-24": 22,
    "25-29": 27,
    "30-34": 32,
    "35-39": 37,
    "40-44": 42,
    "45-49": 47,
    "50-54": 52,
    "55-59": 57,
    "60-64": 62,
    "65+": 70,
}
# ages over which both profiles are normalized before taking the ratio
REFERENCE_AGES = (20, 65)
# the age factor is held at its 60-64 mean from this age (open 65+ band)
OLDEST_MEASURED_AGE = 65


def load_ogusa_e():
    """
    Download the OG-USA ability matrix and group shares.

    Returns:
        e_usa (Numpy array): OG-USA ability matrix, size (80, 10)
        lambdas_usa (Numpy array): OG-USA lifetime-income group shares
    """
    with urllib.request.urlopen(OGUSA_PARAMS_URL) as response:
        usa = json.load(response)
    e_usa = np.asarray(usa["e"], dtype=float)
    if e_usa.ndim == 3:
        e_usa = e_usa[0]
    return e_usa, np.asarray(usa["lambdas"], dtype=float)


def group_midpoints(lambdas):
    """
    Percentile midpoints of the lifetime-income groups.

    Args:
        lambdas (Numpy array): population share of each group, length J

    Returns:
        midpoints (Numpy array): midpoint of each group in (0, 1)
    """
    upper = np.cumsum(lambdas)
    return upper - 0.5 * np.asarray(lambdas)


def model_ages(E, S):
    """
    Midpoint age of each model period.

    Args:
        E (int): age at which agents become economically active
        S (int): number of model periods in a lifetime

    Returns:
        ages (Numpy array): midpoint age of each period, length S
    """
    step = OGUSA_S / S
    return np.linspace(E + 0.5 * step, E + S - 0.5 * step, S)


def interpolate_usa_e(e_usa, lambdas_usa, E, S, lambdas):
    """
    Map the OG-USA ability matrix onto the model's age and group grid.

    Args:
        e_usa (Numpy array): OG-USA ability matrix, size (80, 10)
        lambdas_usa (Numpy array): OG-USA group shares, length 10
        E (int): age at which agents become economically active
        S (int): number of model periods in a lifetime
        lambdas (Numpy array): model group shares, length J

    Returns:
        e_base (Numpy array): OG-USA profiles on the model grid, size (S, J)
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if (
        S == OGUSA_S
        and E == OGUSA_E
        and lambdas.shape == lambdas_usa.shape
        and np.allclose(lambdas, lambdas_usa)
    ):
        return e_usa.copy()
    midp = group_midpoints(lambdas)
    midp_usa = group_midpoints(lambdas_usa)
    if midp.min() < midp_usa.min() or midp.max() > midp_usa.max():
        raise RuntimeError(
            "One or more entries in abilities vector (lambdas) is outside "
            "the allowable bounds for interpolation."
        )
    ages_usa = model_ages(OGUSA_E, OGUSA_S)
    j_mesh, s_mesh = np.meshgrid(midp_usa, ages_usa)
    points = np.column_stack((s_mesh.ravel(), j_mesh.ravel()))
    new_j_mesh, new_s_mesh = np.meshgrid(midp, model_ages(E, S))
    return si.griddata(
        points, e_usa.ravel(), (new_s_mesh, new_j_mesh), method="linear"
    )


def employment_ratio(country, year, ages):
    """
    ILOSTAT employment-to-population ratio interpolated to single ages.

    Args:
        country (str): "ETH" or "US"
        year (int): data year
        ages (Numpy array): ages to interpolate to

    Returns:
        ratio (Numpy array): employed share of the population at each age
    """
    df = pd.read_csv(
        os.path.join(DATA_DIR, "ilostat_employment_ratio_by_age.csv")
    )
    df = df[(df.country == country) & (df.year == year)]
    if df.empty:
        raise ValueError(f"No ILOSTAT employment ratio for {country} {year}")
    x = df.age_band.map(EMPLOYMENT_BAND_MIDPOINTS).to_numpy(dtype=float)
    order = np.argsort(x)
    return np.interp(ages, x[order], df.emp_pop_ratio.to_numpy()[order])


def nta_labor_income(country, ages):
    """
    NTA smoothed labor income per capita at single ages (held past the end).

    Args:
        country (str): "ETH" or "US"
        ages (Numpy array): ages to interpolate to

    Returns:
        income (Numpy array): labor income per capita at each age
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "nta_labor_income_by_age.csv"))
    df = df[(df.country == country) & (df.year == NTA_YEAR[country])]
    if df.empty:
        raise ValueError(f"No NTA labor income profile for {country}")
    df = df.sort_values("age")
    return np.interp(
        ages, df.age.to_numpy(dtype=float), df.labor_income.to_numpy()
    )


def per_worker_earnings(country, ages):
    """
    Labor income per worker by age: NTA per capita over employment ratio.

    Args:
        country (str): "ETH" or "US"
        ages (Numpy array): ages to evaluate at

    Returns:
        earnings (Numpy array): labor income per employed person by age
    """
    return nta_labor_income(country, ages) / employment_ratio(
        country, NTA_YEAR[country], ages
    )


# Weekly hours of the employed by age: LFMS 2021 (Ethiopia's only age
# profile) and ILOSTAT CPS 2010 (US)
HOURS_YEAR = {"ETH": 2021, "US": 2010}


def hours_employed(country, ages):
    """
    Mean weekly hours of the employed by age (band midpoints; 65+ at 67).

    Args:
        country (str): "ETH" or "US"
        ages (Numpy array): ages to evaluate at

    Returns:
        hours (Numpy array): weekly hours of an employed person
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "hours_employed_by_age.csv"))
    df = df[(df.country == country) & (df.year == HOURS_YEAR[country])]
    if df.empty:
        raise ValueError(f"No hours by age for {country}")
    mid = []
    for band in df.age_band:
        if band.endswith("+"):
            mid.append(float(band[:-1]) + 2.0)
        else:
            lo, hi = band.split("-")
            mid.append((float(lo) + float(hi)) / 2)
    x = np.array(mid)
    order = np.argsort(x)
    return np.interp(
        ages, x[order], df.mean_weekly_hours_employed.to_numpy()[order]
    )


def per_hour_earnings(country, ages):
    """
    Labor income per hour worked by age: per-worker income over weekly
    hours of the employed (labor supply n is hours, so e is pay per hour).

    Args:
        country (str): "ETH" or "US"
        ages (Numpy array): ages to evaluate at

    Returns:
        earnings (Numpy array): labor income per hour worked by age
    """
    return per_worker_earnings(country, ages) / hours_employed(country, ages)


def earnings_age_factor(ages):
    """
    Ethiopia-to-US ratio of labor income per hour by age, each normalized
    over REFERENCE_AGES; held from OLDEST_MEASURED_AGE (earnings.md).

    Args:
        ages (Numpy array): model ages, length S

    Returns:
        factor (Numpy array): multiplicative age factor, length S
    """
    ages = np.asarray(ages, dtype=float)
    ref = (ages >= REFERENCE_AGES[0]) & (ages < REFERENCE_AGES[1])
    eth = per_hour_earnings("ETH", ages)
    usa = per_hour_earnings("US", ages)
    factor = (eth / eth[ref].mean()) / (usa / usa[ref].mean())
    old = ages >= OLDEST_MEASURED_AGE
    hold = (ages >= OLDEST_MEASURED_AGE - 5) & ~old
    if old.any() and hold.any():
        factor[old] = factor[hold].mean()
    return factor


def wid_cumulative_shares(country, variable=WID_SHARE_VARIABLE):
    """
    Cumulative shares at the percentile bounds WID reports.

    Args:
        country (str): "ETH" or "US"
        variable (str): WID series (income default, or wealth)

    Returns:
        cumulative (dict): share of the bottom fraction of adults, keyed by
            the fraction (0.25, 0.5, 0.7, 0.8, 0.9, 0.99, 1.0)
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "wid_pretax_income_2021.csv"))
    df = df[
        (df.country == country)
        & (df.year == WID_YEAR)
        & (df.variable == variable)
    ]
    if df.empty:
        raise ValueError(f"No WID {variable} shares for {country}")
    s = df.set_index("percentile").value
    bottom_70 = s["p0p50"] + s["p50p90"] - s["p70p80"] - s["p80p90"]
    return {
        0.25: s["p0p25"],
        0.50: s["p0p50"],
        0.70: bottom_70,
        0.80: bottom_70 + s["p70p80"],
        0.90: s["p0p50"] + s["p50p90"],
        0.99: s["p0p50"] + s["p50p90"] + s["p90p100"] - s["p99p100"],
        1.00: 1.0,
    }


def wid_group_shares(country, lambdas, variable=WID_SHARE_VARIABLE):
    """
    WID share of each lifetime-income group, of pre-tax income by default.

    Args:
        country (str): "ETH" or "US"
        lambdas (Numpy array): population share of each group, length J
        variable (str): WID series (see wid_cumulative_shares)

    Returns:
        shares (Numpy array): share of each group, length J
    """
    cumulative = wid_cumulative_shares(country, variable)
    bounds = np.round(np.cumsum(lambdas), 4)
    missing = [b for b in bounds if b not in cumulative]
    if missing:
        raise ValueError(
            "WID shares are available at the cumulative percentiles "
            f"{sorted(cumulative)}; lambdas imply {list(bounds)}"
        )
    cum_shares = np.array([cumulative[b] for b in bounds])
    return np.diff(np.concatenate(([0.0], cum_shares)))


def wealth_share_targets(lambdas, country="ETH", floor=WEALTH_SHARE_FLOOR):
    """
    WID wealth share of each group, floored and rescaled to sum to one.

    Args:
        lambdas (Numpy array): population share of each group, length J
        country (str): "ETH" or "US"
        floor (float): smallest share any group is asked to hold

    Returns:
        shares (Numpy array): target wealth share of each group, length J
    """
    shares = wid_group_shares(country, lambdas, WID_WEALTH_VARIABLE)
    low = shares < floor
    shares = shares.copy()
    shares[low] = floor
    shares[~low] *= (1.0 - floor * low.sum()) / shares[~low].sum()
    return shares


def implied_wealth_shares(b, age_wgts):
    """
    Share of aggregate household wealth held by each group.

    Args:
        b (Numpy array): savings by age and group, size (S, J)
        age_wgts (Numpy array): joint population distribution, size (S, J)

    Returns:
        shares (Numpy array): wealth share of each group, length J
    """
    wealth = np.asarray(b) * np.asarray(age_wgts)
    return wealth.sum(axis=0) / wealth.sum()


def implied_group_shares(e, age_wgts):
    """
    Income share of each group implied by e at equal labor supply.

    Args:
        e (Numpy array): ability matrix, size (S, J)
        age_wgts (Numpy array): joint population distribution, size (S, J)

    Returns:
        shares (Numpy array): implied income share of each group, length J
    """
    income = e * age_wgts
    return income.sum(axis=0) / income.sum()


def scale_groups_to_shares(e, age_wgts, group_shares):
    """
    Rescale each group's profile so implied income shares hit the targets.

    Args:
        e (Numpy array): ability matrix, size (S, J)
        age_wgts (Numpy array): joint population distribution, size (S, J)
        group_shares (Numpy array): target shares, length J (normalized)

    Returns:
        e_scaled (Numpy array): rescaled ability matrix, size (S, J)
    """
    target = np.asarray(group_shares, dtype=float).flatten()
    if target.shape[0] != e.shape[1] or (target <= 0).any():
        raise ValueError("group_shares must be J positive income shares")
    target = target / target.sum()
    scale = target / implied_group_shares(e, age_wgts)
    return e * scale.reshape(1, -1)


def get_e_interp(
    E,
    S,
    J,
    lambdas,
    age_wgts,
    group_shares=None,
    plot_path=None,
):
    """
    Build the OG-ETH ability matrix from the OG-USA profiles (earnings.md).

    Args:
        E (int): age at which agents become economically active
        S (int): number of model periods in a lifetime
        J (int): number of lifetime-income groups
        lambdas (Numpy array): population share of each group, length J
        age_wgts (Numpy array): population distribution, (S, J) or (S,)
        group_shares (Numpy array): target income shares, length J;
            None uses the WID shares for Ethiopia
        plot_path (str): directory to save a plot of the profiles to

    Returns:
        emat_new_scaled (Numpy array): ability matrix with weighted mean
            1, size (S, J)
    """
    lambdas = np.asarray(lambdas, dtype=float).flatten()
    age_wgts = np.asarray(age_wgts, dtype=float)
    assert lambdas.shape[0] == J
    if age_wgts.ndim == 1:
        age_wgts = age_wgts.reshape(S, 1) * lambdas.reshape(1, J)
    assert age_wgts.shape == (S, J)
    e_usa, lambdas_usa = load_ogusa_e()
    e_base = interpolate_usa_e(e_usa, lambdas_usa, E, S, lambdas)
    ages = model_ages(E, S)
    e_age = e_base * earnings_age_factor(ages).reshape(S, 1)
    if group_shares is None:
        group_shares = wid_group_shares("ETH", lambdas)
    e_new = scale_groups_to_shares(e_age, age_wgts, group_shares)
    emat_new_scaled = e_new / (e_new * age_wgts).sum()
    if plot_path is not None:
        pp.plot_income_data(
            ages,
            group_midpoints(lambdas),
            lambdas,
            emat_new_scaled,
            path=plot_path,
            filesuffix="_eth",
        )
    return emat_new_scaled


def write_json_parameters(json_path, updates):
    """
    Replace parameters in a packaged JSON file, leaving the rest as is.

    Args:
        json_path (str): path to the JSON file
        updates (dict): parameter name -> JSON-serializable value

    Returns:
        None
    """
    with open(json_path, encoding="utf-8") as f:
        params = json.load(f)
    params.update(updates)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
        f.write("\n")
