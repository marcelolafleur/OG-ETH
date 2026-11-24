"""
Labor supply targets (LFMS 2021 hours per person, ESS 2021/22 shape past
65) and the chi_n update used by examples/tune.py (households.md).
"""

import os

import numpy as np
import pandas as pd
from ogcore import household

from ogeth import income

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_PATH, "data")

# OG-USA's convention: labor supply is the share of a 16-hour waking day,
# seven days a week
WEEKLY_TIME_ENDOWMENT = (24 - 8) * 7
# LFMS band midpoints; 65+ is not used as a point (older_hours_decline)
HOURS_BAND_MIDPOINTS = {
    "10-14": 12,
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
    "65+": 67,
}
# age from which the target follows the exponential decline fitted to ESS
OLDER_HOURS_START_AGE = 65
# the target never falls below this share of the time endowment
MIN_LABOR_SUPPLY = 0.02
# ages over which examples/tune.py judges the fit of chi_n
FIT_AGES = (20, 80)
# OG-Core's upper bound on chi_n
MAX_CHI_N = 10000.0


def hours_per_person():
    """
    LFMS 2021 weekly hours per person by age band (employment ratio times
    hours of the employed).

    Returns:
        df (Pandas DataFrame): age_band, emp_pop_ratio,
            mean_weekly_hours_employed, hours_per_person
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "lfms2021_hours_by_age.csv"))
    df["hours_per_person"] = df.emp_pop_ratio * df.mean_weekly_hours_employed
    return df


def older_hours_shape():
    """
    ESS wave 5 (2021/22) hours per person by age band past 65, all work,
    non-workers at zero (World Bank Microdata catalog 6161; tabulated by
    og-analysis/eth-ess/scripts/hours_by_age.py).

    Returns:
        df (Pandas DataFrame): age_band, mean_age, n (people),
            hours_per_person, ratio_to_65plus (relative to ESS 65+ mean)
    """
    return pd.read_csv(
        os.path.join(DATA_DIR, "ess2021_older_hours_by_age.csv")
    )


def older_hours_decline():
    """
    Exponential decline of hours past 65: weighted fit of log hours on age
    through the ESS bands, at the LFMS 65+ level.

    Returns:
        (tuple): level (weekly hours per person at OLDER_HOURS_START_AGE)
            and rate (the fall per year of age, as a log difference)
    """
    level65 = (
        hours_per_person().set_index("age_band").loc["65+", "hours_per_person"]
    )
    old = older_hours_shape()
    hours = level65 * old.ratio_to_65plus.to_numpy()
    slope, intercept = np.polyfit(
        old.mean_age.to_numpy() - OLDER_HOURS_START_AGE,
        np.log(hours),
        1,
        w=np.sqrt(old.n.to_numpy()),
    )
    return float(np.exp(intercept)), float(-slope)


def labor_supply_target(E, S, ltilde=1.0):
    """
    Average labor supply by model age, share of the time endowment: LFMS
    bands, then older_hours_decline from 65 (households.md).

    Args:
        E (int): age at which agents become economically active
        S (int): number of model periods in a lifetime
        ltilde (float): time endowment per period

    Returns:
        target (Numpy array): average labor supply by age, length S
    """
    ages = income.model_ages(E, S)
    df = hours_per_person()
    df = df[df.age_band != "65+"]
    level, rate = older_hours_decline()
    x = np.append(
        df.age_band.map(HOURS_BAND_MIDPOINTS).to_numpy(dtype=float),
        OLDER_HOURS_START_AGE,
    )
    y = np.append(df.hours_per_person.to_numpy(), level)
    order = np.argsort(x)
    hours = np.interp(ages, x[order], y[order])
    old = ages >= OLDER_HOURS_START_AGE
    hours[old] = level * np.exp(-rate * (ages[old] - OLDER_HOURS_START_AGE))
    target = hours / WEEKLY_TIME_ENDOWMENT
    return np.maximum(target * ltilde, MIN_LABOR_SUPPLY * ltilde)


def average_labor_supply(n, omega_SS):
    """
    Population-weighted average labor supply by age.

    Args:
        n (Numpy array): labor supply by age and group, size (S, J)
        omega_SS (Numpy array): joint population distribution, size (S, J)

    Returns:
        n_avg (Numpy array): average labor supply by age, length S
    """
    return (n * omega_SS).sum(axis=1) / omega_SS.sum(axis=1)


def marginal_disutility(n, p):
    """
    Marginal disutility of labor with unit weights (OG-Core elliptical).

    Args:
        n (Numpy array): labor supply, length S
        p (OG-Core Specifications object): model parameters

    Returns:
        mdu (Numpy array): marginal disutility at each labor supply
    """
    n = np.asarray(n, dtype=float)
    return household.marg_ut_labor(n, np.ones_like(n), p)


def chi_n_step(chi_n, n_model, n_target, p):
    """
    Rescale chi_n so the labor FOC holds at the target, at fixed
    consumption and prices; capped at MAX_CHI_N.

    Args:
        chi_n (Numpy array): current disutility weights, length S
        n_model (Numpy array): model average labor supply, length S
        n_target (Numpy array): target average labor supply, length S
        p (OG-Core Specifications object): model parameters

    Returns:
        chi_n_new (Numpy array): updated disutility weights, length S
    """
    return np.minimum(
        chi_n
        * marginal_disutility(n_model, p)
        / marginal_disutility(n_target, p),
        MAX_CHI_N,
    )


def steady_state_chi_n(p):
    """
    The steady-state row of chi_n, whatever its stored shape.

    Args:
        p (OG-Core Specifications object): model parameters

    Returns:
        chi_n (Numpy array): disutility weights by age, length S
    """
    chi_n = np.asarray(p.chi_n, dtype=float)
    if chi_n.ndim == 2:
        chi_n = chi_n[-1]
    return chi_n.copy()
