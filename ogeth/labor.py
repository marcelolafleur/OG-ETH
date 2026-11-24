"""
Labor supply targets and the chi_n calibration for OG-ETH.

chi_n, the age profile of the disutility of labor, is calibrated so that
the model's average labor supply by age matches hours worked per person
in Ethiopia's 2021 Labour Force and Migration Survey (LFMS), with the
shape within the survey's open 65+ band from the Ethiopia Socioeconomic
Survey wave 5 (ESS, 2021/22). examples/tune.py rescales chi_n between
steady-state solves by the ratio of marginal disutilities at the model's
and the data's labor supply (chi_n_step), which is the exact adjustment
when consumption and prices are held fixed.
"""

import os

import numpy as np
import pandas as pd
from ogcore import household

from ogeth import income

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_PATH, "data")

LFMS_YEAR = 2021
# OG-USA's convention: labor supply is the share of a 16-hour waking day,
# seven days a week
WEEKLY_TIME_ENDOWMENT = (24 - 8) * 7
# LFMS five-year bands placed at their midpoint; the open 65+ band is not
# used as a point (from 65 the target follows older_hours_decline)
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
    LFMS 2021 hours worked per week per person by age band: the
    employment-to-population ratio times the mean weekly hours of the
    employed.

    Returns:
        df (Pandas DataFrame): age_band, emp_pop_ratio,
            mean_weekly_hours_employed, hours_per_person

    """
    df = pd.read_csv(os.path.join(DATA_DIR, "lfms2021_hours_by_age.csv"))
    df["hours_per_person"] = df.emp_pop_ratio * df.mean_weekly_hours_employed
    return df


def older_hours_shape():
    """
    Hours worked per person within the LFMS open 65+ band, from the ESS
    wave 5 (2021/22) labor module: hours in the last 7 days on the
    household farm, in own-account or household business, in casual work
    and in paid work, with people who did not work counted at zero,
    household weights. Ages 70-79 are pooled (75-79 alone has 120 people).
    Tabulated by ~/Projects/og-analysis/eth-ess/scripts/hours_by_age.py
    (World Bank Microdata Library, catalog 6161).

    Returns:
        df (Pandas DataFrame): age_band, mean_age (weighted mean age in
            the band), n (people), hours_per_person, ratio_to_65plus
            (the band's hours relative to the ESS 65+ mean)

    """
    return pd.read_csv(
        os.path.join(DATA_DIR, "ess2021_older_hours_by_age.csv")
    )


def older_hours_decline():
    """
    Exponential decline of hours per person past 65: a weighted least
    squares fit of log hours on age through the ESS bands (each at its
    mean age, weighted by its sample size), at the LFMS 65+ level.

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
    Average labor supply by model age, as a share of the time endowment,
    from LFMS hours per person.

    LFMS publishes one open band from 65. From that age the target
    declines exponentially, fitted to the ESS age shape within the band
    (65-69, 70-79, 80+) at the LFMS 65+ level, so the survey sets the
    level and the ESS only the decline with age (about 6 percent a year
    of age). Between the 60-64 band's midpoint and 65 the target is
    linear.

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
    Marginal disutility of labor with unit weights, from OG-Core's
    elliptical utility function.

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
    Rescale chi_n so that, at fixed consumption and prices, the labor
    supply first-order condition holds at the target instead of the model
    value.

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
