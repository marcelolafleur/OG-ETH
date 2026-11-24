from ogeth import input_output as io
from ogeth import macro_params, income
import copy
import os
import numpy as np
import pandas as pd

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
# UN M49 code for Ethiopia, used for the UN population data
UN_COUNTRY_CODE = "231"
# Last year of UN data read; rates are held from it on, so it sets
# long-run population growth. get_pop reads one year past it and the
# offline archive ends in 2099 (demographics.md, The data window).
DEMOG_FINAL_DATA_YEAR = 2098
# Ethiopia's rows of EAPD-DRB/Demographic-Gradients (commit ccf53a0):
# slopes of the log rate on wealth rank (0 to 1) for TFR, IMR, CMR (DHS
# 2024) and adult mortality (2007 census; demographics.md).
GRADIENT_FILE = os.path.join(CUR_PATH, "data", "demographic_gradients_ETH.csv")
# The library's fertility offsets by age of mother, and its income rule and
# age-band offsets for the 45q15 gradient
FERTILITY_OFFSET_FILE = os.path.join(
    CUR_PATH, "data", "demographic_fertility_age_offsets.csv"
)
GENERAL_GRADIENT_FILE = os.path.join(
    CUR_PATH, "data", "demographic_general_gradient.csv"
)
# OG-Core takes a log-odds slope per percentile point: a gradient g at
# OG-Core's own rate p is passed as g / (100 (1 - p))
PERCENT = 100.0
# ages 1-4 take the child-mortality gradient; age 0 is infmort_gradient
CHILD_AGES = (1, 5)
# 45q15 gradient plus band offsets at ages 5-59 (5-14 take 15-29); linear
# fall from the 45-59 value at 60 to 0 at 75
ADULT_BANDS = ((5, 30, "15-29"), (30, 45, "30-44"), (45, 60, "45-59"))
TAPER_AGES = (60, 75)
# Income rule: the 45q15 gradient moves with ln GDP per capita (PPP,
# constant 2021 $, World Bank NY.GDP.PCAP.PP.KD) from the 2007 census along
# the growth path, held above $18,000 (demographics.md)
CENSUS_YEAR = 2007
GDP_PPP_PER_CAPITA = {2007: 1109.14, 2025: 3094.87}
GDP_LATEST_YEAR = 2025
INCOME_RULE_MAX_GDP = 18000.0


def _library_tables(path, offset_path, general_path):
    df = pd.read_csv(path)
    offsets = pd.read_csv(offset_path)
    general = pd.read_csv(general_path)
    rule = general[general.component == "income_rule"].set_index("key").value
    band = general[general.component == "age_offset"].set_index("key").value
    return df, offsets, rule, band


def _gradient(df, indicator, path):
    rows = df[df.indicator == indicator]
    if rows.empty:
        raise ValueError(f"No {indicator} gradient in {path}")
    return float(rows.slope.iloc[0])


def fertility_gradient_by_age(E, S, df, offsets, path=GRADIENT_FILE):
    """
    Fertility gradient by single age: TFR gradient plus age offset.

    Returns:
        Numpy array: length E+S
    """
    ages = np.arange(E + S)
    tfr = _gradient(df, "TFR", path)
    off = offsets.sort_values("age_lo")
    lo, val = off.age_lo.to_numpy(), off.offset.to_numpy()
    idx = np.clip(np.searchsorted(lo, ages, side="right") - 1, 0, len(lo) - 1)
    return tfr + val[idx]


def mortality_gradient_by_age(E, S, df, band, shift=0.0, path=GRADIENT_FILE):
    """
    Mortality gradient by single age: 0 at age 0, CMR at 1-4, 45q15 plus
    ``shift`` and band offsets at 5-59, tapering to 0 at 75.

    Returns:
        Numpy array: length E+S
    """
    ages = np.arange(E + S)
    mort = np.zeros(E + S)
    mort[(ages >= CHILD_AGES[0]) & (ages < CHILD_AGES[1])] = _gradient(
        df, "CMR", path
    )
    adult = df[(df.indicator == "AMR") & (df.measure == "45q15")]
    if adult.empty:
        raise ValueError(f"No 45q15 adult-mortality gradient in {path}")
    g45 = float(adult.slope.iloc[0]) + shift
    for lo, hi, key in ADULT_BANDS:
        mort[(ages >= lo) & (ages < hi)] = g45 + float(band[key])
    top = g45 + float(band["45-59"])
    taper = (ages >= TAPER_AGES[0]) & (ages < TAPER_AGES[1])
    mort[taper] = (
        top * (TAPER_AGES[1] - ages[taper]) / (TAPER_AGES[1] - TAPER_AGES[0])
    )
    return mort


def income_rule_shift(years, g_y_annual, rule):
    """
    Change in the 45q15 gradient since the census year by the income rule.
    """
    years = np.asarray(years, dtype=float)
    gdp = GDP_PPP_PER_CAPITA[GDP_LATEST_YEAR] * (1 + g_y_annual) ** (
        years - GDP_LATEST_YEAR
    )
    gdp = np.minimum(gdp, INCOME_RULE_MAX_GDP)
    return float(rule["slope_ln_income"]) * np.log(
        gdp / GDP_PPP_PER_CAPITA[CENSUS_YEAR]
    )


def to_ogcore_units(gradient, rate):
    """
    gradient / (100 (1 - p)); 0 where the rate is 1 (OG-Core's final age).
    """
    rate = np.asarray(rate, dtype=float)
    out = np.zeros(np.broadcast(gradient, rate).shape)
    ok = rate < 1.0
    g = np.broadcast_to(gradient, out.shape)
    p = np.broadcast_to(rate, out.shape)
    out[ok] = g[ok] / (PERCENT * (1 - p[ok]))
    return out


def demographic_gradients(
    E,
    S,
    g_y_annual=None,
    rates=None,
    path=GRADIENT_FILE,
    offset_path=FERTILITY_OFFSET_FILE,
    general_path=GENERAL_GRADIENT_FILE,
):
    """
    Income gradients in fertility and mortality, in OG-Core's units.

    Args:
        E (int): age at which agents become economically active
        S (int): number of model periods in a lifetime
        g_y_annual (float): annual growth of income per head
        rates (dict): output of ``demographic_rates``; None takes p = 0
            and the census-year gradients
        path, offset_path, general_path (str): the library's tables

    Returns:
        gradients (dict): fert_gradient, mort_gradient, infmort_gradient
            for get_pop_objs
    """
    df, offsets, rule, band = _library_tables(path, offset_path, general_path)
    fert = fertility_gradient_by_age(E, S, df, offsets, path)
    infmort = _gradient(df, "IMR", path)
    if rates is None:
        return {
            "fert_gradient": fert / PERCENT,
            "infmort_gradient": infmort / PERCENT,
            "mort_gradient": mortality_gradient_by_age(
                E, S, df, band, 0.0, path
            )
            / PERCENT,
        }
    years = rates["first_year"] + np.arange(rates["fert"].shape[0])
    if g_y_annual is None:
        shifts = np.zeros(len(years))
    else:
        shifts = income_rule_shift(years, g_y_annual, rule)
    mort = np.vstack(
        [mortality_gradient_by_age(E, S, df, band, s, path) for s in shifts]
    )
    mort[:, -1] = 0.0  # final age: everyone dies
    return {
        "fert_gradient": to_ogcore_units(fert.reshape(1, -1), rates["fert"]),
        "infmort_gradient": to_ogcore_units(infmort, rates["infmort"]),
        "mort_gradient": to_ogcore_units(mort, rates["mort"]),
    }


def demographic_rates(E, S, T, start_year, download_path=None):
    """
    OG-Core's fertility and mortality rates on get_pop_objs' rows (from
    start_year - 2, held to T + S rows), to convert the gradients.
    """
    from ogcore import demographics

    first = start_year - 2
    fert = demographics.get_fert(
        E + S,
        0,
        99,
        UN_COUNTRY_CODE,
        first,
        DEMOG_FINAL_DATA_YEAR,
        download_path=download_path,
    )
    mort, infmort = demographics.get_mort(
        E + S,
        0,
        99,
        UN_COUNTRY_CODE,
        first,
        DEMOG_FINAL_DATA_YEAR,
        download_path=download_path,
    )

    def hold(x):
        x = np.asarray(x, dtype=float)
        n = T + S - x.shape[0]
        if n <= 0:
            return x[: T + S]
        return np.concatenate([x, np.repeat(x[-1:], n, axis=0)], axis=0)

    return {
        "fert": hold(fert),
        "mort": hold(mort),
        "infmort": hold(np.asarray(infmort).reshape(-1)),
        "first_year": first,
    }


def demographic_params(p, download_path=None):
    """
    Rebuild the demographics (UN data, income gradients) and the earnings
    matrix e. The allocations and fiscal paths that depend on them come
    from macro_params.baseline_params (update_baseline.py).

    Args:
        p (OG-Core Specifications object): model parameters
        download_path (str): directory to save the demographic data to

    Returns:
        dict: OG-Core demographic parameters and e
    """
    from ogcore import demographics

    d = demographics.get_pop_objs(
        p.E,
        p.S,
        p.T,
        0,
        99,
        country_id=UN_COUNTRY_CODE,
        initial_data_year=p.start_year - 1,
        final_data_year=DEMOG_FINAL_DATA_YEAR,
        income_percentiles=p.lambdas.flatten(),
        GraphDiag=False,
        download_path=download_path,
        **demographic_gradients(
            p.E,
            p.S,
            p.g_y_annual,
            rates=demographic_rates(
                p.E, p.S, p.T, p.start_year, download_path
            ),
        ),
    )

    # earnings profiles, scaled on the joint population distribution
    d["e"] = (
        income.get_e_interp(
            p.E,
            p.S,
            p.J,
            p.lambdas,
            d["omega_SS"],
        )
        * macro_params.EARNINGS_SCALE
    )
    return d


class Calibration:
    """
    The parameters OG-ETH rebuilds from data, in the form the OG-Core
    country models share. With ``update_from_api=False`` (the default) it
    rebuilds nothing: the packaged JSON already holds the calibration.
    """

    def __init__(self, p, update_from_api=False, demographic_data_path=None):
        """
        Args:
            p (OG-Core Specifications object): model parameters
            update_from_api (bool): rebuild the demographics and e from the
                UN data, everything macro_params.baseline_params derives
                from them, and (for a multi-industry model) alpha_c and
                io_matrix from the SAM
            demographic_data_path (str): directory to save the UN data to
        """
        self.params = {}
        if not update_from_api:
            return
        demog = demographic_params(p, download_path=demographic_data_path)
        p_new = copy.deepcopy(p)
        p_new.update_specifications(
            {k: np.asarray(v).tolist() for k, v in demog.items()}
        )
        self.params.update(demog)
        self.params.update(macro_params.baseline_params(p_new))
        # single-industry values; a multi-industry model keeps its own
        if p.M > 1:
            for key in ("gamma", "gamma_g"):
                self.params.pop(key)
        if p.I > 1:
            self.params.pop("c_min")
        if p.I > 1:
            self.params["alpha_c"] = list(io.get_alpha_c().values())
        if p.M > 1:
            self.params["io_matrix"] = io.get_io_matrix().values.tolist()

    def get_dict(self):
        """The rebuilt parameters, for Specifications.update_specifications."""
        return dict(self.params)
