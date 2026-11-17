from ogeth import macro_params, income
from ogeth import input_output as io
import os
import numpy as np
import pandas as pd
import datetime
import warnings

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
# UN M49 code for Ethiopia, used for the UN population data
UN_COUNTRY_CODE = "231"
# Last year of UN data the demographics read. ogcore holds fertility,
# mortality and immigration at final_data_year forever after, so the window
# sets the long-run population growth rate: with start_year + 1 the model
# froze Ethiopia's current growth (2.6 percent a year) as its steady state.
# Reading the UN projection to its end lets growth fall as the projection
# says. get_pop reads final_data_year + 1, and the Population-Data archive
# (used when no UN API token is registered) ends in 2099, so 2098 is the
# last usable year; ogcore also caps it at start_year + 74.
DEMOG_FINAL_DATA_YEAR = 2098
# Ethiopia's rows of the EAPD-DRB/Demographic-Gradients library (commit
# ccf53a0, 2026-11-16): DHS 2024 wealth-quintile gradients in fertility
# (TFR), infant mortality (IMR) and child mortality at ages 1-4 (CMR), and
# the 2007 census household-deaths gradients in adult mortality (45q15 and
# age bands, four wealth groups). A gradient is the slope of the log rate on
# household wealth rank (0 to 1).
GRADIENT_FILE = os.path.join(CUR_PATH, "data", "demographic_gradients_ETH.csv")
# The library's pooled fertility offsets by age of mother (added to the TFR
# gradient so that the timing of births differs across groups, not only
# their number) and its general-gradient file: the income rule for the
# 45q15 gradient and the offsets by age band added to it
FERTILITY_OFFSET_FILE = os.path.join(
    CUR_PATH, "data", "demographic_fertility_age_offsets.csv"
)
GENERAL_GRADIENT_FILE = os.path.join(
    CUR_PATH, "data", "demographic_general_gradient.csv"
)
# OG-Core reads a log-odds slope per centered percentile point, so a
# library gradient g at a rate p is passed as g / (100 (1 - p)), p being
# OG-Core's own rate at that age and year (births per person per year for
# fertility, the annual probability of death for mortality); see the
# library's calibration guide, "Converting a gradient to OG-Core's units"
PERCENT = 100.0
# ages 1-4 take the child-mortality gradient; age 0 is infmort_gradient
CHILD_AGES = (1, 5)
# the 45q15 gradient plus the band offsets sets ages 5-59 (5-14 take the
# 15-29 value); from 60 the 45-59 value falls in a straight line to 0 at 75,
# and it is 0 from 75 on (the library's recipe; the measured 60-74 band is
# not used)
ADULT_BANDS = ((5, 30, "15-29"), (30, 45, "30-44"), (45, 60, "45-59"))
TAPER_AGES = (60, 75)
# The adult-mortality gradient steepens as a country gets richer: the
# library's income rule is
#     gradient(45q15) = intercept + slope ln(GDP per capita, PPP, constant
#                       2021 international $; World Bank NY.GDP.PCAP.PP.KD),
# fitted on GDP per capita between about $1,100 and $18,000 and not
# extrapolated beyond. Ethiopia's measurement is from the 2007 census, so
# the measured gradient is shifted by the rule's change since 2007 and keeps
# shifting along the model's growth path, held where the rule's range ends.
# (The library also allows holding it fixed; its ruling on the choice is
# pending, and the two bracket the treatment.)
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
    The library's fertility gradient by single age: the TFR gradient plus
    the pooled offset of the age group (ages below 15 take the 15-19
    offset, ages 50 and above the 45-49 offset), in the library's units.

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
    The library's mortality gradient by single age, in the library's units:
    0 at age 0 (infant mortality is separate), the child-mortality gradient
    at ages 1-4, the 45q15 gradient (plus ``shift``) plus the band offsets
    at ages 5-59, a straight-line fall from the 45-59 value at 60 to 0 at
    75, and 0 from 75 on.

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
    Change in the 45q15 gradient since the census year along the income
    rule, with GDP per head growing at g_y_annual from its latest value and
    held at the top of the rule's range.
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
    T=None,
    start_year=None,
    g_y_annual=None,
    rates=None,
    path=GRADIENT_FILE,
    offset_path=FERTILITY_OFFSET_FILE,
    general_path=GENERAL_GRADIENT_FILE,
):
    """
    Income gradients in fertility and mortality for OG-Core's demographics,
    from the Demographic-Gradients library's values for Ethiopia, by age
    and year, in OG-Core's units.

    Args:
        E (int): age at which agents become economically active
        S (int): number of model periods in a lifetime
        T (int): number of periods in the transition path
        start_year (int): calendar year of the first model period
        g_y_annual (float): annual growth of income per head
        rates (dict): OG-Core's own rates on the rows of its rate paths
            (``demographic_rates``): ``fert`` and ``mort`` (n, E+S),
            ``infmort`` (n,), and ``first_year`` of the first row. Without
            them the conversion takes p = 0 (division by 100 only) and the
            gradients are those of the census year.
        path, offset_path, general_path (str): the library's tables

    Returns:
        gradients (dict): fert_gradient, mort_gradient (period by age, or
            by age without rates) and infmort_gradient (by period, or a
            scalar without rates), ready for get_pop_objs
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
    OG-Core's own fertility and mortality rates on the rows of the rate
    paths get_pop_objs builds (from start_year - 2, the year before the
    initial data year, to DEMOG_FINAL_DATA_YEAR, then the last year held to
    T + S rows), so that each gradient is converted at the rate it is
    applied to.
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


# Discount-factor step used by examples/tune.py, which calibrates chi_n to
# hours by age and the relative discount factors to WID wealth shares in the
# steady state, and the remaining dials to first-year (FY2024/25) moments:
# a group's steady-state wealth is very elastic to its beta, so the update is
# damped and each step bounded.
BETA_DAMPING = 0.3
BETA_STEP = 1.03
# Upper bound above 1 for the sigma check: at sigma near 3 and g_y 4.7
# percent the top group needs beta near 1.03 (beta * exp(-sigma g_y) < 1).
BETA_BOUNDS = (0.5, 1.1)
# OG-USA's bequest weight, kept for every group
CHI_B_DEFAULT = 80.0


class Calibration:
    """OG-ETH calibration class"""

    def __init__(
        self,
        p,
        macro_data_start_year=datetime.datetime(1947, 1, 1),
        macro_data_end_year=datetime.datetime(2024, 12, 31),
        demographic_data_path=None,
        output_path=None,
        # Set True to update from World Bank and UN APIs
        update_from_api=False,
    ):
        """
        Constructor for the Calibration class.

        Args:
            p (OG-Core Specifications object): model parameters
            demographic_data_path (str): path to save demographic data
            output_path (str): path to save output to
            update_from_api (bool): Set True to pull updated macro data
                from World Bank and UN APIs

        Returns:
            None

        """
        # Create output_path if it doesn't exist
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        # Initialize attributes — populated only when update succeeds
        self.macro_params = {}
        self.demographic_params = {}
        self.e = None
        self.alpha_c = np.array([1.0]) if p.I == 1 else None
        self.io_matrix = np.array([[1.0]]) if p.M == 1 else None

        if not update_from_api:
            return

        # --- Online path: try each source independently ---

        # Macro estimation
        try:
            self.macro_params = macro_params.get_macro_params(
                macro_data_start_year,
                macro_data_end_year,
                update_from_api=update_from_api,
            )
        except Exception as exc:
            warnings.warn(f"Macro params update failed: {exc}", stacklevel=2)

        # io matrix and alpha_c (multi-sector only)
        if p.I > 1:
            try:
                alpha_c_dict = io.get_alpha_c()
                assert p.I == len(list(alpha_c_dict.keys()))
                self.alpha_c = np.array(list(alpha_c_dict.values()))
            except Exception as exc:
                warnings.warn(f"alpha_c update failed: {exc}", stacklevel=2)
        if p.M > 1:
            try:
                io_df = io.get_io_matrix()
                assert p.M == len(list(io_df.keys()))
                self.io_matrix = io_df.values
            except Exception as exc:
                warnings.warn(f"io_matrix update failed: {exc}", stacklevel=2)

        # Demographics + income (atomic — e depends on demographic output)
        try:
            self.update_demographics(p, demographic_data_path, output_path)
        except Exception as exc:
            warnings.warn(
                f"Demographics/income update failed: {exc}", stacklevel=2
            )
            self.demographic_params = {}
            self.e = None

    def update_demographics(
        self, p, demographic_data_path=None, output_path=None
    ):
        """
        Rebuild the demographics from the UN data, with Ethiopia's income
        gradients in fertility and mortality, and everything derived from
        them: the earnings matrix, the remittance growth path and
        allocation, and the transfer allocation.

        Args:
            p (OG-Core Specifications object): model parameters
            demographic_data_path (str): path to save demographic data
            output_path (str): path to save plots to

        Returns:
            None

        """
        from ogcore import demographics

        self.demographic_params = demographics.get_pop_objs(
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
            download_path=demographic_data_path,
            **demographic_gradients(
                p.E,
                p.S,
                p.T,
                p.start_year,
                p.g_y_annual,
                rates=demographic_rates(
                    p.E, p.S, p.T, p.start_year, demographic_data_path
                ),
            ),
        )

        # earnings profiles, reshaped with Ethiopian data and scaled on
        # the model's own joint age-by-group population distribution
        self.e = income.get_e_interp(
            p.E,
            p.S,
            p.J,
            p.lambdas,
            self.demographic_params["omega_SS"],
            plot_path=output_path,
        )

        # the remittance growth path and allocation matrix are derived
        # from these demographics, so they travel with them
        self.demographic_params.update(
            macro_params.derive_remittance_params(
                p.g_y,
                self.demographic_params["g_n"],
                self.demographic_params["omega_SS"],
                p.lambdas,
            )
        )
        # so is the transfer allocation matrix (per capita within the
        # targeted groups, pensions to the retired)
        retire_idx = int(np.asarray(p.retirement_age).flatten()[0]) - int(p.E)
        self.demographic_params["eta"] = macro_params.transfer_eta(
            self.demographic_params["omega_SS"], p.lambdas, retire_idx
        ).tolist()

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        d = {}
        d.update(self.macro_params)
        d.update(self.demographic_params)
        if self.e is not None:
            d["e"] = self.e
        if self.alpha_c is not None:
            d["alpha_c"] = self.alpha_c
        if self.io_matrix is not None:
            d["io_matrix"] = self.io_matrix
        return d
