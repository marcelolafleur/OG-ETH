"""
This script creates tables and figures from the OG-ETH documentation.
"""

# import
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files
import json
from ogcore.parameters import Specifications
from ogcore import parameter_plots as pp
from ogcore import parameter_tables as pt
from ogcore import demographics as demog
from ogeth import calibrate

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
plot_path = os.path.join(CUR_DIR, "book", "content", "calibration", "images")
EXAMPLE_OUTPUT = os.path.join(
    CUR_DIR, "..", "examples", "OG-ETH-Example", "OUTPUT_BASELINE"
)


def main():
    # update path for demographics graphdiag plots
    demog.OUTPUT_DIR = plot_path

    # Use a custom matplotlib style file for plots
    plt.style.use("ogcore.OGcorePlots")

    """
    Load specifications object with default parameters
    """
    p = Specifications()
    # Update parameters for baseline from default json file
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    defaults = json.loads(content)
    p.update_specifications(defaults)
    YEAR_TO_PLOT = int(p.start_year)
    """
    Demographics chapter
    """
    # Fertility rates
    _, _ = demog.get_fert(
        totpers=100,
        min_age=0,
        max_age=99,
        country_id=calibrate.UN_COUNTRY_CODE,
        start_year=YEAR_TO_PLOT,
        end_year=YEAR_TO_PLOT,
        graph=True,
        plot_path=None,
        download_path=None,
    )
    plt.savefig(os.path.join(plot_path, "fert_rates.png"), dpi=300)
    # Mortality rates
    _, _, _ = demog.get_mort(
        totpers=100,
        min_age=0,
        max_age=99,
        country_id=calibrate.UN_COUNTRY_CODE,
        start_year=YEAR_TO_PLOT,
        end_year=YEAR_TO_PLOT,
        graph=True,
        plot_path=None,
        download_path=None,
    )
    plt.xlabel(r"Age ($s$)")
    plt.ylabel(r"Mortality rate ($\rho_s$)")
    plt.savefig(os.path.join(plot_path, "mort_rates.png"), dpi=300)
    # Immigration rates
    _, _ = demog.get_imm_rates(
        totpers=100,
        min_age=0,
        max_age=99,
        fert_rates=None,
        mort_rates=None,
        infmort_rates=None,
        pop_dist=None,
        country_id=calibrate.UN_COUNTRY_CODE,
        start_year=YEAR_TO_PLOT,
        end_year=YEAR_TO_PLOT + 50,
        graph=True,
        plot_path=None,
        download_path=None,
    )
    plt.xlabel(r"Age ($s$)")
    plt.ylabel(r"Immigration rate ($i_s$)")
    # give a little more before the plot source note

    plt.savefig(os.path.join(plot_path, "imm_rates.png"), dpi=300)
    # Fixed versus original population distribution
    demog.get_pop_objs(
        E=20,
        S=80,
        T=320,
        min_age=0,
        max_age=99,
        fert_rates=None,
        mort_rates=None,
        infmort_rates=None,
        imm_rates=None,
        infer_pop=False,
        pop_dist=None,
        country_id=calibrate.UN_COUNTRY_CODE,
        initial_data_year=YEAR_TO_PLOT - 1,
        final_data_year=YEAR_TO_PLOT + 2,
        GraphDiag=True,
        download_path=None,
    )

    # Population growth
    pp.plot_pop_growth(
        p,
        start_year=YEAR_TO_PLOT,
        num_years_to_plot=150,
        include_title=False,
        path=None,
    )
    # Add average growth rate with this
    plt.plot(
        np.arange(YEAR_TO_PLOT, YEAR_TO_PLOT + 150),
        np.ones(150) * np.mean(p.g_n[:150]),
        linestyle="-",
        linewidth=1,
        color="red",
    )
    plt.xlabel(r"Model Period ($t$)")
    plt.ylabel(r"Population Growth Rate ($g_{n,t}$)")
    plt.savefig(
        os.path.join(plot_path, "population_growth_rates.png"), dpi=300
    )

    # Population distribution at different points in time
    pp.plot_population(
        p,
        years_to_plot=[
            YEAR_TO_PLOT,
            YEAR_TO_PLOT + 25,
            YEAR_TO_PLOT + 50,
            YEAR_TO_PLOT + 100,
        ],
        include_title=False,
        path=plot_path,
    )
    """
    Income chapter
    """
    # USA profiles
    pp.plot_ability_profiles(
        p, p2=None, t=None, log_scale=True, include_title=False, path=plot_path
    )

    """
    Create table for exogenous parameters
    """
    pt.param_table(
        p,
        table_format="md",
        path=os.path.join(plot_path, "exogenous_parameters_table.md"),
    )

    """
    Figures built from the packaged parameters and data
    """
    earnings_vs_data()
    demographics_by_income_group()
    statutory_tax_functions()
    # needs the output of examples/run_og_eth.py, newer than the packaged
    # parameters (an older run would plot a superseded calibration)
    ss_path = os.path.join(EXAMPLE_OUTPUT, "SS", "SS_vars.pkl")
    json_path = str(files("ogeth").joinpath("ogeth_default_parameters.json"))
    if os.path.exists(ss_path) and os.path.getmtime(
        ss_path
    ) > os.path.getmtime(json_path):
        labor_supply_vs_data()
        results_tables()
        fiscal_path_panel()
        consumption_vs_target()


def earnings_vs_data(save_path=None):
    """
    Plot the calibrated ability matrix against its data sources: the age
    profile against the NTA per-worker earnings profiles and the group
    means against the WID income shares.

    Args:
        save_path (str): where to save the figure; defaults to the docs
            images folder

    Returns:
        matplotlib Figure
    """
    from ogeth import income

    if save_path is None:
        save_path = os.path.join(plot_path, "earnings_vs_data.png")
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    params = json.loads(content)
    e = np.asarray(params["e"])
    omega = np.asarray(params["omega_SS"])
    lambdas = np.asarray(params["lambdas"])
    S, J = e.shape
    ages = income.model_ages(20, S)
    e_usa, lambdas_usa = income.load_ogusa_e()
    e_base = income.interpolate_usa_e(e_usa, lambdas_usa, 20, S, lambdas)

    def age_profile(mat):
        prof = (mat * omega).sum(axis=1) / omega.sum(axis=1)
        return prof / prof[(ages >= 30) & (ages < 40)].mean()

    def per_worker(country):
        y = income.per_worker_earnings(country, ages)
        return y / y[(ages >= 30) & (ages < 40)].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    shown = ages < income.OLDEST_MEASURED_AGE
    ax.plot(ages, age_profile(e), label="OG-ETH $e$ (calibrated)")
    ax.plot(
        ages, age_profile(e_base), "--", label="OG-USA profiles, unadjusted"
    )
    ax.plot(
        ages[shown],
        per_worker("ETH")[shown],
        "o",
        ms=3,
        color="C3",
        label="NTA Ethiopia 2005, labor income per worker",
    )
    ax.plot(
        ages[shown],
        per_worker("US")[shown],
        "s",
        ms=3,
        color="C7",
        label="NTA United States 2006, labor income per worker",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Relative to the mean at ages 30-39")
    ax.set_title("Age profile of earnings")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    x = np.arange(J)
    width = 0.27
    ax.bar(
        x - width,
        100 * income.implied_group_shares(e, omega),
        width,
        label="OG-ETH $e$, implied income share",
    )
    ax.bar(
        x,
        100 * income.wid_group_shares("ETH", lambdas),
        width,
        color="C3",
        label="WID Ethiopia 2021",
    )
    ax.bar(
        x + width,
        100 * income.wid_group_shares("US", lambdas),
        width,
        color="C7",
        label="WID United States 2021",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["0-25", "25-50", "50-70", "70-80", "80-90", "90-99", "Top 1"]
    )
    ax.set_xlabel("Lifetime-income group (percentiles)")
    ax.set_ylabel("Share of income (%)")
    ax.set_title("Income shares by lifetime-income group")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def demographics_by_income_group(save_path=None):
    """
    Plot the income gradients in fertility and mortality: the survey and
    census measurements they come from, the gradient applied by age, and
    the mortality and survival by lifetime-income group they produce in
    the packaged steady state.

    Args:
        save_path (str): where to save the figure; defaults to the docs
            images folder

    Returns:
        matplotlib Figure
    """
    import pandas as pd
    from ogeth import calibrate, income

    if save_path is None:
        save_path = os.path.join(plot_path, "demographics_by_income.png")
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    params = json.loads(content)
    E, S = 20, int(params["S"])
    lambdas = np.asarray(params["lambdas"])
    rho = np.asarray(params["rho"])[-1]
    omega = np.asarray(params["omega_SS"])
    ages = income.model_ages(E, S)
    lib = pd.read_csv(calibrate.GRADIENT_FILE)
    gradients = calibrate.demographic_gradients(E, S)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    quintiles = np.arange(1, 6)
    ax = axes[0, 0]
    for indicator, label, color in [
        ("TFR", "Total fertility rate (children per woman)", "C0"),
        ("IMR", "Infant mortality (per 1,000 births, ÷ 10)", "C3"),
    ]:
        row = lib[lib.indicator == indicator].iloc[0]
        values = np.array([row[f"q{q}"] for q in quintiles], dtype=float)
        if indicator == "IMR":
            values = values / 10
        ax.plot(
            quintiles,
            values,
            "o-",
            color=color,
            label=f"{label}, tilt {row.slope:.2f}",
        )
    ax.set_xticks(quintiles)
    ax.set_xticklabels(["poorest", "2", "3", "4", "richest"])
    ax.set_xlabel("Wealth quintile, DHS 2024")
    ax.set_title("Survey data: fertility and infant mortality by wealth")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    bands = lib[(lib.indicator == "AMR") & (lib.measure == "mx")]
    groups = int(bands.groups.iloc[0])
    for row in bands.itertuples():
        values = np.array(
            [getattr(row, f"q{g}") for g in range(1, groups + 1)], dtype=float
        )
        ax.plot(
            np.arange(1, groups + 1),
            values / values[0],
            "o-",
            label=f"ages {int(row.age_lo)}-{int(row.age_hi)}, "
            f"gradient {row.slope:+.2f}",
        )
    g45 = lib[(lib.indicator == "AMR") & (lib.measure == "45q15")].iloc[0]
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.set_xticks(np.arange(1, groups + 1))
    ax.set_xticklabels(
        ["poorest"] + [str(g) for g in range(2, groups)] + ["richest"]
    )
    ax.set_xlabel("Asset-index group, 2007 census household deaths")
    ax.set_ylabel("Death rate relative to the poorest group")
    ax.set_title(
        f"Census 2007: adult mortality by wealth\n(45q15 gradient "
        f"{g45.slope:+.2f}; the bands are shown, not applied)"
    )
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    all_ages = np.arange(E + S)
    df, offsets, rule, band = calibrate._library_tables(
        calibrate.GRADIENT_FILE,
        calibrate.FERTILITY_OFFSET_FILE,
        calibrate.GENERAL_GRADIENT_FILE,
    )
    start = int(params["start_year"])
    for year, color, ls in [
        (calibrate.CENSUS_YEAR, "C7", ":"),
        (start, "C3", "-"),
        (2063, "C3", "--"),
    ]:
        shift = (
            0.0
            if year == calibrate.CENSUS_YEAR
            else float(
                calibrate.income_rule_shift(
                    [year], float(params["g_y_annual"]), rule
                )[0]
            )
        )
        ax.step(
            all_ages,
            calibrate.mortality_gradient_by_age(E, S, df, band, shift),
            where="post",
            color=color,
            ls=ls,
            label=f"mortality, {year}",
        )
    fertile = (all_ages >= 15) & (all_ages < 50)
    ax.step(
        all_ages[fertile],
        100 * gradients["fert_gradient"][fertile],
        where="post",
        color="C0",
        label="fertility (by age of mother)",
    )
    ax.axhline(
        100 * gradients["infmort_gradient"],
        color="C1",
        ls=":",
        label="infant mortality (age 0)",
    )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("Age")
    ax.set_ylabel("Change in log rate, poorest to richest")
    ax.set_title("Gradients passed to OG-Core (before ÷ 100 (1 − p))")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    survival = np.cumprod(1 - rho, axis=0)
    for j, label, color in [
        (0, "bottom 25%", "C0"),
        (2, "50-70%", "C2"),
        (6, "top 1%", "C3"),
    ]:
        ax.plot(ages, 100 * survival[:, j], color=color, label=label)
    ax.set_xlabel("Age")
    ax.set_ylabel("Survivors from age 20 (%)")
    ax.set_title("Model steady state: survival by lifetime-income group")
    shares = omega.sum(axis=0)
    text = "adult population share vs. birth share:\n" + "\n".join(
        f"  group {j + 1}: {100 * shares[j]:.2f}% vs {100 * lambdas[j]:.0f}%"
        for j in range(len(lambdas))
    )
    ax.text(
        0.02,
        0.05,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        family="monospace",
        va="bottom",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def statutory_tax_functions(save_path=None):
    """
    Plot the statutory income-tax schedules against the Gouveia-Strauss
    functions fitted to them, in effective and marginal rates.

    Args:
        save_path (str): where to save the figure; defaults to the docs
            images folder

    Returns:
        matplotlib Figure
    """
    from ogeth import income
    from ogeth import macro_params as mp

    if save_path is None:
        save_path = os.path.join(plot_path, "statutory_tax_functions.png")
    params = mp.statutory_tax_params()
    growth = (
        mp.NOMINAL_GDP_REBASED_BIRR_BN["FY2025/26"]
        / mp.NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"]
    ) / (1 + mp.POPULATION_GROWTH)
    incomes = np.geomspace(*mp.TAX_FIT_INCOME_RANGE, 400)
    packaged = json.loads(
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    group_means = income.implied_group_shares(
        np.asarray(packaged["e"]), np.asarray(packaged["omega_SS"])
    ) / np.asarray(packaged["lambdas"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rate_type, title in [
        (axes[0], "etr", "Effective rate"),
        (axes[1], "mtr", "Marginal rate"),
    ]:
        for label, schedule, scale, phi, color in [
            (
                "Proclamation 979/2016 (FY2024/25)",
                mp.PIT_SCHEDULE_2016,
                1.0,
                params["etr_params"][0][0],
                "C0",
            ),
            (
                "Proclamation 1395/2025 (from FY2025/26), in FY2024/25 birr",
                mp.PIT_SCHEDULE_2025,
                1 / growth,
                params["etr_params"][1][0],
                "C3",
            ),
        ]:
            etr, mtr = mp.schedule_tax_rates(incomes, schedule, scale)
            ax.plot(
                incomes / 12000,
                etr if rate_type == "etr" else mtr,
                color=color,
                lw=1,
                label=f"statutory, {label}",
            )
            ax.plot(
                incomes / 12000,
                mp.gs_rates(incomes, phi, rate_type),
                "--",
                color=color,
                lw=2,
                label="Gouveia-Strauss fit",
            )
        for j, m in enumerate(group_means, start=1):
            x = m * params["mean_income_data"] / 12000
            ax.axvline(x, color="grey", lw=0.5, alpha=0.6)
            if rate_type == "etr":
                ax.text(
                    x, 0.36, f"g{j}", fontsize=7, ha="center", color="grey"
                )
        ax.set_xscale("log")
        ax.set_xlabel("Monthly income, thousand birr (log scale)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}: statutory schedule vs. fitted function")
        ax.set_ylim(0, 0.38)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "Vertical lines: mean income of the seven lifetime-income groups "
        "(mean_income_data times the group means of e)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def labor_supply_vs_data(output_dir=None, save_path=None):
    """
    Plot the steady-state average labor supply by age against hours worked
    per person in the 2021 Labour Force and Migration Survey.

    Args:
        output_dir (str): OUTPUT_BASELINE directory of an example run;
            defaults to examples/OG-ETH-Example/OUTPUT_BASELINE
        save_path (str): where to save the figure; defaults to the docs
            images folder

    Returns:
        matplotlib Figure
    """
    from ogcore.utils import safe_read_pickle
    from ogeth import income, labor

    if output_dir is None:
        output_dir = EXAMPLE_OUTPUT
    if save_path is None:
        save_path = os.path.join(plot_path, "labor_supply_vs_data.png")
    ss = safe_read_pickle(os.path.join(output_dir, "SS", "SS_vars.pkl"))
    params = safe_read_pickle(os.path.join(output_dir, "model_params.pkl"))
    ages = income.model_ages(params.E, params.S)
    n_model = labor.average_labor_supply(np.asarray(ss["n"]), params.omega_SS)
    target = labor.labor_supply_target(params.E, params.S, params.ltilde)
    hours = labor.hours_per_person()
    hours = hours[hours.age_band.map(labor.HOURS_BAND_MIDPOINTS) >= 20]
    scale = labor.WEEKLY_TIME_ENDOWMENT / params.ltilde

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(ages, scale * n_model, label="OG-ETH steady state")
    ax.plot(
        ages,
        scale * target,
        "--",
        color="C3",
        label="Target (LFMS 2021; decline from 65 fitted to ESS 2021/22)",
    )
    ax.plot(
        hours.age_band.map(labor.HOURS_BAND_MIDPOINTS),
        hours.hours_per_person,
        "o",
        color="C3",
        label="LFMS 2021: employment ratio x weekly hours of the employed",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Weekly hours worked per person")
    ax.set_title("Labor supply by age: model vs. survey")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    chi_n = labor.steady_state_chi_n(params)
    ax.semilogy(ages, chi_n, label="OG-ETH, calibrated to hours by age")
    ax.axhline(
        labor.MAX_CHI_N, color="0.5", ls=":", label="OG-Core's upper bound"
    )
    ax.set_xlabel("Age")
    ax.set_ylabel(r"$\chi^n_s$ (log scale)")
    ax.set_title("Disutility of labor by age")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


# Rows of the FY2024/25 validation table: label, moment, how it is used
VALIDATION_ROWS = [
    ("Personal income tax", "PIT/Y", "tuned (compliance scale)", "NBE"),
    (
        "Corporate and capital-income taxes",
        "CIT/Y",
        "tuned (CIT collections factor)",
        "NBE",
    ),
    (
        "Consumption taxes, fees and sales",
        "cons tax/Y",
        "tuned (`tau_c`)",
        "NBE",
    ),
    ("Land-use fees", "wealth tax/Y", "tuned (`p_wealth`)", "NBE"),
    (
        "Pension contributions",
        "payroll/Y",
        "tuned (payroll coverage scale)",
        "pension schemes",
    ),
    (
        "Old-age pensions",
        "pensions/Y",
        "tuned (pension coverage scale)",
        "ILO",
    ),
    (
        "Foreign-owned share of private capital",
        "K_f/K",
        "tuned (`zeta_K`)",
        "IMF BoP; PWT",
    ),
    ("Household wealth", "B/Y", "imposed", "PWT; IMF"),
    ("General-government debt", "D/Y", "imposed", "IMF"),
    ("Remittances", "RM/Y", "imposed", "IMF"),
    (
        "Gross capital formation (private and public)",
        "(I+I_g)/Y",
        "check",
        "MoNS",
    ),
    ("Private consumption", "C/Y purch.", "check", "MoNS"),
    ("Government consumption", "G/Y", "check", "MoNS"),
    ("Net exports", "NX/Y", "check", "MoNS"),
    (
        "Payments abroad on capital and debt",
        "factor payments/Y",
        "check",
        "IMF",
    ),
]
HOURS_BANDS = [
    (20, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, 60),
    (60, 65),
    (65, 70),
    (70, 80),
    (80, 90),
    (90, 100),
]
CONSUMPTION_BANDS = [
    (20, 25),
    (25, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 65),
    (65, 70),
    (70, 80),
    (80, 86),
]


def _band(lo, hi):
    return f"{lo}\u2013{hi - 1}"


def _num(x, digits, sign=False):
    """A number for a table, with a typographic minus sign."""
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return text.replace("-", "\u2212")


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def results_tables(output_dir=None, path=None):
    """
    Write the results tables of the calibration chapters from an example
    run and the packaged parameters: the FY2024/25 and long-run validation
    tables (macro.md), hours by age (households.md) and consumption and net
    family transfers by age (households.md).

    Args:
        output_dir (str): OUTPUT_BASELINE directory of an example run;
            defaults to examples/OG-ETH-Example/OUTPUT_BASELINE
        path (str): folder for the tables; defaults to the docs images
            folder

    Returns:
        None
    """
    from ogcore.utils import safe_read_pickle
    from ogeth import income, labor, targets
    from ogeth import macro_params as mp

    output_dir = EXAMPLE_OUTPUT if output_dir is None else output_dir
    path = plot_path if path is None else path
    params = safe_read_pickle(os.path.join(output_dir, "model_params.pkl"))
    ss = safe_read_pickle(os.path.join(output_dir, "SS", "SS_vars.pkl"))
    tpi = safe_read_pickle(os.path.join(output_dir, "TPI", "TPI_vars.pkl"))

    # validation: first model year and steady state
    m0 = targets.tpi_moments(tpi, 0)
    mss = targets.ss_moments(ss)
    lines = [
        "| Ratio to GDP, FY2024/25 (percent) | Model | Data "
        "| How it is used | Source |",
        "|:---|:---:|:---:|:---|:---|",
    ]
    for label, key, use, source in VALIDATION_ROWS:
        model, data = 100 * m0[key], 100 * targets.TRANSITION_TARGETS[key][0]
        digits = 2 if abs(data) < 5 else 1
        lines.append(
            f"| {label} | {_num(model, digits)} | {_num(data, digits)} "
            f"| {use} | {source} |"
        )
    lines += [
        "",
        "| Long run (steady state) | Model | Target |",
        "|:---|:---:|:---:|",
        f"| Owners' return on capital (real, percent a year) | "
        f"{100 * mss['r_K']:.2f} | at least "
        f"{100 * mp.OWNERS_RETURN_FLOOR:.2f} |",
        f"| Firms' return on capital (real, percent a year) | "
        f"{100 * mss['r']:.2f} | outcome |",
        f"| Rate on market debt (real, percent a year) | "
        f"{100 * mss['r_gov']:.2f} | {100 * mp.LONG_RUN_R_GOV:.2f} |",
        f"| Private capital / GDP | {mss['K/Y']:.2f} | outcome "
        f"({m0['K/Y']:.2f} in FY2024/25) |",
        f"| Foreign-owned share of capital | {100 * mss['K_f/K']:.1f} "
        "| outcome |",
        f"| Private consumption / GDP | {100 * mss['C/Y purch.']:.1f} "
        "| outcome |",
    ]
    _write(os.path.join(path, "validation_table.md"), lines)

    # first-tier moments over the early transition
    rows = [
        ("Gross capital formation", "(I+I_g)/Y"),
        ("Private consumption", "C/Y purch."),
        ("Net exports", "NX/Y"),
        ("General-government debt", "D/Y"),
        ("Total tax", "total tax/Y"),
        ("Foreign-owned share of private capital", "K_f/K"),
        ("Firms' return on capital", "r"),
    ]
    years = [0, 2, 5, 10]
    m_t = [targets.tpi_moments(tpi, t) for t in years]
    lines = [
        "| Percent of GDP (return: percent a year) | "
        + " | ".join(str(int(params.start_year) + t) for t in years)
        + " |",
        "|:---|" + ":---:|" * len(years),
    ]
    for label, key in rows:
        lines.append(
            f"| {label} | "
            + " | ".join(_num(100 * m[key], 1) for m in m_t)
            + " |"
        )
    _write(os.path.join(path, "early_transition_table.md"), lines)

    # hours by age, steady state
    ages = income.model_ages(params.E, params.S)
    scale = labor.WEEKLY_TIME_ENDOWMENT / params.ltilde
    n_model = scale * labor.average_labor_supply(
        np.asarray(ss["n"]), params.omega_SS
    )
    n_target = scale * labor.labor_supply_target(
        params.E, params.S, params.ltilde
    )
    capped = labor.steady_state_chi_n(params) >= labor.MAX_CHI_N
    lines = [
        "| Age | Target: hours per person per week (LFMS 2021; ESS 2021/22 "
        "shape from 65) | Model steady state |",
        "|---|---|---|",
    ]
    for lo, hi in HOURS_BANDS:
        k = (ages >= lo) & (ages < hi)
        note = " ($\\chi^n$ at its cap)" if capped[k].all() else ""
        lines.append(
            f"| {_band(lo, hi)}{note} | {n_target[k].mean():.1f} "
            f"| {n_model[k].mean():.1f} |"
        )
    _write(os.path.join(path, "hours_by_age_table.md"), lines)

    # consumption per adult and net family transfers by age, steady state
    a, model, data, gap = targets.consumption_profile(ss, params)
    phi = mp.family_transfers_ss()
    a_phi = np.arange(phi.shape[0]) + int(params.E)
    lines = [
        "| Age | Consumption per adult, model | Target | Net family "
        "transfers received, percent of output |",
        "|---|---|---|---|",
    ]
    for lo, hi in CONSUMPTION_BANDS:
        k = (a >= lo) & (a < hi)
        top = 101 if hi == 86 else hi
        kp = (a_phi >= lo) & (a_phi < top)
        label = f"{lo}+ (consumption to 85)" if hi == 86 else _band(lo, hi)
        lines.append(
            f"| {label} | {model[k].mean():.2f} | {data[k].mean():.2f} "
            f"| {_num(100 * phi[kp].sum(), 1, sign=True)} |"
        )
    lines.append(
        f"| Mean absolute gap, ages 20\u201385 | {gap:.3f} | | "
        f"total received {100 * phi[phi > 0].sum():.1f} |"
    )
    _write(os.path.join(path, "consumption_by_age_table.md"), lines)


def fiscal_path_panel(output_dir=None, save_path=None, years=36):
    """
    The near-term fiscal path: general-government debt, primary balance,
    tax revenue and public investment as percent of GDP at market prices,
    model path against the FY2024/25 outturn and the FY2025/26 budget
    (IMF CR 26/174, NBE Annual Report 2024/25, rebased).

    Args:
        output_dir (str): OUTPUT_BASELINE directory of an example run;
            defaults to examples/OG-ETH-Example/OUTPUT_BASELINE
        save_path (str): where to save the figure; defaults to the docs
            images folder
        years (int): number of transition years to plot

    Returns:
        matplotlib Figure
    """
    from ogcore.utils import safe_read_pickle
    from ogeth import macro_params as mp
    from ogeth import targets

    output_dir = EXAMPLE_OUTPUT if output_dir is None else output_dir
    if save_path is None:
        save_path = os.path.join(plot_path, "fiscal_path_panel.png")
    tpi = safe_read_pickle(os.path.join(output_dir, "TPI", "TPI_vars.pkl"))
    params = safe_read_pickle(os.path.join(output_dir, "model_params.pkl"))
    t = np.arange(years)
    yr = int(params.start_year) + t

    def ratio(key):
        y = (
            np.asarray(tpi["Y"])[:years]
            + np.asarray(tpi["cons_tax_revenue"])[:years]
        )
        return 100 * np.asarray(tpi[key])[:years] / y

    revenue = ratio("total_tax_revenue")
    primary = revenue - ratio("total_primary_government_outlays")
    if "foreign_aid" in tpi:
        primary = primary + ratio("foreign_aid")
    # actuals: cash accounts, rebased; primary balance = all revenue and
    # grants less recurrent spending net of interest and capital spending
    rev_all = (
        100
        * sum(mp.REVENUE_FY2425_BIRR_BN.values())
        / (mp.NOMINAL_GDP_REBASED_BIRR_BN["FY2024/25"])
    )
    pb_actual = rev_all - (
        mp.IMF_RECURRENT[0] - mp.IMF_INTEREST[0] + mp.IMF_CAPITAL[0]
    )
    tax_actual = 100 * targets.TRANSITION_TARGETS["total tax/Y"][0]
    panels = [
        (
            "General-government debt",
            ratio("D") + ratio("D_ext"),
            mp.IMF_PUBLIC_DEBT,
        ),
        ("Primary balance (incl. grants)", primary, [pb_actual, None]),
        ("Tax revenue (mapped instruments)", revenue, [tax_actual, None]),
        ("Public investment", ratio("I_g"), mp.IMF_CAPITAL),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, path, data) in zip(axes.ravel(), panels):
        ax.plot(yr, path, color="C0", label="OG-ETH baseline")
        for i, (v, lab) in enumerate(
            zip(data, ["FY2024/25 outturn", "FY2025/26 budget"])
        ):
            if v is not None:
                ax.plot(
                    yr[i], v, "o" if i == 0 else "D", color="C3", label=lab
                )
        lo, hi = (
            min(path.min(), *[v for v in data if v is not None]),
            max(path.max(), *[v for v in data if v is not None]),
        )
        pad = 0.15 * max(hi - lo, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Percent of GDP")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def consumption_vs_target(output_dir=None, save_path=None):
    """
    Plot steady-state consumption per adult by age against the consumption
    target (targets.consumption_target), relative to ages 30-49.

    Args:
        output_dir (str): OUTPUT_BASELINE directory of an example run;
            defaults to examples/OG-ETH-Example/OUTPUT_BASELINE
        save_path (str): where to save the figure; defaults to the docs
            images folder

    Returns:
        matplotlib Figure
    """
    from ogcore.utils import safe_read_pickle
    from ogeth import targets

    output_dir = EXAMPLE_OUTPUT if output_dir is None else output_dir
    if save_path is None:
        save_path = os.path.join(plot_path, "consumption_vs_target.png")
    ss = safe_read_pickle(os.path.join(output_dir, "SS", "SS_vars.pkl"))
    params = safe_read_pickle(os.path.join(output_dir, "model_params.pkl"))
    ages, model, data, gap = targets.consumption_profile(ss, params)
    k = ages <= 85
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ages[k], model[k], label="OG-ETH steady state")
    ax.plot(ages[k], data[k], "--", color="C3", label="Target")
    ax.set_xlabel("Age")
    ax.set_ylabel("Consumption per adult, relative to ages 30-49")
    ax.set_title(f"Consumption by age: model vs. target (mean gap {gap:.3f})")
    ax.legend(loc="lower center", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
