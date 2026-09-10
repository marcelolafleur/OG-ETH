"""
This script creates tables and figures from the OG-ETH documentation.
"""

# import
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files
import json
from ogcore.parameters import Specifications
from ogcore import parameter_plots as pp
from ogcore import parameter_tables as pt
from ogcore import demographics as demog

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
UN_COUNTRY_CODE = "231"
plot_path = os.path.join(CUR_DIR, "book", "content", "calibration", "images")


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
        country_id=UN_COUNTRY_CODE,
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
        country_id=UN_COUNTRY_CODE,
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
        country_id=UN_COUNTRY_CODE,
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
        country_id=UN_COUNTRY_CODE,
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


if __name__ == "__main__":
    main()


def calibration_vs_data(output_dir=None, save_path=None, program=None):
    """
    Plot the baseline transition against the IMF program path (Country
    Report 26/174) for the calibrated fiscal and external moments.

    Args:
        output_dir (str): OUTPUT_BASELINE directory of an example run;
            defaults to examples/OG-ETH-Example/OUTPUT_BASELINE
        save_path (str): where to save the figure; defaults to the docs
            images folder
        program (module): module carrying the IMF program constants
            (defaults to ogeth.macro_params)

    Returns:
        matplotlib Figure
    """
    from ogcore.utils import safe_read_pickle
    from ogeth import macro_params as mp

    if program is None:
        program = mp
    if output_dir is None:
        output_dir = os.path.join(
            CUR_DIR, "..", "examples", "OG-ETH-Example", "OUTPUT_BASELINE"
        )
    if save_path is None:
        save_path = os.path.join(plot_path, "calibration_vs_program.png")
    tpi = safe_read_pickle(os.path.join(output_dir, "TPI", "TPI_vars.pkl"))
    params = safe_read_pickle(os.path.join(output_dir, "model_params.pkl"))
    start_year = int(params.start_year)
    n = len(program.PROGRAM_YEARS)
    years_prog = np.arange(start_year, start_year + n)
    horizon = n + 9
    years = np.arange(start_year, start_year + horizon)

    def ratio(key):
        return (
            np.asarray(tpi[key])[:horizon].reshape(horizon, -1).sum(axis=1)
            / np.asarray(tpi["Y"])[:horizon]
        )

    Y = np.asarray(tpi["Y"])[: horizon + 1]
    growth = (Y[1:] / Y[:-1]) * np.exp(params.g_y) * (
        1 + np.asarray(params.g_n)[1 : horizon + 1]
    ) - 1
    panels = [
        ("Public debt (% of GDP)", 100 * ratio("D"), program.IMF_PUBLIC_DEBT),
        (
            "Tax revenue (% of GDP)",
            100 * ratio("total_tax_revenue"),
            program.IMF_TAX_REVENUE,
        ),
        (
            "Primary expenditure (% of GDP)",
            100 * (ratio("G") + ratio("TR") + ratio("I_g")),
            list(
                np.array(program.IMF_EXPENDITURE)
                - np.array(program.IMF_INTEREST)
            ),
        ),
        (
            "Remittances (% of GDP)",
            100 * ratio("RM"),
            program.IMF_PRIVATE_TRANSFERS,
        ),
        (
            "Trade balance, model resource-constraint residual (% of GDP)",
            100
            * (1 - ratio("C") - ratio("I_total") - ratio("I_g") - ratio("G")),
            program.IMF_TRADE_BALANCE,
        ),
        ("Real GDP growth (%)", 100 * growth, program.IMF_REAL_GDP_GROWTH),
        (
            "Gross investment incl. public (% of GDP)",
            100 * (ratio("I_total") + ratio("I_g")),
            program.IMF_GROSS_INVESTMENT,
        ),
        (
            "Public debt held abroad (% of GDP)",
            100 * ratio("D_f"),
            list(
                np.array(program.IMF_PUBLIC_DEBT)
                - np.array(program.IMF_DOMESTIC_DEBT)
            ),
        ),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    for ax, (title, model, data) in zip(axes.flat, panels):
        ax.plot(years[: len(model)], model, label="OG-ETH baseline")
        ax.plot(
            years_prog, data, "o--", label="IMF CR 26/174 program", color="C3"
        )
        ax.set_title(title)
        ax.axvspan(years_prog[0], years_prog[-1], color="grey", alpha=0.08)
    axes.flat[0].legend(loc="best")
    fig.suptitle(
        "OG-ETH baseline transition vs. the IMF program path (shaded: program "
        "years, model periods held at program values)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


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
        output_dir = os.path.join(
            CUR_DIR, "..", "examples", "OG-ETH-Example", "OUTPUT_BASELINE"
        )
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
        label="Calibration target (LFMS bands interpolated, NTA taper)",
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
    ax.semilogy(ages, chi_n, label="OG-ETH, calibrated to LFMS hours")
    with urllib.request.urlopen(income.OGUSA_PARAMS_URL) as response:
        chi_n_usa = np.asarray(json.load(response)["chi_n"], dtype=float)
    if chi_n_usa.ndim == 2:
        chi_n_usa = chi_n_usa[-1]
    ax.semilogy(
        income.model_ages(income.OGUSA_E, income.OGUSA_S),
        chi_n_usa,
        "--",
        label="OG-USA (shipped before)",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel(r"$\chi^n_s$ (log scale)")
    ax.set_title("Disutility of labor by age")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig
