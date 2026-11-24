"""
An extra example, not part of the calibration: the IMF's program for
Ethiopia as an alternative baseline.

The packaged parameters are OG-ETH's proposed baseline (run it with
examples/run_og_eth.py). This script overlays the paths of the IMF's fifth
review of Ethiopia's Extended Credit Facility (IMF Country Report 26/174)
for FY2024/25-FY2030/31: spending, revenue measures, grants, borrowing and
remittances, and the program's long-run debt ratio, all built by
``ogeth.macro_params.scenario_params(p, "program")``. It then runs the same
corporate-tax-cut reform as the official example. Results are written under
examples/OG-ETH-IMF-Program.
"""

# imports
import multiprocessing
from distributed import Client
import os
import json
import time
import copy
from importlib.resources import files
import matplotlib.pyplot as plt
from ogeth import macro_params, warm_start
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle
import dask

dask.config.set(scheduler="synchronous")

# Use a custom matplotlib style file for plots
plt.style.use("ogcore.OGcorePlots")


def main():
    """
    Run the IMF program baseline and a corporate-tax-cut reform.
    """
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    client = Client(n_workers=num_workers, threads_per_worker=1)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "OG-ETH-IMF-Program")
    base_dir = os.path.join(save_dir, "OUTPUT_BASELINE")
    reform_dir = os.path.join(save_dir, "OUTPUT_REFORM")

    """
    ---------------------------------------------------------------------------
    Run baseline policy: the packaged parameters with the IMF program paths
    ---------------------------------------------------------------------------
    """
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    with (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .open("r") as file
    ):
        defaults = json.load(file)
    p.update_specifications(defaults)
    p.update_specifications(macro_params.scenario_params(p, "program"))
    # The steady state starts from the shipped solved state of the official
    # baseline (ogeth/warm_start.py), with transfers at this scenario's
    # share of output.
    if warm_start.enable():
        print("Warm start enabled (ogeth/data/ss_warm_start.pkl).")
    else:
        print("No warm-start seed found; cold-starting the steady state.")

    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    """
    ---------------------------------------------------------------------------
    Run reform policy
    ---------------------------------------------------------------------------
    """
    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir
    p2.update_specifications({"cit_rate": [[0.25]]})  # CIT rate to 25%

    start_time = time.time()
    runner(p2, time_path=True, client=client)
    print("run time = ", time.time() - start_time)
    client.close()

    """
    ---------------------------------------------------------------------------
    Save some results of simulations
    ---------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )
    op.plot_all(
        base_dir, reform_dir, os.path.join(save_dir, "OG-ETH_example_plots")
    )
    print("Percentage changes in aggregates:", ans)
    ans.to_csv(os.path.join(save_dir, "OG-ETH_example_output.csv"))


if __name__ == "__main__":
    # execute only if run as a script
    main()
