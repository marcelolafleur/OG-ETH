"""
One step of fitting the steady-state net family transfers (phi_FT) so that
each lifetime-income group's consumption by age follows the consumption
target (targets.consumption_target). Solves nothing: give it a
steady state solved with the current transfers (examples/tune.py), then
solve again with the overrides it writes, and repeat until the gap stops
falling (households.md, Net family transfers).

Rule, per group j, at ages 20 and over: the target is k_j times the target
profile, with k_j keeping the group's total consumption unchanged; the
transfer to age s moves by the gap times the group's population at s over
output. The step sums to zero within each group.

    python examples/fit_family_transfers.py SS_vars.pkl [overrides.json]
        --next next.json [--damp 1.0] [--write]

--next writes the overrides (the phi_FT path) for the next tune.py run;
--write writes the matrix in use (the settled values) to ogeth/data.
"""

import argparse
import json
import os
from importlib.resources import files

import numpy as np
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle

from ogeth import macro_params as mp
from ogeth import targets

FIRST_AGE = 20


def load_params(overrides=None):
    p = Specifications(baseline=True)
    p.update_specifications(
        json.loads(
            files("ogeth")
            .joinpath("ogeth_default_parameters.json")
            .read_text(encoding="utf-8")
        )
    )
    if overrides:
        p.update_specifications(
            {k: v for k, v in overrides.items() if not k.startswith("_")}
        )
    return p


def fit_step(ss, p, damp=1.0, first_age=FIRST_AGE):
    """The next steady-state phi_FT, (S, J), and the current profile gap."""
    omega = np.asarray(p.omega_SS, dtype=float)
    c = np.asarray(ss["c"]) + float(np.asarray(p.c_min).ravel()[0])
    y = float(np.asarray(ss["Y"]).sum())
    ages, _, _, gap = targets.consumption_profile(ss, p)
    target = targets.consumption_target(ages)
    phi = np.asarray(p.phi_FT, dtype=float)
    phi = phi[-1] if phi.ndim == 3 else phi
    on = ages >= first_age
    new = phi.copy()
    for j in range(phi.shape[1]):
        m = omega[on, j]
        k = (m * c[on, j]).sum() / (m * target[on]).sum()
        new[on, j] += damp * (k * target[on] - c[on, j]) * m / y
    return new, gap


def write_matrix(phi, path):
    S, J = phi.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Net family transfers in the steady state, shares of the "
            "model's output Y by age (rows)\n# and lifetime-income group "
            "(columns), fitted to the consumption target by\n"
            "# examples/fit_family_transfers.py\n"
        )
        f.write("age," + ",".join(f"j{j}" for j in range(J)) + "\n")
        for s in range(S):
            f.write(
                f"{s + 20}," + ",".join(f"{x:.10e}" for x in phi[s]) + "\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ss")
    ap.add_argument("overrides", nargs="?", default=None)
    ap.add_argument("--next", default=None)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ov = {}
    if a.overrides and os.path.exists(a.overrides):
        with open(a.overrides, encoding="utf-8") as f:
            ov = json.load(f)
    p = load_params(ov)
    new, gap = fit_step(safe_read_pickle(a.ss), p, a.damp)
    print(f"consumption profile gap, ages 20-85: {gap:.4f}")
    if a.next:
        ov["phi_FT"] = mp.family_transfer_path(p, new).tolist()
        with open(a.next, "w", encoding="utf-8") as f:
            json.dump(ov, f)
    if a.write:
        phi = np.asarray(p.phi_FT, dtype=float)
        write_matrix(
            phi[-1] if phi.ndim == 3 else phi,
            str(files("ogeth").joinpath("data", mp.FAMILY_TRANSFERS_FILE)),
        )


if __name__ == "__main__":
    main()
