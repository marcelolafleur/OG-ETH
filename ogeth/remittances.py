"""
Remittance calibration for OG-ETH.

Remittances enter OG-Core as an exogenous share of GDP (``alpha_RM_1`` in the
first period, ``alpha_RM_T`` in the long run) that is distributed across
households by the ``eta_RM`` matrix. Two of the shipped values are *derived*
from the packaged demographics rather than hand-set, so they have to be
rebuilt whenever the demographics are:

* ``g_RM`` -- ``aggregates.get_RM`` advances detrended remittances by the
  factor ``(1 + g_RM[t]) / (exp(g_y) * (1 + g_n[t-1]))`` between the first
  period and ``tG1``, so ``RM/Y`` only stays at the calibrated share when
  that factor is one. ``g_n`` moves along the transition, so the growth rate
  that holds the share flat is a path, not a scalar (see ``flat_share_g_RM``).
* ``eta_RM`` -- the share of aggregate remittances each household receives,
  shaped (S, J). OG-Core's default hands every lifetime-income group exactly
  its population share. ``remittance_eta`` instead maps a data distribution
  of remittance value across income quintiles onto the model's J groups and
  spreads it per capita across ages within each group.

``main()`` rewrites exactly those two keys in
``ogeth_default_parameters.json``; ``update_baseline.main`` calls the same
functions after a full regeneration.
"""

import json
import os
from importlib.resources import files

import numpy as np
from ogcore.parameters import Specifications

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
    Remittance growth path that holds aggregate remittances at a constant
    share of GDP.

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


def derived_remittance_params(p):
    """
    The remittance parameters that depend on the demographics in ``p``.

    Args:
        p (Specifications): parameters carrying ``g_y``, ``g_n``,
            ``omega_SS`` and ``lambdas``

    Returns:
        dict: ``{"g_RM": list, "eta_RM": nested list}`` ready for
            ``Specifications.update_specifications``
    """
    g_n = np.asarray(p.g_n, dtype=float)[: p.T + p.S]
    return {
        "g_RM": flat_share_g_RM(p.g_y, g_n).tolist(),
        "eta_RM": remittance_eta(p.omega_SS, p.lambdas).tolist(),
    }


def main():
    """
    Rewrite ``g_RM`` and ``eta_RM`` in the packaged JSON from its own
    demographics, leaving every other value untouched.
    """
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(cur_dir, "ogeth_default_parameters.json")
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    defaults = json.loads(content)
    p = Specifications(baseline=True)
    p.update_specifications(defaults)
    defaults.update(derived_remittance_params(p))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    main()
