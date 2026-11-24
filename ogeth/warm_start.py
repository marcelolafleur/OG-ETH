"""
Warm-started steady-state solves. ``enable()`` seeds every solve from
``data/ss_warm_start.pkl`` (regenerate with examples/save_warm_start.py);
``solve_ss_from()`` solves from a solved state in memory (examples/tune.py).
"""

import os
import pickle

import numpy as np
import scipy.optimize as opt
from ogcore import SS, firm

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
SEED_PATH = os.path.join(CUR_PATH, "data", "ss_warm_start.pkl")


def save_seed(ss_vars, path=SEED_PATH):
    """
    Store the parts of a solved steady state needed to warm-start another.

    Args:
        ss_vars (dict): a solved ``SS_vars.pkl``
        path (str): where to write the seed

    Returns:
        str: the path written
    """

    def scalar(x):
        return float(np.asarray(x).sum()) if np.ndim(x) else float(x)

    seed = {
        "b": np.asarray(ss_vars["b_sp1"], dtype=float),
        "n": np.asarray(ss_vars["n"], dtype=float),
        "r": scalar(ss_vars["r"]),
        "r_p": scalar(ss_vars["r_p"]),
        "TR": scalar(ss_vars["TR"]),
        "factor": scalar(ss_vars["factor"]),
        "BQ": np.asarray(ss_vars["BQ"], dtype=float).ravel(),
        # solved output, wage and prices
        "w": scalar(ss_vars["w"]),
        "Y": scalar(ss_vars["Y"]),
        "p_m": np.atleast_1d(np.asarray(ss_vars["p_m"], dtype=float)).ravel(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(seed, fh)
    return path


def enable(path=SEED_PATH):
    """
    Patch ``SS.SS_initial_guesses`` to start from the stored seed; no-op
    (cold start) if no seed is stored.

    Args:
        path (str): seed written by :func:`save_seed`

    Returns:
        bool: True if a warm start is now active
    """
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        seed = pickle.load(fh)

    original = SS.SS_initial_guesses

    def warm(p, b_val=None, n_val=None, r_tr_scalars=[1.0, 1.0]):
        if seed["b"].shape != (p.S, p.J) or seed["BQ"].size != p.J:
            print("Warm-start seed is for other dimensions; cold start.")
            return original(p, b_val, n_val, r_tr_scalars)

        # scale by ogcore's retry-ladder factors (DEV_FACTOR_LIST)
        r_g = r_tr_scalars[0] * seed["r"]
        r_p_g = r_tr_scalars[0] * seed["r_p"]
        BQ_g = seed["BQ"] * r_tr_scalars[1]
        BQ_items = [BQ_g.sum()] if p.use_zeta else list(BQ_g)
        if seed["p_m"].size == p.M:
            # solved Y, w and p_m; TR is the run's own share of Y, and w
            # moves with r as the firms' condition implies
            Y_g = seed["Y"]
            TR_g = r_tr_scalars[1] * p.alpha_T[-1] * Y_g
            w_g = (
                seed["w"]
                * firm.get_w_from_r(r_g, p, "SS")
                / firm.get_w_from_r(seed["r"], p, "SS")
            )
            p_m_g = list(seed["p_m"])
        else:
            # a run with another number of industries (the seed is
            # single-industry): rebuild Y and w from TR and r
            TR_g = r_tr_scalars[1] * seed["TR"]
            Y_g = TR_g / p.alpha_T[-1]
            w_g = firm.get_w_from_r(r_g, p, "SS")
            p_m_g = list(np.ones(p.M))

        guesses = [r_p_g, r_g, w_g] + p_m_g + [Y_g] + BQ_items + [TR_g]
        if p.baseline:
            # passed directly: as a parameter it is capped at 500,000
            guesses = guesses + [seed["factor"]]
        return guesses, seed["b"].copy(), seed["n"].copy()

    SS.SS_initial_guesses = warm
    return True


def solve_ss_from(p, prev, client=None):
    """
    Solve the baseline steady state starting from a previous solution.

    Args:
        p (OG-Core Specifications object): model parameters, baseline
        prev (dict): a solved steady state (``SS_vars.pkl`` contents or the
            output of ``SS.run_SS``) for nearby parameters
        client (Dask client object): client for parallel computation

    Returns:
        output (dict): as ``SS.run_SS``; a cold start if this fails
    """
    assert p.baseline, "warm start is for baseline solves"
    BQ = np.atleast_1d(np.squeeze(prev["BQ"]))
    guesses = (
        [float(np.squeeze(prev["r_p"])), float(np.squeeze(prev["r"]))]
        + [float(np.squeeze(prev["w"]))]
        + list(np.atleast_1d(np.squeeze(prev["p_m"])))
        + [float(np.squeeze(prev["Y"]))]
        + list(BQ)
        + [float(np.squeeze(prev["TR"])), float(np.squeeze(prev["factor"]))]
    )
    b_guess, n_guess = prev["b_sp1"], prev["n"]
    scattered_p = SS.scatter_params(p, client)
    args = (b_guess, n_guess, None, None, None, None, p, client, scattered_p)
    sol = opt.root(
        SS.SS_fsolve,
        guesses,
        args=args,
        method=p.SS_root_method,
        tol=p.mindist_SS,
    )
    if not sol.success:
        print("warm start did not converge; falling back to a cold start")
        return SS.run_SS(p, client=client)
    M = p.M
    r_p, r, w = sol.x[0], sol.x[1], sol.x[2]
    p_m = sol.x[3 : 3 + M]
    BQ = sol.x[3 + M + 1 : -2]
    TR, factor = sol.x[-2], sol.x[-1]
    Y = TR / p.alpha_T[-1]  # SS_solver recomputes it, as in SS.run_SS
    return SS.SS_solver(
        b_guess,
        n_guess,
        r_p,
        r,
        w,
        p_m,
        Y,
        BQ,
        TR,
        None,
        None,
        factor,
        p,
        client,
        True,
        scattered_p,
    )
