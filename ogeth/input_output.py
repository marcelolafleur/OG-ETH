import pandas as pd
import numpy as np
import os

from ogeth.constants import CONS_DICT, PROD_DICT


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sam_path = os.path.join(CUR_DIR, "data", "IFPRI_SAM_ETH_2022_SAM.csv")


def read_SAM():
    """
    Read the packaged Ethiopia SAM file.

    Returns:
        pd.DataFrame: parsed SAM table
    """
    sam = pd.read_csv(sam_path, index_col=1, thousands=",").dropna(how="all")
    # first column is labels; the rest are values
    value_cols = sam.columns.drop(sam.columns[0])
    sam[value_cols] = (
        sam[value_cols]
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .fillna(0)
    )
    return sam


def get_alpha_c(sam=None, cons_dict=CONS_DICT):
    """
    Household expenditure share of each consumption good (alpha_c).

    Args:
        sam (pd.DataFrame): SAM table
        cons_dict (dict): consumption goods to SAM accounts

    Returns:
        alpha_c (dict): expenditure share by good
    """
    if sam is None:
        sam = read_SAM()

    hh_cols = [
        "hhd-r1",
        "hhd-r2",
        "hhd-r3",
        "hhd-r4",
        "hhd-r5",
        "hhd-u1",
        "hhd-u2",
        "hhd-u3",
        "hhd-u4",
        "hhd-u5",
    ]
    alpha_c = {}
    overall_sum = 0
    for key, value in cons_dict.items():
        category_total = (
            sam.loc[sam.index.isin(value), hh_cols].values.astype(float).sum()
        )
        alpha_c[key] = category_total
        overall_sum += category_total
    for key, value in cons_dict.items():
        alpha_c[key] = alpha_c[key] / overall_sum

    return alpha_c


def get_io_matrix(sam=None, cons_dict=CONS_DICT, prod_dict=PROD_DICT):
    """
    Share of each industry in each consumption good (io_matrix; firms.md).

    Args:
        sam (pd.DataFrame): SAM table
        cons_dict (dict): consumption goods to SAM accounts
        prod_dict (dict): industries to SAM accounts

    Returns:
        io_df (pd.DataFrame): goods by industries, rows sum to one
    """
    if sam is None:
        sam = read_SAM()
    # goods x industries, filled below
    io_dict = {}
    for key in prod_dict.keys():
        io_dict[key] = np.zeros(len(cons_dict.keys()))
    io_df = pd.DataFrame(io_dict, index=cons_dict.keys())
    # a SAM cell is a payment from the column account to the row account:
    # goods from rows, industries from columns
    for ck, cv in cons_dict.items():
        for pk, pv in prod_dict.items():
            io_df.loc[io_df.index == ck, pk] = sam.loc[
                sam.index.isin(cv), pv
            ].values.sum()
    # levels to shares (rows sum to one)
    io_df = io_df.div(io_df.sum(axis=1), axis=0)

    return io_df
