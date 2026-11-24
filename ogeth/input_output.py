import pandas as pd
import numpy as np
import os


# Map the SAM's micro categories to the model's broad goods and industries
CONS_DICT = {
    "Food": [
        "cmaiz",
        "crice",
        "cocer",
        "cpuls",
        "coils",
        "croot",
        "cvege",
        "csugr",
        "ctoba",
        "ccott",
        "cfrui",
        "ccoff",
        "cocrp",
        "ccatt",
        "cpoul",
        "coliv",
        "cfish",
        "cfood",
        "cbeve",
    ],
    "Energy and extraction": [
        "cmine",
        "cfore",
        "cchem",
        "cnmet",
    ],
    "Non-durables": [
        "ctext",
        "cwood",
    ],
    "Durables": [
        "cmetl",
        "cmach",
        "coman",
        "ccons",
    ],
    "Services": [
        "ctrad",
        "ctran",
        "chotl",
        "ccomm",
        "cfsrv",
        "creal",
        "cbsrv",
        "cpadm",
        "ceduc",
        "cheal",
        "cosrv",
        "celec",
        "cwatr",
    ],
}
PROD_DICT = {
    "Primary": [
        "amaiz",
        "arice",
        "aocer",
        "apuls",
        "aoils",
        "aroot",
        "avege",
        "asugr",
        "atoba",
        "acott",
        "afrui",
        "acoff",
        "aocrp",
        "acatt",
        "apoul",
        "aoliv",
        "afore",
        "afish",
        "amine",
    ],
    "Energy": [
        "aelec",
    ],
    "Tertiary": [
        "atrad",
        "atran",
        "ahotl",
        "acomm",
        "afsrv",
        "areal",
        "absrv",
        "apadm",
        "aeduc",
        "aheal",
        "aosrv",
    ],
    "Secondary Ex Energy": [
        "afood",
        "abeve",
        "atext",
        "awood",
        "achem",
        "anmet",
        "ametl",
        "amach",
        "aoman",
        "awatr",
        "acons",
    ],
}


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sam_path = os.path.join(CUR_DIR, "data", "IFPRI_SAM_ETH_2022_SAM.csv")


def read_SAM():
    """
    Read the packaged Ethiopia SAM file.

    Returns:
        pd.DataFrame | None: parsed SAM table, or None if unavailable
    """
    try:
        sam = pd.read_csv(
            sam_path,
            index_col=1,
            thousands=",",
        ).dropna(how="all")

        # first column is labels; the rest are values
        label_col = sam.columns[0]
        value_cols = sam.columns.drop(label_col)

        sam[value_cols] = (
            sam[value_cols]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .fillna(0)
        )

        return sam
    except Exception as exc:
        print(f"Failed to read packaged SAM file: {exc}")
        return None


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
    if sam is None:
        raise RuntimeError("SAM data is unavailable. Cannot compute alpha_c.")

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
    if sam is None:
        raise RuntimeError(
            "SAM data is unavailable. Cannot compute io_matrix."
        )
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
