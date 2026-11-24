"""
Tests of input_output.py module
"""

import pandas as pd
import pytest

from ogeth import input_output as io

HH_COLS = [
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


@pytest.fixture
def sam_df():
    data = {col: [0.0, 0.0] for col in HH_COLS}
    data["hhd-r1"] = [10.0, 30.0]
    data["hhd-u1"] = [20.0, 40.0]
    data["A"] = [2.0, 1.0]
    data["B"] = [1.0, 3.0]
    return pd.DataFrame(data, index=["beer", "car"])


def test_get_alpha_c(sam_df):
    cons_dict = {"Food": ["beer"], "Non-food": ["car"]}

    test_dict = io.get_alpha_c(sam=sam_df, cons_dict=cons_dict)

    assert isinstance(test_dict, dict)
    assert sorted(test_dict.keys()) == sorted(["Food", "Non-food"])
    assert test_dict["Food"] == pytest.approx(0.3)
    assert test_dict["Non-food"] == pytest.approx(0.7)


def test_get_io_matrix(sam_df):
    cons_dict = {"Food": ["beer"], "Non-food": ["car"]}
    prod_dict = {"Primary": ["A"], "Secondary": ["B"]}

    test_df = io.get_io_matrix(
        sam=sam_df, cons_dict=cons_dict, prod_dict=prod_dict
    )

    assert isinstance(test_df, pd.DataFrame)
    assert sorted(test_df.columns) == sorted(["Primary", "Secondary"])
    assert sorted(test_df.index) == sorted(["Food", "Non-food"])
    assert test_df.loc["Food", "Primary"] == pytest.approx(2 / 3)
    assert test_df.loc["Food", "Secondary"] == pytest.approx(1 / 3)
    assert test_df.loc["Non-food", "Primary"] == pytest.approx(1 / 4)
    assert test_df.loc["Non-food", "Secondary"] == pytest.approx(3 / 4)


def test_multi_industry_parameters_load():
    """The packaged parameters, the multi-industry overlay and the SAM's
    consumption shares and input-output matrix load together (the setup of
    examples/run_og_eth_multi_industry.py)."""
    import json
    from importlib.resources import files

    from ogcore.parameters import Specifications

    p = Specifications()
    for name in (
        "ogeth_default_parameters.json",
        "ogeth_multi_industry_parameters.json",
    ):
        p.update_specifications(
            json.loads(files("ogeth").joinpath(name).read_text())
        )
    alpha_c, io_matrix = io.get_alpha_c(), io.get_io_matrix()
    p.update_specifications(
        {
            "alpha_c": list(alpha_c.values()),
            "io_matrix": io_matrix.values.tolist(),
        }
    )
    assert not p.errors
    assert p.io_matrix.shape == (p.I, p.M)
