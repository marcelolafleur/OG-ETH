"""
Rebuild the demographics and the earnings matrix and rewrite them, with
everything macro_params.baseline_params derives, in the packaged JSON; other
entries are left as they are.

    python -m ogeth.update_baseline
"""

import json
import os
from importlib.resources import files

import numpy as np
from ogcore.parameters import Specifications

from ogeth.calibrate import demographic_params
from ogeth.income import write_json_parameters
from ogeth.macro_params import baseline_params


def _jsonable(value):
    """Turn a parameter value into something json.dump accepts."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    """Rebuild the demographic-dependent parameters and write them."""
    json_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "ogeth_default_parameters.json",
    )
    p = Specifications(baseline=True)
    p.update_specifications(
        json.loads(
            files("ogeth")
            .joinpath("ogeth_default_parameters.json")
            .read_text(encoding="utf-8")
        )
    )
    d = demographic_params(p)
    p.update_specifications({k: _jsonable(v) for k, v in d.items()})
    # the fiscal, transfer and remittance paths depend on the demographics
    d.update(baseline_params(p))
    write_json_parameters(json_path, {k: _jsonable(v) for k, v in d.items()})


if __name__ == "__main__":
    main()
