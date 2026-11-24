"""
Rebuild the demographics, the earnings matrix and everything
macro_params.baseline_params derives, and write them into the packaged JSON;
other entries are left as they are.

    python -m ogeth.update_baseline
"""

import json
import os
from importlib.resources import files

import numpy as np
from ogcore.parameters import Specifications

from ogeth.calibrate import Calibration


def _jsonable(value):
    """Turn a parameter value into something json.dump accepts."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json_parameters(json_path, updates):
    """Overwrite the given keys of a parameter JSON, keeping the others."""
    with open(json_path, encoding="utf-8") as f:
        params = json.load(f)
    params.update(updates)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
        f.write("\n")


def main():
    """Rebuild the parameters that derive from data and write them."""
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
    d = Calibration(p, update_from_api=True).get_dict()
    write_json_parameters(json_path, {k: _jsonable(v) for k, v in d.items()})


if __name__ == "__main__":
    main()
