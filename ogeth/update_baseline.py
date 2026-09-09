import os
import json
from importlib.resources import files
from ogeth.calibrate import Calibration
from ogeth.macro_params import derived_remittance_params
from ogcore.parameters import Specifications
from ogcore.utils import params_to_json


def main():
    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))

    # Set up baseline parameterization
    p = Specifications(baseline=True)
    # Update parameters for baseline from default json file
    content = (
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    defaults = json.loads(content)
    p.update_specifications(defaults)
    c = Calibration(
        p,
        update_from_api=True,
    )
    d = c.get_dict()
    # update parameters
    p.update_specifications(d)
    # remittance growth path and allocation are derived from the demographics
    # just regenerated, so they are rebuilt here rather than left stale
    p.update_specifications(derived_remittance_params(p))
    # save to json file
    params_to_json(p, os.path.join(CUR_DIR, "ogeth_default_parameters.json"))


if __name__ == "__main__":
    main()
