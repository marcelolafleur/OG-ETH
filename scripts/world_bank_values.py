"""
Look up the latest World Bank value of a calibration input, to compare with
the packaged one when updating the calibration. Nothing is written: adopt a
new value by editing the packaged JSON and documenting its source.

    python scripts/world_bank_values.py

g_y_annual is the mean annual growth of GDP per capita (WDI NY.GDP.PCAP.KD,
constant US dollars) over 2016-2025 (macro.md, Economic Assumptions).
"""

import json
from importlib.resources import files

import pandas as pd
import requests

G_Y_YEARS = (2016, 2025)


def world_bank_series(indicator, country="ETH", years=(1990, 2025)):
    """Annual World Bank WDI series as a pandas Series indexed by year."""
    response = requests.get(
        f"https://api.worldbank.org/v2/country/{country}/indicator/"
        f"{indicator}",
        params={
            "date": f"{years[0]}:{years[1]}",
            "format": "json",
            "per_page": 1000,
        },
        timeout=60,
    )
    response.raise_for_status()
    rows = response.json()[1]
    data = {int(r["date"]): r["value"] for r in rows if r["value"] is not None}
    return pd.Series(data).sort_index()


def main():
    gdp = world_bank_series("NY.GDP.PCAP.KD")
    first, last = G_Y_YEARS
    growth = gdp.pct_change().loc[first:last]
    packaged = json.loads(
        files("ogeth")
        .joinpath("ogeth_default_parameters.json")
        .read_text(encoding="utf-8")
    )
    print(
        f"g_y_annual: World Bank {growth.mean():.4f} "
        f"({growth.index.min()}-{growth.index.max()}), "
        f"packaged {packaged['g_y_annual']:.4f}"
    )


if __name__ == "__main__":
    main()
