"""
Reproduce the sourced values that ogeth/macro_params.py holds as constants:
the world interest rate, the capital stocks on the rebased accounts, the
foreign-owned capital stock and the sovereign-rate scale. Each was derived
once from the data files in ogeth/data; this script shows how, and prints
the values to compare with the constants.

    python scripts/derive_sourced_values.py

Method and sources: docs/book/content/calibration/macro.md and firms.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ogeth import macro_params as mp

DATA = Path(mp.__file__).parent / "data"

# Capital rebuild: old-base investment share and real growth for the years
# after PWT's last (FY2023/24 interpolated, FY2024/25 the old-base share)
OLD_BASE_GCF_SHARE = 0.201
OLD_BASE_EXTENSION = {2024: (0.2115, 0.081), 2025: (OLD_BASE_GCF_SHARE, 0.092)}
CAPITAL_REVISION_START_YEAR = 2016
# FDI stock: dollar price growth of capital goods, and the flow assumed for
# the years the IMF does not report (1995-2002), USD million
FDI_PRICE_GROWTH = 0.025
FDI_MISSING_YEARS_FLOW = 250.0


def peer_fdi_return(years=(2015, 2024)):
    """Pooled nominal return on inward FDI in fourteen African peers: direct-
    investment income paid abroad over average direct-investment
    liabilities (IMF BOP and IIP)."""
    df = pd.read_csv(DATA / "imf_fdi_income_africa_peers.csv")
    df = df.sort_values(["country", "year"])
    df["avg"] = df.groupby("country")["di_liabilities_usd_m"].transform(
        lambda s: (s + s.shift(1)) / 2
    )
    used = df[(df.year >= years[0]) & (df.year <= years[1])].dropna(
        subset=["di_income_debit_usd_m", "avg"]
    )
    return float(used.di_income_debit_usd_m.sum() / used.avg.sum())


def pwt_domestic_capital_output_ratio(year=2023):
    """PWT 11.0 net capital stock over GDP, both at current national
    prices (old-base GDP)."""
    detail = pd.read_csv(
        DATA / "pwt110_eth_capital_detail.csv", index_col="year"
    )
    na = pd.read_csv(DATA / "pwt110_eth_capital.csv", index_col="year")
    return float(detail.net_capital_stock[year] / na.v_gdp[year])


def rebasing_investment_gap():
    """Investment the old accounts missed in FY2024/25, over old-base GDP:
    rebased investment restated over old-base GDP less the old-base share."""
    return (
        mp.GROSS_CAPITAL_FORMATION_REBASED / 100 / mp.rebase_factors()[0]
        - OLD_BASE_GCF_SHARE
    )


def rebased_capital_output_ratio(revision_start=CAPITAL_REVISION_START_YEAR):
    """Total capital over rebased GDP in FY2024/25: perpetual inventory on
    PWT investment with the missed investment phased in linearly from
    ``revision_start`` (None: no revision), in old-base GDP, converted."""
    pwt = pd.read_csv(DATA / "pwt110_eth_capital.csv", index_col="year")
    share = pwt.v_gfcf / pwt.v_gdp
    real_per_nominal = (pwt.q_gfcf / pwt.q_gdp) / share
    years = list(pwt.index) + list(OLD_BASE_EXTENSION)
    q_gdp, shares = list(pwt.q_gdp), list(share)
    for s_old, growth in OLD_BASE_EXTENSION.values():
        q_gdp.append(q_gdp[-1] * (1 + growth))
        shares.append(s_old)
    rpn = list(real_per_nominal) + [real_per_nominal.iloc[-1]] * len(
        OLD_BASE_EXTENSION
    )
    delta = float(pwt.delta.iloc[-1])
    gap, last = rebasing_investment_gap(), years[-1]

    def stock_output(with_gap):
        ky, k = {}, None
        for y, qy, s, r in zip(years, q_gdp, shares, rpn):
            extra = 0.0
            if with_gap and revision_start is not None:
                frac = (y - revision_start) / (last - revision_start)
                extra = gap * min(max(frac, 0.0), 1.0)
            inv = (s + extra) * r * qy
            k = inv / (delta + 0.03) if k is None else k * (1 - delta) + inv
            ky[y] = k / qy
        return ky

    old_base = (
        pwt_domestic_capital_output_ratio()
        * stock_output(True)[last]
        / stock_output(False)[2023]
    )
    return old_base * mp.rebase_factors()[0]


def fdi_stock_usd_bn(delta=None):
    """Net replacement-cost FDI stock in mid-FY2024/25, USD billion, by
    perpetual inventory on IMF equity inflows."""
    if delta is None:
        delta = mp.DELTA_ANNUAL
    flows = pd.read_csv(
        DATA / "imf_fdi_inflows_eth.csv", index_col="year"
    ).fdi_equity_inflows_usd_m
    k = 0.0
    for year in range(int(flows.index.min()), int(flows.index.max()) + 1):
        flow = flows.get(year, FDI_MISSING_YEARS_FLOW)
        k = k * (1 - delta) * (1 + FDI_PRICE_GROWTH) + flow
    return k * (1 - delta) * (1 + FDI_PRICE_GROWTH) ** 0.5 / 1e3


def sovereign_rate_scale():
    """r_gov_scale: invert the sovereign-corporate yield relationship of
    Li, Magud and Werner (2023, Table 8 column 2) by OLS of the sovereign
    yield on the fitted corporate yield, for sovereign yields of 2-12
    percent."""
    sov = np.arange(20, 120) / 10
    corp = 8.199 - 2.975 * sov + 0.478 * sov**2
    slope, intercept = np.polyfit(corp, sov, 1)
    return float(slope)


def nta_old_age_consumption():
    """NTA 2005 private consumption per person at 65-69, 70-79 and 80-85
    relative to ages 30-49 (simple means over single ages): a check on the
    fitted net family transfers (households.md)."""
    c = pd.read_csv(
        DATA / "nta_consumption_by_age.csv", comment="#"
    ).set_index("age")["private_consumption"]
    base = c.loc[30:49].mean()
    return {
        f"{a}-{b - 1}": float(c.loc[a : b - 1].mean() / base)
        for a, b in ((65, 70), (70, 80), (80, 86))
    }


def main():
    rows = [
        ("peer FDI return, nominal", peer_fdi_return(), None),
        (
            "world_int_rate_annual (less US inflation)",
            round(peer_fdi_return() - mp.US_INFLATION, 2),
            mp.WORLD_INT_RATE_ANNUAL,
        ),
        (
            "PWT capital-output ratio, domestic prices, old base",
            pwt_domestic_capital_output_ratio(),
            mp.PWT_CAPITAL_OUTPUT_RATIO,
        ),
        (
            "investment gap, old-base GDP",
            rebasing_investment_gap(),
            mp.REBASING_INVESTMENT_GAP,
        ),
        (
            "total capital / rebased GDP",
            rebased_capital_output_ratio(),
            mp.TOTAL_CAPITAL_OUTPUT_RATIO,
        ),
        ("  no revision", rebased_capital_output_ratio(None), None),
        ("  gap since 1970", rebased_capital_output_ratio(1970), None),
        ("  gap from 2011", rebased_capital_output_ratio(2011), None),
        ("  gap from 2019", rebased_capital_output_ratio(2019), None),
        ("FDI stock, USD bn", fdi_stock_usd_bn(), mp.FDI_STOCK_USD_BN),
        ("r_gov_scale", sovereign_rate_scale(), None),
    ] + [
        (f"NTA consumption {band} / 30-49", ratio, None)
        for band, ratio in nta_old_age_consumption().items()
    ]
    for name, value, constant in rows:
        held = "" if constant is None else f"   (macro_params: {constant})"
        print(f"{name:52s} {value:10.4f}{held}")


if __name__ == "__main__":
    main()
