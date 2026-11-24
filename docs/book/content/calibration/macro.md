---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

```{code-cell} ipython3
:tags: ["remove-cell"]

from importlib.resources import files
import json
from myst_nb import glue
from ogeth import macro_params as mp

params = json.loads(
    files("ogeth")
    .joinpath("ogeth_default_parameters.json")
    .read_text(encoding="utf-8")
)
glue("ky_total", f"{mp.TOTAL_CAPITAL_OUTPUT_RATIO:.2f}", display=False)
glue("ky_private", f"{mp.PRIVATE_CAPITAL_OUTPUT_RATIO:.2f}", display=False)
glue("ky_public", f"{mp.PUBLIC_CAPITAL_OUTPUT_RATIO:.3f}", display=False)
glue("investment_gap", f"{100 * mp.REBASING_INVESTMENT_GAP:.1f}",
     display=False)
glue("fdi_stock", f"{mp.FDI_STOCK_TO_GDP:.3f}", display=False)
glue("foreign_share", f"{100 * mp.FOREIGN_CAPITAL_SHARE:.1f}", display=False)
glue("zeta_K", f"{float(params['zeta_K'][0]):.3f}", display=False)
glue("domestic_debt", f"{100 * mp.DOMESTIC_DEBT_TO_GDP:.1f}", display=False)
glue("wealth_market",
     f"{mp.INITIAL_WEALTH_RATIO / mp.MODEL_PER_MARKET_GDP:.2f}",
     display=False)
glue("wealth_model", f"{mp.INITIAL_WEALTH_RATIO:.3f}", display=False)
glue("world_r", f"{mp.WORLD_INT_RATE_ANNUAL:.2f}", display=False)
glue("us_inflation", f"{100 * mp.US_INFLATION:.0f}", display=False)
glue("pwt_ky", f"{mp.PWT_CAPITAL_OUTPUT_RATIO:.2f}", display=False)
glue("kg_old", f"{mp.PUBLIC_CAPITAL_OUTPUT_RATIO_OLD_BASE:.3f}", display=False)
glue("fdi_usd", f"{mp.FDI_STOCK_USD_BN:.1f}", display=False)
```

## Calibration vintage and refresh policy

The macroeconomic parameters below are calibrated to Ethiopian fiscal year **2024/25** (Ethiopian FY runs 8 July–7 July, so FY2024/25 ended 7 July 2025), which is the OG-ETH model start year of 2025.  This is the first full-year vintage that reflects the July 2024 exchange-rate float, and it materially changes the debt block relative to the pre-float FY2023/24 numbers used in earlier calibrations.

Following the approach used in the sibling models OG-PHL and OG-IDN, only parameters whose documented source is genuinely a live data feed are refreshed when `Calibration(update_from_api=True)` runs.  For OG-ETH that is `g_y_annual` (World Bank WDI) alone; the capital share `gamma` is a triangulated value ({ref}`Chap_FirmCalib`) and is off the live path, so a refresh cannot overwrite it with the naive ILOSTAT labour share.  The example script makes no live calls.  Every other parameter on this page is a documented, point-in-time value held in [`ogeth_default_parameters.json`](https://github.com/EAPD-DRB/OG-ETH/blob/main/ogeth/ogeth_default_parameters.json) so that a live update cannot silently overwrite a value sourced from a report table or a national statistical office.  In particular the World Bank Quarterly Public Sector Debt pull and the IMF Government Finance Statistics pull are switched off (the QPSD database has no Ethiopia coverage, and the IMF SDMX endpoint returns only 2002-vintage data for Ethiopia).

(Sec_Rebasing)=
### GDP basis: the rebased national accounts

Every ratio to GDP in this calibration is to Ethiopia's **rebased** national accounts.  In November 2026 the Ministry of National Strategy (formerly the Ministry of Planning and Development), with IMF technical assistance, moved the base year from 2015/16 to **2024/25** (2017 EFY), adopted the 2008 System of National Accounts and ISIC Rev. 5, and widened coverage using the 2025 agricultural and economic-establishment censuses, the integrated household survey, 37 federal and all regional state-owned enterprises, non-profit institutions and digital finance ({cite}`MoNS:2026`).  FY2024/25 nominal GDP went from 15,094 to **24,031 billion birr** (USD 126.5 to 201.4 billion at the NBE's average rate of 119.3; Table 2), and FY2025/26 (preliminary) is **31,934 billion birr** (Table 5).  The report's text also gives 31.86 and 31.97 trillion for FY2025/26; we use Table 5's figure because it is the sum of the table's components and the denominator of the report's own sector shares.  The figures used here are in `ogeth/data/mons_rebased_gdp_fy2024_25.csv`.  Nothing real changed overnight: the economy was always that large, and the old series did not see about 37 percent of it.

What the rebasing does to a calibration depends on the numerator:

- **Fiscal flows and stocks in birr, and balance-of-payments flows in dollars, keep their numerators.**  Their rebased ratio is the published ratio times old GDP over new GDP of the same year: 15,094/24,031 = 0.628 for FY2024/25 and 19,268/31,934 = 0.603 for FY2025/26 (`macro_params.rebase_factors`).  The IMF tables are kept exactly as published (on the old base) and converted in code, so every number remains traceable to its table.  Years the rebasing has not yet reached — FY2026/27 onward, still old-base IMF projections — take the FY2025/26 factor until the IMF re-bases its projections.  Public debt falls from 50.5 to 31.7 percent of GDP on the program's wide perimeter, or from 43.3 to 27.2 percent for general government (below); tax revenue from 7.8 to 4.9 percent.
- **National-accounts components were revised in their numerators too**, so they are not rescaled but taken from the rebased accounts directly ({cite}`MoNS:2026`, Table 4, FY2024/25): private consumption 64.4 percent of GDP (80.9 on the old base; households 61.6 and non-profit institutions serving households 2.8), gross capital formation **35.3** percent (20.1), government consumption 4.7 percent (5.8) and net exports −4.4 percent (−6.9).  The rebased net exports agree with the IMF's balance-of-payments goods-and-services balance over rebased GDP (−4.5).
- **Growth rates and household surveys are unaffected**: hours, earnings and wealth distributions, demographics and productivity growth do not depend on the GDP level.  The rebased back series for 2016–2024 has not been published, so `g_y_annual` stays on the old-base growth rates.
- **Capital stocks are rebuilt on the rebased investment.**  The Penn World Table and the IMF capital-stock dataset are built on the old accounts, and the rebased accounts revise investment up far more than GDP: 35.3 against 20.1 percent of GDP in FY2024/25, investment by regional state-owned enterprises and non-profit institutions the old accounts missed.  The old accounts had investment at 35–38 percent of GDP in 2013–2017, falling to 20 percent by FY2024/25, so the undercount built up after the old base year.  We rebuild the total stock by perpetual inventory from PWT 11.0's investment and GDP at constant national prices, add the investment the old accounts missed, phased in linearly from FY2015/16, and scale PWT's 2023 ratio of net capital stock to GDP, both at current national prices ({glue:text}`pwt_ky` of old-base GDP), by the revised over the unrevised stock-output ratio (`scripts/derive_sourced_values.py`).  The inventory runs in units of old-base GDP, so the missed investment is measured there too: rebased investment, 35.3 percent of rebased GDP, is 56.2 percent of old-base GDP, so the gap is {glue:text}`investment_gap` points of old-base GDP in FY2024/25 rather than the 15.2-point difference between the two published shares (`scripts/derive_sourced_values.py`); the resulting stock is then converted to rebased GDP like every other ratio.  Total capital is **{glue:text}`ky_total`** times GDP, in a range of 1.39 (no revision before FY2024/25: the old-base stock rolled forward, over the larger rebased GDP) to 2.94 (the gap in every year since 1970); phasing in from 2019 or 2011 gives 2.00 or 2.42.  Public capital is the IMF ICSD general-government stock at current prices, {glue:text}`kg_old` of old-base GDP in 2019 (the ratio at domestic current prices, matching the total stock), a stock of accumulated budget spending in birr that converts like other fiscal stocks ({glue:text}`ky_public` of rebased GDP), so private capital is {glue:text}`ky_private`.  The stocks remain the calibration's provisional element until the Ministry publishes capital stocks or back series, and the capital-output ratio is reported as an outcome rather than targeted ({ref}`Sec_Validation`).

The calibration also uses GDP **at market prices**, the basis of every published ratio.  OG-Core's output $Y$ excludes the consumption tax (it is collected on top of the producer price), so shares OG-Core applies to its own $Y$ — spending, transfers, public investment, grants, remittances, the debt ratios, initial wealth and public capital — are the data share times $1/(1-\tau^{c}\text{-revenue share}) = 1.031$ (`macro_params.MODEL_PER_MARKET_GDP`), and model moments are scored over $Y$ plus the consumption tax (`ogeth/targets.py`).

## Economic Assumptions

As the default rate of labor-augmenting technological change, $g_y$, we use a value of **4.7%**.  This is the average annual growth rate of GDP per capita in Ethiopia over **2016–2025**, according to [data from the World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.KD.ZG?locations=ET) (series `NY.GDP.PCAP.KD`).

$g_y$ is the *permanent* balanced-growth-path rate of labor-augmenting technical change, so it should reflect Ethiopia's sustainable long-run growth rather than a historical episode.  We deliberately do not use the longer 2006–2024 window, whose ~6.0% average embeds the 2004–2015 state-led investment boom that the [IMF Country Report 26/20](https://www.imf.org/en/publications/cr/issues/2026/01/29/the-federal-democratic-republic-of-ethiopia-fourth-review-under-the-extended-573522) Debt Sustainability Analysis explicitly assumes will not repeat ("growth slower than historical rates of around 10 percent per annum over the two decades prior to 2019").  The post-2015 decade averages over the post-boom normalization, the 2020–2022 conflict-and-COVID dip, and the recent gold-export-driven recovery, giving a more defensible long-run anchor.  The IMF's own medium-term path (~8% real GDP growth) implies roughly 5.6% per-capita growth on UN population projections, which we treat as a program-optimistic upper bound rather than the balanced-growth rate.

## Open Economy Parameters

### Foreign holding of government debt in the initial period

OG-Core's foreign-held debt $D^f$ is the part of **market debt**, the debt issued in the domestic market, held by non-residents; debt the government contracts abroad is modelled separately at its own interest rate ({ref}`Sec_ExternalDebt`).  General-government external debt at end-June 2025 (FY2024/25) is 31.8 percent of old-base GDP less the 7.2 points of NBE long-term foreign liabilities and Ethio Telecom external debt that the IMF's program perimeter includes and the budget does not service, 56.8 percent of general-government gross debt of 43.3 percent (IMF World Economic Outlook, October 2026); all of it is debt contracted abroad.  Market debt is the domestic debt, 18.7 percent of old-base GDP, in birr Treasury bills and bonds held by the central bank, banks and pension funds ({cite}`IMFCR26174:2026`); Ethiopia's securities market opened to non-residents only in 2025, so `initial_foreign_debt_ratio` is 0 (`macro_params.MARKET_FOREIGN_DEBT_SHARE`).  The July 2024 float (the birr moved from about 57 to about 135 per US dollar) mechanically raised the birr value of external debt, so the external share of public debt jumped from about 45 percent (FY2023/24) to about 63 percent (FY2024/25) on the wider perimeter.

### Foreign purchases of newly issued debt

$\zeta_D$ is the share of new **market** debt bought by non-residents, and we set it to 0, like the initial share (`macro_params.zeta_D_path`).  New borrowing abroad enters through the path of external debt instead ({ref}`Sec_ExternalDebt`): the program sets a zero limit on new non-concessional external borrowing by the government, the central bank and public enterprises, with a single capped exception for the Koysha dam ({cite}`IMFCR26174:2026`, continuous performance criterion), and new concessional loans are part of the DSA's external debt path.  Before external debt was modelled at its own rate, $\zeta_D$ carried the external share of all new borrowing (0.15, with a program path for the share of the debt decline borne by external creditors).

### Foreign holdings of excess capital

$\zeta_K$ is tuned to the foreign-owned share of private capital in FY2024/25, {glue:text}`foreign_share` percent: the inward FDI stock at replacement cost over the private capital-output ratio of {glue:text}`ky_private` on the rebased accounts ({ref}`Sec_Rebasing`).  Ethiopia publishes no international investment position.  UNCTAD's inward FDI stock (World Investment Report 2026: USD 42.5 billion at end-2024) is essentially the cumulative IMF balance-of-payments equity inflows at historical cost, with no depreciation; the model's foreign-owned capital is a net stock at replacement cost like its $K$, so we accumulate the same inflows by perpetual inventory at the private depreciation rate, revalued 2.5 percent a year for the dollar price of capital goods: USD {glue:text}`fdi_usd` billion in mid-FY2024/25, {glue:text}`fdi_stock` of rebased GDP (`scripts/derive_sourced_values.py`).  Unrecorded reinvested earnings bias both measures down.  The share is scored in the first transition year, where the data are, because with initial wealth anchored to the data the first-year split between domestic and foreign capital depends on $\zeta_K$ (OG-PHL's practice).  The tuned value, {glue:text}`zeta_K`, sits below the prior from Ethiopia's **normalized Chinn-Ito capital-account-openness index**, which is 0.162 in the 2023 vintage and flat since the 1990s (the [Chinn-Ito KAOPEN dataset](https://web.pdx.edu/~ito/Chinn-Ito_website.htm)).  This places Ethiopia in the second-lowest openness bin, just above fully-closed economies and well below the investment-grade siblings OG-PHL (0.45) and OG-IDN (0.42).  It is cross-checked against the flow evidence: foreign direct investment of about USD 4.0 billion in FY2024/25 ([Ethiopian Investment Commission](https://unctad.org/publication/world-investment-report-2025)) against gross fixed capital formation of about 20 percent of a roughly USD 126 billion economy implies foreign financing of new capital in the low-teens percent (about 11–16 percent depending on the window), which the FDI-stock level now replaces as the target.  Ethiopia is a near-closed capital account: the [IMF Annual Report on Exchange Arrangements and Exchange Restrictions](https://www.imf.org/en/Publications/Annual-Report-on-Exchange-Arrangements-and-Exchange-Restrictions) documents binding foreign-exchange controls until the July 2024 reform, no resident access to international portfolio markets, and only nascent foreign participation via the Ethiopian Securities Exchange (launched January 2025) and the 2024 banking-sector opening.  The index locates the plausible range; the FDI stock sets the level.

### World interest rate

We set `world_int_rate_annual =` {glue:text}`world_r`, the real return foreign investors earn on direct investment in economies like Ethiopia's.  In OG-Core the world rate is not a portfolio benchmark: it fixes $K^{open}$, the capital stock firms would demand at that return, and foreign capital fills a share $\zeta_K$ of the gap between it and what domestic wealth finances, $K_f = \zeta_K\,(K^{open} - K_d)$.  Ethiopia reports almost no direct-investment income in its balance of payments (no reinvested earnings since 2005), so its own return cannot be measured.  We use its African peers instead: direct-investment income paid abroad over the average of opening and closing direct-investment liabilities, pooled over countries and the years 2015–2024 (IMF balance of payments and international investment position, `ogeth/data/imf_fdi_income_africa_peers.csv`), is 6.1 percent nominal, about {glue:text}`world_r` real after {glue:text}`us_inflation` percent dollar inflation (`scripts/derive_sourced_values.py`).  The same rate is the long-run target for the return on capital that sets patience ({ref}`Chap_HouseholdCalib`).  Ethiopia is a distressed sovereign (it defaulted on its single Eurobond and is still restructuring), so its own cost of funds is calibrated separately below.

## Government Debt, Spending and Transfers

### Government Debt

The path of government debt is endogenous, but the initial value and the steady-state (long-run) value are exogenous.  To avoid converting between model units and birr, we calibrate the initial debt-to-GDP ratio rather than the birr value of the debt.  The debt has two parts: **market debt**, OG-Core's $D$, the domestic debt, whose initial ratio is `initial_debt_ratio` (0.121 in model units) and which the closure rule acts on, and **debt contracted abroad**, `alpha_D_ext` (0.159 in FY2024/25), at its own interest rate ({ref}`Sec_ExternalDebt`).  Together they are general-government debt, and the sovereign premium prices them together.

The debt is **general-government** gross debt at end-June 2025 (FY2024/25): 43.3 percent of old-base GDP in the IMF's World Economic Outlook (October 2026), **27.2 percent of rebased GDP** ({ref}`Sec_Rebasing`), and 0.280 in the model's output units.  That is the perimeter of the budget the model's government runs — the revenue, spending and interest the fiscal block is calibrated to are general-government figures — while the fifth ECF review's wider public debt, 50.5 percent of old-base GDP (31.7 rebased), adds about 7.2 points of NBE long-term foreign liabilities and Ethio Telecom external debt that the budget does not service ({cite}`IMFCR26174:2026`, Table 1).  On either perimeter the jump from FY2023/24 (35.5 percent of old-base GDP) is almost entirely valuation: the July 2024 float raised the birr value of external debt.  The debt block is calibrated on the post-float basis throughout.

The steady-state ratio is `debt_ratio_ss`.  It anchors market debt, and external debt is repaid by the late 2040s, so in the long run all debt is market debt and the anchor is a ratio of all debt.  The baseline sets it to the measured FY2024/25 ratio of all general-government debt, 0.280: the long-run anchor is a measurement, and OG-Core's closure rule stabilizes debt where it starts rather than at a plan, refinancing the falling external stock with market debt along the way.

OG-Core's debt-closure rule, which adjusts government consumption toward `debt_ratio_ss`, starts at `tG1 = 7` (FY2031/32).  Before that the baseline holds its inputs at the FY2025/26 budget values and lets debt fall or rise as the budget identity dictates.

(Sec_ExternalDebt)=
#### Debt contracted abroad

All of Ethiopia's general-government external debt is borrowed directly from foreign lenders, outside the domestic bond market, and OG-Core (PSLmodels/OG-Core#1279, for Issue #1125: `alpha_D_ext` and `r_D_ext_annual`) models it as a stock held abroad at its own interest rate: each year the government receives the new external borrowing that keeps the stock at its share of GDP and pays interest on it, and the interest is paid abroad.  It is not part of households' portfolio, so it does not earn the portfolio return $r_p$; without it every unit of debt held abroad earned that return, which in the first year is far above anything Ethiopia pays on its external debt.  Most of it is concessional: IDA holds about 39 percent and official non-Paris Club creditors 31 percent, and non-official creditors hold 10 percent, chiefly the defaulted USD 1 billion Eurobond ({cite}`IMFCR26174:2026`, DSA).

- **The stock.**  0.159 of model output in FY2024/25 (`macro_params.EXTERNAL_DEBT_RATIO`).  It moves with the public external debt ratio of the program's debt sustainability analysis, 31.8 percent of old-base GDP in FY2024/25, 16.8 in FY2030/31, 9.2 in 2036 and 4.1 in 2045 ({cite}`IMFCR26174:2026`, Table 2), as amortization outpaces new concessional loans and growth erodes the ratio.  Ethiopia's income was 74 percent of IDA's operational cut-off when the program was approved, and the grant element of new external borrowing stays near 47 percent to FY2044/45, so concessional terms last through the DSA horizon and end with IDA graduation after it: beyond 2045 the ratio falls to zero at its 2036–2045 pace, by the early 2050s (`macro_params.external_debt_path`).  The concessional subsidy is therefore temporary; in the steady state there is none, and all debt is market debt.
- **Its rate.**  `r_D_ext_annual` is the DSA's average nominal interest rate on all external debt, the Eurobond included, less 2 percent US inflation, with the real exchange rate held constant after the float (the model has no exchange rate, so a rate on foreign-currency debt is stated in units of output this way): $-1.3$ percent real in FY2024/25 (restructuring relief), rising to 0.6 percent in FY2026/27 as the program's IMF and blended loans are drawn and the interest relief ends, and back to about $-0.4$ by FY2030/31 and $-1.1$ by 2045 as IDA credits dominate (`macro_params.external_rate_path`).  This path is how the grant element enters the model: concessionality is a low interest rate on the stock, not a transfer, and `alpha_FA` holds only the budget's cash grants (IMF Table 2b), so nothing is counted twice.
- **The premium and the closure.**  OG-Core prices the debt-elastic sovereign premium on all government debt, market plus external, so its anchor is a ratio of all debt; the closure rule and `debt_ratio_ss` act on market debt, which in the long run is all debt.  The stock is a path, so a reform that changes borrowing changes market debt only unless it also changes `alpha_D_ext`.
- **The dashboard** scores total debt, $D + D^{ext}$, against the data, and counts interest on external debt in the payments abroad (`ogeth/targets.py`).

#### Interest rates on government debt

We assume a wedge between the real rate of return on private capital and the real interest rate on government debt.  OG-Core models this as a scale term, a level shift, and (optionally) a debt-elastic premium:

```{math}
:label: eqn:r_gov
    r_{gov,t} = \max\left\{(1-\tau_{d})r_t - \mu_{d,t} + \beta_1\frac{D_t}{Y_t} + \beta_2\left(\frac{D_t}{Y_t}\right)^2,\; 0\right\}
```

where $\tau_d$ is the scale parameter (`r_gov_scale`), $\mu_{d,t}$ is the level shift (`r_gov_shift`, a time path), and $\beta_1$ (`r_gov_DY`), $\beta_2$ (`r_gov_DY2`) govern the debt-elastic premium.  The scale is set to 0.245 by inverting the estimated relationship between corporate and sovereign yields in {cite}`LMW2023` (Table 8, Column 2): we simulate corporate yields for sovereign yields between 2 and 12 percent and estimate the scale and shift that best fit these simulated data by ordinary least squares (reproduced by `scripts/derive_sourced_values.py`).  The **level**, however, is anchored to Ethiopia's actual real effective interest rate on public debt rather than to the cross-country intercept, because it multiplies the whole debt stock in the government's interest bill and so decides what primary balance the debt path requires.

With external debt at its own rate, $r_{gov}$ is the rate on **market debt**, the domestic Treasury bills and bonds, and its level is the measured real rate on that debt.  The DSA publishes it: the average real interest rate on domestic debt was $-11.0$ percent in FY2024/25 under financial repression (mandatory purchases of Treasury bonds at sub-market rates, ended June 2025) and rises to $-9.8$, $-5.1$, $-0.6$, 1.0, 2.6 and 3.3 percent over FY2025/26–FY2030/31 and 4.6 percent in 2036 as the legacy instruments roll over at market rates ({cite}`IMFCR26174:2026`, public DSA table; policy and Treasury-bill rates turned positive in real terms by 2026, ¶12).  It follows from the stock rolling over at the market rates already in place, not from future policy.  We calibrate $\mu_{d,t}$ so that $r_{gov}$ follows it year by year to 2036 (`macro_params.market_rate_targets`, `market_r_gov_shift_path`), and lower `r_gov_floor` to $-12$ percent so the first-year rate binds.  The **long-run** rate is 4.7 percent, the mean of the DSA's terms for new domestic borrowing (4.9 percent real on medium- and long-term debt, 4.5 on short-term debt, its baseline ten-year projections), which the DSA's average domestic rate reaches by 2036.  Its 2045 value, 6.6 percent, is not used: by then the DSA has public debt down to 13.6 percent of GDP and a thin domestic market, while the model holds debt at its FY2024/25 ratio.  The blended rate on all FY2024/25 debt, market at 4.7 and external near $-1$ percent, is close to the 2 percent long-run effective rate this calibration used before external debt was modelled at its own rate.

Following OG-PHL and OG-IDN, we also turn on a **debt-elastic sovereign premium** — the crowding-out-via-risk channel that a distressed sovereign like Ethiopia faces but that OG-Core leaves off by default.  We use the convex form of {cite}`SchmittGrohe:2003`, $\beta_2\left(D_t/Y_t - \bar{D}\right)^2$ with $\beta_2 = 0.04$, and **center it on the steady-state debt ratio** $\bar{D} =$ `debt_ratio_ss` so the premium is exactly zero at the target and does not move the steady state; it only prices deviations of debt from target.  Expanding the square and folding the constant into the level shift (OG-Core subtracts `r_gov_shift`) gives

```{math}
:label: eqn:r_gov_center
    \beta_1 = -2\beta_2\bar{D} = -0.022, \qquad
    \mu_{d,t} = \tau_d\,\bar{r} - r^{target}_{gov,t} + \beta_2\left(d_t - \bar{D}\right)^2 - \beta_2\bar{D}^2,
```

evaluated in the baseline at $d_t = \bar{D}$ in every year, so the premium term drops out of the shift and prices only the deviations of the solved debt path from the anchor, with $\bar{r}$ the steady-state return on capital (`macro_params.R_SS_FOR_R_GOV`).  The transition's $r$ starts about 5 percentage points above its steady state (13.1 against 8.0 percent), which the 0.245 scale turns into a sovereign rate about 1.2 points above its target in the first year: $-9.8$ percent against the measured $-11.0$.  The sign of the last term follows from OG-Core's formula, $\beta_1 d + \beta_2 d^2 = \beta_2(d-\bar{D})^2 - \beta_2\bar{D}^2$: the shift must *subtract* $\beta_2\bar{D}^2$ to cancel the expansion's constant.  (The earlier calibration added it, which had put the long-run sovereign rate about 0.7 percentage points below the level it documented.)  The path is built by `macro_params.market_r_gov_shift_path` and pinned by `tests/test_fiscal_program.py`.


### Aggregate transfers

$\alpha_T$ is the ratio of cash transfers to households to GDP. Figures in this section are percent of *old-base* GDP as the IMF publishes them; the calibration converts them to rebased GDP (times 0.628 for FY2024/25 and 0.603 for FY2025/26, {ref}`Sec_Rebasing`) and to the model's output units (times 1.031).  The IMF's general-government accounts bundle these into recurrent spending together with the wage bill, goods and services and interest, so the household-transfer part has to be built up from its components ({cite}`IMFCR26174:2026`, ¶9, ¶20–22 and the social-safety-net box): explicit fuel and fertilizer subsidies of about 1 percent of GDP in FY2024/25 — fuel subsidies were effectively eliminated by March 2026, with a capped envelope in FY2026/27 to smooth the 2026 oil-price shock, while fertilizer imports (about 1 percent of GDP) remain subsidized — the Productive Safety Net Program (about 0.4 percent of GDP), and public pension payouts, which the program's figures count at a round half percent of GDP.  That gives 2.0, 1.8 and 1.5 percent of GDP for FY2024/25–FY2026/27 and 1.5 percent thereafter.  The pension schemes' own accounts give much less: the public servants' scheme paid 22.2 billion birr in 2024 to 750,812 pensioners, and with the private scheme's 43,347 at the same average benefit all public pensions were about 0.20 percent of old-base GDP (ILO Social Security Inquiry), of which old-age pensions are about 60 percent and the rest survivor and invalidity benefits.  We replace the program's half percent with that 0.20.  Old-age pensions are paid by OG-Core's pension block ({ref}`Chap_Tax`), so they are taken out of $\alpha_T$ (`macro_params.program_spending_paths`), which holds the other cash transfers, survivor and invalidity benefits included: 1.02 and 0.86 percent of the model's output in FY2024/25 and FY2025/26, held thereafter.

The transfers are targeted, not spread evenly.  OG-Core's `eta` matrix allocates aggregate transfers across ages and lifetime-income groups and defaults to population shares; we build it by programme (`macro_params.transfer_eta`), year by year from each year's population (`macro_params.transfer_eta_path`), so that each person's transfer keeps its intended size as the population ages (OG-Core accepts `eta` as a path over time, PSLmodels/OG-Core#1297).  The safety net (0.4 percent of GDP, the whole programme, of which the government finances about 30 percent) goes to the poorest quarter of households by its own design: a household with an able-bodied adult is a public-works client, paid for six months a year in exchange for work, and one without receives Permanent Direct Support, paid for twelve months at the same monthly rate per person (PSNP IV Programme Implementation Manual, 2014; PSNP5 project appraisal document, 2020).  The model has no households or disability, so direct support is proxied by age, the poorest group's members from 60: they receive twice as much per person a year as the younger members, which gives direct support 19 percent of the programme's value in 2025 against about 26 percent in the programme (the difference is disabled and chronically ill clients under 60).  The fertilizer subsidy (0.6 percent) goes per capita to the bottom 70 percent — smallholder agriculture — and survivor and invalidity benefits to the households of the two formal groups the pension schemes cover.  The poorest quarter receives 57 percent of transfers rather than its 25 percent population share, the next two groups 20 and 16 percent, the two formal groups 7 percent between them and groups 4 and 5 nothing.  The composition is the long-run one; the fuel subsidy that inflates FY2024/25–FY2025/26 is spread with the rest.

The IMF's *poverty-reducing expenditure* line (2.3 rising to 3.5 percent of GDP) is not a household transfer: it is education, health, agriculture, roads and water spending, which in the national accounts is government consumption and public investment and is therefore already inside $\alpha_G$ and $\alpha_I$.  Likewise the federal block grant to the regions (about 1.8 percent of GDP) is an intergovernmental transfer, not a household one.


### Government expenditures

$\alpha_G$ is government consumption of goods and services relative to GDP. Figures in this section are percent of *old-base* GDP as the IMF publishes them; the calibration converts them to rebased GDP (times 0.628 for FY2024/25 and 0.603 for FY2025/26, {ref}`Sec_Rebasing`) and to the model's output units (times 1.031).  We take it from the same general-government cash accounts as the rest of the fiscal block, so that the government budget identity the debt path obeys is the one the data describe: recurrent expenditure net of interest (which the model pays through $r_{gov}$) and net of the cash transfers in $\alpha_T$.  From {cite}`IMFCR26174:2026`, Table 2b, this gives 4.6 percent of old-base GDP in FY2024/25 and 5.4 percent in the FY2025/26 budget, which restores real public-sector wages; the baseline holds the budget value.  On the rebased national accounts government consumption is 4.7 percent of GDP in FY2024/25 ({cite}`MoNS:2026`, Table 4), against 2.9 for $\alpha_G$: the national-accounts figure also counts the consumption of fixed capital on public capital, which the dashboard adds to the model's $G$, and spending that the cash accounts record elsewhere.  Public investment is calibrated separately via $\alpha_I$, the general-government capital expenditure, 5.0 percent of old-base GDP in FY2024/25 and 5.8 in the FY2025/26 budget, held (see {ref}`Chap_FirmCalib` for the public-capital block).

Built along the IMF program's years, the three spending paths reproduce the program's primary expenditure (total expenditure less interest) to within a tenth of a percent of GDP in every year, which `tests/test_fiscal_program.py` checks; it confirms the mapping from the IMF tables to the model's instruments.


> **A note on federal versus general government.**  Ethiopia is a federation, and the regional states execute much of the pro-poor spending and most of the education and health wage bill, financed largely by a federal block grant (about ETB 214 billion in FY2023/24, 1.8 percent of old-base GDP — an *intergovernmental* transfer, not a household transfer, and correctly excluded from $\alpha_T$).  Every fiscal figure used here is **general government**, which consolidates federal and regional accounts and nets the block grant out: revenue by line from the NBE's general-government table (rural and urban land-use fees, for instance, are regional), spending from the IMF's general-government tables, and debt on the general-government perimeter.  Separately, much of Ethiopia's public investment runs *off-budget through state-owned enterprises*, federal and regional, so general-government spending (12 percent of old-base GDP, 7.5 percent of rebased GDP, in FY2024/25) understates the broader public sector — which is why the wide public-sector debt (31.7 percent of rebased GDP, including the National Bank and Ethio Telecom) exceeds general-government debt (27.2).  The rebasing's large upward revision to investment is, by the Ministry's account, mostly investment by regional state-owned enterprises that the old accounts missed; the model counts it in private capital $K$, since those enterprises sit outside general government, and keeps $\alpha_I$ and public capital $K_g$ on the on-budget general-government figures.

### Foreign aid

Ethiopia is one of the world's largest aid recipients, and on-budget grants to the government were 1.7 percent of GDP in FY2024/25, elevated by post-reform donor budget support.  Figures in this section are percent of *old-base* GDP as the IMF publishes them; the calibration converts them to rebased GDP (times 0.628 for FY2024/25 and 0.603 for FY2025/26, {ref}`Sec_Rebasing`) and to the model's output units (times 1.031).  OG-Core's `alpha_FA` parameter routes foreign aid to the domestic government as a share of GDP: it is a government resource (it funds spending and reduces the deficit) and an external inflow in the resource constraint.  It accepts a time path, and we set it to the FY2024/25 outturn, 1.7 percent of old-base GDP, and the FY2025/26 budget, 1.3 percent ({cite}`IMFCR26174:2026`, Table 2b), held thereafter.

(Sec_Scenarios)=
### The baseline

A calibration has to assume something about every fiscal and external input in every future year.  Measured history and the start year pin down only the starting values; for the years after, the baseline holds the latest official values rather than following anyone's projections.

The packaged parameters (`ogeth_default_parameters.json`; `macro_params.fiscal_params`) are the baseline OG-ETH proposes, and `examples/run_og_eth.py` runs it together with a corporate-tax-cut reform.  Every input takes an official outturn or enacted value and is held there: the FY2024/25 outturn for model period 0 and the enacted FY2025/26 budget for period 1 ({cite}`IMFCR26174:2026`, Table 2b, the budget-year column), then the budget-year value for the long run.  The income tax uses the statutory schedule in force, Proclamation 1395/2025 from FY2025/26, because it is law; nothing else that the program *projects* — revenue measures, base broadening, the decline of grants, the borrowing mix, the fall in remittances, the debt target — is imposed.  Debt, growth, the current account and the trade balance are model outcomes, compared with the IMF projections under {ref}`Sec_Validation` but not fitted to them.  Two inputs are forward-looking by necessity the real rate on market debt follows the DSA's average real rate on domestic debt, because it follows from the existing stock rolling over at market rates already in place rather than from future policy (*Interest rates on government debt* above); and the external-debt stock and its rate follow the DSA's external-debt path ({ref}`Sec_ExternalDebt`).

| Input | Baseline (packaged) | Source |
|:---|:---|:---|
| `alpha_G`, `alpha_T`, `alpha_I` (% rebased GDP) | 2.9, 1.0, 3.1 (FY2024/25); 3.3, 0.8, 3.5 (FY2025/26 budget), held | IMF CR 26/174, Table 2b, rebased |
| `alpha_FA` (% rebased GDP) | 1.1; 0.8 (budget), held | IMF CR 26/174, Table 2b, rebased |
| `tau_c`, CIT collections factor | 0.040, 0.376 (tuned to FY2024/25 collections), held | NBE AR 2024/25 |
| Income-tax compliance | FY2024/25 calibration, held | {ref}`Chap_Tax` |
| Income-tax schedule | Proclamation 979/2016 in FY2024/25, 1395/2025 from FY2025/26 | law |
| `zeta_D` | 0 (market debt held at home) | NBE; CR 26/174 |
| `alpha_D_ext`, `r_D_ext_annual` | DSA external-debt path and rate ({ref}`Sec_ExternalDebt`) | IMF CR 26/174, Table 2 |
| `debt_ratio_ss` (model units) | 0.280 (measured FY2024/25 general-government debt, 27.2 percent of GDP) | IMF WEO Oct 2026 |
| Real rate on market debt | DSA real rate on domestic debt, −11.0% in FY2024/25 to 4.6% in 2036, then 4.7% | IMF CR 26/174, Table 2 |
| `alpha_RM_1`, `alpha_RM_T` (% rebased GDP) | 3.5, 3.5 | IMF CR 26/174, Table 4a, rebased |
| `tG1` | 7 | |

`macro_params.baseline_params(p)` builds the set from the tables, `update_baseline.py` writes it, and `tests/test_fiscal_program.py` pins it.

(Sec_InitialWealth)=
### Initial household wealth

The transition starts from a predetermined stock of household wealth.  By default OG-Core sets it to the steady-state level, which for Ethiopia would mean starting the economy with the capital stock it will only reach after decades of catch-up.  OG-Core's `initial_wealth_ratio` anchors aggregate initial wealth to the data instead: household wealth in the model is domestically owned capital plus domestically held government debt, so its data counterpart is private capital of {glue:text}`ky_private` times GDP on the rebased accounts ({ref}`Sec_Rebasing`), less the inward FDI stock of {glue:text}`fdi_stock` of GDP (above), plus domestically held general-government debt of {glue:text}`domestic_debt` percent of rebased GDP (18.7 percent of old-base GDP; {cite}`IMFCR26174:2026`, Table 1) — {glue:text}`wealth_market` times GDP at market prices, **{glue:text}`wealth_model`** in the model's output units (`ogeth.macro_params.INITIAL_WEALTH_RATIO`, set in the packaged parameters).  With OG-Core's Anderson outer loop the anchor is accelerated together with the prices (PSLmodels/OG-Core#1280); damped on its own it took about three times as many transition iterations.

The level leaves the *shape* open: how initial wealth is spread over ages and lifetime-income groups (`initial_hhwealth_dist`).  OG-Core's default, the steady-state shape, describes households that have lived under the long-run demographics, and here it handed first-year households a windfall.  Ethiopia has no wealth-by-age data (its National Transfer Accounts cover labour income only), so the shape is imputed from the model by a burn-in (`ogeth/initial_wealth.py`, applied by `examples/tune.py --tpi`): each tuning round moves the starting shape 30 percent of the way toward the savings households choose in the first period, at ages below 80 (the oldest cohorts' first-period savings are mostly bequests, and feeding them back stalled the transition).  The burn-in ran with the tuning rounds and stopped with them, when every fiscal and foreign-capital moment was on target and stable to within 0.03 points between rounds.  It is not converged cell by cell.  In the baseline the largest relative difference between the starting shape and the first-period choice (`initial_wealth.shape_gap`, over cells holding at least 0.1 percent of wealth) is 29 percent, at age 22 in the second group, a cell holding 0.13 percent of initial wealth; weighted by wealth the difference is 4.6 percent, the ages 21–24 that carry most of it hold 2.2 percent of wealth, and aggregate detrended wealth changes 0.33 percent from the first period to the second.  The equilibrium itself is solved to the usual tolerances; this gap is about the imputed starting distribution, and early-year reform results inherit that uncertainty.


### Remittances

Remittances are a defining feature of Ethiopian household income, and the model calibrates their level, their growth along the transition, and how they are shared out across households.  The theory is in the OG-Core chapter "[Remittances](https://pslmodels.github.io/OG-Core/content/theory/households.html#remittances)".

#### Level

The IMF's fifth ECF review ({cite}`IMFCR26174:2026`, Table 4a) records net private transfers of USD 7,037 million in FY2024/25 — 5.6 percent of old-base GDP, larger than FDI — the first full post-float year and the vintage the rest of this page uses.  On rebased GDP (USD 201.4 billion) the same inflow is **3.5 percent of GDP**, and the baseline holds it there: `alpha_RM_1 = alpha_RM_T` = 0.0363 in the model's output units.  The World Bank WDI personal-remittances series (the BPM6 measure: personal transfers plus compensation of employees) puts the same inflow at USD 7.14 billion for calendar 2024; its lower GDP ratio (4.8 percent) reflects only a dollar-GDP denominator that straddles the July 2024 float, so the two sources agree on the flow.  Two caveats on the level.  The IMF line does not separate household remittances from transfers to NGOs, so it is an upper bound on household receipts; and formal channels capture only part of what the diaspora sends — market studies put informal flows at 65–80 percent of the total ({cite}`UNCDF:2022`) — so the measured figure is a floor on the true inflow.  We use the balance-of-payments value because it is the one consistent with the trade and current-account data the open-economy block is calibrated to.  The IMF projects the ratio peaking near 6 percent of old-base GDP in FY2025/26 and easing to 4.1 percent by FY2030/31 as GDP outgrows transfers; the baseline holds the measured share.  Turning remittances on is what lets the model run a trade deficit at all: like aid, remittances are an external inflow that finances imports in excess of exports (how the model's trade balance and current account compare with the data is taken up under Validation).

#### Growth path

`g_RM` is not the growth rate of remittances in dollars.  Before $t_{G1}$ OG-Core advances *detrended* remittances by $(1+g_{RM,t})/\left(e^{g_y}(1+g_{n,t-1})\right)$, independently of that period's output, so remittances keep pace with trend GDP only when that factor is exactly one.  With Ethiopia's productivity and population growth (about 4.7 and 2–3.5 percent a year) a scalar `g_RM = 0` shrinks remittances relative to trend GDP by about 7 percent a year — from 5.6 percent toward 1.5 percent over the first twenty years — before OG-Core blends them back to `alpha_RM_T`.  $\alpha_{RM,1}$ and $\alpha_{RM,T}$ pin only the endpoints.  We therefore build $g_{RM,t}$ on $e^{g_y}(1+g_{n,t-1}) - 1$, the path derived from the packaged `g_n` that keeps remittances exactly on trend GDP (because $g_n$ falls over the transition no single constant does this).  Along the transition $RM/Y$ also moves inversely with output's deviation from trend; output stays close to trend, so the baseline holds 3.5 percent of rebased GDP to within 0.1 point in each of the first seven years.  The path is rebuilt with the demographics by `update_baseline.py` (the derivation lives in `ogeth.macro_params`), and `tests/test_remittances.py` checks it.

#### Distribution across households

`eta_RM` allocates aggregate remittances across ages and lifetime-income groups.  OG-Core's default gives every group exactly its population share; the Ethiopian evidence says receipt is concentrated among better-off, urban households.  The World Bank's national remittance survey ({cite}`WorldBankFAR:2010`; 2,412 adults, margin of error 2 points) found 14 percent of adults receiving international remittances regularly; 35 percent of recipients were urban against 16 percent of the population; and recipients' monthly household income was 9 percent below ETB 1,000, 47 percent at ETB 1,000–3,500 and 37 percent above ETB 3,500 (7 percent did not answer).  Against 2010 GDP per capita of ETB 4,262 (WDI) — about ETB 1,600 a month for a household of 4.6 — those brackets sit below, around and well above the mean, and we map them to the bottom two, middle two and top quintile of households.  Renormalizing over respondents gives remittance value shares by quintile of **4.8, 4.8, 25.3, 25.3 and 39.8 percent** from poorest to richest (the survey reports amounts per transfer without a systematic income gradient, so value shares follow recipient shares).  A 2011 household panel finds the same positive selection: recipient households are more educated and rated their economic position higher before receiving ({cite}`Andersson:2014`).

The quintile shares are mapped onto the model's seven lifetime-income groups by linear interpolation of the cumulative distribution (uniform density within each quintile), and spread per capita across ages within each group, so every household in a group receives the same amount.  Like `eta`, the matrix is built for each year from that year's population (`macro_params.allocation_path`; OG-Core accepts `eta_RM` as a path over time, PSLmodels/OG-Core#1297), so the amount per person keeps its intended size as the population ages.  The resulting group shares are 6.0, 16.3, 25.3, 12.6, 19.9, 17.9 and 2.0 percent against population shares of 25, 25, 20, 10, 10, 9 and 1: the bottom quarter of households receives 6 percent of remittance value rather than 25, and the top fifth 40 percent rather than 20.  Three caveats.  The survey is a 2010 vintage and inflows have since roughly doubled in dollars; the brackets are self-reported *current* income, so a household can sit in a higher bracket *because* it receives remittances, which overstates concentration relative to the lifetime-income concept the model's groups represent; and mapping every household above ETB 3,500 a month to the whole top quintile is conservative in the other direction (in 2010 that threshold was well above the 80th percentile).  The survey's age profile of recipients (over-represented at ages 25–34 and 65+) is not used, because its cells are coarse and the 65+ cell tiny.  The matrix is rebuilt with the demographics by `update_baseline.py`, and `tests/test_remittances.py` pins the quintile shares to the survey and the tilt toward higher-income groups.

(Sec_Validation)=
## Validation

Every target is a measured FY2024/25 value on the rebased national accounts at market prices (`ogeth/targets.py`), and the fiscal and foreign-capital dials are scored in the **first model year**, where the data are: Ethiopia's population is far from stationary and its initial wealth is anchored to the data, so a steady-state moment would describe a different economy.  The steady state is matched to hours by age and to the long-run return.  The tables compare the solved transitions and steady states of both scenarios (`examples/run_og_eth.py` and `--scenario program`, from an installer copy of this branch, offline) with the data; each scenario, baseline and reform together, took 10 to 18 minutes.  Revenue lines are the mapped instruments of {ref}`Chap_Tax`.

```{figure} ./images/calibration_vs_program.png
---
scale: 60%
align: center
name: CalibVsProgram
---
OG-ETH baseline (the observed economy, current policy) against the IMF fifth-review program path, both on rebased GDP.  Shaded years are the program horizon; the baseline imposes nothing from the program after the FY2025/26 budget.  The investment panel shows the one rebased observation (FY2024/25).
```

| Ratio to GDP, FY2024/25 (percent) | Baseline | IMF program scenario | Data | How it is used |
|:---|:---:|:---:|:---:|:---|
| Personal income tax | 0.81 | 0.81 | 0.81 | tuned (compliance scale) |
| Corporate and capital-income taxes | 1.26 | 1.26 | 1.26 | tuned (CIT collections factor) |
| Consumption taxes, fees and sales | 3.01 | 2.99 | 3.01 | tuned (`tau_c`) |
| Land-use fees | 0.18 | 0.18 | 0.18 | tuned (`p_wealth`) |
| Pension contributions | 0.29 | 0.29 | 0.29 | tuned (payroll coverage scale) |
| Old-age pensions | 0.08 | 0.08 | 0.08 | tuned (pension coverage scale) |
| Foreign-owned share of private capital | 10.2 | 10.2 | 10.2 | tuned (`zeta_K`) |
| Household wealth | 160.7 | 160.8 | 160.7 | imposed |
| General-government debt | 27.2 | 27.2 | 27.2 | imposed |
| Remittances | 3.5 | 3.5 | 3.5 | imposed |
| Gross capital formation (private and public) | 27.5 | 27.3 | 35.3 | check |
| Private consumption | 68.2 | 67.7 | 64.4 | check |
| Government consumption | 3.95 | 3.95 | 3.67 | check |
| Net exports | 1.5 | 2.1 | -4.5 | check |
| Payments abroad on capital and debt | 1.16 | 1.16 | 0.37 | check |

| Fiscal accounts, FY2024/25 (percent of GDP) | Baseline | IMF program scenario | Data |
|:---|:---:|:---:|:---:|
| Revenue excluding grants | 5.54 | 5.53 | 5.78 |
| Primary spending | 7.10 | 7.10 | 7.04 |
| Interest | -1.46 | -1.46 | 0.50 (nominal) |
| Debt | 27.2 | 27.2 | 27.2 |

| Long run (steady state) | Baseline | IMF program scenario | Target |
|:---|:---:|:---:|:---:|
| Return on capital (real, percent a year) | 4.00 | 3.82 | 4.00 (world rate) |
| Private capital / GDP | 1.85 | 1.85 | outcome (1.66 in FY2024/25) |
| Foreign-owned share of capital | 6.0 | 5.7 | outcome |
| Household consumption / GDP | 74.9 | 72.8 | outcome |

**What the calibration matches.**  Every dial lands on its own moment in FY2024/25: the five revenue instruments, pensions and the foreign-owned share of private capital.  Debt, household wealth and remittances are imposed at their measured values, and the one discount factor puts the long-run return at the world rate.  Hours per person match the 2021 Labour Force and Migration Survey at every fitted age below 75.  Interest is negative in the model because it is real: domestic debt pays about $-11$ percent under financial repression, and the data's 0.50 percent is nominal.

**What the subsistence floor and the lower curvature changed.**  With $\sigma = 3$ and no floor ({ref}`Chap_HouseholdCalib`), patience sat at its upper bound and the long-run return stayed at 4.75 percent, above the world rate; first-year investment was 23.2 percent of GDP and private consumption 79.7.  With $\sigma = 1.5$ above a subsistence floor, patience is 0.903, the return reaches the world rate, investment rises to 27.5 percent and consumption falls to 68.2.

**What neither scenario matches, and why.**  *Investment* is 8 points of GDP below the rebased accounts.  Households alone finance the model's capital, while in Ethiopia much of it is financed by the state and state enterprises (through state banks and retained earnings) and from abroad; the measurable state channels close less than a point (state-enterprise retained earnings about 0.7 percent of GDP, repression beyond the government's below-market rate about 0.2), and the rest needs a state-enterprise sector the single-industry model does not have.  Foreign loans to state enterprises (about 5 percent of rebased GDP, mostly guaranteed by the government) are not counted in foreign-owned capital, which follows the usual practice of the FDI stock.  *Net exports* have the wrong sign: with investment short and consumption close to the data, output exceeds domestic demand and the model runs a surplus of 1.5 percent of GDP against a deficit of 4.5 financed from abroad.  *Payments abroad* are three times net primary income because OG-Core pays foreign-owned capital the household portfolio return, which includes the rent on public capital (`aggregates.get_r_p`), while most of Ethiopia's foreign liabilities pay concessional or FDI returns.  *The capital-output ratio* is not targeted: the rebuilt stock is provisional.

**Resource constraint.**  The transition's resource-constraint error is at most $2.3 \times 10^{-7}$ in every period except period 120, where OG-Core switches to the fixed steady-state population (`fixper`) and the error is about $1.9 \times 10^{-3}$, a known OG-Core bookkeeping break ({ref}`Chap_Demog`).  `RC_TPI` is set to $2 \times 10^{-3}$ to accommodate that single period.

(SecMacro_footnotes)=
## Footnotes
The following are the footnotes for this section.
