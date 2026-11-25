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

(Chap_HouseholdCalib)=
# Calibration of Household Preference Parameters

```{code-cell} ipython3
:tags: ["remove-cell"]

from importlib.resources import files
from pathlib import Path
import json
from myst_nb import glue
from ogeth import macro_params as mp

params = json.loads(
    files("ogeth")
    .joinpath("ogeth_default_parameters.json")
    .read_text(encoding="utf-8")
)
beta = float(params["beta_annual"][0])
glue("beta", f"{beta:.3f}", display=False)
glue("beta_rate", f"{100 * (1 / beta - 1):.0f}", display=False)
glue("chi_b_1", f"{float(params['chi_b'][0]):.2f}", display=False)
glue("frisch", f"{float(params['frisch'])}", display=False)
glue("sigma", f"{float(params['sigma'])}", display=False)
glue("c_min", f"{float(params['c_min'][0]):.4f}", display=False)
for label, t in [("ss", -1), ("2025", 0)]:
    phi = params["phi_FT"][t]
    received = sum(v for row in phi for v in row if v > 0)
    glue(f"phi_ft_net_{label}", f"{100 * received:.1f}", display=False)

```

## Behavioral Assumptions

### Elasticity of labor supply

As discussed in the [OG-Core household theory documentation](https://pslmodels.github.io/OG-Core/content/theory/households.html), we use the elliptical disutility of labor function developed by {cite}`EvansPhillips:2017`, whose parameters OG-Core fits to the marginal disutility of a constant Frisch elasticity.  `OG-ETH` sets that elasticity to {glue:text}`frisch` (`FRISCH` in `ogeth/macro_params.py`).  OG-Core's households choose hours per person over the life cycle, so the relevant value is the micro Frisch elasticity of total hours.  Corrected for publication bias, a meta-analysis of more than 700 estimates on each margin puts the intensive margin at about 0.2 and total hours at about 0.25 ({cite}`Elminejad:2023`); {cite}`Chetty:2013` bound the intensive margin below 0.47.  In a setting close to Ethiopia's, randomly offered daily wages in rural Malawi give 0.15 to 0.16 ({cite}`Goldberg:2016`), and off-farm hours in western Ethiopia respond with elasticities of 0.13 to 0.22.  Participation in Ethiopia is near universal and does not vary with education, so the extensive margin adds little, and scarce credit and rationed lean-season work lower the substitution response further.  The range for sensitivity analysis is 0.15 to 0.4.

### Disutility of labor by age

The age profile of the disutility of labor, $\chi^n_s$, sets how much labor households of each age supply in equilibrium, and it is calibrated to Ethiopian hours by age (below).

The target is hours worked per week per person by age from the 2021 Labour Force and Migration Survey of the Ethiopian Statistics Service {cite}`ESS:2021`: the employment-to-population ratio (Table 5.1) times the mean weekly hours of the employed (Table 5.16a), in five-year age bands. Per person rather than per worker because the model has no extensive margin — every household of an age supplies some labor — so the model's average labor supply by age corresponds to hours per person in the data, with the non-employed counted at zero. Following OG-USA, labor supply is measured as a share of a 16-hour waking day, seven days a week, so a target of 25 hours a week is $n \approx 0.22$. Ethiopian hours per person rise from 18 a week at ages 20–24 to about 25 at 30–44, fall to 20 in the fifties and 17 at 60–64, and are still 10.6 at 65 and over: participation stays high into old age because most work is self-employed farming, but hours per worker are low (32 a week at their peak) and participation among the young is held down by schooling and underemployment.

The survey's last band is open (65 and over), so the shape within it comes from the 2021/22 Ethiopia Socioeconomic Panel Survey ({cite}`ESS:ESPS2024`): hours in the last seven days on the household's farm, in its own or household business, and in casual and paid work, with non-workers at zero, household weights, 815 people aged 65 and over.  Hours per person are 14.0 at 65–69, 9.6 at 70–79 and 4.4 at 80 and over, at mean ages of 66.6, 73.0 and 86.1.  From 65 the target is an exponential decline fitted to those three bands (weighted least squares of log hours on each band's mean age, weights its sample size) at the labour-force survey's level, so that survey sets the level and the panel survey only the decline with age: 16.4 hours a week at 65, falling 5.9 percent a year of age, within half a percent of each band, and 2.2 hours at 100 (`ogeth.labor.older_hours_decline`).  Between the 60–64 band's midpoint and 65 the target is linear, and it never goes below two percent of the time endowment.

`examples/tune.py` sets $\chi^n_s$ with the other dials.  Between steady-state solves each $\chi^n_s$ is rescaled by the ratio of the marginal disutility of labor at the model's average labor supply to that at the target, which is the exact adjustment to the household's labor first-order condition when consumption and prices are held fixed.  The fit is on target through the late sixties.  From about 68 $\chi^n_s$ sits at OG-Core's cap of 10,000, and older households work more than the target (table below).  The excess is concentrated in the poorest group: its consumption is close to the subsistence floor, where the marginal utility of consumption is high, so as pay falls with age it works more rather than less, and its hours barely respond to the disutility weight (with a cap ten times higher, people over 90 still worked 11.8 hours a week against a target of 2.8 in steady-state probes).  The Frisch elasticity is not moved to close the gap, since its value is set by the evidence above.  Net family transfers ({ref}`SecHH_FamilyTransfers`) narrow it by about half in the seventies and eighties; the remainder is documented here rather than fitted.

```{include} ./images/hours_by_age_table.md
```

```{figure} ./images/labor_supply_vs_data.png
---
height: 350px
name: FigLaborVsData
---
Steady-state average labor supply by age against LFMS 2021 hours per person, and the calibrated $\chi^n_s$ with OG-Core's upper bound.
```

The calibrated weights are lowest in the thirties (about 1,300 at their minimum, near 40), where Ethiopians work most, and rise with age to OG-Core's upper bound of 10,000 from about 68; at 20 the weight is also at the bound.  The level of labor supply is absorbed by the `factor` that converts model units to birr, so the calibrated ratios in {ref}`Chap_MacroCalib` do not depend on it.

Two limits. Hours per person mix the decision to work with hours conditional on working, and the elliptical disutility function governs both through one intensive margin, so the Frisch elasticity above should be read as applying to total hours per person. And $\chi^n_s$ has no lifetime-income dimension in OG-Core, so differences in hours between the formal and informal groups (OG-ZAF's #95 looks at the same limit) cannot be represented; the model's groups differ in hours only through their wealth and productivity.

### Intertemporal elasticity of substitution and the subsistence floor

OG-Core's households face no income risk, so the curvature of utility $\sigma$ acts as the inverse of the intertemporal elasticity of substitution (EIS).  OG-Core also lets households buy a subsistence quantity $c_{min}$ before the rest of their consumption enters utility (Stone–Geary preferences).  With that floor the EIS of total consumption is $(1/\sigma)(1 - c_{min}/c)$ ({cite}`OgakiOstryReinhart:1996`): low near subsistence, rising toward $1/\sigma$ as households get richer.  `OG-ETH` uses both: $\sigma =$ {glue:text}`sigma`, OG-Core's default (an EIS of 0.67, below the 0.8 above which {cite}`Havranek:2015` finds calibrations inconsistent with the evidence), and a floor at the cost of an energy-sufficient diet.

The floor comes from the World Bank and FAO's Food Prices for Nutrition data, USD 1.17 a day at 2021 PPP in Ethiopia ({cite}`WB:FoodPrices2021`), 12.2 percent of mean consumption per person on the rebased accounts.  Applied as that share of private consumption per adult, it would be 0.136 in model units in FY2024/25.  The value used, $c_{min} =$ {glue:text}`c_min`, is 80 percent of it: the floor is the same for every age and lifetime-income group, and the poorest group, a quarter of the population, has to clear it at every age: in the steady state its consumption above the floor is at least 0.11, at the oldest ages.  In the data about 6 percent of people consume less than the diet costs.  OG-Core states $c_{min}$ in growth-adjusted units, so the floor rises with labor productivity, a constant share of average income.  In the solved steady state the floor is 33 percent of the poorest group's consumption and 1 percent of the top percent's, so the effective $\sigma$ is 2.2 for the poorest group, 1.64 at mean consumption and 1.5 for the richest.

A single high curvature for every household, the low EIS measured near subsistence applied to everyone, would make the richer households, who hold most of the capital, want to consume now against their higher future income as labor productivity grows 4.7 percent a year.  Ethiopian households do not behave that way.  In the 2021/22 Ethiopia Socioeconomic Panel Survey ({cite}`ESS:ESPS2024`), 26 percent of households borrowed in the past year, three-quarters of the loans from relatives (51 percent) and neighbours (23 percent) and 0.1 percent from commercial banks; the main purposes were food and other everyday spending (29 percent of loans), inputs for food crops (24 percent) and health and education (9 percent).  A third of adults saved, rising from 19 percent in the poorest consumption quintile to 52 percent in the richest, and 80 percent of savers named emergencies as the main reason, 0.9 percent old age.  Credit is informal and smooths shocks; saving is precautionary and rises with income, which is the pattern the subsistence floor produces.  The tabulations are reproduced by `scripts/ess_credit_saving.py` from the public-use microdata.

(SecHH_FamilyTransfers)=
### Net family transfers

In Ethiopia older people live mostly on support from their children and relatives, usually in the same household, rather than on savings or public programmes: the public pension schemes cover a small formal workforce, and only 0.9 percent of savers in the panel survey above save for old age.  In OG-Core each household lives off its own budget, and nothing moves resources between households except bequests, so without that support people over 65 in the model consumed 40 to 55 percent of what prime-age adults consume (0.45 at ages 70–79), against 84 percent in Ethiopia's National Transfer Accounts, and worked long past what the surveys show.

OG-Core's net family transfers (`phi_FT`, PSLmodels/OG-Core#1298) add that channel: a share of output received by each age and lifetime-income group, positive for those who receive support and negative for those who give it.  The shares sum to zero in every year, so the transfers only move resources between households and leave aggregate resources and the government budget unchanged.  No Ethiopian source measures private transfers by age, so they are calibrated to consumption by age.

**The target.**  OG-Core's households start at age 20 (`E`) and have no children, while the national accounts' private consumption includes children's.  The target is therefore consumption per adult by age with each adult also carrying the consumption of the people under 20 they support (`ogeth/data/consumption_by_age.csv`, `targets.consumption_target`), relative to ages 30–49.  It is the default consumption-by-age target for OG-Core countries, one method from public data:

- own private consumption per person by age from a cross-country model of the 70 National Transfer Accounts profiles: an eight-knot piecewise-linear shape whose knots are regressed on log GDP per capita at purchasing-power parity and the old-age dependency ratio (World Bank, 2024: 2,892 and 5.60 for Ethiopia); its average error on countries left out of the fit is about 0.07;
- the consumption of each age under 20 carried by its parents, with the mother's age drawn from UN fertility by age times the population and the father's from the DHS cross-country median distribution of husband–wife age gaps, and by the grandparents' generation for the 12.6 percent of children who live with neither parent (DHS cross-country median), weighted by survival; the 85+ group is held at the age-84 value.

The target is low for young adults, peaks in the forties, where parents carry their children, and declines gently in old age.  Against Ethiopia's own survey, ESPS wave 5 (2021/22), built the same way from its household roster, it is off by about 0.06 on average, lower than the survey in the twenties and higher at 80 and over.

**The fit.**  `examples/fit_family_transfers.py` takes a solved steady state and moves the transfer to each age of a group by the gap between the group's consumption and the target (scaled to the group's total consumption) times the group's population at that age, so each step only reshuffles resources within the group; repeated with `examples/tune.py`, it settles in about ten steady-state rounds together with the other dials.  The fitted matrix is `ogeth/data/family_transfers_ss.csv` (`macro_params.family_transfers_ss`).

- each lifetime-income group's transfers net to zero within the group, and every age from 20 is fitted, so there is no imposed boundary between givers and receivers;
- adults aged 30–59 are net givers and elders net receivers, while young adults roughly balance (table below);
- over the transition each receiving age keeps its transfer per person relative to output per person, and the givers of each group pay the group's total in their steady-state proportions (`macro_params.family_transfer_path`), so the total follows the population: {glue:text}`phi_ft_net_2025` percent of output in 2025 and {glue:text}`phi_ft_net_ss` percent in the steady state.

With the fitted transfers, steady-state consumption per adult follows the target closely (relative to ages 30–49, the floor included):

```{figure} ./images/consumption_vs_target.png
---
name: FigConsumptionTarget
---
Steady-state consumption per adult by age (the floor included) against the target, both relative to ages 30–49.  The target is the consumption by age that the National Transfer Accounts of 70 countries predict for a country at Ethiopia's income and age structure, with children's consumption counted with the adults who support them (`docs/create_doc_figures.py`).
```

```{include} ./images/consumption_by_age_table.md
```

The transfers also lower old-age hours, since elders do not have to work for all of their own consumption, though hours after 70 remain above the targets (table above).  The fit treats all differences between the model's consumption profile and the target as family support: shared consumption within households, which is what the transfers stand in for, but also anything else that shapes consumption by age and the model lacks.

### Discount factor

Every household has the same discount factor, and it sets household saving: today's returns are fixed by technology and the measured capital stock, not by patience.  The national accounts show more saving than the model's households produce at any patience the return evidence allows, so the discount factor is as high as that evidence allows: the owners' long-run return on capital (the firms' return plus the rent on public capital that OG-Core passes to them, government debt excluded) stays at or above 6 percent, the bottom of the 6–10 percent range ({ref}`Chap_FirmCalib`).  The value is {glue:text}`beta` (`macro_params.BETA_ANNUAL`, a constant of the calibration rather than a tuned dial), an annual discount rate of about {glue:text}`beta_rate` percent, and the owners' long-run return is 6.1 percent.  The floor applies present-day evidence to the long run, which is a choice; along the path owners earn about 8 percent until mid-century.  From the measured stock, private capital builds up faster than output for a decade, and the firms' return falls from 5 percent to the 4 percent world rate by about 2035 and stays there through mid-century.  The steady state, much later, describes an old economy with a shrinking population (the UN projection): the firms' return is 2.7 percent, below the world rate, and foreign-owned capital is under 1 percent of capital ({ref}`Sec_Validation`).  It stays positive because OG-Core sets the capital demanded at the world rate without its public capital, so that demand exceeds the actual stock.  More patience would put the owners' long-run return below the evidence and, from about 0.96, make the long-run economy a net exporter of capital; patience set so that the steady-state firms' return equals the world rate instead would be about 0.89, with households saving far less than the national accounts show.  OG-Core's households save only for retirement and bequests, motives the survey above shows Ethiopian savers rarely report; the missing precautionary motive is the main thing this parameter stands in for.

The World Inequality Database's wealth shares for Ethiopia ({cite}`WID:2024`) are a check rather than a target: matching them with one discount factor per lifetime-income group would require patience above one for the top group and at the lower bound of 0.5 for the bottom, and the shares themselves are imputed from consumption surveys with a borrowed top correction.  The bottom quarter's bequest weight $\chi^b_1$ is {glue:text}`chi_b_1` (80 for the other groups): with 80, the poorest elderly, whose transfers and remittances are several times their consumption, saved nearly all of them for bequests and left the first-year household problem too badly conditioned for a reform's transition to solve.

The one experimental measurement of time preference in Ethiopia, among 262 farm households in the highlands, found median annual discount rates of 43 to 106 percent, rising with poverty ({cite}`YesufBluffstone:2019`); a meta-analysis of 927 experimental estimates finds that publication bias roughly doubles reported rates, with a corrected mean of about 33 percent a year ({cite}`Matousek:2022`), and the literature reads the high rates measured in poor households largely as credit constraints and cash shortages rather than preference.  The model's rate is well below those measurements; its households face no risk and no cash shortages, so their patience is a calibration device for saving and the long-run return rather than an estimate of preference.  The capital-output ratio is an outcome, not a target ({ref}`Sec_Validation`): on the rebased accounts the capital stock is provisional ({ref}`Sec_Rebasing`).
