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
glue("phi_ft_ss", f"{100 * mp.FAMILY_TRANSFER_SHARE_SS:.0f} percent",
     display=False)
for label, t in [("ss", -1), ("2025", 0)]:
    phi = params["phi_FT"][t]
    received = sum(v for row in phi for v in row if v > 0)
    glue(f"phi_ft_net_{label}", f"{100 * received:.1f}", display=False)

```

## Behavioral Assumptions

### Elasticity of labor supply

As discussed in the [OG-Core household theory documentation](https://pslmodels.github.io/OG-Core/content/theory/households.html), we use the elliptical disutility of labor function developed by {cite}`EvansPhillips:2017`, whose parameters OG-Core fits to the marginal disutility of a constant Frisch elasticity.  `OG-ETH` sets that elasticity to {glue:text}`frisch` (`FRISCH` in `ogeth/macro_params.py`), below OG-USA's 0.4 from {cite}`Altonji:1986`.  OG-Core's households choose hours per person over the life cycle, so the relevant value is the micro Frisch elasticity of total hours.  Corrected for publication bias, a meta-analysis of more than 700 estimates on each margin puts the intensive margin at about 0.2 and total hours at about 0.25 ({cite}`Elminejad:2023`); {cite}`Chetty:2013` bound the intensive margin below 0.47.  In a setting close to Ethiopia's, randomly offered daily wages in rural Malawi give 0.15 to 0.16 ({cite}`Goldberg:2016`), and off-farm hours in western Ethiopia respond with elasticities of 0.13 to 0.22.  Participation in Ethiopia is near universal and does not vary with education, so the extensive margin adds little, and scarce credit and rationed lean-season work lower the substitution response further.  The range for sensitivity analysis is 0.15 to 0.4.

### Disutility of labor by age

The age profile of the disutility of labor, $\chi^n_s$, sets how much labor households of each age supply in equilibrium. Every OG-Core country model has so far shipped the eighty values estimated for the United States in OG-USA unchanged, `OG-ETH` included, and none has documented that fact; the estimation code inherited with them did not run (issue #71). `OG-ETH` now calibrates $\chi^n_s$ to Ethiopian hours.

The target is hours worked per week per person by age from the 2021 Labour Force and Migration Survey of the Ethiopian Statistics Service {cite}`ESS:2021`: the employment-to-population ratio (Table 5.1) times the mean weekly hours of the employed (Table 5.16a), in five-year age bands. Per person rather than per worker because the model has no extensive margin — every household of an age supplies some labor — so the model's average labor supply by age corresponds to hours per person in the data, with the non-employed counted at zero. Following OG-USA, labor supply is measured as a share of a 16-hour waking day, seven days a week, so a target of 25 hours a week is $n \approx 0.22$. Ethiopian hours per person rise from 18 a week at ages 20–24 to about 25 at 30–44, fall to 20 in the fifties and 17 at 60–64, and are still 10.6 at 65 and over: participation stays high into old age because most work is self-employed farming, but hours per worker are low (32 a week at their peak against 40 in the United States) and participation among the young is held down by schooling and underemployment.

The survey's last band is open (65 and over), so the shape within it comes from the 2021/22 Ethiopia Socioeconomic Panel Survey ({cite}`ESS:ESPS2024`): hours in the last seven days on the household's farm, in its own or household business, and in casual and paid work, with non-workers at zero, household weights, 815 people aged 65 and over.  Hours per person are 14.0 at 65–69, 9.6 at 70–79 and 4.4 at 80 and over, at mean ages of 66.6, 73.0 and 86.1.  From 65 the target is an exponential decline fitted to those three bands (weighted least squares of log hours on each band's mean age, weights its sample size) at the labour-force survey's level, so that survey sets the level and the panel survey only the decline with age: 16.4 hours a week at 65, falling 5.9 percent a year of age, within half a percent of each band, and 2.2 hours at 100 (`ogeth.labor.older_hours_decline`).  Between the 60–64 band's midpoint and 65 the target is linear, and it never goes below two percent of the time endowment.  An earlier version extrapolated beyond 67 with the National Transfer Accounts labor-income profile, which falls much faster than measured hours because it also carries the fall in pay with age.

`examples/tune.py` sets $\chi^n_s$ with the other dials.  Between steady-state solves each $\chi^n_s$ is rescaled by the ratio of the marginal disutility of labor at the model's average labor supply to that at the target, which is the exact adjustment to the household's labor first-order condition when consumption and prices are held fixed.  The fit is on target at every age through 68.  From 69 $\chi^n_s$ sits at OG-Core's cap of 10,000, and older households work more than the target.  The excess is concentrated in the poorest group: its consumption is close to the subsistence floor, where the marginal utility of consumption is high, so as pay falls with age it works more rather than less, and its hours barely respond to the disutility weight (with a cap ten times higher, people over 90 still worked 11.8 hours a week against a target of 2.8 in steady-state probes).  The Frisch elasticity is not moved to close the gap, since its value is set by the evidence above.  Net family transfers ({ref}`SecHH_FamilyTransfers`) halve it in the seventies and narrow it by 40 percent in the eighties; the remainder is documented here rather than fitted.

| Age | Target: hours per person per week (LFMS 2021; ESS 2021/22 shape from 65) | Model steady state |
|---|---|---|
| 20–24 | 18.6 | 18.6 |
| 25–29 | 23.2 | 23.2 |
| 30–34 | 24.9 | 24.9 |
| 35–39 | 24.9 | 24.9 |
| 40–44 | 24.4 | 24.4 |
| 45–49 | 23.3 | 23.3 |
| 50–54 | 20.3 | 20.3 |
| 55–59 | 19.1 | 19.1 |
| 60–64 | 16.9 | 16.9 |
| 65–69 | 14.2 | 14.3 |
| 70–79 ($\chi^n$ at its cap) | 9.3 | 12.7 |
| 80–89 | 5.3 | 12.1 |
| 90–99 | 3.1 | 11.5 |

```{figure} ./images/labor_supply_vs_data.png
---
height: 350px
name: FigLaborVsData
---
Steady-state average labor supply by age against LFMS 2021 hours per person, and the calibrated $\chi^n_s$ against the OG-USA profile.
```

The calibrated profile is far from the US one, in level and in shape. With the US values Ethiopian households in the model worked about 46 percent of their time endowment at every age from 20 to 64, twice the survey's hours; the disutility weights that bring them to the data are about 1,300 to 1,500 in the prime ages against 20–25 in OG-USA and about 9,700 at age 20, and they climb steeply after 50 to OG-Core's upper bound of 10,000 at age 69. The level of labor supply halves, which the model absorbs through the `factor` that converts model units to birr; the calibrated ratios in {ref}`Chap_MacroCalib` are not affected by it, and the steady-state return on capital is unchanged to the third decimal.

Two limits. Hours per person mix the decision to work with hours conditional on working, and the elliptical disutility function governs both through one intensive margin, so the Frisch elasticity above should be read as applying to total hours per person. And $\chi^n_s$ has no lifetime-income dimension in OG-Core, so differences in hours between the formal and informal groups (OG-ZAF's #95 looks at the same limit) cannot be represented; the model's groups differ in hours only through their wealth and productivity.

### Intertemporal elasticity of substitution and the subsistence floor

OG-Core's households face no income risk, so the curvature of utility $\sigma$ acts as the inverse of the intertemporal elasticity of substitution (EIS).  OG-Core also lets households buy a subsistence quantity $c_{min}$ before the rest of their consumption enters utility (Stone–Geary preferences).  With that floor the EIS of total consumption is $(1/\sigma)(1 - c_{min}/c)$ ({cite}`OgakiOstryReinhart:1996`): low near subsistence, rising toward $1/\sigma$ as households get richer.  `OG-ETH` uses both: $\sigma =$ {glue:text}`sigma`, OG-Core's default (an EIS of 0.67, below the 0.8 above which {cite}`Havranek:2015` finds calibrations inconsistent with the evidence), and a floor at the cost of an energy-sufficient diet.

The floor comes from the World Bank and FAO's Food Prices for Nutrition data, USD 1.17 a day at 2021 PPP in Ethiopia ({cite}`WB:FoodPrices2021`), 12.2 percent of mean consumption per person on the rebased accounts.  Applied as that share of private consumption per adult, it would be 0.136 in model units in FY2024/25.  The value used, $c_{min} =$ {glue:text}`c_min`, is 80 percent of it: the floor is the same for every age and lifetime-income group, and the poorest group, a quarter of the population, has to clear it at every age: in the steady state its consumption above the floor is at least 0.11, at the oldest ages.  In the data about 6 percent of people consume less than the diet costs.  OG-Core states $c_{min}$ in growth-adjusted units, so the floor rises with labor productivity, a constant share of average income.  In the solved steady state the floor is 32 percent of the poorest group's consumption and 1 percent of the top percent's, so the effective $\sigma$ is 2.2 for the poorest group, 1.63 at mean consumption and 1.5 for the richest.

An intermediate working value during this recalibration, $\sigma = 3$ for every household, took the low EIS measured near subsistence and applied it to everyone.  With labor productivity growing 4.7 percent a year, that curvature made the richer households, who hold most of the capital, want to consume now against their higher future income, and patience had to sit at its upper bound to hold the long-run return at the world rate.  Ethiopian households do not behave that way.  In the 2021/22 Ethiopia Socioeconomic Panel Survey ({cite}`ESS:ESPS2024`), 26 percent of households borrowed in the past year, three-quarters of the loans from relatives (51 percent) and neighbours (23 percent) and 0.1 percent from commercial banks; the main purposes were food and other everyday spending (29 percent of loans), inputs for food crops (24 percent) and health and education (9 percent).  A third of adults saved, rising from 19 percent in the poorest consumption quintile to 52 percent in the richest, and 80 percent of savers named emergencies as the main reason, 0.9 percent old age.  Credit is informal and smooths shocks; saving is precautionary and rises with income, which is the pattern the subsistence floor produces.  The tabulations are reproduced by `scripts/ess_credit_saving.py` from the public-use microdata.

(SecHH_FamilyTransfers)=
### Net family transfers

In Ethiopia older people live mostly on support from their children and relatives, usually in the same household, rather than on savings or public programmes: the public pension schemes cover a small formal workforce, and only 0.9 percent of savers in the panel survey above save for old age.  In OG-Core each household lives off its own budget, and nothing moves resources between households except bequests, so without that support people over 65 in the model consumed 40 to 55 percent of what prime-age adults consume (0.45 at ages 70–79), against 84 percent in Ethiopia's National Transfer Accounts, and worked long past what the surveys show.

OG-Core's net family transfers (`phi_FT`, PSLmodels/OG-Core#1298) add that channel: a share of output received by each age and lifetime-income group, positive for those who receive support and negative for those who give it.  The shares sum to zero in every year, so the transfers only move resources between households and leave aggregate resources and the government budget unchanged.  Ethiopia's National Transfer Accounts (2005) measure consumption and labor income by age but not private transfers (no African account in the NTA database has a private-transfer profile), so the transfers follow a simple rule with one size (`macro_params.family_transfers`):

- the givers are ages 25 to 64, each in proportion to their earnings capacity (population times the earnings matrix), and the receivers are ages 65 and over, equal per person;
- each lifetime-income group's transfers net to zero within the group: working-age adults support their own group's elders, and a group's share of the total is its share of the givers' earnings capacity;
- a group's elders receive from the family the group's share less the government cash transfers they already get (the safety net's direct support and the subsidies in `eta`); old-age pensions, about 0.075 percent of GDP, are not netted;
- per elder, the transfer is held fixed relative to output per person, so the total grows with the elderly share of the population: net of the government transfers, {glue:text}`phi_ft_net_2025` percent of output in 2025 and {glue:text}`phi_ft_net_ss` percent in the steady state, when people aged 65 and over are 29.5 percent of adults (6.5 percent in 2025).

The size is set to the NTA consumption profile: with family transfers at {glue:text}`phi_ft_ss` of the model's output before the government transfers elders receive are netted ({glue:text}`phi_ft_net_ss` percent of output after; 16.5 and 16.2 percent of GDP at market prices), steady-state consumption of people aged 65 and over relative to ages 30–49 is 0.88 at 65–69, 0.91 at 70–79 and 0.87 at 80–85 (total consumption, the floor included), against NTA private consumption of 0.87, 0.84 and 0.80 (`ogeth/data/nta_consumption_by_age.csv`; `scripts/derive_sourced_values.py`).  The transfers also lower old-age hours, since elders no longer have to work for their own consumption: hours per person fall from 16.2 to 12.7 a week at ages 70–79 and from 16.9 to 12.1 at 80–89 (targets 9.3 and 5.3).  The NTA profile is from 2005, the only Ethiopian account; the rule's age boundaries and the equal split among elders are assumptions, and the one size is the only calibrated value.

### Discount factor

Every household has the same discount factor, set so that the steady-state return on capital equals the world rate, 4 percent ({ref}`Sec_Validation`): a small open economy's domestic return converges to what foreign investors earn as capital controls and financial repression are unwound.  The calibrated value is {glue:text}`beta`, an annual discount rate of about {glue:text}`beta_rate` percent.  OG-Core's households save only for retirement and bequests, motives the survey above shows Ethiopian savers rarely report; the missing precautionary motive is the main thing this parameter stands in for.  The level is tuned by `examples/tune.py`, which moves it by the gap between the steady-state return and the world rate times the measured response of the return to patience, damped (`R_LEVEL_SLOPE`, `LEVEL_DAMPING` and `LEVEL_STEP`), within bounds of 0.5 and 0.995.

Earlier calibrations set the discount factors by lifetime-income group to the World Inequality Database's wealth shares for Ethiopia ({cite}`WID:2024`), as OG-USA does for the United States.  Matching them required patience above one for the top group and at the lower bound of 0.5 for the bottom, and the shares themselves are imputed from consumption surveys with a borrowed top correction, so they are now a check rather than a target.  The bottom quarter's bequest weight $\chi^b_1$ is {glue:text}`chi_b_1` (80, OG-USA's value, for the other groups): with 80, the poorest elderly, whose transfers and remittances are several times their consumption, saved nearly all of them for bequests and left the first-year household problem too badly conditioned for a reform's transition to solve.

The one experimental measurement of time preference in Ethiopia, among 262 farm households in the highlands, found median annual discount rates of 43 to 106 percent, rising with poverty ({cite}`YesufBluffstone:2019`); a meta-analysis of 927 experimental estimates finds that publication bias roughly doubles reported rates, with a corrected mean of about 33 percent a year ({cite}`Matousek:2022`), and the literature reads the high rates measured in poor households largely as credit constraints and cash shortages rather than preference.  The model's rate is well below those measurements; its households face no risk and no cash shortages, so their patience is a calibration device for the long-run return rather than an estimate of preference.  Releases up to 0.2.0 used OG-USA's 0.96 for every group, and 0.3.0 set one factor per group to the WID wealth shares; the Penn World Table capital-output ratio, an intermediate target during this recalibration, is now an outcome ({ref}`Sec_Validation`).  On the rebased accounts the Penn World Table ratio is itself provisional ({ref}`Sec_Rebasing`), which is another reason not to target it.
