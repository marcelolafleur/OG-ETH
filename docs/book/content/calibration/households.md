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
from ogeth.calibrate import BETA_DAMPING, BETA_STEP

params = json.loads(
    files("ogeth")
    .joinpath("ogeth_default_parameters.json")
    .read_text(encoding="utf-8")
)
beta = [float(b) for b in params["beta_annual"]]
for j, b in enumerate(beta, start=1):
    glue(f"beta_{j}", f"{b:.3f}", display=False)
glue("beta_low_rate", f"{100 * (1 / beta[0] - 1):.0f}", display=False)
glue("chi_b_1", f"{float(params['chi_b'][0]):.1f}", display=False)
glue("beta_damping", f"{BETA_DAMPING}", display=False)
glue("frisch", f"{float(params['frisch'])}", display=False)
glue("beta_step", f"{100 * (BETA_STEP - 1):.0f} percent", display=False)

# written by docs/create_doc_figures.py from the run behind wealth_vs_data.png
shares = json.loads(Path("images/wealth_shares.json").read_text())

def pcts(values):
    return ", ".join(f"{100 * v:.1f}" for v in values)

glue("wealth_model", pcts(shares["model"][1:]), display=False)
glue("wealth_target", pcts(shares["target"][1:]), display=False)
glue("wealth_floor", f"{100 * shares['model'][0]:.1f}", display=False)
```

## Behavioral Assumptions

### Elasticity of labor supply

As discussed in the [OG-Core household theory documentation](https://pslmodels.github.io/OG-Core/content/theory/households.html), we use the elliptical disutility of labor function developed by {cite}`EvansPhillips:2017`, whose parameters OG-Core fits to the marginal disutility of a constant Frisch elasticity.  `OG-ETH` sets that elasticity to {glue:text}`frisch` (`FRISCH` in `ogeth/macro_params.py`), below OG-USA's 0.4 from {cite}`Altonji:1986`.  OG-Core's households choose hours per person over the life cycle, so the relevant value is the micro Frisch elasticity of total hours.  Corrected for publication bias, a meta-analysis of more than 700 estimates on each margin puts the intensive margin at about 0.2 and total hours at about 0.25 ({cite}`Elminejad:2023`); {cite}`Chetty:2013` bound the intensive margin below 0.47.  In a setting close to Ethiopia's, randomly offered daily wages in rural Malawi give 0.15 to 0.16 ({cite}`Goldberg:2016`), and off-farm hours in western Ethiopia respond with elasticities of 0.13 to 0.22.  Participation in Ethiopia is near universal and does not vary with education, so the extensive margin adds little, and scarce credit and rationed lean-season work lower the substitution response further.  The range for sensitivity analysis is 0.15 to 0.4.

### Disutility of labor by age

The age profile of the disutility of labor, $\chi^n_s$, sets how much labor households of each age supply in equilibrium. Every OG-Core country model has so far shipped the eighty values estimated for the United States in OG-USA unchanged, `OG-ETH` included, and none has documented that fact; the estimation code inherited with them did not run (issue #71). `OG-ETH` now calibrates $\chi^n_s$ to Ethiopian hours.

The target is hours worked per week per person by age from the 2021 Labour Force and Migration Survey of the Ethiopian Statistics Service {cite}`ESS:2021`: the employment-to-population ratio (Table 5.1) times the mean weekly hours of the employed (Table 5.16a), in five-year age bands. Per person rather than per worker because the model has no extensive margin — every household of an age supplies some labor — so the model's average labor supply by age corresponds to hours per person in the data, with the non-employed counted at zero. Following OG-USA, labor supply is measured as a share of a 16-hour waking day, seven days a week, so a target of 25 hours a week is $n \approx 0.22$. Ethiopian hours per person rise from 18 a week at ages 20–24 to about 25 at 30–44, fall to 20 in the fifties and 17 at 60–64, and are still 10.6 at 65 and over: participation stays high into old age because most work is self-employed farming, but hours per worker are low (32 a week at their peak against 40 in the United States) and participation among the young is held down by schooling and underemployment. The survey's last band is open (65+), so beyond age 67 the target tapers with the National Transfer Accounts labor-income profile used in {ref}`Chap_LfEarn`, which falls faster than hours alone, and it never goes below two percent of the time endowment.

The calibration solves the steady state repeatedly. Between solves each $\chi^n_s$ is rescaled by the ratio of the marginal disutility of labor at the model's average labor supply to that at the target, which is the exact adjustment to the household's labor first-order condition when consumption and prices are held fixed; the steady state then re-solves with the new profile, and the loop stops when every age between 20 and 79 is within two percent of its target. Starting from the OG-USA profile it converges in six solves, and from a previous calibrated profile in one or two, the last within 1 percent at every age below 75; from 75 on $\chi^n_s$ sits at OG-Core's cap of 10,000, those ages are left out of the fit, and households over 80 work about three and a half hours a week more than the extrapolated target (`python -m ogeth.labor` runs it and writes the result into the packaged parameters).

| Age | Target: LFMS 2021 hours per person per week, by single year of age | Model steady state |
|---|---|---|
| 20–24 | 18.6 | 18.6 |
| 25–29 | 23.2 | 23.2 |
| 30–34 | 24.9 | 24.9 |
| 35–39 | 24.9 | 24.9 |
| 40–44 | 24.4 | 24.4 |
| 45–49 | 23.3 | 23.3 |
| 50–54 | 20.3 | 20.3 |
| 55–59 | 19.1 | 19.1 |
| 60–64 | 15.9 | 15.9 |
| 65–69 | 10.3 | 10.3 |
| 70–79 (NTA taper) | 6.7 | 7.4 |
| 80–99 (NTA taper, $\chi^n$ at its cap) | 3.4 | 7.0 |

```{figure} ./images/labor_supply_vs_data.png
---
height: 350px
name: FigLaborVsData
---
Steady-state average labor supply by age against LFMS 2021 hours per person, and the calibrated $\chi^n_s$ against the OG-USA profile.
```

The calibrated profile is far from the US one, in level and in shape. With the US values Ethiopian households in the model worked about 46 percent of their time endowment at every age from 20 to 64, twice the survey's hours; the disutility weights that bring them to the data are about 300–400 in the prime ages against 20–25 in OG-USA and about 1,700 at age 20, and they climb steeply after 60 to OG-Core's upper bound of 10,000 at age 75, where the target is an extrapolation and little labor is supplied in any case. The level of labor supply halves, which the model absorbs through the `factor` that converts model units to birr; the calibrated ratios in {ref}`Chap_MacroCalib` are not affected by it, and the steady-state return on capital is unchanged to the third decimal.

Two limits. Hours per person mix the decision to work with hours conditional on working, and the elliptical disutility function governs both through one intensive margin, so the Frisch elasticity of 0.4 above should be read as applying to total hours per person. And $\chi^n_s$ has no lifetime-income dimension in OG-Core, so differences in hours between the formal and informal groups (OG-ZAF's #95 looks at the same limit) cannot be represented; the model's groups differ in hours only through their wealth and productivity.

### Intertemporal elasticity of substitution

The default value for the intertemporal elasticity of substitution, $\sigma$, is taken from {cite}`ABMW:1999`.  We set $\sigma=1.5$.

### Discount factors by lifetime-income group and the distribution of wealth

Households in OG-Core save for retirement and to leave bequests, and the strength of those motives by lifetime-income group is what shapes the distribution of wealth beyond what the earnings profiles imply.  OG-USA matches the US wealth distribution with **heterogeneous discount factors** $\beta_j$ — richer groups discount the future less and so save a larger share of their income — and `OG-ETH` follows that convention.  The bequest weight $\chi^b_j$ stays at OG-USA's 80 for every group except the bottom quarter (below).

The targets are the World Inequality Database's net personal wealth shares for Ethiopia {cite}`WID:2024` (equal-split adults, 2021): the bottom half of adults hold 4.8 percent of wealth, the middle 40 percent 37.1, the top decile 58.1 and the top percent 23.6; the wealth Gini is 0.735.  The seven groups' targets are read off the cumulative shares at the 25th, 50th, 70th, 80th, 90th and 99th percentiles, as for the income shares in {ref}`Chap_LfEarn`.  The bottom quarter's share is negative in the data (−0.5 percent; they are net debtors), which the model's households, who cannot borrow, cannot reproduce, so that group is given a floor of half a percent.  Its discount factor is tied to the second group's, and its share is matched instead with its own bequest weight $\chi^b_1$.  With OG-USA's 80 the poorest elderly, whose transfers and remittances are several times their consumption, saved nearly all of them for bequests: the bottom quarter held 2.3 percent of wealth, the oldest of them ten times the group's average, and that hoarding left their first-year household problem so badly conditioned that a reform's transition could not solve it (OG-Core's check on households alive in the first year caught it).  A weight of {glue:text}`chi_b_1` brings the group to its floor, and the reform's transition then solves with Euler errors near 1e-10.  The wealth shares pin down *relative* patience across groups.  The *level* of the discount factors is set so that the steady-state return on capital equals the world rate, 4 percent ({ref}`Sec_Validation`): a small open economy's domestic return converges to what foreign investors earn as capital controls and financial repression are unwound.  An earlier calibration tuned the level to gross capital formation in the first model year, 35.3 percent of rebased GDP.  That made households finance all of it, including state-enterprise investment financed through state banks, and required a long-run return of about 2 percent, below the world rate, with foreign-owned capital leaving.  The measurable state channels (state-enterprise retained earnings, about 0.7 percent of GDP, and financial repression beyond the below-market government borrowing rate the model already carries, about 0.2 percent) close less than a point of that gap, so first-year investment is reported as a measured miss rather than tuned.  The cost is a first-year consumption share above the data: patience tuned to the long run makes today's households consume more of today's income.

The calibration runs through `examples/tune.py`, which solves the steady state and the transition from the packaged parameters and after each run moves every fitted group's $\beta_j$ by the ratio of its target to its model wealth share, then restores the population-weighted mean of $\log\beta_j$ so that the level is untouched; the level moves separately, by the gap between the steady-state return and the world rate times the measured response of the return to patience.  A group's steady-state wealth is very elastic to its discount factor and the shares interact through the interest rate, so those updates are damped (an exponent of {glue:text}`beta_damping`, no more than {glue:text}`beta_step` a solve, halved for a group whenever its gap changes sign), and the level update by a factor of 0.8 and no more than four percent a run (`R_LEVEL_SLOPE`, `LEVEL_DAMPING` and `LEVEL_STEP` in `examples/tune.py`).

```{figure} ./images/wealth_vs_data.png
---
height: 350px
name: FigWealthVsData
---
Left: the steady-state share of household wealth held by each lifetime-income group against the WID target (with the bottom quarter floored) and the raw WID shares. Right: the calibrated annual discount factors $\beta_j$ against the single 0.88 used before.
```

The calibrated discount factors are {glue:text}`beta_1` for the two groups in the bottom half, {glue:text}`beta_3`, {glue:text}`beta_4` and {glue:text}`beta_5` for the 50th–90th percentiles, {glue:text}`beta_6` for the 90th–99th and {glue:text}`beta_7` for the top percent, with a population-weighted level set by the long-run return (the fit is in the validation table of {ref}`Sec_Validation`).  The resulting steady-state wealth shares are {glue:text}`wealth_model` percent for groups 2–7 against targets of {glue:text}`wealth_target`, and {glue:text}`wealth_floor` for the floored bottom quarter.  The spread is wide, and the low end has independent support: the bottom half's discount factor is an annual discount rate of about {glue:text}`beta_low_rate` percent, and the one experimental measurement of time preference in Ethiopia, among 262 farm households in the highlands, found median annual rates of 43 to 106 percent depending on the horizon, rising with poverty ({cite}`YesufBluffstone:2019`).  A meta-analysis of 927 experimental estimates finds that publication bias roughly doubles reported rates, with a corrected mean of about 33 percent a year ({cite}`Matousek:2022`), and the literature reads the high rates measured in poor households largely as credit constraints and cash shortages rather than preference.  The top percent's {glue:text}`beta_7` is in the range OG-USA estimates for the richest US groups.  Two cautions: the wealth shares WID reports rest on the same consumption surveys as its income shares and a borrowed top correction, and with heterogeneous patience the model's response to a reform is no longer that of a representative saver — the top percent, which holds a quarter of the wealth, reacts much more to the return on capital than the bottom half does.

### Rate of time preference

Earlier versions set one discount factor for every household against the Penn World Table capital-output ratio (0.96 inherited from OG-USA, then 0.92, 0.90 and 0.88).  The calibration above replaces that: relative patience across groups is identified by the wealth distribution and its level by the long-run return, and the capital-output ratio is an outcome ({ref}`Sec_Validation`).  On the rebased accounts the Penn World Table ratio is itself provisional ({ref}`Sec_Rebasing`), which is another reason not to target it.
