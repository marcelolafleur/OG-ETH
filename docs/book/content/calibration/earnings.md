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

(Chap_LfEarn)=
# Lifetime Earnings Profiles

Among households in `OG-ETH`, we model variations in labor productivity over the lifecycle and between households of different skill groups. Together, these variations generate a distribution of earnings. This chapter describes how the lifecycle earnings profiles are calibrated to Ethiopian data on the age profile of labor income and on the distribution of income across households.

Differences among workers' productivity is one of the key dimensions of heterogeneity to model in a micro-founded macroeconomy. We characterize this heterogeneity as deterministic lifetime productivity paths to which new cohorts of agents in the model are randomly assigned. Households' labor income comes from the equilibrium wage and the agent's endogenous quantity of labor supply, augmented by an individual productivity $e_{j,s}$, where $j$ is the index of the ability type or path of the individual and $s$ is the age of the individual with that ability path.

```{math}
:label: EqLaborIncome
  \text{labor income:}\quad x_{j,s,t}\equiv w_t e_{j,s}n_{j,s,t} \quad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
```

In this specification, $w_t$ is an equilibrium wage representing a portion of labor income that is common to all workers. Individual quantity of labor supply is $n_{j,s,t}$, and $e_{j,s}$ represents a labor productivity factor that augments or diminishes the productivity of a worker's labor supply relative to average productivity.

## Approach

Estimating lifetime productivity paths requires long panels of individual earnings, which do not exist for Ethiopia. As in the other OG-Core country models, we therefore start from the $J=10$ profiles estimated for the United States in [OG-USA](https://pslmodels.github.io/OG-USA/content/calibration/earnings.html) {cite}`DeBackerEtAl:2017`, interpolate them onto the seven lifetime-income groups used here, $\boldsymbol{\lambda}=[0.25, 0.25, 0.20, 0.10, 0.10, 0.09, 0.01]$, and then reshape them with Ethiopian data. The reshaping acts separately on the two dimensions of the matrix:

```{math}
  :label: eqnEarningsCalib
    e^{ETH}_{j,s} = e^{USA}_{j,s}\; A_s\; G_j, \qquad \sum_{s,j} e^{ETH}_{j,s}\,\omega_{s,j} = 1
```

where $A_s$ is an **age factor** built from National Transfer Accounts labor-income profiles and $G_j$ is a **group factor** built from World Inequality Database income shares, and $\omega_{s,j}$ is the steady-state population distribution over age and lifetime-income group. Both factors are ratios of the same series for Ethiopia and the United States, so that what carries over from the US estimates is the *within-country shape* of the profiles, while the level of every age relative to other ages, and of every group relative to other groups, is Ethiopia's relative to the United States'. Working with ratios of like-for-like series is also what keeps the concepts consistent: the OG-USA profiles measure lifetime labor earnings of workers, while the survey aggregates below measure annual income per adult, and those differences cancel to first order when Ethiopia is compared with the United States in the same source.

This replaces the earlier calibration, shared with several sibling models, that tilted the US profiles with a single scalar so that the model reproduced the ratio of the two countries' Gini coefficients. That approach used one number, and in `OG-ETH` the number was a World Bank *consumption* Gini for Ethiopia (31.1) set against an *income* Gini for the United States (41.5), which understated Ethiopian inequality (issue #33). The two-factor reshaping uses seven income shares and eighty ages instead of one statistic, and draws both countries' inputs from the same database and concept.

The code is in `ogeth/income.py`; `python -m ogeth.income` regenerates the packaged matrix from the packaged demographics and prints the comparison with the data.

### Age factor from National Transfer Accounts

The National Transfer Accounts project {cite}`NTA:2022` publishes smoothed mean labor income per capita by single year of age. Ethiopia's only profile is for 2005 (prepared by Terefe Degefa); the US profile closest in time is 2006. Labor income per capita mixes two things, how many people of an age work and how much each earns, and the model's $e_{j,s}$ is productivity per unit of labor supplied, with labor supply decided separately (see the labor supply section of {ref}`Chap_Households`). We therefore divide each per-capita profile by the ILOSTAT employment-to-population ratio of the same country and year (five-year age bands, interpolated), to obtain labor income per *worker* by age. Each per-worker profile is normalized to its mean over ages 20–64 and the age factor is their ratio:

```{math}
  A_s = \frac{y^{ETH}_s / \bar{y}^{ETH}}{y^{USA}_s / \bar{y}^{USA}}, \qquad y^c_s = \frac{\text{NTA labor income per capita}^c_s}{\text{employment ratio}^c_s}
```

Beyond 65 the employment ratio is a single open band in both countries, so per-worker earnings cannot be measured by age there; from 65 the factor is held at its mean over ages 60–64, and the shape of the profiles at older ages is the US shape (itself an arctan extrapolation for ages 80–100 in {cite}`DeBackerEtAl:2017`). Little labor is supplied at those ages, so this matters little for the results.

Ethiopian earnings per worker peak earlier and fall faster after 50 than US earnings. The factor is about 1.5 at age 20, 1.3 at 25, 1.2 through the thirties, 1.05 at 45, 0.8 at 55 and 0.55 from 60. As a result the population-weighted age profile of $e$ relative to the thirties is 0.69 in the twenties, 1.21 in the forties, 1.06 in the fifties and 0.72 in the sixties, where the US shape gives 0.60, 1.43, 1.69 and 1.63. This is the difference between an economy of long formal careers with returns to seniority and one where most work is physical and self-employed.

### Group factor from the World Inequality Database

The World Inequality Database (WID) {cite}`WID:2024` publishes pre-tax national income shares by percentile group for equal-split adults, for both countries in the same year. Ethiopia's series (2021 is the latest estimate) is built from the household consumption surveys with a correction for top incomes; the US series is built from tax and survey data. We read off the shares of the seven lifetime-income groups directly from the cumulative shares WID reports at the 25th, 50th, 70th, 80th, 90th and 99th percentiles, and the group factor is the ratio of each group's mean income relative to the economy-wide mean:

```{math}
  G_j = \frac{\text{share}^{ETH}_j / \lambda_j}{\text{share}^{USA}_j / \lambda_j} = \frac{\text{share}^{ETH}_j}{\text{share}^{USA}_j}
```

| Lifetime-income group | $\lambda_j$ | WID share, Ethiopia 2021 | WID share, United States 2021 | $G_j$ |
|---|---|---|---|---|
| 1 (bottom 25%) | 0.25 | 4.4% | 3.4% | 1.30 |
| 2 (25–50%) | 0.25 | 13.0% | 10.1% | 1.28 |
| 3 (50–70%) | 0.20 | 16.0% | 14.7% | 1.09 |
| 4 (70–80%) | 0.10 | 11.0% | 10.8% | 1.02 |
| 5 (80–90%) | 0.10 | 15.2% | 15.0% | 1.01 |
| 6 (90–99%) | 0.09 | 30.0% | 26.2% | 1.15 |
| 7 (top 1%) | 0.01 | 10.5% | 19.9% | 0.53 |

Relative to the United States, Ethiopia's bottom half earns a larger share of the total and its top percent about half as much. WID's Gini coefficients of pre-tax national income are 0.516 for Ethiopia and 0.583 for the United States.

## Results

```{figure} ./images/ability_profiles.png
---
height: 350px
name: FigLogAbil
---
Exogenous life cycle income ability paths $\log(e_{j,s})$ with $S=80$ and $J=7$
```

```{figure} ./images/earnings_vs_data.png
---
height: 350px
name: FigEarnVsData
---
Left: the population-weighted age profile of $e$ against the unadjusted OG-USA profile and the two NTA per-worker profiles it is reshaped with (each relative to its mean at ages 30–39). Right: the income share of each lifetime-income group implied by $e$ against the WID shares for Ethiopia and the United States.
```

{numref}`Figure %s <FigLogAbil>` shows the resulting $J=7$ deterministic lifetime ability paths $e_{j,s}$, and {numref}`Figure %s <FigEarnVsData>` compares them with the data they are built from. The implied shares do not reproduce Ethiopia's WID shares, and are not meant to: they sit between the US profiles' shares and Ethiopia's, because the group factor moves each group by Ethiopia's distance from the United States rather than to Ethiopia's level. The lifetime means of the seven groups are 0.54, 0.68, 0.82, 1.01, 1.25, 2.71 and 5.87, so the top percent's ability is 10.8 times the bottom quarter's, against 28.5 in the US profiles interpolated to the same groups and 8.8 in the previous Gini-tilted calibration.

The Gini coefficient of $e$ over the model population is 0.347. Two comparisons put that number in context. The family's Gini-ratio approach with concept-consistent inputs (WID pre-tax income for both countries) would target a model Gini of about 0.40–0.42, depending on whether the US anchor is the OG-USA matrix on its own population (0.479) or interpolated to seven groups on Ethiopia's population (0.448). Matching the WID Ethiopian shares directly, as `OG-ZAF` does, would give about 0.48. The two-factor result is below both because the share ratios cut the top percent's weight by half while lifting the bottom half, and the Gini is more sensitive to the top than the seven shares are to the Gini. We keep the share-based result: it matches seven moments rather than one, and the alternatives either mix concepts or ignore the wedge between annual income per adult and lifetime labor earnings of workers that the ratio to the United States controls for.

## Limits

Three limits are worth stating plainly. First, WID's pre-tax national income includes capital income, which is more concentrated than labor income, so the US ratio removes this difference only to the extent that capital's share of top incomes is similar in the two countries; WID publishes labor-income shares for the United States but not for Ethiopia, so the correction cannot be made country by country. Second, the model has no non-employment: every household of working age supplies labor, whereas WID's adult population includes people with no income. Ethiopia's employment ratios of 80–90 percent make this less of a distortion than it would be for the United States, but it is the reason we compare shares across countries rather than impose Ethiopia's shares directly. Third, the Ethiopian NTA profile is from 2005 and WID's Ethiopian estimate rests on consumption surveys with a borrowed top correction; no Ethiopian earnings panel exists to check either. The 2021/22 Ethiopia Socioeconomic Panel Survey has individual wages and hours and would allow a cross-sectional check of the age profile when its microdata are used.
