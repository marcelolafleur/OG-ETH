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

(Chap_FirmCalib)=
# Calibration of Firm Parameters

```{code-cell} ipython3
:tags: ["remove-cell"]

from importlib.resources import files
import json
import numpy as np
from myst_nb import glue
from ogcore.parameters import Specifications
from ogeth import macro_params as mp

params = json.loads(
    files("ogeth")
    .joinpath("ogeth_default_parameters.json")
    .read_text(encoding="utf-8")
)
p = Specifications()
p.update_specifications(params)
alpha_I = float(np.asarray(p.alpha_I)[-1])
kg_ss = ((1 - mp.PUBLIC_INVESTMENT_EFFICIENCY_GAP) * alpha_I
         / (np.exp(p.g_y) * (1 + p.g_n_ss) - (1 - mp.DELTA_G_ANNUAL)))
glue("gamma", f"{float(params['gamma'][0]):.3f}", display=False)
glue("gamma_g", f"{float(params['gamma_g'][0]):.2f}", display=False)
glue("labour_share", f"{1 - mp.GAMMA - mp.GAMMA_G:.2f}", display=False)
glue("firm_r", f"{100 * mp.FIRM_RETURN_TARGET:.0f}", display=False)
glue("world_r", f"{100 * mp.WORLD_INT_RATE_ANNUAL:.0f}", display=False)
glue("tau_b", f"{mp.BUSINESS_TAX_WEDGE:.3f}", display=False)
glue("delta_tau", f"{mp.DELTA_TAU_ANNUAL:.3f}", display=False)
glue("delta", f"{float(params['delta_annual']):.4f}", display=False)
glue("delta_g", f"{mp.DELTA_G_ANNUAL:.2f}", display=False)
glue("pwt_delta", f"{mp.PWT_DEPRECIATION_RATE:.4f}", display=False)
glue("ky_total", f"{mp.TOTAL_CAPITAL_OUTPUT_RATIO:.2f}", display=False)
glue("ky_private", f"{mp.PRIVATE_CAPITAL_OUTPUT_RATIO:.2f}", display=False)
glue("ky_private_model", f"{mp.to_model(mp.PRIVATE_CAPITAL_OUTPUT_RATIO):.2f}",
     display=False)
glue("kg_old", f"{mp.PUBLIC_CAPITAL_OUTPUT_RATIO_OLD_BASE:.3f}", display=False)
glue("kg_rebased", f"{mp.PUBLIC_CAPITAL_OUTPUT_RATIO:.3f}", display=False)
glue("pub_share", f"{mp.PUBLIC_CAPITAL_OUTPUT_RATIO / mp.TOTAL_CAPITAL_OUTPUT_RATIO:.3f}",
     display=False)
glue("leakage", f"{float(params['infra_investment_leakage_rate']):.2f}", display=False)
glue("kg0", f"{float(params['initial_Kg_ratio']):.3f}", display=False)
glue("alpha_I_lr", f"{alpha_I:.4f}", display=False)
glue("g_n_ss", f"{p.g_n_ss:.4f}", display=False)
glue("kg_ss", f"{kg_ss:.3f}", display=False)
```

## Aggregate Production Function and Capital Accumulation

The [OG-Core firm theory documentation](https://pslmodels.github.io/OG-Core/content/theory/firms.html) outlines the constant returns to scale, constant elasticity of substitution production function of the representative firm.  This function has two parameters; the elasticity of substitution and capital's share of output.

The production function is given as:

```{math}
:label: EqFirmsCESprodfun
  \begin{split}
    Y_{m,t} &= F(K_{m,t}, K_{g,m,t}, L_{m,t}) \\
    &\equiv Z_{m,t}\biggl[(\gamma_m)^\frac{1}{\varepsilon_m}(K_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + (\gamma_{g,m})^\frac{1}{\varepsilon_m}(K_{g,m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + \\
    &\quad\quad\quad\quad\quad(1-\gamma_m-\gamma_{g,m})^\frac{1}{\varepsilon_m}(e^{g_y t}L_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\biggr]^\frac{\varepsilon_m}{\varepsilon_m-1} \quad\forall m,t
  \end{split}
```

  This production function has the following parameters:
  * $\varepsilon_m$ is the elasticity of substitution between capital, labor, and infrastructure in sector $m$.
  * $\gamma_m$ is the share of capital in sector $m$.
  * $\gamma_{g,m}$ is the share of government capital in sector $m$.
  * $Z_{m,t}$ is the total factor productivity in sector $m$ at time $t$.

### Elasticity of substitution

`OG-ETH`'s default parameterization has an elasticity of substitution of $\varepsilon=1.0$, which implies a Cobb-Douglas production function.

### Factor shares of output

The private capital share is $\gamma_m =$ {glue:text}`gamma` and the public capital share $\gamma_{g,m} =$ {glue:text}`gamma_g`, so OG-Core's labour share, $1 - \gamma_m - \gamma_{g,m}$, is {glue:text}`labour_share`.  The labour share here is household income from work, land and own enterprise: in OG-Core, everything that is not a return to reproducible private or public capital.

#### Calibrating the capital share

Ethiopia publishes no income-side GDP, so the capital share cannot be measured directly, and the two available measures both misclassify the self-employed, who are about three-quarters of employment.  The IFPRI 2022 social accounting matrix books their mixed income as capital (58 percent of value added).  The [ILOSTAT labour income share](https://rshiny.ilo.org/dataexplorer41/?lang=en&segment=indicator&id=SDG_1041_NOC_RT_A) (0.385 in 2025) imputes their labour income at employee earnings, which in Ethiopia fall far below the self-employed's value added per worker.  Taken at face value, the ILOSTAT share would need a real return on capital above 20 percent at Ethiopia's capital stock.

We therefore set $\gamma_m$ from the firms' first-order condition at a target marginal return, the standard fallback when profits cannot be measured ({cite}`CaselliFeyrer:2007`):

```{math}
:label: EqGammaFOC
\gamma_m = \frac{r + \delta - \tau_b\,\delta_\tau}{1 - \tau_b}\;\frac{K}{Y},
```

with the target firms' return $r$ at {glue:text}`firm_r` percent, just above the {glue:text}`world_r` percent world rate ({ref}`Chap_MacroCalib`); private depreciation $\delta =$ {glue:text}`delta` (below); the effective business tax rate $\tau_b =$ {glue:text}`tau_b` (the CIT rate times the corporate share of assets times the CIT collections factor, {ref}`Chap_Tax`) and its depreciation allowance $\delta_\tau =$ {glue:text}`delta_tau`; and private capital of {glue:text}`ky_private` times GDP at market prices on the rebased accounts, {glue:text}`ky_private_model` in the model's output units ({ref}`Sec_Rebasing`).  This gives $\gamma_m =$ {glue:text}`gamma` (`macro_params.GAMMA`).  Because OG-Core passes public capital's rent to the owners of private capital (`aggregates.get_r_p`), they earn about 9.5 percent, inside the 6–10 percent range of the evidence: realized FDI returns in African peers of about 4 percent real after tax plus a marginal premium of 2–4 points, and Caselli and Feyrer's returns corrected for natural capital.  The implied labour share, {glue:text}`labour_share`, is inside {cite}`Gollin:2002`'s range of 0.65–0.80 for labour shares that count the self-employed's income.  Land in Ethiopia is state-owned and worked by households, so its return belongs with labour income too.

The capital share is decided here and is not moved to close a capital-output gap.  Patience is set by the owners' long-run return instead ({ref}`Chap_HouseholdCalib`).  The SAM-based per-industry capital shares of the multi-industry calibration, which separate mixed income industry by industry, are the way to refine it.

#### The public capital share

$\gamma_{g,m} =$ {glue:text}`gamma_g` is the low end of {cite}`BomLigthart:2014`'s meta-analysis of 578 estimates of the output elasticity of public capital (0.083 in the short run, 0.122 in the long run, for capital supplied by central government).  The low end is deliberate: OG-Core passes public capital's rent to the owners of private capital, so their return exceeds the firms' marginal return by about $(1-\tau_b)\gamma_g Y/K$, and at Ethiopia's capital-output ratio a larger $\gamma_g$ would push the firms' return below the world rate, where foreign capital would leave.

### Public-investment efficiency

OG-Core's law of motion for public capital is

```{math}
:label: EqPublicCapitalLOM
K_{g,m,t+1} = (1 - \delta_g)\,K_{g,m,t} + (1 - \phi_g)\,I_{g,m,t}
```

where $\delta_g$ is the depreciation rate of public capital (`delta_g_annual`, {glue:text}`delta_g`) and $\phi_g$ (`infra_investment_leakage_rate`) is the fraction of public investment that does not become productive capital.  We set $\phi_g =$ {glue:text}`leakage`, the hybrid efficiency gap the IMF-World Bank Public Investment Management Assessment for Ethiopia estimates (June 2025, section II: access and quality of infrastructure per head, against the efficiency frontier).  The same gap discounts the measured stock at the start (below); it does not scale $\gamma_g$, which multiplies the productive stock.

Public investment flow is set as a share of GDP, $I_{g,t} = \alpha_{I,t}\,Y_t$.  `alpha_I` is general-government capital expenditure from the same fiscal accounts as the rest of the budget ({cite}`IMFCR26174:2026`, Table 2b), converted from old-base GDP to the model's output ({ref}`Sec_Rebasing`): 0.032 in FY2024/25, the outturn (5.0 percent of old-base GDP), and 0.036 from FY2025/26, the enacted budget (5.8 percent), held ({ref}`Sec_Scenarios`).  Much of Ethiopia's public investment runs off-budget through state-owned enterprises, so this on-budget general-government $\alpha_I$ understates total public-sector investment.

### Initial public capital to GDP ratio

The parameter `initial_Kg_ratio` sets the productive public capital stock in the start year.  The measured stock is the [IMF Investment and Capital Stock Dataset](https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.FAD:ICSD(1.0.0))'s general-government capital at current prices over GDP at current prices, {glue:text}`kg_old` of old-base GDP in 2019, its latest year (the PIMA quotes the same 84 percent), held at that ratio to FY2024/25.  It is accumulated budget spending in birr, so it converts like other fiscal stocks, to {glue:text}`kg_rebased` of rebased GDP.  Only the efficient share of it is productive, so the model starts from $(1-\phi_g)$ times that, {glue:text}`kg0` in the model's output units (`macro_params.INITIAL_KG_RATIO`).

The steady state implied by the stationarized law of motion, at the long-run $\alpha_I$ of {glue:text}`alpha_I_lr` and steady-state population growth of {glue:text}`g_n_ss`,

```{math}
:label: EqInitialKgSS
\bar{K}_g / \bar{Y} = \frac{(1-\phi_g)\,\alpha_I}{e^{g_y}(1+\bar{g}_n) - (1-\delta_g)},
```

is {glue:text}`kg_ss`, close to the starting stock; along the way it falls to about 0.18 of output in the first decades, because output grows faster than the public investment that adds to the stock, and recovers over the following century.  The large measured stock, built by the state-led and largely SOE-financed investment of the 2010s, enters at its productive share; the rest is the inefficiency the PIMA measures.

### Depreciation

OG-Core's `delta_annual` is the depreciation rate of private capital; public capital depreciates at its own $\delta_g$.  The Penn World Table 11.0 depreciation rate of Ethiopia's total capital stock is {glue:text}`pwt_delta` in 2023.  With public capital {glue:text}`pub_share` of the rebased total ({glue:text}`kg_rebased` over {glue:text}`ky_total`, {ref}`Sec_Rebasing`) depreciating at $\delta_g =$ {glue:text}`delta_g`, the private rate that reproduces the total is $\delta =$ {glue:text}`delta` (`macro_params.DELTA_ANNUAL`).

### Total factor productivity

In the case of the single production sector, we can normalize $Z_{m,t}=1.0$.
