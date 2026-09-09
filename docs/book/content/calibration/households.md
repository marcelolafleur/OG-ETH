(Chap_HouseholdCalib)=
# Calibration of Household Preference Parameters

## Behavioral Assumptions

### Elasticity of labor supply

As discussed in the [OG-Core household theory documentation](https://pslmodels.github.io/OG-Core/content/theory/households.html), we use the elliptical disutility of labor function developed by {cite}`EvansPhillips:2017`.  We then fit the parameters of the elliptical utility function to match the marginal disutility from a constant Frisch elasticity function.  `OG-ETH` users enter the constant Frisch elasticity as a parameter.  {cite}`Peterman:2016` finds a range of Frisch elasticities estimated from microeconomic and macroeconomic data.  These range from 0 to 4.  Peterman makes the case that in lifecycle models without an extensive margin for employment the  Frisch elasticity should be higher. For `OG-ETH` we take a default value of 0.4 from {cite}`Altonji:1986`.

### Intertemporal elasticity of substitution

The default value for the intertemporal elasticity of substitution, $\sigma$, is taken from {cite}`ABMW:1999`.  We set $\sigma=1.5$.

### Rate of time preference

The rate of time preference, $\beta$, is calibrated to Ethiopia's capital-output ratio rather than borrowed from the US literature.  With the US value of $\beta = 0.96$ ({cite}`Carroll:2009`), which the OG-Core country models inherit by default, the OG-ETH steady state has a capital-output ratio near 2.9 and gross investment above 40 percent of GDP, against a Penn World Table capital-output ratio of about 2.2 and investment of 20–28 percent of GDP in the IMF program ({cite}`IMFCR26174:2026`, Table 1).  In the steady state the return on capital is close to $1/\beta - 1$ and the capital-output ratio is $K/Y = \gamma/(r + \delta)$, so matching $K/Y \approx 2.2$ with the calibrated capital share $\gamma = 0.30$ and depreciation of 5 percent requires $r \approx 8.6$ percent and hence $\beta \approx 0.92$.  We set $\beta = 0.92$ (annual) for all lifetime-income groups.  A high return on capital is the flip side of Ethiopia's scarce capital, and the value is within the range used in low-income-country OLG calibrations; the steady-state validation table in {ref}`Chap_MacroCalib` reports the resulting capital-output ratio, investment share and return.
