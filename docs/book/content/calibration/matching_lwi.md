(Chap_matchingLWI)=
# Matching Labor, Wealth, and Income Moments

The calibration matches the distributions of income, labor and wealth with three sets of parameters.  The lifetime productivity profiles $e_{j,s}$ match the World Inequality Database's income shares for the seven lifetime-income groups ({ref}`Chap_LfEarn`); the disutility of labor $\chi^n_s$ matches hours worked per person by age in the 2021 Labour Force and Migration Survey; and the discount factors $\beta_j$, with the bottom quarter's bequest weight $\chi^b_1$, match the WID wealth shares ({ref}`Chap_HouseholdCalib`).

(SecLWI_avghhinc)=
## Average household income
We use the average household income parameter (`mean_income_data`) to map the tax functions in country currency to model units.  It is rebased FY2024/25 GDP per adult, about 360,000 birr; the derivation is in {ref}`Chap_Tax`.
