(Chap_matchingLWI)=
# Matching Labor, Wealth, and Income Moments

The calibration matches the distributions of income, labor and wealth with these parameters.  The lifetime productivity profiles $e_{j,s}$ match the World Inequality Database's income shares for the seven lifetime-income groups ({ref}`Chap_LfEarn`); the disutility of labor $\chi^n_s$ matches hours worked per person by age (the 2021 Labour Force and Migration Survey, with the decline from 65 from the 2021/22 panel survey) through age 68; and one discount factor for every group sets the long-run return to the world rate, with the WID wealth shares as a check rather than a target ({ref}`Chap_HouseholdCalib`).

(SecLWI_avghhinc)=
## Average household income
We use the average household income parameter (`mean_income_data`) to map the tax functions in country currency to model units.  It is household income per adult in FY2024/25, about 310,000 birr: rebased GDP per adult times the household-income share of GDP in the model. The derivation is in {ref}`Chap_Tax`.
