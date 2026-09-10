# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-09 12:00:00

### Changed
* Made **informality the baseline tax calibration**, modeling it as graded income-tax *non-compliance* by lifetime-income group rather than a flat blended rate. Ethiopia's ~85% informal employment (ILO 2021) was previously approximated by a single low effective rate on everyone, which imposed a spurious work-and-saving wedge on the ~90% of households that in fact remit no income tax. Using OG-Core's by-`(t,j)` noncompliance dials, the bottom five of seven lifetime-income groups (90% of households by population weight) are set fully non-compliant, group 6 half, and the top group fully compliant. Five parameters change in `ogeth_default_parameters.json`: `etr_params` 0.03 → **0.1313** (the compliant-group effective rate, solved from a revenue identity to hit the PIT anchor), `mtrx_params` 0.20 → **0.35** (Ethiopia's statutory top PIT rate), `labor_income_tax_noncompliance_rate` and `capital_income_tax_noncompliance_rate` both `[0,…,0]` → **`[1,1,1,1,1,0.5,0]`**, and `adjustment_factor_for_cit_receipts` 0.2 → **0.327** (re-anchored to actual corporate-tax collections); `income_tax_filer` stays all 1.0 (non-compliance, not non-filing). Steady-state revenue now matches the data instrument by instrument — PIT 1.39% of GDP (target 1.4), CIT 1.71% (1.7), direct taxes 3.10% (3.1); sources IMF SIP 2025/108 ¶9 and CR 26/20 Table 2b — with the correct incidence (average income-tax rate by group `[0,0,0,0,0,0.066,0.131]`). Removing the flat-rate wedge lifts steady-state output ~7% (a recalibration effect, not a welfare claim). Documented in `calibration/taxes.md`; refs #71. Lifetime income is a proxy for formality (high-income informality is not representable), and time-varying formalization *reforms* are a separate exercise.
* Recalibrated the real and fiscal blocks to bring the steady state substantially closer to the actual Ethiopian economy. (1) **Capital share**: `gamma` 0.515 → 0.30, replacing the naive ILOSTAT labour share (0.385, biased down by Ethiopia's pervasive self-employment) with a Gollin-adjusted / sectorally-weighted labour share of 0.60; this cuts the model capital-output ratio from ~3.5 toward the Penn World Table's ~2.2 and the foreign-owned capital stock from ~1.3× to ~0.29× GDP (against a ~0.24× FDI stock). (2) **Remittances turned on**: `alpha_RM` 0 → 0.056 (IMF CR 26/20 balance-of-payments net private transfers, ~5.6% of GDP), which flips the steady-state trade balance from a wrong-signed +5.7% surplus to a −6.8% deficit, matching Ethiopia's actual −6.9% goods-trade deficit. (3) **Foreign aid turned on**: `alpha_FA` 0 → 0.01 (a modest long-run grant level; World Bank DPO on-budget grants were ~1.7% of GDP in FY2024/25, declining per the IMF program; superseded by the program path below), routing grants to the government so it can fund its spending — Ethiopia's low domestic revenue (~9% of GDP) cannot otherwise support positive government consumption on a debt-stable path. (4) **Payroll tax**: `tau_payroll` 0.18 → 0.03, the coverage-weighted *effective* pension-contribution rate (the 18% statutory rate covers only the small formal sector), correcting tax/GDP from an implausible 13% to ~9.2% (matching general-government revenue excluding grants). (5) `alpha_I` set to Ethiopia's actual on-budget public investment (~0.05, from the aspirational 0.065 recovery path). (6) `initial_Kg_ratio` 0.50 → **0.67**, Ethiopia's measured public-capital stock (IMF ICSD, 2019), so the model starts from the real state/SOE-built infrastructure and lets it depreciate toward the sustainable steady state (~0.28) over the transition — the honest counterfactual for an over-built stock that current revenue cannot maintain. This lifts baseline GDP ~10% in the start year (fading over a generation) versus starting near the steady state; the steady state itself is unchanged (initial_Kg_ratio is an initial condition only). `initial_guess_r_SS` retuned to 0.052. Fiscal aggregates are the IMF/World Bank general-government figures (federal + regional consolidated); the federal-to-regional block grant is excluded from `alpha_T`.
* Refreshed the single-industry baseline calibration to Ethiopian FY2024/25 (model start 2025), the first full-year vintage after the July 2024 exchange-rate float. Macro and open-economy block: `initial_debt_ratio` 0.327 → 0.50 and `initial_foreign_debt_ratio` 0.42 → 0.63 (the float revalued external debt; IMF fourth-ECF-review DSA / MoF Debt Bulletin No. 56, end-June 2025), `debt_ratio_ss` 0.40 → 0.30 (IMF fifth-review medium-term path to ~28.6% by FY2030/31), `zeta_D` 0.12 → 0.15, `zeta_K` 0.20 → 0.16 (Ethiopia's normalized Chinn-Ito index, 0.162, cross-checked against FDI/GFCF), `alpha_G` 0.055 → 0.058 (WDI `NE.CON.GOVT.ZS`, 2025), `alpha_T` 0.05 → 0.04, `alpha_I` path raised to 0.05 → 0.065 and `initial_Kg_ratio` 0.40 → 0.50 (IMF public-investment path), `tau_c` 0.07 → 0.06 (effective consumption-tax rate, FY2024/25 indirect taxes over private consumption). `world_int_rate_annual` stays 0.04 (Ethiopia is a distressed, near-closed sovereign; its integration is captured through low `zeta_K` and the debt-elastic rate).
* Moved the `g_y_annual` growth window from 2006-2024 to **2016-2025** (`G_Y_START_YEAR`/`G_Y_END_YEAR`), refreshing `g_y_annual` 0.0595 → 0.0470. The longer window embedded the unrepeatable 2004-2015 investment boom; the post-2015 decade is a more defensible balanced-growth-path rate. `gamma` refreshed 0.5169 → 0.51475 (latest ILOSTAT labour-share vintage, 2025).
* Turned on a **centered debt-elastic sovereign premium** (`r_gov_DY = -0.024`, `r_gov_DY2 = 0.04`), following OG-PHL/OG-IDN. The premium `r_gov_DY2·(D/Y − debt_ratio_ss)²` is re-centered on the steady-state debt ratio (0.30) so it is exactly zero at target — the steady state is unchanged — and only prices the transition-path debt overshoot. `r_gov_shift` is re-centered accordingly (-0.03377 → -0.03737).
* Froze the `r_gov_*` parameters in the live `update_from_api` path (they are no longer returned by `get_macro_params`): the committed `r_gov_shift` is the re-centered value, and returning the raw Li-Magud-Werner shift would silently un-center the premium and move the steady state. The Li-Magud-Werner derivation plus the re-centering arithmetic is preserved as the reproducible `macro_params.estimate_r_gov` helper.
* Retuned `initial_guess_r_SS` (0.0648 → 0.093) to the refreshed steady state.
* Calibrated the transition path to the IMF program (Country Report 26/174, fifth ECF review, Tables 1, 2b and 4b) instead of holding FY2024/25 spending ratios fixed for twenty years. Previously the baseline ran a primary deficit of 5.6–6.1 percent of GDP for its first two decades (government consumption 5.8, transfers 4.0 and public investment 5.0 percent of GDP against 8.7 percent of revenue), debt climbed from 50 to 84 percent of GDP by year 10, and the closure rule later pushed government consumption below zero. The fiscal block now follows the program for FY2024/25–FY2030/31 (model periods 0–6) and holds the last value: government consumption (`alpha_G`, recurrent spending net of interest and cash transfers) 4.3 → 5.9 percent of GDP, cash transfers (`alpha_T`: fuel and fertilizer subsidies, the safety net, public pensions) 2.0 → 1.5 percent, public investment (`alpha_I`, capital expenditure) 5.0 → 5.1 percent, grants (`alpha_FA`, now a path) 1.7 → 0.3 percent, and revenue rising with the program from 9.2 to 12.3 percent of GDP through formalization (below), the consumption-tax rate (`tau_c` 0.060 → 0.082) and CIT collections (`adjustment_factor_for_cit_receipts` 0.327 → 0.511). `initial_debt_ratio` is 0.505 (the program's FY2024/25 figure) and the closure rule starts after the program (`tG1` 20 → 7). Reproducing the program's debt decline with primary balances near zero requires a negative real effective interest rate on the legacy debt — the counterpart of inflation and concessional terms eroding it — so `r_gov_shift` is a path that puts the sovereign rate on the program-implied real rates (−3.5 to −1.4 percent) and converges over four periods to a long-run 2 percent (the previous 4.5 percent long-run rate exceeded even the program's FY2030/31 nominal effective rate of 5.2 percent). OG-Core clips the sovereign rate at `r_gov_floor` (a parameter since PSLmodels/OG-Core#1203, hard-coded at zero before), and the example lowers the floor to −10 percent where the installed OG-Core has the parameter so the negative rates bind. The centering constant of the debt-elastic premium is now subtracted from the shift, as OG-Core's formula requires (`r_gov_DY d + r_gov_DY2 d² = r_gov_DY2 (d − D̄)² − r_gov_DY2 D̄²`); the previous calibration added it, which put the long-run sovereign rate about 0.7 percentage points below the level it documented. On an OG-Core without the parameter the rate sits at zero through the program and the debt ratio declines more slowly than the program's — to about 36 rather than 29 percent of GDP by FY2030/31 — reaching the 30 percent target through the closure rule afterwards; the gap is the inflation-tax channel a real model with a non-negative rate leaves out.
* Extended the informality calibration from the steady state to the program and to pensions. The program's revenue gains are direct-tax heavy — in the first nine months of FY2025/26 federal direct taxes grew 78 percent against 41 percent for domestic VAT and 5 percent for import taxes (IMF CR 26/174, p. 15) — so part of the gain is now the tax base broadening rather than higher rates: over the seven program years lifetime-income group 6 moves from half to full income-tax compliance and group 5 from none to a fifth (`labor_income_tax_noncompliance_rate` and `capital_income_tax_noncompliance_rate` are paths, `[1,1,1,1,1,0.5,0]` → `[1,1,1,1,0.8,0,0]`), worth about 0.7 points of GDP of personal income tax, and the remaining gain is split 60/40 between consumption taxes and CIT collections (`tau_c` 0.060 → 0.082, `adjustment_factor_for_cit_receipts` 0.327 → 0.511) instead of two thirds / one third. Pensions now go only to the formal groups: `replacement_rate_adjust` is `[0,0,0,0,0,0.5,1]` by lifetime-income group, mirroring the compliance structure, where before every household drew a US-style public pension although Ethiopia's PSSSA and POESSA schemes cover only formal employees.
* Calibrated the capital block to Ethiopia's capital-output ratio and foreign-owned capital stock. The inherited US patience (`beta_annual` 0.96) put the steady-state capital-output ratio near 2.9 and gross investment above 40 percent of GDP, against the Penn World Table's 2.2 and the program's 20–28 percent; and the 4 percent world rate — which in OG-Core fixes the capital stock foreign investors are willing to hold, `K_f = zeta_K (K_open − K_d)` — put the foreign-owned capital stock at 0.37 of GDP against a ~0.24 FDI stock. Two levers, both set to data: `beta_annual` 0.96 → 0.90 (household wealth near 2.4 times GDP, close to the 2.15 initial-wealth anchor) and `world_int_rate_annual` 0.04 → 0.063, UNCTAD's measured return on inward FDI in Africa (World Investment Report 2018). The steady state now has a capital-output ratio of 2.4 (2.1 at the start of the transition), a foreign-owned capital stock of 0.20 of GDP, a return on capital of 6.8 percent and investment of 33 percent of GDP. Patience alone (0.92 with the 4 percent world rate) had taken the capital-output ratio only to 2.6 while pushing the foreign-owned stock up; `households.md` and the world-rate section of `macro.md` document the arithmetic.
* Anchored initial household wealth to the data where the installed OG-Core supports it (PSLmodels/OG-Core#1189, `initial_wealth_ratio`): B(0)/Y = 2.15, the Penn World Table capital-output ratio (2.2) less the foreign-owned capital stock (0.24) plus domestically held public debt (0.19). The example applies it when available and otherwise starts from steady-state wealth, as before; the value moves into the packaged JSON once that OG-Core release is the floor.
* Completed the remittance calibration, following OG-PHL's approach (EAPD-DRB/OG-PHL#85). The level is re-anchored to the observed value: the IMF fifth ECF review (Country Report 26/174, Table 4a) records FY2024/25 net private transfers of USD 7,037 million, 5.6 percent of GDP, so `alpha_RM_1 = alpha_RM_T = 0.056` is now the measured FY2024/25 figure rather than a program anchor (the World Bank WDI series puts the same inflow at USD 7.14 billion). `g_RM` is now a path derived from the packaged `g_n` (`g_RM_t = exp(g_y)(1 + g_n_{t-1}) − 1`) so remittances grow with trend GDP and keep a constant share on the balanced growth path: before `tG1` OG-Core compounds detrended remittances by `(1 + g_RM)/(exp(g_y)(1 + g_n))` regardless of that period's output, and with the shipped scalar `g_RM = 0` Ethiopia's growth rates shrank them relative to trend by about 7 percent a year — from 5.6 to about 1.5 percent of GDP over the first twenty periods — before OG-Core blended them back. Along the baseline transition RM/Y now holds at 5.6–5.7 percent of GDP. `eta_RM` is no longer population-proportional: the World Bank's 2010 national remittance survey shows recipients concentrated in better-off, urban households (recipients' household income 9 percent below ETB 1,000 a month, 47 percent at ETB 1,000–3,500, 37 percent above; 35 percent urban against 16 percent of the population), which maps to remittance value shares by income quintile of 4.8, 4.8, 25.3, 25.3 and 39.8 percent, spread per capita across ages within each lifetime-income group. Both derived objects are rebuilt with the demographics by `update_baseline.py`, and `tests/test_remittances.py` pins the level to its source and asserts the share stays flat along the whole transition.
* Require `ogcore>=0.20.0` (the current OG-Core release), which carries the fixes this calibration depends on: 0.19.1's corrected payroll-tax calculation and bequest split under income-group demographics, and 0.19.2's removal of a payroll-revenue double count that made any model with a nonzero `tau_payroll` fail the steady-state resource constraint. With `tau_payroll = 0.03` the baseline steady state now solves with a resource-constraint error of 2e-14.

## [0.2.0] - 2026-08-31 14:00:00

### Changed
* Require `ogcore>=0.18.0` and migrate the calibration to its income-group-varying demographics (PSLmodels/OG-Core#1165): the packaged demographic arrays (`omega`, `omega_SS`, `rho`, `imm_rates` and their preTP seeds) are regenerated in the new age-by-income shape with the new `update_baseline_demographics` tool (macro parameters untouched, enforced by the tool's clobber guard), and both `get_pop_objs` call sites pass `income_percentiles=p.lambdas.flatten()` as 0.18 requires. OG-ETH's demographics do not vary by income group, so the new arrays are the old ones spread across groups by `lambdas`: the age distribution and the regenerated earnings matrix reproduce the previous values to machine precision, and model results are unchanged. `income.get_e_interp` now reads the OG-USA snapshot's raw JSON values instead of loading them through a `Specifications` object, which decouples it from the installed ogcore's array schema (the 0.18 schema rejects OG-USA's not-yet-migrated shapes) and accepts age weights in either the 1-D or the new age-by-income shape. The multisector JSON's demographic arrays (an older data vintage than the single-industry file) are expanded to the new shape mechanically — distributions scaled by `lambdas`, rates replicated across groups — so their values are bit-for-bit preserved rather than re-downloaded.
* Raised the Python floor to `>=3.12` (matching CI and ogcore's own `>=3.12` requirement; classifiers and the ruff target follow) and relocked to a single `ogcore 0.16.3`, matching OG-ZAF and OG-IDN. This removes the stale Python 3.11 resolution branch that pinned an older ogcore.
* Regenerated the baseline demographics in `ogeth_default_parameters.json` under ogcore 0.16.3, which reworks the pre-time-path population distribution (PSLmodels/OG-Core#1073): the transition-path arrays (`omega`, `g_n`, `imm_rates`, `rho`) shift by one period and three period-0 seeds (`g_n_preTP`, `imm_rates_preTP`, `rho_preTP`) are added.
* Limited the `update_from_api` macro calibration to the sources that are authoritative for Ethiopia: `g_y_annual` (World Bank WDI) and `gamma` (UN ILOSTAT) still update, while the World Bank QPSD debt pull and the IMF `alpha_T`/`alpha_G` pull are switched off (QPSD has no Ethiopia data; the IMF series returns only 2002 values). Debt ratios, `alpha_G`, `alpha_T`, and `r_gov_*` stay at the documented values in `calibration/macro.md`. This refreshes `g_y_annual` (0.060 → 0.0595) and `gamma` (0.518 → 0.517).

### Fixed
* Fixed the demographic `country_id` in `calibrate.py`, which pulled South Africa (UN code 710) data instead of Ethiopia (231), and regenerated the baseline demographics in `ogeth_default_parameters.json`. Steady-state population growth corrects from 0.4% to 2.0%; macro parameters are unchanged.
* Brought all installation instructions in line with the uv workflow the project migrated to in 0.1.0, matching the same fix in OG-PHL and OG-ZAF. The README now documents two supported paths, each as per-platform copy-paste blocks verified end to end: the OG family's universal installer (`install.sh --repo og-eth`, from PSLmodels/OG-Core) and a manual install (install uv, clone, `uv run python examples/run_og_eth.py`). The PyPI install section is dropped: `pip install ogeth` fails outright on the Python that ships with macOS (3.9), silently installs an ogcore older than the tested one on Python 3.11, and does not pin the tested ogcore even on a supported Python. The contributor guide and the UN tutorial no longer instruct readers to build the deleted `ogeth-dev` conda environment (`environment.yml` was removed in 0.1.0, so those steps failed at the first command); both now use `uv sync --extra dev` and `uv run`, the contributor guide's test command matches CI (`pytest -m "not local"` instead of OG-USA's `needs_puf` suite), stale `master`-branch references now say `main`, and the 3-period-model solutions page points at OG-ETH instead of OG-IDN.

## [0.1.0] - 2026-05-20 12:00:00

### Changed
* Migrated the project from conda to uv. Install with `uv sync --extra dev`; `pyproject.toml` is the single source of truth for dependencies and `uv.lock` pins exact versions.
* CI uses `astral-sh/setup-uv`, and ruff replaces black for formatting and linting (`check_format.yml` -> `check_ruff.yml`).
* Updated README, AGENTS.md, and the Makefile to the uv workflow.

### Removed
* `setup.py`, `environment.yml`, and `pytest.ini` (their settings moved into `pyproject.toml`).

## [0.0.8] - 2026-05-18 23:00:00

### Added
* Reads the SAM file from `ogeth/data/` instead of fetching it from GitHub at runtime, so offline runs work
* Adds a `pip-import-smoke` CI job that installs the package and imports it from a temp directory, catching packaging issues invisible from the source tree

### Fixed
* Fixes `alpha_c` to sum only the ten household columns of the SAM (instead of total - row, which included government, investment, and intermediate use), matching OG-IDN and OG-PHL

## [0.0.7] - 2026-05-12 00:50:00

### Fixed

- Fixed bug in `calibrate.py` where the `income.get_e_interp` function was not being called with the correct parameters. This was causing an error when running the `calibrate.py` script.

## [0.0.6] - 2026-04-15 15:50:00

### Added

- Updates connections to API calls to the World Bank, IMF, and UN in `macro_params.py` and `calibrate.py` to allow for updating the exogenous parameters from the APIs. This is currently set to `False` by default, but can be set to `True` to update the parameters from the APIs when running the `calibrate.py` script. The documentation in `exogenous_parameters.md` has also been updated to reflect this change.
- Updates how the SAM file is loaded in `input_output.py`
- Adds an `update_baseline.py` script that updates the default parameters in `ogeth_default_parameters.json` based on the output of the `calibrate.py` script. This allows us to easily update the default parameters in the JSON file when we run the calibration script.

## [0.0.5] - 2025-11-17 23:40:00

### Added

- Updates average household income `mean_income_data` to ETB 157,845 and the corresponding documentation in `matching_lwi.md`
- Updates initial debt-to-GDP and the corresponding documentation in `macro.md`

## [0.0.4] - 2025-11-17 18:30:00

### Added

- Updates the TPI resource constraint `RC_TPI=0.01`

## [0.0.3] - 2025-11-17 13:00:00

### Added

- Updates default parameters

## [0.0.2] - 2025-11-16 13:00:00

### Added

- Fixes black formatting in `income.py` and `input_output.py`
- Fixes a typo in `constants.py`
- Fixes an error in the `deploy_docs.yml` and `docs_check.yml` files
- Adds Jason as a core maintainer in `intro.md`. This also allows us to see if the documentation GH Actions work.
- Removed `test_income.py` and `test_input_output.py` tests

## [0.0.1] - 2025-11-16 12:30:00

### Added

- Adds 3 logo files to the `./docs/` directory: `OG-ETH_logo_gitfig.png`, `OG-ETH_logo_long.png`, and `OG-ETH_logo.png`.
- Updates a `.gitignore` file.
- Fixes references in `./docs/book/content/OGETH_references.md`, `./docs/create_doc_figures.py`, `PSL_catalog.json,` and `./docs/README.md`
- Fixes badges in `README.md` and `intro.md`
- Pins the `environment.yml` package `jupyter-book<2.0.0` so that the book can build with `jb build ...` command.
- Updates the functions in `input_output.rst` and `utils.rst`
- Updates the Jupyter metadata in `earnings.md` and `exogenous_parameters.md`. This is what was stopping the Jupyter Book from compiling (once we pinned `jupyter-book<2.0.0`).
- Adds GH Action files `build_and_tes.yml`, `check_format.yml`, `deploy_docs.yml`, `docs_check.yml`, `publish_to_pypi.yml`, `ISSUE_TEMPLATE.md`, and `PULL_REQUEST_TEMPLATE.md`. These files required me to add OG-ETH to Codecov.io, add a repository secret for Codecov, create the gh-pages branch with the files for the Jupyter Book and publish it as a GitHub pages site, create and upload the first version of the `ogeth` package to PyPI.org, and add a repository secret for PYPI.

## [0.0.0] - 2025-10-06 12:00:00

### Added

- This version is a pre-release alpha. The example run script OG-ETH/examples/run_og_eth.py runs, but the model is not currently calibrated to represent the Ethiopian economy and population.


[0.2.0]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.8...v0.1.0
[0.0.8]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/EAPD-DRB/OG-ETH/compare/v0.0.0...v0.0.1
