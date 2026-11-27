# RiskLab — Quantitative Asset & Risk Management II

**Updated 27 November 2026 — final deployed client dashboard edition.**

RiskLab is a multi-asset portfolio management and risk-budgeting application developed for the Quantitative Asset & Risk Management II (QARM II) course at HEC Lausanne, University of Lausanne.

The project includes seven ETFs, six portfolio strategies, quarterly historical backtesting, constrained risk-budgeting optimization, covariance shrinkage, robustness analysis, raw and cleaned input exports, solver diagnostics, a readable offline report, and an interactive four-tab Streamlit application.

## Live application

**[Launch RiskLab](https://risklab-unil.streamlit.app/)**

RiskLab is deployed on Streamlit Community Cloud using Python 3.12. The application includes the historical ETF dataset and is ready to use without local installation or manual data uploads. The online dashboard has been tested against the local historical backtest and reproduces the displayed baseline financial results.

The application provides four interactive sections:

1. **Client mandate** — investment objectives, investment universe, portfolio constraints and historical data coverage.
2. **Portfolio optimizer** — target portfolio allocations, estimated risk contributions and constraint diagnostics.
3. **Risk analytics** — historical risk-budget deviations, diversification analysis, drawdowns and stress-period comparisons.
4. **Backtest & benchmarks** — historical performance, benchmark comparisons, portfolio holdings and reproducible result exports.

The application and source code repository are publicly accessible. No account or invitation is required to explore the project.

## Historical dataset

The repository includes the historical dataset used to produce the project results in `data/etf_prices.csv`.

| Property | Description |
|---|---|
| Source | Yahoo Finance via yfinance |
| Assets | SPY, VGK, EEM, IEF, LQD, GLD, DBC |
| Price frequency | Daily |
| Price adjustment | Dividend- and split-adjusted closing prices |
| Currency | USD |
| First common price observation | 6 February 2006 |
| Last price observation | 30 October 2026 |
| Common price observations | 5,217 |
| Baseline estimation window | 504 trading days |
| Robustness estimation window | 252 trading days |

The dataset is kept fixed to ensure reproducibility of the empirical results. Historical prices have undergone structural validation and external spot-checks. These checks provide supporting evidence of data quality but do not constitute independent authentication of every observation.

The external data verification record is available in [validation/RiskLab_Data_Verification.md](validation/RiskLab_Data_Verification.md).

The dataset is provided for authorized academic use within the project. No broader market-data redistribution permission is implied.

## Run the project

The repository contains the Python source code, historical ETF dataset, application, documentation, automated tests and validation records required to reproduce the project locally.

Provider archives, generated backtest outputs, virtual environments and supplied course materials are deliberately excluded.

Python **3.12 is recommended** for compatibility with the pinned project dependencies.

On Windows, use Python 3.12 (recommended), then double-click `SETUP_WINDOWS.bat` once. After setup, double-click `START_ANALYSIS.bat` or `START_APP.bat`. Alternatively, open PowerShell in this folder and create your new environment once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the batch launchers or invoke `.\.venv\Scripts\python.exe` directly; alternatively allow script execution only in the current terminal with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activation.

Install all dependencies once in the activated virtual environment:

```bash
python -m pip install -r requirements.txt
python run.py
python run.py --app
```

`python run.py` reuses `data/etf_prices.csv` if present; otherwise it downloads adjusted ETF prices from Yahoo Finance, subject to applicable provider terms. It validates the input and runs all six strategies twice (504 and 252 observations), saving `outputs_real_504/` and `outputs_real_252/`. The requested dates are fixed at 2005-01-01 through 2026-10-31 (exclusive endpoint 2026-11-01); common available history starts later. The existing CSV is not refreshed automatically. `--open` opens the baseline report after both runs finish. An explicitly requested `--refresh-data` attempts a new download for the fixed period and preserves the previous snapshot on successful replacement. Provider failure or invalid data stops the analysis; no artificial prices are generated or substituted.

The app uses that same saved CSV and the 504-day default, and independently recalculates when settings change. It does not download data on page load. If the saved file is missing, run the analysis first or upload a suitable CSV in the sidebar. The four tabs now include client mandate summary, grouped capital/risk bars, historical risk-budget comparison, percentage drawdowns, annual stress comparisons, dynamic performance cards and stacked holdings. Raw data and technical diagnostics remain available in expandable sections. `python -m streamlit run app.py` is equivalent to launching via `--app`.

Windows shortcuts: `SETUP_WINDOWS.bat` (once), `START_ANALYSIS.bat`, `START_APP.bat`. Unix equivalents: `sh SETUP_UNIX.sh`, `sh START_ANALYSIS.sh`, `sh START_APP.sh`.

## Use a different historical file

```bash
python run.py --input data/etf_prices.csv --window 504 --output outputs_my_data
```

Required columns: `Date,SPY,VGK,EEM,IEF,LQD,GLD,DBC`. Supply USD dividend- and split-adjusted closing prices and enough common history. An explicit input never triggers a replacement download. A custom `--output` requires a single `--window`; omit both to run the configured pair. Both CLI and app use the defaults in `risklab/config.py`.

Results calculated from user-supplied files remain **USER-SUPPLIED — NOT INDEPENDENTLY VERIFIED**. Matching source-archive bytes allow provider attribution but do not authenticate prices. The retired demo is no longer supported, and exact copies of it are rejected. See [REAL_DATA_README.md](REAL_DATA_README.md).

## What a run saves

| Output | Purpose |
|---|---|
| `raw_input.csv` | Exact imported CSV bytes; provider-table serialization for the optional Yahoo route |
| `cleaned_prices.csv`, `cleaning_report.json` | Actual analysis input and original cleaning history |
| `metrics.csv`, `wealth_index.csv`, `daily_returns.csv` | Strategy statistics, net wealth and net return histories |
| `trades.csv`, `target_weights.csv` | Separate signal/execution dates, planned targets, executed turnover and costs |
| Six `weights_*.csv` files | Daily post-execution holdings for each strategy |
| `risk_tracking.csv`, `next_quarter_realized_risk.csv` | Estimation-date risk shares and later fixed-target attribution |
| `solver_diagnostics.csv` and `.json` | Convergence, constraints, objective, risk residuals and candidate outcomes |
| `covariance_estimates.csv` | Sample, constant-correlation target and blended covariance at each signal |
| `calendar_year_returns.csv` | Each observed year independently, with actual date coverage |
| `source_provenance.json`, `run_metadata.json`, `run_status.json` | Data labels, hashes, settings, source fingerprints and run status |
| `report.html`, three `.svg` charts | A local readable report and exported charts |

The run keeps its input and result files together. A failed run records an error, never substitutes an input silently, and must not be interpreted using an earlier successful report. The export system preserves prior-run files when reusing an existing run folder.

## Models and project choices

- **Unconstrained ERC:** the supplied LN5 log-constraint optimization followed by rescaling.
- **Constrained risk budgeting:** minimize squared deviations from equal percentage risk shares under the client caps. This is an explicitly labelled project approximation; exact equal risk may be impossible.
- **Covariance:** the supplied LN3 constant-correlation target, blended with sample covariance. The fixed intensity `delta=0.5` is a visible project assumption, not an automatically estimated Ledoit–Wolf coefficient.
- **Trade timing:** signal at quarter-end close; execute at the next observed close; new holdings earn returns afterwards. Initial cash earns zero before execution.
- **Transaction costs:** 10 basis points per dollar traded by default, applied as a simple proportional NAV haircut. Purchases and sales both count.
- **Optional turnover penalty:** defaults to zero; for constrained strategies only, penalizes full absolute planned weight changes from holdings known at the signal. It is not a hard limit on later executed turnover.
- **Benchmarks:** 60/40, equal weight and unconstrained ERC are not forced to obey client caps. Limits apply to the two constrained targets at execution; holdings can drift beyond them between trades.

Read [docs/METHODS.md](docs/METHODS.md) for formulas, course pages, and exact distinctions between course content and implementation choices. Read [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) to follow one run from input through output. [docs/CHANGES.md](docs/CHANGES.md) maps the audit findings to fixes.

## Change a setting from the command line

```bash
python run.py --window 252 --shrinkage-intensity 0.25 --cost-bps 10 --output outputs_sensitivity
python run.py --window 504 --turnover-penalty 0.05 --output outputs_penalty
```

These commands reuse the saved historical snapshot unless `--input` selects another file. These are configuration examples, not recommended or empirically optimized settings. For every comparison, retain the input hash and settings. CLI numerical flags configure report runs; use the sidebar controls for the interactive app.

## Run the regression checks

```bash
python -m unittest discover -s tests -v
```

The test runner is part of Python. Checks include course identities, solver-failure rejection, cash/execution timing, future-data independence, drift/turnover/cost accounting, input provenance, export failure handling, and app control flow. The current verification record is in `validation/CLIENT_DASHBOARD_VALIDATION.txt`; the external data spot-check record is in `validation/RiskLab_Data_Verification.md`.

## Code map

| File | Responsibility |
|---|---|
| `run.py` | One-command entry point and optional app launcher |
| `risklab/config.py` | Shared settings, strategy names and result structure |
| `risklab/portfolio.py` | Covariance, risk contributions, constrained optimization and diagnostics |
| `risklab/core.py` | Price validation, chronological simulation and performance measures |
| `risklab/data.py` | Cached historical input, provider archive and validation |
| `risklab/export.py` | Complete run exports and local HTML/SVG report |
| `app.py` | Four-tab client interface; imports the shared calculations |
| `data/` | Fixed historical ETF price dataset |
| `tests/` | Standard-library regression checks |
| `validation/` | Historical data spot-check and prior local validation records |
| `docs/` | Mathematical methodology, implementation walkthrough and change history |
| `requirements.txt` | Project Python dependencies |
| `.streamlit/` | Streamlit application appearance and configuration |

## Cloud deployment

The application is successfully deployed on Streamlit Community Cloud.

**Live application:** [risklab-unil.streamlit.app](https://risklab-unil.streamlit.app/)

**Source repository:** [github.com/sye3124/RiskLab](https://github.com/sye3124/RiskLab)

### Deployment configuration

| Parameter | Value |
|---|---|
| Hosting platform | Streamlit Community Cloud |
| GitHub repository | `sye3124/RiskLab` |
| Branch | `main` |
| Main application file | `app.py` |
| Python version | 3.12 |
| Dependencies | `requirements.txt` |
| Historical dataset | `data/etf_prices.csv` |

Python 3.12 should be selected explicitly in the advanced deployment settings to maintain compatibility with the validated project dependencies.

The deployed application automatically loads the included historical dataset. If no saved dataset is present, the application defaults to CSV upload.

The application does not require a live market-data connection during normal operation.

The cloud dashboard has been visually checked across all four tabs, and its displayed baseline financial results have been compared with the local backtest. Automatic loading of the hosted historical dataset has also been tested successfully.

Supply and share only data you are authorized to use.

## Scope and limitations

RiskLab is an academic portfolio management and quantitative risk analysis project.

Historical backtest results are conditional on the selected investment universe, historical sample, covariance estimator, optimization objectives, estimation windows, transaction-cost assumptions and portfolio constraints.

Historical performance does not guarantee future performance. Estimated risk contributions do not guarantee equal subsequently realized risk contributions.

The application is intended for educational purposes and investment strategy analysis, not live trading or personalized investment advice.

---

**RiskLab — QARM II**

Quantitative Asset & Risk Management II  
HEC Lausanne — University of Lausanne  
2026
