# RiskLab — runnable QARM II project

**Updated 26 November 2026 — client dashboard edition.** Seven ETFs, six portfolio strategies, a quarterly backtest, raw and cleaned input exports, solver diagnostics, a readable offline report, and the four-tab Streamlit application in the supplied blueprint.

## Run the project

This is the code-only repository edition. Historical market prices, provider archives, generated outputs and supplied course materials are deliberately excluded. Use your own authorized adjusted-price CSV or, where permitted by provider terms, run the optional downloader. The separately retained local team snapshot is identified in the data spot-check audit. No market-data redistribution permission is implied.

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

`python run.py` reuses `data/etf_prices.csv` if present; otherwise it downloads adjusted ETF prices from Yahoo Finance. It validates the input and runs all six strategies twice (504 and 252 observations), saving `outputs_real_504/` and `outputs_real_252/`. The requested dates are fixed at 2005-01-01 through 2026-10-31 (exclusive endpoint 2026-11-01); common available history starts later. The existing CSV is not refreshed automatically. `--open` opens the baseline report after both runs finish. An explicitly requested `--refresh-data` attempts a new download for the fixed period and preserves the previous snapshot on successful replacement. Provider failure or invalid data stops the analysis; no artificial prices are generated or substituted.

The app uses that same saved CSV and the 504-day default, and independently recalculates when settings change. It does not download data on page load. If the saved file is missing, run the analysis first or upload a suitable CSV in the sidebar. The four tabs now include client mandate summary, grouped capital/risk bars, historical risk-budget comparison, percentage drawdowns, annual stress comparisons, dynamic performance cards and stacked holdings. Raw data and technical diagnostics remain available in expandable sections. `python -m streamlit run app.py` is equivalent to launching via `--app`.

Windows shortcuts: `SETUP_WINDOWS.bat` (once), `START_ANALYSIS.bat`, `START_APP.bat`. Unix equivalents: `sh SETUP_UNIX.sh`, `sh START_ANALYSIS.sh`, `sh START_APP.sh`.

## Use a different historical file

```bash
python run.py --input data/etf_prices.csv --window 504 --output outputs_my_data
```

Required columns: `Date,SPY,VGK,EEM,IEF,LQD,GLD,DBC`. Supply USD dividend- and split-adjusted closing prices and enough common history. An explicit input never triggers a replacement download. A custom `--output` requires a single `--window`; omit both to run the configured pair. Both CLI and app use the defaults in `risklab/config.py`.

Results remain **USER-SUPPLIED — NOT INDEPENDENTLY VERIFIED**. Matching source-archive bytes allow provider attribution but do not authenticate prices. The retired demo is no longer supported, and exact copies of it are rejected. See [REAL_DATA_README.md](REAL_DATA_README.md).

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
| `tests/` | Standard-library regression checks |
| `validation/` | Historical data spot-check and prior local validation records |

## Cloud deployment

Deploy `app.py` from the repository root using Streamlit Community Cloud with Python 3.12. The app defaults to CSV upload when no saved dataset is present. Supply only data you are authorized to use and share; a default hosted dataset is not included. The cloud deployment itself has not yet been tested.
