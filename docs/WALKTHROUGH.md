# Understand one complete run

## 1. Open the result before reading the code

Open `outputs_real_504/report.html`. It includes the input prices, six wealth curves, drawdowns, statistics, risk diagnostics, trade dates, solver records and links to the underlying files. The report uses the archived historical input; source authenticity remains independently unverified.

The 252-day report is available in `outputs_real_252/report.html`. Compare both over common dates for robustness.

## 2. Follow the input

`run.py` selects `data/etf_prices.csv` when no `--input` is provided, downloading only if missing or explicitly refreshed. Defaults are in `risklab/config.py`.

`risklab/data.py` reads the exact bytes, computes their SHA-256 fingerprint and calls `clean_prices` in `risklab/core.py`. It stores the untouched input and keeps the original cleaning report. This is why you can inspect both `raw_input.csv` and `cleaned_prices.csv` after a run.

The cleaner may trim early dates before all ETFs have data. It will not remove an incomplete date from the middle of the history and silently turn the next return into a multi-day observation. An explicit failure includes affected dates.

## 3. Look at one signal

Open `target_weights.csv`. Choose a row for `ERC (shrinkage, constrained)` and note its signal and execution dates.

The signal uses only the previous 504 return observations by default. The source matrix is in `covariance_estimates.csv`: you can see the sample covariance, constant-correlation target and their fixed 50/50 blend. That blend is a deliberate configurable assumption, not a fitted optimal coefficient.

The optimizer then chooses capital weights. Its goal is to bring the percentage risk shares close to 1/7 while respecting the caps. The percentages in `risk_tracking.csv` show whether it succeeded and how much mismatch remains. A 20% capital weight is not necessarily a 20% risk share.

Read the matching row in `solver_diagnostics.json`. It records the optimizer's status and every attempted candidate. A constrained result can be numerically converged and feasible while still having a nonzero risk-budget error. That is visible rather than hidden.

## 4. Follow the execution and first return

The selected allocation waits until the next observed close. At that close the old holdings earn their final return interval, then the target is installed and costs are deducted.

For the initial investment, the portfolio is still in cash before execution. Under the default cost assumption:

```text
Initial portfolio value:                  USD 10,000,000
Initial purchase fraction:                100%
Cost rate:                                10 bps = 0.001
Initial cost in this simplified model:    USD 10,000
Wealth index immediately after purchase:  0.999
```

The execution day's initial net return is the purchase cost only. The next return interval is the first one earned by the new ETF holdings. This fixes the previous version's initial-entry timing inconsistency.

After a market move, holdings drift. Find the same date in a `weights_*.csv` file: those are end-of-day holdings after any scheduled execution, not a repeated constant target. At the next rebalance the code compares the new target with these drifted positions to calculate traded notional and costs.

## 5. Read the performance table with its definitions

`metrics.csv` reports the actual input kind in its first data column. Historical CSVs use `user_supplied`, which means the code has not authenticated the source.

- CAGR describes growth over the declared calendar span, including costs.
- Volatility and Sharpe summarize the net observed return series under the stated 252-observation convention.
- Drawdown includes losses from prior NAV peaks, starting from initial wealth one.
- Turnover distinguishes the initial purchase from subsequent rebalancing.
- Risk-budget errors describe risk matching, not a guarantee of higher returns.

You can reconstruct every wealth value from the cumulative product of `1 + daily_returns`. The regression suite checks that identity and independently checks the execution/cost arithmetic.

## 6. Change one thing at a time

For a one-year calibration instead of two:

```bash
python run.py --window 252 --output outputs_window_252
```

For a different fixed shrinkage mix:

```bash
python run.py --window 504 --shrinkage-intensity 0.25 --output outputs_delta_025
```

For the optional planned-turnover preference:

```bash
python run.py --window 504 --turnover-penalty 0.05 --output outputs_penalty_005
```

These commands reuse the saved historical input. They demonstrate parameter effects without establishing which settings work best on real markets. Input hashes should match when comparing settings on the same file. Different calibration windows also change the first eligible signal and evaluation start, so compare their date coverage before comparing aggregate metrics.

## 7. Use the app to explain the same run

In **Client mandate**, inspect source data and settings. In **Portfolio optimizer**, compare the newly computed preview with the last executed target and final held positions; those are deliberately separate. In **Risk analytics**, compare fitted risk shares with subsequent fixed-target diagnostics. In **Backtest & benchmarks**, inspect the full path and export all tables.

The latest preview uses the final available observation and is not a trade already recorded in the historical simulation. Its hypothetical execution would happen at a later observed close. The UI does not invent that future price.

## 8. What changes when a different historical file is supplied?

The models and accounting stay the same; the input changes. Use:

```bash
python run.py --input data/etf_prices.csv --window 504 --output outputs_my_data
```

The exact selected CSV is preserved, and the results refer to that file. A missing or invalid file stops the run. No replacement data are substituted. If you reuse an output folder, earlier generated outputs are preserved separately and the current status makes success/failure explicit.

Authenticity, dividend/split adjustment, coverage and the research interpretation still need review. The package makes these checks possible; it cannot establish facts absent from the supplied observations.
