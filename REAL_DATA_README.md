# Historical price data and provenance

The default analysis uses `data/etf_prices.csv`. Retain this exact snapshot and `data/etf_prices_source_archive/` for reproducibility. The current team snapshot contains 5,217 common observations from 2006-02-06 to 2026-10-30, downloaded via yfinance with `auto_adjust=True`. The update package does not redistribute the price data; keep your existing files locally.

## Normal use

```bash
python run.py
python run.py --app
```

The main runner downloads from Yahoo Finance only if the CSV is absent or `--refresh-data` is supplied. The fixed requested period is 2005-01-01 to the exclusive endpoint 2026-11-01. No automatic fallback exists. Explicit custom inputs and app uploads are not replaced. If a saved file is invalid, correct it or deliberately refresh it.

## Source evidence

The source archive contains the full returned provider table (CSV serialization, not original HTTP bytes), pre-cleaning closes, cleaned prices, cleaning report and source metadata. The analysis exports exact imported bytes and source attribution when the archive's cleaned bytes match. This is not independent verification of authenticity or adjustments. Keep the unverified label; investigate extreme returns and compare against an independent historical source before making empirical claims.

A CSV must contain `Date,SPY,VGK,EEM,IEF,LQD,GLD,DBC` and adjusted USD closes. Only leading pre-common-history dates may be trimmed; interior missing or invalid prices and ambiguous dates are rejected. Observed-date gaps are reported, not authenticated against an exchange calendar. Exact copies of the retired demonstration are rejected, but software cannot infer the authenticity of arbitrary uploaded CSVs.

Before publishing to GitHub or deploying the app, confirm the provider's redistribution rights. The supplied `.gitignore` excludes local datasets, provider archives and generated output directories. It does not untrack files already committed to Git. For a deployed app, arrange authorized access to the CSV or upload it in the sidebar.

The 504- and 252-day backtests begin at different dates. Compare performance over a common evaluation period when assessing window sensitivity. No historical strategy superiority is assumed.
