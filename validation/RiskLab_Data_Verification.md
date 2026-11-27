# RiskLab — historical data spot-check audit

Audit date: 25 November 2026. Dataset: `data/etf_prices.csv` in the complete local RiskLab ZIP.

## Scope and identity

- SHA-256 of exact CSV: `d0d02b74bcf8d264fbd394717d2702db998ec7388b65ffa68537cb4dd7f4af7e`.
- Source metadata: Yahoo Finance via yfinance, `auto_adjust=True`, requested 2005-01-01 to exclusive 2026-11-01.
- 5,217 rows, seven ETFs, 2006-02-06 to 2026-10-30 inclusive. No missing or nonpositive prices, duplicate dates or weekend observations.
- `exchange_calendars` 4.13.1, XNYS: 5,217 sessions in that interval, zero missing or unexpected dates.
- SHA-256 of archived provider CSV matches the source provenance: `1c62457305387677f336d322d85e8f2fe5fa88386d3b68dcdcd0c9bb4696f4d8`.

## External spot checks

Daily simple return = adjusted closing price / preceding observed adjusted close - 1. Annual return = last adjusted close of 2022 / last adjusted close of 2021 - 1. The reference is PortfoliosLab's dividend-adjusted ETF performance page (accessed 25 November 2026). It is an external publication; its upstream data source has not been independently established.

| ETF | Largest absolute daily return date | CSV daily return | Published daily return | CSV 2022 return | Published 2022 return | Reference |
|---|---|---:|---:|---:|---:|---|
| SPY | 2008-10-13 | +14.52% | +14.5% | -18.18% | -18.18% | https://portfolioslab.com/symbol/SPY |
| VGK | 2008-10-13 | +14.19% | +14.2% | -15.98% | -15.98% | https://portfolioslab.com/symbol/VGK |
| EEM | 2008-10-13 | +22.77% | +22.8% | -20.56% | -20.56% | https://portfolioslab.com/symbol/EEM |
| IEF | 2009-03-18 | +3.43% | +3.4% | -15.16% | -15.15% | https://portfolioslab.com/symbol/IEF |
| LQD | 2008-09-30 | +9.77% | +9.8% | -17.92% | -17.92% | https://portfolioslab.com/symbol/LQD |
| GLD | 2008-09-17 | +11.29% | +11.3% | -0.77% | -0.77% | https://portfolioslab.com/symbol/GLD |
| DBC | 2022-03-09 | -7.94% | -7.9% | +19.34% | +19.34% | https://portfolioslab.com/symbol/DBC |

All seven daily extremes match at the published one-decimal precision, and 2022 returns differ by at most 0.01 percentage point at displayed precision. No observations were removed or modified. This is a structural/calendar audit and selected external return corroboration, not independent authentication of every price, corporate action, or the full historical dataset. Keep the app's unverified source label. No redistribution rights are established by this audit.

## Regression review

The complete local package with the subsequently supplied UTF-8 `tests/test_pipeline.py` replacement was extracted and tested under Python 3.13.5, NumPy 2.3.5, pandas 2.2.3 and SciPy 1.17.0. `python -m unittest discover -s tests -q`: 43 tests, OK. App control flow was tested using its recording double, not a real browser. The team reported successful local Windows execution and app launch. Final browser interaction, current local exports and cloud deployment still require confirmation.
