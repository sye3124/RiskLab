"""Validated inputs, delayed-execution backtest, and transparent result measures.

Portfolio mathematics lives in portfolio.py. No function here fetches data.
Signal at quarter-end close; execute at the next supplied close; old holdings
earn the execution day's return, new holdings earn subsequent returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import TICKERS, EQUITIES, ALTERNATIVES, STRATEGIES, Mandate, BacktestResult
from .portfolio import covariance_estimates, portfolio_risk, weights_feasible, optimize, strategy_weights


class PriceValidationError(ValueError):
    """A rejected input with a serializable report for the data trail."""
    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trim unavailable leading history; reject incomplete or ambiguous interior rows.

    An exchange calendar would be needed to detect dates absent from every
    ticker. Never fill or silently delete an explicit missing-price date.
    """
    report = {"policy": "trim leading pre-common history; reject duplicate, invalid or incomplete interior dates",
              "no_missing_price_imputation": True, "dates_removed_for_missing_prices": 0,
              "duplicate_dates_removed": 0,
              "calendar_completeness": "Not independently verified; supplied complete dates are trading observations."}
    def reject(message):
        raise PriceValidationError(message, report)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        reject("Price data are empty. Use the bundled demo or supply an adjusted-price CSV.")
    prices = raw.copy()
    report["input_price_rows"] = len(prices)
    prices.columns = [str(c).strip().upper() for c in prices.columns]
    if prices.columns.duplicated().any():
        reject("Duplicate ticker column names are ambiguous; supply one column per ETF.")
    missing_cols = sorted(set(TICKERS) - set(prices.columns))
    if missing_cols:
        report["missing_ticker_columns"] = missing_cols
        reject(f"Missing required ticker columns: {missing_cols}")
    prices = prices.loc[:, list(TICKERS)]
    original_index = prices.index
    parsed = pd.to_datetime(original_index, errors="coerce", utc=True)
    if parsed.isna().any():
        report["invalid_date_values"] = [str(x) for x in original_index[parsed.isna()]]
        reject("Invalid dates in input; correct the dates rather than dropping rows.")
    prices.index = parsed.tz_convert(None).normalize()
    duplicates = prices.index[prices.index.duplicated(keep=False)].unique()
    if len(duplicates):
        report["duplicate_dates"] = [str(x.date()) for x in duplicates]
        reject("Duplicate dates are ambiguous: " + ", ".join(report["duplicate_dates"][:5]))
    report["input_was_sorted"] = bool(prices.index.is_monotonic_increasing)
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    prices = prices.mask((prices <= 0) | ~np.isfinite(prices))
    empty_tickers = [t for t in TICKERS if prices[t].first_valid_index() is None]
    if empty_tickers:
        report["tickers_without_valid_prices"] = empty_tickers
        reject(f"ETFs without valid positive prices: {empty_tickers}")
    first_common = max(prices[t].first_valid_index() for t in TICKERS)
    leading = prices.index[prices.index < first_common]
    report["leading_dates_trimmed"] = [str(x.date()) for x in leading]
    report["leading_rows_trimmed"] = len(leading)
    prices = prices.loc[first_common:].copy()
    incomplete = prices.isna().any(axis=1)
    if incomplete.any():
        report["incomplete_dates"] = [str(x.date()) for x in prices.index[incomplete]]
        report["invalid_prices_by_ticker"] = {
            t: [str(x.date()) for x in prices.index[prices[t].isna()]] for t in TICKERS if prices[t].isna().any()}
        reject("Missing or invalid prices inside the common sample on " + ", ".join(report["incomplete_dates"][:5])
               + ". No dates were dropped; supply corrected observations or an explicitly reviewed dataset.")
    if len(prices) < 300:
        reject("Not enough common ETF observations (minimum 300 price dates; the default backtest needs more).")
    changes = prices.pct_change(fill_method=None).iloc[1:]
    extreme = changes.abs() > .80
    if extreme.any().any():
        report["extreme_return_dates"] = [str(x.date()) for x in changes.index[extreme.any(axis=1)]]
        reject("Price change exceeds the project's 80% sanity threshold. Review adjustment and source data.")
    gaps = prices.index.to_series().diff().dt.days
    report["calendar_intervals_over_three_days"] = [
        {"previous_date": str(prices.index[i-1].date()), "date": str(prices.index[i].date()),
         "calendar_days": int(gaps.iloc[i])} for i in range(1,len(prices)) if gaps.iloc[i] > 3]
    report.update(first_common_price=str(prices.index[0].date()), last_price=str(prices.index[-1].date()),
                  usable_price_days=len(prices), ticker_columns=list(TICKERS))
    prices.attrs = dict(raw.attrs)
    return prices, report


def evaluation_table(return_series: pd.DataFrame, wealth: pd.DataFrame, trades: pd.DataFrame,
                     rf_annual: float, wealth_initial: float = 10_000_000,
                     period_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Calendar-span CAGR, 252-observation volatility/Sharpe and explicit turnover.

    Backtest supplies its initial signal close as period_start. A legacy direct
    call without it uses observation-count annualization.
    """
    if return_series.empty or wealth.empty:
        raise ValueError("Performance evaluation needs at least one observation.")
    years = ((wealth.index[-1]-pd.Timestamp(period_start)).total_seconds()/(365.25*86400)
             if period_start is not None else len(return_series)/252)
    if years <= 0:
        raise ValueError("Performance evaluation needs a positive elapsed period.")
    daily_rf = (1+rf_annual)**(1/252)-1
    rows = []
    for strategy in return_series.columns:
        r, v = return_series[strategy], wealth[strategy]
        std = r.std(ddof=1)
        peak = v.cummax().clip(lower=1.0)
        t = trades.loc[trades["strategy"].eq(strategy)]
        turnover = t.loc[t["trade_type"].eq("rebalance"),"turnover"].sum()
        initial = t.loc[t["trade_type"].eq("initial"),"turnover"].sum()
        rows.append({"strategy": strategy, "CAGR": float(v.iloc[-1]**(1/years)-1),
                     "annual_volatility": float(std*np.sqrt(252)),
                     "Sharpe": float((r.mean()-daily_rf)*np.sqrt(252)/std) if std > 0 else np.nan,
                     "max_drawdown": float((v/peak-1).min()),
                     "total_one_way_turnover": float(t["turnover"].sum()),
                     "rebalance_one_way_turnover": float(turnover),
                     "annualized_rebalance_turnover": float(turnover/years),
                     "initial_purchase_fraction": float(initial),
                     "transaction_cost_USD": float(t["cost_nav"].sum()*wealth_initial),
                     "ending_value_USD": float(v.iloc[-1]*wealth_initial)})
    return pd.DataFrame(rows).set_index("strategy").reindex(STRATEGIES)


def backtest(prices: pd.DataFrame, mandate: Mandate = Mandate()) -> BacktestResult:
    """Quarterly signals, next-observation execution, and initial zero-return cash.

    Signal t uses returns through t. At close t+1, old holdings earn that day's
    return before trading. New holdings first earn the return ending t+2.
    Planned-turnover penalties use holdings known at the signal close; actual
    executed turnover is measured after the intervening market return.
    """
    original_quality = prices.attrs.get("data_quality")
    prices, current_quality = clean_prices(prices)
    quality = original_quality or current_quality
    daily = prices.pct_change(fill_method=None).iloc[1:]
    dates = daily.index
    signal_dates = [dates[i] for i in range(len(dates)-1)
                    if dates[i].to_period("Q") != dates[i+1].to_period("Q")
                    and i+1 >= mandate.estimation_days]
    if not signal_dates:
        raise ValueError("Not enough data for the calibration window followed by a completed quarterly signal.")
    initial_signal = signal_dates[0]
    eval_days = dates[dates > initial_signal]
    if len(eval_days) < 10:
        raise ValueError("At least ten observations after the first qualifying signal are needed.")
    execution_for = {dt: dates[dates.get_loc(dt)+1] for dt in signal_dates}
    signal_for_execution = {execution: signal for signal,execution in execution_for.items()}
    signals = set(signal_dates)
    covs_by_date, targets_by_signal = {}, {}
    risk_rows, diagnostic_rows, target_rows, trade_rows = [], [], [], []

    def prepare_signal(signal, known_weights=None):
        training = daily.loc[:signal].iloc[-mandate.estimation_days:]
        covs = covariance_estimates(training, shrinkage_intensity=mandate.shrinkage_intensity)
        covs_by_date[signal] = covs
        try:
            target, diagnostics = strategy_weights(covs, mandate, previous_weights=known_weights, return_diagnostics=True)
        except Exception as exc:
            if hasattr(exc, "diagnostics"):
                exc.diagnostics = {**exc.diagnostics, "signal_date": str(signal.date()),
                                   "training_start": str(training.index[0].date()),
                                   "training_end": str(training.index[-1].date()),
                                   "training_observations": len(training)}
            raise
        targets_by_signal[signal] = target
        execution = execution_for[signal]
        for row in diagnostics:
            diagnostic_rows.append({**row, "signal_date": signal, "execution_date": execution,
                                    "training_start": training.index[0], "training_end": training.index[-1],
                                    "training_observations": len(training)})
        for name, weights in target.items():
            key = "sample" if name == "ERC (sample)" else "shrinkage"
            pct = portfolio_risk(weights, covs[key])["percentage"]
            risk_rows.append({"signal_date": signal, "execution_date": execution, "strategy": name,
                              "ERC_deviation_L2": float(np.linalg.norm(pct-1/len(TICKERS))),
                              **{f"pRC_{t}": float(pct[i]) for i,t in enumerate(TICKERS)}})
            target_rows.append({"signal_date": signal, "execution_date": execution, "strategy": name,
                                **{t: float(weights[i]) for i,t in enumerate(TICKERS)}})

    prepare_signal(initial_signal)
    weights_now = {name: np.zeros(len(TICKERS)) for name in STRATEGIES}
    nav_now = {name: 1.0 for name in STRATEGIES}
    cost_rate = mandate.transaction_cost_bps/10000
    wealth_rows, return_rows = [], []
    weight_rows = {name: [] for name in STRATEGIES}
    for dt in eval_days:
        asset_return = daily.loc[dt].to_numpy(dtype=float)
        execution_signal = signal_for_execution.get(dt)
        row_wealth, row_return = {}, {}
        for name in STRATEGIES:
            previous_nav = nav_now[name]
            held = weights_now[name]
            gross_return = float(held @ asset_return)
            if 1+gross_return <= 0:
                raise ValueError("Nonpositive portfolio wealth after an asset return.")
            nav_now[name] *= 1+gross_return
            drifted = held*(1+asset_return)/(1+gross_return)
            if execution_signal is not None:
                target = targets_by_signal[execution_signal][name]
                traded_notional = float(np.abs(target-drifted).sum())
                initial = execution_signal == initial_signal
                cost_fraction = cost_rate*traded_notional
                cost_nav = nav_now[name]*cost_fraction
                nav_before_cost = nav_now[name]
                nav_now[name] -= cost_nav
                weights_now[name] = target.copy()
                trade_rows.append({"signal_date": execution_signal, "date": dt, "strategy": name,
                                   "trade_type": "initial" if initial else "rebalance",
                                   "turnover": traded_notional if initial else traded_notional/2,
                                   "traded_notional": traded_notional, "cost_fraction": cost_fraction,
                                   "cost_nav": cost_nav, "nav_before_cost": nav_before_cost,
                                   "nav_after_cost": nav_now[name]})
            else:
                weights_now[name] = drifted
            row_wealth[name] = nav_now[name]
            row_return[name] = nav_now[name]/previous_nav-1
            weight_rows[name].append(weights_now[name].copy())
        wealth_rows.append(row_wealth)
        return_rows.append(row_return)
        if dt in signals:
            prepare_signal(dt, known_weights={name: w.copy() for name,w in weights_now.items()})

    wealth = pd.DataFrame(wealth_rows,index=eval_days).reindex(columns=STRATEGIES)
    daily_returns = pd.DataFrame(return_rows,index=eval_days).reindex(columns=STRATEGIES)
    weight_frames = {name: pd.DataFrame(rows,index=eval_days,columns=TICKERS) for name,rows in weight_rows.items()}
    trades, tracking = pd.DataFrame(trade_rows), pd.DataFrame(risk_rows)
    metrics = evaluation_table(daily_returns,wealth,trades,mandate.risk_free_annual,period_start=initial_signal)
    metrics["mean_ERC_deviation_L2"] = metrics.index.map(tracking.groupby("strategy")["ERC_deviation_L2"].mean())
    realized_rows = []
    for i,signal in enumerate(signal_dates):
        execution = execution_for[signal]
        next_execution = execution_for[signal_dates[i+1]] if i+1 < len(signal_dates) else dates[-1]
        future = daily.loc[(daily.index > execution) & (daily.index <= next_execution)]
        if len(future) < 20:
            continue
        realized_cov = np.cov(future.to_numpy(dtype=float),rowvar=False,ddof=1)
        for name,target in targets_by_signal[signal].items():
            pct = portfolio_risk(target,realized_cov)["percentage"]
            realized_rows.append({"signal_date": signal, "execution_date": execution,
                                  "evaluation_start": future.index[0], "evaluation_end": future.index[-1],
                                  "observations": len(future), "partial_period": i == len(signal_dates)-1,
                                  "strategy": name, "next_quarter_RCd_L2": float(np.linalg.norm(pct-1/len(TICKERS))),
                                  **{f"realized_pRC_{t}": float(pct[j]) for j,t in enumerate(TICKERS)}})
    realized_columns = ["signal_date","execution_date","evaluation_start","evaluation_end","observations",
                        "partial_period","strategy","next_quarter_RCd_L2",*[f"realized_pRC_{t}" for t in TICKERS]]
    realized = pd.DataFrame(realized_rows,columns=realized_columns)
    metrics["mean_next_quarter_RCd_L2"] = (metrics.index.map(realized.groupby("strategy")["next_quarter_RCd_L2"].mean())
                                            if not realized.empty else np.nan)
    return BacktestResult(daily_returns=daily_returns,wealth=wealth,weights=weight_frames,trades=trades,metrics=metrics,
                          initial_signal=initial_signal,cov_estimates=covs_by_date,risk_tracking=tracking,
                          realized_risk_tracking=realized,solver_diagnostics=pd.DataFrame(diagnostic_rows),
                          target_weights=pd.DataFrame(target_rows),data_quality=quality)
