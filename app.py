"""Offline-first RiskLab dashboard: streamlit run app.py.

Portfolio calculations live in risklab. Streamlit is imported only by main(),
so formatting and export helpers can be checked without the UI runtime.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

from risklab import (TICKERS, STRATEGIES, Mandate, backtest,
                     covariance_estimates, portfolio_risk, read_price_csv,
                     strategy_weights, weights_feasible)

from risklab.config import DATA_PATH, ESTIMATION_WINDOWS
from risklab.data import local_price_provenance, safe_provenance

DEFAULT_MANDATE = Mandate()


def csv_bytes(frame: pd.DataFrame, *, index: bool = True,
              index_label: str | None = None) -> bytes:
    """Retain nested candidate diagnostics as JSON cells in CSV exports."""
    serializable = frame.copy()
    for column in serializable.columns:
        if serializable[column].dtype == object:
            serializable[column] = serializable[column].map(
                lambda value: json.dumps(value, default=str, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value)
    return serializable.to_csv(index=index, index_label=index_label).encode("utf-8")


def format_table(frame: pd.DataFrame, formats: str | dict) -> pd.DataFrame:
    """Format display copies without requiring pandas' optional Jinja2 Styler."""
    display = frame.copy()
    if isinstance(formats, str):
        formats = {column: formats for column in frame.select_dtypes(include="number").columns}
    for column, template in formats.items():
        if column in display:
            display[column] = display[column].map(
                lambda value, pattern=template: "—" if pd.isna(value) else pattern.format(value))
    return display


def stress_window_table(daily_returns: pd.DataFrame, year: int) -> pd.DataFrame:
    """Check each requested year independently and label available observations."""
    subset = daily_returns.loc[daily_returns.index.year == year]
    if subset.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Net return over available dates": (1 + subset).prod() - 1,
        "Observations": len(subset),
        "First observation": str(subset.index.min().date()),
        "Last observation": str(subset.index.max().date()),
    })


def allocation_snapshot(result, strategy: str,
                        preview_weights: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Separate hypothetical latest targets from executed targets and holdings."""
    history = result.weights[strategy]
    held_date = pd.Timestamp(history.index[-1])
    held = history.iloc[-1].reindex(TICKERS).to_numpy(dtype=float)
    trades = result.trades.loc[result.trades["strategy"] == strategy].copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.loc[trades["date"] <= held_date].sort_values("date")
    last_trade = trades.iloc[-1] if not trades.empty else None
    last_target = np.full(len(TICKERS), np.nan)
    last_signal = last_execution = None
    if last_trade is not None:
        last_execution = pd.Timestamp(last_trade["date"])
        last_signal = pd.Timestamp(last_trade["signal_date"])
        targets = result.target_weights
        if not targets.empty:
            matching = targets.loc[
                (targets["strategy"] == strategy)
                & (pd.to_datetime(targets["signal_date"]) == last_signal)
                & (pd.to_datetime(targets["execution_date"]) == last_execution)
            ]
            if not matching.empty:
                last_target = matching.iloc[-1].reindex(TICKERS).to_numpy(dtype=float)
    table = pd.DataFrame({
        "Latest computed target (preview)": np.asarray(preview_weights),
        "Backtested holdings at final close": held,
        "Most recent executed target": last_target,
    }, index=TICKERS)
    return table, {"held_date": held_date, "last_signal": last_signal,
                   "last_execution": last_execution}


def build_run_zip(result, prices: pd.DataFrame, mandate: Mandate,
                  raw_content: bytes, provenance: dict) -> bytes:
    """Package the shared exporter output, including its HTML report."""
    from risklab.export import export_run

    with tempfile.TemporaryDirectory(prefix="risklab-export-") as temporary:
        directory = Path(temporary)
        export_run(result, prices, mandate, directory,
                   raw_content=raw_content, provenance=provenance)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(directory).as_posix())
        return output.getvalue()


def _date_text(value) -> str:
    return str(pd.Timestamp(value).date()) if value is not None else "No executed trade"


STRATEGY_COLORS = ["#64748b", "#d97706", "#7356a8", "#60a5fa", "#2563eb", "#0d9488"]
ASSET_COLORS = ["#1d4ed8", "#60a5fa", "#93c5fd", "#0f766e", "#5bb6a9", "#d97706", "#f2bf63"]
SHORT_NAMES = dict(zip(STRATEGIES, ["60/40", "Equal weight", "Min. variance", "ERC sample", "ERC shrinkage", "Constrained ERC"]))


def time_chart(st, frame, title, *, percent=False, holdings=False):
    """Explicit date axis and stable colors; stack capital weights, never risk shares."""
    names = list(TICKERS if holdings else STRATEGIES)
    labels = names if holdings else [SHORT_NAMES[n] for n in names]
    colors = ASSET_COLORS if holdings else STRATEGY_COLORS
    wide = frame.reindex(columns=names).rename(columns={} if holdings else SHORT_NAMES)
    long = wide.rename_axis("Date").reset_index().melt("Date", var_name="Series", value_name="Value")
    long["Date"] = pd.to_datetime(long["Date"]).dt.strftime("%Y-%m-%d")
    long["Order"] = long["Series"].map({name:i for i,name in enumerate(labels)})
    # Zero-based stacking retains actual weights and does not disguise cash or missing capital.
    y = {"field":"Value", "type":"quantitative", "title":title,
         "axis":{"format":".0%" if percent or holdings else ".1f"}}
    if holdings:
        y.update(stack="zero", scale={"domain":[0,1]})
    spec = {"height":300, "mark":{"type":"area" if holdings else "line", "opacity":0.9 if holdings else 1},
        "encoding":{"x":{"field":"Date", "type":"temporal", "title":None,
                           "axis":{"format":"%Y", "tickCount":8}},
                    "y":y, "color":{"field":"Series", "type":"nominal", "title":None,
                        "scale":{"domain":labels, "range":colors}, "legend":{"orient":"bottom", "columns":3}},
                    "tooltip":[{"field":"Date","type":"temporal","format":"%Y-%m-%d"},
                               {"field":"Series","type":"nominal"},
                               {"field":"Value","type":"quantitative","format":".2%" if percent or holdings else ".4f"}]}}
    if holdings:
        spec["encoding"]["order"] = {"field":"Order","type":"quantitative"}
    st.vega_lite_chart(long, spec, use_container_width=True)


def allocation_chart(st, table):
    long = table[["Capital weight", "Volatility risk share"]].rename_axis("ETF").reset_index().melt("ETF", var_name="Measure", value_name="Share")
    colors = ["#2563eb", "#0d9488"]
    fields = ["Capital weight", "Volatility risk share"]
    spec = {"height":300,"layer":[
        {"mark":{"type":"bar","cornerRadiusEnd":2},"encoding":{
            "x":{"field":"ETF","type":"nominal","sort":list(TICKERS),"title":None,"axis":{"labelAngle":0}},
            "xOffset":{"field":"Measure","sort":fields},
            "y":{"field":"Share","type":"quantitative","stack":None,"title":"Share","axis":{"format":".0%"}},
            "color":{"field":"Measure","type":"nominal","scale":{"domain":fields,"range":colors},"legend":{"title":None,"orient":"bottom"}},
            "tooltip":[{"field":"ETF"},{"field":"Measure"},{"field":"Share","type":"quantitative","format":".2%"}]}},
        {"mark":{"type":"rule","strokeDash":[5,4],"color":"#64748b"},"encoding":{"y":{"datum":1/len(TICKERS)}}}]}
    st.vega_lite_chart(long, spec, use_container_width=True)
    st.caption("Dashed line: equal-risk reference (1/7). Signed risk shares can be negative.")


def risk_summary_chart(st, metrics):
    frame = metrics[["mean_next_quarter_RCd_L2"]].rename(columns={"mean_next_quarter_RCd_L2":"Deviation"}).rename_axis("Strategy").reset_index()
    frame["Strategy"] = frame["Strategy"].map(SHORT_NAMES)
    names = list(SHORT_NAMES.values())
    st.vega_lite_chart(frame, {"height":240,"mark":"bar","encoding":{
        "y":{"field":"Strategy","type":"nominal","sort":names,"title":None},
        "x":{"field":"Deviation","type":"quantitative","title":"Mean L2 deviation (lower is closer to equal risk)"},
        "color":{"field":"Strategy","type":"nominal","scale":{"domain":names,"range":STRATEGY_COLORS},"legend":None},
        "tooltip":[{"field":"Strategy"},{"field":"Deviation","type":"quantitative","format":".4f"}]}},use_container_width=True)


def stress_chart(st, tables):
    rows = []
    for year, table in tables.items():
        for name, row in table.iterrows():
            rows.append({"Year":str(year),"Strategy":SHORT_NAMES[name],"Return":row["Net return over available dates"]})
    if not rows:
        return
    names = list(SHORT_NAMES.values())
    st.vega_lite_chart(pd.DataFrame(rows), {"height":280,"mark":"bar","encoding":{
        "x":{"field":"Year","type":"nominal","title":None},"xOffset":{"field":"Strategy","sort":names},
        "y":{"field":"Return","type":"quantitative","stack":None,"axis":{"format":".0%"},"title":"Net return"},
        "color":{"field":"Strategy","type":"nominal","scale":{"domain":names,"range":STRATEGY_COLORS},"legend":{"title":None,"orient":"bottom","columns":3}},
        "tooltip":[{"field":"Year"},{"field":"Strategy"},{"field":"Return","type":"quantitative","format":".2%"}]}},use_container_width=True)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RiskLab | QARM II", page_icon="📊", layout="wide")
    st.title("RiskLab")
    st.caption("Multi-asset portfolio management | Risk budgeting & mandate-aware allocation | QARM II")

    with st.sidebar:
        st.header("Data and settings")
        source = st.radio("Input", ("Saved historical ETF prices", "Upload a price CSV") if DATA_PATH.is_file() else ("Upload a price CSV", "Saved historical ETF prices"))
        uploaded = None
        source_note = ""
        if source == "Upload a price CSV":
            uploaded = st.file_uploader("CSV: Date plus all seven ETF columns", type=["csv"])
            source_note = st.text_input("Source / adjustment note (optional)",
                                        help="Record the source and split/dividend adjustment convention.")
        with st.form("mandate_settings"):
            st.subheader("Client limits")
            etf_cap = st.slider("Maximum ETF weight", 0.15, 0.70, DEFAULT_MANDATE.max_etf_weight, 0.01)
            equity_cap = st.slider("Maximum total equities", 0.00, 1.00, DEFAULT_MANDATE.max_equity_weight, 0.01)
            alt_cap = st.slider("Maximum gold + commodities", 0.00, 0.70, DEFAULT_MANDATE.max_alternatives_weight, 0.01)
            window = st.selectbox("Estimation window (trading days)", list(ESTIMATION_WINDOWS))
            st.subheader("Covariance and trading")
            intensity = st.slider("Fixed shrinkage intensity δ", 0.0, 1.0, DEFAULT_MANDATE.shrinkage_intensity, 0.05,
                                  help="LN3 constant-correlation target. δ is a fixed assumption, not an estimated Ledoit–Wolf intensity.")
            cost_bps = st.slider("Cost per dollar traded (basis points)", 0, 100, int(DEFAULT_MANDATE.transaction_cost_bps), 1)
            penalty = st.number_input("Planned-turnover penalty λ", min_value=0.0,
                                      max_value=10.0, value=DEFAULT_MANDATE.turnover_penalty, step=0.001, format="%.4f",
                                      help="Optional extension for constrained strategies using signal-date holdings. Objective scales differ by strategy, so λ has no universal economic unit.")
            rf = st.number_input("Constant annual risk-free assumption (%)",
                                 min_value=-5.0, max_value=15.0, value=100 * DEFAULT_MANDATE.risk_free_annual, step=0.25) / 100
            st.form_submit_button("Apply settings", use_container_width=True)
        st.caption("Quarter-end close signal → next observed close execution → new holdings earn subsequent returns.")

    mandate = Mandate(max_etf_weight=etf_cap, max_equity_weight=equity_cap,
                      max_alternatives_weight=alt_cap,
                      transaction_cost_bps=float(cost_bps), estimation_days=window,
                      risk_free_annual=rf, shrinkage_intensity=intensity,
                      turnover_penalty=penalty)

    if source == "Saved historical ETF prices":
        if not DATA_PATH.is_file():
            st.error("Historical data are missing. Run python run.py from the project folder, or upload a historical CSV using the sidebar.")
            st.stop()
        raw_content = DATA_PATH.read_bytes()
        provenance = local_price_provenance(DATA_PATH, raw_content)
        provenance["dataset_label"] = DATA_PATH.name
        st.info(f"Saved historical ETF prices | {provenance['source']} | Source not independently verified.")
    else:
        if uploaded is None:
            st.info("Upload a CSV with Date, SPY, VGK, EEM, IEF, LQD, GLD and DBC. The app reads your file locally and does not fetch market data.")
            st.stop()
        raw_content = uploaded.getvalue()
        provenance = {"kind": "user_supplied", "dataset_label": uploaded.name,
                      "source": source_note or "User-supplied CSV; provenance not independently verified"}
        st.info("USER-SUPPLIED DATA — results describe this file. Source authenticity and dividend/split adjustment have not been independently verified.")
    provenance = safe_provenance(provenance)
    prefix = "risklab_HISTORICAL"

    strategy = st.selectbox("Portfolio to inspect", STRATEGIES, index=5)
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Client mandate", "2. Portfolio optimizer",
        "3. Risk analytics", "4. Backtest & benchmarks",
    ])

    # Keep the original input accessible when validation or optimization stops.
    with tab1:
        st.subheader("Client mandate | USD 10 million family office")
        st.write("**Investment objective:** diversify portfolio risk across global equities, bonds and alternatives while respecting the client’s capital limits. The goal is balanced risk, not maximum historical return.")
        a, b, c = st.columns(3)
        a.metric("Initial capital", "$10M")
        b.metric("Investment universe", "7 ETFs")
        c.metric("Rebalancing", "Quarterly")
        a, b, c = st.columns(3)
        a.metric("Individual ETF cap", f"{mandate.max_etf_weight:.0%}")
        b.metric("Total equities cap", f"{mandate.max_equity_weight:.0%}")
        c.metric("Alternatives cap", f"{mandate.max_alternatives_weight:.0%}")
        st.caption("Fully invested and long-only at execution. Caps apply to the two constrained strategies; holdings can drift between rebalances. Other strategies are references.")
        try:
            raw_preview = pd.read_csv(io.BytesIO(raw_content))
            prices = read_price_csv(raw_content, source_metadata=provenance)
        except Exception as exc:
            st.error(f"Price-data validation stopped: {exc}")
            if hasattr(exc, "report"):
                st.json(json.dumps(exc.report, default=str))
            st.download_button("Download original input CSV", raw_content, f"{prefix}_raw_prices.csv", "text/csv", key="invalid_raw_download")
            st.stop()
        quality = prices.attrs.get("data_quality", {})
        provenance["input_sha256"] = hashlib.sha256(raw_content).hexdigest()
        st.subheader("Historical data coverage")
        a, b, c = st.columns(3)
        a.metric("First usable price", _date_text(prices.index[0]))
        b.metric("Last usable price", _date_text(prices.index[-1]))
        c.metric("Usable price observations", f"{len(prices):,}")
        classification = pd.DataFrame({"Ticker": TICKERS,
            "Asset class": ["US equity", "European equity", "Emerging equity", "US Treasuries", "Corporate bonds", "Gold", "Broad commodities"]}).set_index("Ticker")
        st.dataframe(classification, use_container_width=True)
        with st.expander("Historical prices and CSV downloads"):
            st.download_button("Download original input CSV", raw_content, f"{prefix}_raw_prices.csv", "text/csv", key="raw_download")
            left, right = st.columns(2)
            with left:
                st.markdown("**Original input: first 50 rows**")
                st.dataframe(raw_preview.head(50), use_container_width=True)
                st.caption(f"Original file: {len(raw_preview):,} rows. Download retains exact input bytes.")
            with right:
                st.markdown("**Cleaned prices: first 50 rows**")
                st.dataframe(prices.head(50), use_container_width=True)
                st.download_button("Download cleaned prices", csv_bytes(prices, index_label="Date"), f"{prefix}_cleaned_prices.csv", "text/csv", key="clean_download")
        with st.expander("Source, input fingerprint and data-quality report"):
            st.json(json.dumps({"provenance": provenance, "data_quality": quality}, default=str))
            st.caption("Validation checks structure and consistency; it does not authenticate every market price.")
        with st.expander("Method choices and course references"):
            st.markdown("**Minimum variance:** minimize portfolio variance under capital limits (LN1, slides 13–16 and 21).")
            st.markdown("**Covariance:** mix sample covariance with the constant-correlation target in LN3, slides 9–10. δ is fixed by the slider; no automatic intensity estimator is claimed.")
            st.markdown("**Unconstrained ERC:** LN5 log-constraint formulation followed by rescaling (slides 23–28). **Constrained risk budgeting:** a project approximation minimizing squared differences from equal percentage risk shares; exact equality may be infeasible.")
            st.markdown("**Optional turnover penalty:** an extension based on the full sum of planned absolute weight changes (LN3, slides 21–22). It uses signal-date holdings; drift before execution can change actual turnover.")
            st.json(asdict(mandate))

    @st.cache_data(show_spinner="Calculating quarterly allocations and backtest…")
    def cached_backtest(input_prices: pd.DataFrame, settings: Mandate,
                        input_fingerprint: str):
        # Dataframe-value hashing alone can overlook different original input
        # rows/metadata that clean to the same prices. Keep their audit separate.
        return backtest(input_prices, settings)

    try:
        results = cached_backtest(prices, mandate, provenance["input_sha256"])
        training = prices.pct_change(fill_method=None).iloc[1:].iloc[-mandate.estimation_days:]
        covs = covariance_estimates(training, shrinkage_intensity=mandate.shrinkage_intensity)
        held_mapping = {name: results.weights[name].iloc[-1].to_numpy(dtype=float)
                        for name in STRATEGIES}
        latest_targets, latest_diagnostics = strategy_weights(
            covs, mandate, previous_weights=held_mapping, return_diagnostics=True)
    except Exception as exc:
        st.error(f"The portfolio calculation stopped: {exc}")
        if hasattr(exc, "diagnostics"):
            st.json(json.dumps(exc.diagnostics, default=str))
        st.info("Original input, cleaned prices and validation remain available in Client mandate. Check the estimation window and feasibility of capital limits.")
        st.stop()

    preview = latest_targets[strategy]
    fitted_cov = covs["sample"] if strategy == "ERC (sample)" else covs["shrinkage"]
    risk = portfolio_risk(preview, fitted_cov)
    pct = np.asarray(risk["percentage"])
    composition, allocation_dates = allocation_snapshot(results, strategy, preview)
    last_price_date = pd.Timestamp(prices.index[-1])
    latest_diag_frame = pd.DataFrame(latest_diagnostics)

    with tab2:
        st.subheader("Latest target preview and historical holdings")
        st.write(f"The preview uses information through **{_date_text(last_price_date)}**. It is a newly computed target, not a recorded trade. Under the execution policy it would trade at a later observed close; that future price is not in this preview.")
        a, b, c = st.columns(3)
        a.metric("Backtested holdings as of", _date_text(allocation_dates["held_date"]))
        b.metric("Last executed signal", _date_text(allocation_dates["last_signal"]))
        c.metric("Last execution close", _date_text(allocation_dates["last_execution"]))
        st.dataframe(format_table(composition, "{:.2%}"), use_container_width=True)
        st.download_button("Download target / holdings comparison", csv_bytes(composition, index_label="ETF"),
                           f"{prefix}_allocation_comparison.csv", "text/csv")
        target_compliant = weights_feasible(preview, mandate)
        held_compliant = weights_feasible(held_mapping[strategy], mandate)
        st.write(f"Preview capital-limit compliance: **{'within limits' if target_compliant else 'outside client limits'}**. Final backtested holdings: **{'within limits' if held_compliant else 'outside client limits'}**.")
        if not target_compliant:
            st.warning("This reference strategy does not meet every client cap. The constrained strategies are the mandate-compliant target comparisons.")
        if not held_compliant and "constrained" in strategy:
            st.info("Recorded holdings have drifted outside a capital limit since execution. The model does not rebalance continuously.")
        st.subheader("Risk of the latest computed target")
        risk_table = pd.DataFrame({"Capital weight": preview,
                                   "Volatility risk share": pct,
                                   "Equal-risk reference": np.full(len(TICKERS), 1 / len(TICKERS))}, index=TICKERS)
        allocation_chart(st, risk_table)
        st.dataframe(format_table(risk_table, "{:.2%}"), use_container_width=True)
        max_deviation = float(np.max(np.abs(pct - 1 / len(TICKERS))))
        if "ERC" in strategy and max_deviation > 1e-4:
            st.warning(f"Equal risk is not reached: the largest deviation is {100 * max_deviation:.3f} percentage points. Treat this allocation as approximate risk-budget matching.")
        a, b, c = st.columns(3)
        a.metric("Estimated annual volatility", f"{float(risk['volatility']) * np.sqrt(252):.2%}")
        b.metric("Largest risk-share deviation", f"{100 * max_deviation:.3f} pp")
        c.metric("Sum of risk shares", f"{pct.sum():.4%}")
        st.caption("Percentage volatility contribution = wᵢ(Σw)ᵢ / (wᵀΣw), from LN5. Equal risk is a reference for the benchmarks, not their objective.")
        with st.expander("Technical details: latest solver diagnostics"):
            st.dataframe(latest_diag_frame.drop(columns=["candidates"], errors="ignore"), use_container_width=True)
            st.download_button("Download latest solver diagnostics", csv_bytes(latest_diag_frame, index=False), f"{prefix}_latest_solver_diagnostics.csv", "text/csv")
            st.json(json.dumps(latest_diagnostics, default=str))

    with tab3:
        st.subheader("Risk estimation and historical diversification")
        st.write(f"Shrinkage uses the LN3 constant-correlation target with **fixed δ = {mandate.shrinkage_intensity:.2f}**. Compare sample versus shrinkage covariance, then the effect of capital limits.")
        st.subheader("How closely did the portfolios track equal risk?")
        risk_summary_chart(st, results.metrics)
        base = results.metrics.loc[STRATEGIES[0], "mean_next_quarter_RCd_L2"]
        proposed = results.metrics.loc[STRATEGIES[-1], "mean_next_quarter_RCd_L2"]
        if pd.notna(base) and pd.notna(proposed) and base > 0:
            st.metric("Constrained ERC: relative deviation reduction vs 60/40", f"{1-proposed/base:.1%}")
        st.caption("Mean of available subsequent-period fixed-target risk attribution, including a partial final period where present. Lower deviation indicates closer equal-risk tracking, not higher return.")
        risk_rows = []
        for name in STRATEGIES:
            local_cov = covs["sample"] if name == "ERC (sample)" else covs["shrinkage"]
            local_risk = portfolio_risk(latest_targets[name], local_cov)
            risk_rows.append({"Strategy": name,
                              "Annual volatility": float(local_risk["volatility"]) * np.sqrt(252),
                              "Largest signed risk share": float(np.max(local_risk["percentage"])),
                              "Equal-risk deviation L2": float(np.linalg.norm(np.asarray(local_risk["percentage"]) - 1 / len(TICKERS)))})
        st.dataframe(format_table(pd.DataFrame(risk_rows).set_index("Strategy"), {
            "Annual volatility": "{:.2%}", "Largest signed risk share": "{:.2%}",
            "Equal-risk deviation L2": "{:.5f}"}), use_container_width=True)
        st.caption("Latest-target measures use each strategy's fitted covariance. Signed risk contributions may be negative; the largest contribution is not a downside-loss measure.")
        if not results.risk_tracking.empty:
            st.subheader("Risk-budget fit at historical signals")
            time_chart(st, results.risk_tracking.pivot(index="signal_date", columns="strategy", values="ERC_deviation_L2"), "L2 deviation")
            st.caption("Good fit to the estimation matrix is not evidence of better subsequent diversification.")
        if not results.realized_risk_tracking.empty:
            st.subheader("Subsequent holding-period risk: evaluation only")
            time_chart(st, results.realized_risk_tracking.pivot(index="signal_date", columns="strategy", values="next_quarter_RCd_L2"), "L2 deviation")
            with st.expander("Subsequent-period risk attribution table"):
                st.dataframe(results.realized_risk_tracking, use_container_width=True)
            st.caption("Uses post-execution returns unavailable at the signal. Targets stay fixed for this diagnostic while actual holdings drift. The table identifies incomplete final periods; lower error means closer equal risk, not necessarily higher returns.")
        else:
            st.info("No subsequent holding period has enough observations for realized risk attribution.")
        st.subheader("Observed drawdowns")
        time_chart(st, results.wealth.div(results.wealth.cummax().clip(lower=1.0)).sub(1), "Drawdown", percent=True)
        st.subheader("2020 and 2022: separate available-date comparisons")
        tables = {year: stress_window_table(results.daily_returns, year) for year in (2020, 2022)}
        stress_chart(st, tables)
        for year, stress in tables.items():
            st.markdown(f"**{year}**")
            if stress.empty:
                st.info(f"No post-warm-up return observations in {year} for this run.")
            else:
                st.dataframe(format_table(stress, {"Net return over available dates": "{:.2%}"}), use_container_width=True)
        st.caption("Each year is checked independently and covers only dates shown. The 2020 calendar-year return is not the return during the COVID crash alone.")

    with tab4:
        st.subheader("Backtest, benchmarks and complete outputs")
        st.write(f"First signal: **{_date_text(results.initial_signal)}**. Evaluation observations: **{_date_text(results.daily_returns.index[0])}** to **{_date_text(results.daily_returns.index[-1])}**.")
        st.info(results.execution_policy)
        st.caption("The portfolio stays in cash until its initial execution close. The initial purchase cost is charged then; new asset holdings earn returns only afterwards.")
        st.subheader(f"Historical performance | {SHORT_NAMES[strategy]}")
        selected = results.metrics.loc[strategy]
        a, b, c = st.columns(3)
        a.metric("Final portfolio value", f"${selected['ending_value_USD']/1e6:.2f}M")
        b.metric("Annualized return (CAGR)", f"{selected['CAGR']:.2%}")
        c.metric("Annualized volatility", f"{selected['annual_volatility']:.2%}")
        a, b, c = st.columns(3)
        a.metric("Maximum drawdown", f"{selected['max_drawdown']:.2%}")
        b.metric("Sharpe ratio", f"{selected['Sharpe']:.3f}")
        c.metric("Annualized rebalancing turnover", f"{selected['annualized_rebalance_turnover']:.2%}")
        st.caption("Historical simulation net of modeled trading costs; not a forecast. Initial capital: $10M.")
        time_chart(st, results.wealth, "Net wealth index (initial capital = 1)")
        formats = {"CAGR": "{:.2%}", "annual_volatility": "{:.2%}", "Sharpe": "{:.3f}",
                   "max_drawdown": "{:.2%}", "total_one_way_turnover": "{:.3f}",
                   "annualized_rebalance_turnover": "{:.3f}",
                   "transaction_cost_USD": "${:,.0f}", "ending_value_USD": "${:,.0f}",
                   "mean_ERC_deviation_L2": "{:.5f}", "mean_next_quarter_RCd_L2": "{:.5f}"}
        summary_cols = ["CAGR", "annual_volatility", "Sharpe", "max_drawdown", "ending_value_USD"]
        summary = format_table(results.metrics[summary_cols], formats).rename(columns={"annual_volatility":"Volatility", "max_drawdown":"Max drawdown", "ending_value_USD":"Ending value"})
        st.dataframe(summary, use_container_width=True)
        with st.expander("Full performance and trading metrics"):
            st.dataframe(format_table(results.metrics, formats), use_container_width=True)
        st.caption("Costs cover purchases and sales. Ordinary one-way turnover is half the full traded-weight sum in LN3; the initial purchase is identified separately. The risk-free input is a constant assumption, not an observed Treasury series.")
        st.subheader("Backtested holdings after each observed close")
        time_chart(st, results.weights[strategy], "Capital allocation", holdings=True)
        st.caption("Stacked actual holdings after each observed close. Any unallocated space represents capital not assigned to these ETFs; weights are not artificially normalized.")
        downloads = [
            ("Metrics", results.metrics, True, "metrics"),
            ("Wealth index", results.wealth, True, "wealth_index"),
            ("Daily net returns", results.daily_returns, True, "daily_returns"),
            ("Selected holdings", results.weights[strategy], True, "selected_holdings"),
            ("Trade log", results.trades, False, "trades"),
            ("Executed target history", results.target_weights, False, "target_weights"),
            ("Ex-ante risk tracking", results.risk_tracking, False, "risk_tracking"),
            ("Subsequent-period risk", results.realized_risk_tracking, False, "realized_risk"),
            ("Historical solver diagnostics", results.solver_diagnostics, False, "solver_diagnostics"),
        ]
        with st.expander("Download individual tables"):
            for i in range(0, len(downloads), 3):
                columns = st.columns(3)
                for column, (label, frame, include_index, filename) in zip(columns, downloads[i:i+3]):
                    with column:
                        st.download_button(label, csv_bytes(frame, index=include_index), f"{prefix}_{filename}.csv", "text/csv", key=f"export_{filename}", use_container_width=True)
        with st.expander("Historical solver diagnostics"):
            st.dataframe(results.solver_diagnostics.drop(columns=["candidates"], errors="ignore"), use_container_width=True)
        st.subheader("Complete reproducible run")
        st.write("The package contains original and cleaned data, all strategy histories, diagnostics, settings, source notes and a readable report.")
        run_key = hashlib.sha256(raw_content + json.dumps({"mandate": asdict(mandate), "provenance": provenance}, sort_keys=True, default=str).encode()).hexdigest()
        if st.button("Prepare complete run ZIP", use_container_width=True):
            try:
                with st.spinner("Preparing the complete run…"):
                    archive = build_run_zip(results, prices, mandate, raw_content, provenance)
                st.session_state["risklab_export"] = {"run_key": run_key, "content": archive}
            except Exception as exc:
                st.error(f"The complete export could not be prepared: {exc}")
        prepared = st.session_state.get("risklab_export", {})
        if prepared.get("run_key") == run_key:
            st.download_button("Download complete run ZIP", prepared["content"],
                               f"{prefix}_complete_run.zip", "application/zip", key="complete_run")

    st.caption("QARM II basis: supplied LN1, LN3 and LN5 materials. Client limits, fixed δ, costs and constrained approximations are explicit project choices. This app does not retrieve external data.")


if __name__ == "__main__":
    main()
