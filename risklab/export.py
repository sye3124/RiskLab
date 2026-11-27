"""Self-contained run exports: inputs, diagnostics, numerical results and HTML."""
from __future__ import annotations
import hashlib
import html
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from .config import Mandate
from .data import read_price_csv, _write_failure, reject_known_demo, local_price_provenance, safe_provenance

COLORS = ('#155e75', '#db6b34', '#5662b3', '#32936f', '#9a4c89', '#1f2937', '#8a6d3b')
SLUGS = {
    '60/40 SPY/IEF': 'sixty_forty', 'Equal weight': 'equal_weight',
    'Minimum variance (constrained)': 'minimum_variance', 'ERC (sample)': 'erc_sample',
    'ERC (shrinkage)': 'erc_shrinkage', 'ERC (shrinkage, constrained)': 'erc_constrained',
}
GENERATED_FILES = {
    'raw_input.csv', 'input_prices_before_cleaning.csv', 'cleaned_prices.csv',
    'cleaning_report.json', 'daily_returns.csv', 'wealth_index.csv', 'trades.csv',
    'metrics.csv', 'risk_tracking.csv', 'next_quarter_realized_risk.csv',
    'solver_diagnostics.csv', 'solver_diagnostics.json', 'target_weights.csv',
    'calendar_year_returns.csv', 'covariance_estimates.csv', 'input_prices.svg',
    'wealth_index.svg', 'drawdowns.svg', 'source_provenance.json', 'run_metadata.json',
    'RUN_CONTEXT.txt', 'report.html', 'run_status.json', 'failure_diagnostics.json',
} | {f'weights_{slug}.csv' for slug in SLUGS.values()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _begin_run(folder, source, raw_content=None):
    """Move only prior RiskLab-produced files aside, preserving unrelated files."""
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    known_run = False
    for filename in ('run_status.json', 'run_metadata.json'):
        marker = folder / filename
        if marker.is_file():
            try:
                info = json.loads(marker.read_text())
                known_run |= (info.get('application') == 'RiskLab' or
                              (info.get('schema_version') == 1 and 'source_code_sha256' in info))
            except (ValueError, OSError):
                pass
    existing = [folder/name for name in sorted(GENERATED_FILES)
                if (folder/name).exists() or (folder/name).is_symlink()]
    if existing and not known_run:
        raise ValueError('Output folder contains reserved filenames without a RiskLab run marker. Choose a new output folder to preserve existing files.')
    if any(path.is_dir() and not path.is_symlink() for path in existing):
        raise ValueError('A reserved output filename is an existing directory. Choose a new output folder.')
    if existing:
        previous = folder / 'previous_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        previous.mkdir(parents=True)
        for path in existing:
            if path.is_file() or path.is_symlink():
                path.rename(previous/path.name)
    _write_json(folder/'run_status.json', {'application': 'RiskLab', 'status': 'running', 'source': source})
    if raw_content is not None:
        (folder/'raw_input.csv').write_bytes(raw_content)
    return folder


def _failed_run(folder, exc, source):
    """A failed attempt must not expose an older success as the current report."""
    folder = Path(folder)
    _write_failure(folder, exc, source)
    _write_json(folder/'run_status.json', {
        'application': 'RiskLab', 'status': 'failed', 'kind': source.get('kind'),
        'message': str(exc), 'synthetic_fallback_used': False,
    })
    message = html.escape(str(exc))
    (folder/'report.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>RiskLab — run failed</title><body style="font:16px/1.6 Arial,sans-serif;max-width:780px;margin:60px auto;padding:20px">'
        '<h1>RiskLab run failed</h1><p>No results from this attempt are available. No replacement dataset was substituted.</p>'
        f'<pre style="white-space:pre-wrap;background:#fff1e8;padding:20px">{message}</pre>'
        '<p><a href="failure_diagnostics.json">Failure diagnostics</a> · <a href="run_status.json">Run status</a></p>'
        '<p>Any earlier RiskLab output files were preserved in the previous_runs subfolder. They are not results for this input.</p></body></html>',
        encoding='utf-8',
    )


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _runtime_versions():
    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for package in ('numpy', 'pandas', 'scipy', 'streamlit', 'yfinance'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [root / 'run.py', root / 'app.py']
    files += list((root / 'risklab').glob('*.py')) + list((root / 'scripts').glob('*.py'))
    files += list(root.glob('requirements*.txt'))
    return {str(path.relative_to(root)): _sha(path.read_bytes())
            for path in sorted(set(files)) if path.is_file()}


def period_return_table(daily_returns):
    """Each observed calendar year is independent; partial years are identified."""
    rows = []
    for year in sorted(set(daily_returns.index.year)):
        segment = daily_returns.loc[daily_returns.index.year == year]
        for strategy, value in ((1 + segment).prod() - 1).items():
            rows.append({
                'year': int(year), 'strategy': strategy, 'net_return': float(value),
                'observations': len(segment), 'first_return_date': segment.index.min(),
                'last_return_date': segment.index.max(),
                'period_label': 'Observed portion of calendar year; check first/last dates',
            })
    return pd.DataFrame(rows)


def line_chart_svg(frame, title, *, percent=False):
    """Small static SVG with exact dates and labeled axes; no external resources."""
    data = frame.select_dtypes(include='number').replace([np.inf, -np.inf], np.nan)
    if data.empty or not np.isfinite(data.to_numpy()).any():
        return '<p class="muted">No observations are available for this chart.</p>'
    width, height = 1060, 390
    left, right, top, bottom = 75, 25, 40, 55
    chart_w, chart_h = width-left-right, height-top-bottom
    values = data.to_numpy(dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(high-low, abs(high)*0.04, 0.01)
    low, high = low-span*0.06, high+span*0.06
    time = pd.DatetimeIndex(data.index).asi8.astype(float)
    time_span = max(time[-1]-time[0], 1.0)
    x = left + chart_w*(time-time[0])/time_span
    y = lambda value: top + chart_h*(high-value)/(high-low)
    safe_title = html.escape(title)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{safe_title}">',
           f'<title>{safe_title}</title>', '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="22" font-family="Arial,sans-serif" font-size="15" font-weight="bold" fill="#182c42">{safe_title}</text>']
    for value in np.linspace(low, high, 5):
        label = f'{value:.1%}' if percent else f'{value:,.2f}'
        yy = y(value)
        svg += [f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e5eaf0"/>',
                f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#506377">{label}</text>']
    for j in np.unique(np.linspace(0, len(data)-1, min(6, len(data))).astype(int)):
        label = pd.Timestamp(data.index[j]).strftime('%Y-%m-%d')
        svg.append(f'<text x="{x[j]:.2f}" y="{height-22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#506377">{label}</text>')
    for i, col in enumerate(data.columns):
        # Retain every observation, including short-lived peaks and drawdowns.
        indices = np.arange(len(data))
        points = ' '.join(f'{x[j]:.2f},{y(values[j,i]):.2f}' for j in indices if np.isfinite(values[j,i]))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[i % len(COLORS)]}" stroke-width="1.8"><title>{html.escape(str(col))}</title></polyline>')
    svg.append('</svg>')
    return ''.join(svg)


def _legend(frame):
    return '<div class="legend">' + ''.join(
        f'<span><i style="background:{COLORS[i % len(COLORS)]}"></i>{html.escape(str(column))}</span>'
        for i, column in enumerate(frame.columns)) + '</div>'


def _table(frame, *, limit=None, index=False):
    if frame is None or frame.empty:
        return '<p class="muted">No rows were produced for this diagnostic.</p>'
    view = frame.head(limit) if limit is not None else frame
    return '<div class="table-wrap">' + view.to_html(
        index=index, border=0, escape=True, float_format=lambda n: f'{n:,.6g}',
        classes='data-table', na_rep='—') + '</div>'


def _metrics_table(metrics):
    view = metrics.copy()
    percent_columns = {'CAGR', 'annual_volatility', 'max_drawdown'}
    for column in view:
        if column in percent_columns:
            view[column] = view[column].map(lambda v: f'{v:.2%}' if pd.notna(v) else '—')
        elif pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda v: f'{v:,.3f}' if pd.notna(v) else '—')
    return _table(view, index=True)


def _report_html(result, prices, mandate, metadata, raw_content, annual, artifacts):
    label = 'USER-SUPPLIED DATA — UNVERIFIED'
    explanation = 'Results use the exact historical-price CSV archived with this run. Structural validation does not authenticate prices or their dividend/split adjustment convention.'
    start, end = prices.index.min().date(), prices.index.max().date()
    wealth = result.wealth
    dd = wealth.div(wealth.cummax().clip(lower=1.0)).sub(1)
    normalized = prices.div(prices.iloc[0]).mul(100)
    raw_preview = html.escape('\n'.join(raw_content.decode('utf-8', errors='replace').splitlines()[:12]))
    quality = html.escape(json.dumps(_jsonable(metadata['original_data_quality']), indent=2))
    provenance = html.escape(json.dumps(_jsonable(metadata), indent=2))
    links = ''.join(f'<li><a href="{html.escape(name, quote=True)}">{html.escape(name)}</a></li>' for name in artifacts)
    yearly = annual.copy()
    if not yearly.empty:
        yearly['net_return'] = yearly['net_return'].map(lambda v: f'{v:.2%}')
    year_notes = []
    for year in (2020, 2022):
        subset = annual.loc[annual['year'] == year] if not annual.empty else annual
        year_notes.append(f'<p><b>{year}:</b> ' + (
            'included in the table; its first and last return dates show the evaluated coverage.'
            if not subset.empty else 'no out-of-sample returns are available in this input/run.') + '</p>')
    solver = getattr(result, 'solver_diagnostics', pd.DataFrame())
    targets = getattr(result, 'target_weights', pd.DataFrame())
    target_tail = targets.tail(12) if not targets.empty else targets
    risk = result.risk_tracking
    realised = result.realized_risk_tracking
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RiskLab — {label}</title>
<style>
:root{{--ink:#182c42;--muted:#526779;--accent:#155e75;--line:#dfe6ec}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f5;color:var(--ink);font:15px/1.55 Arial,Helvetica,sans-serif}}
a{{color:var(--accent)}}header,main,footer{{max-width:1260px;margin:auto}}header{{padding:48px 28px 24px}}
.eyebrow{{letter-spacing:.13em;text-transform:uppercase;font-size:12px;font-weight:bold;color:var(--accent)}}h1{{font-size:40px;line-height:1.15;margin:8px 0 16px}}h2{{font-size:24px;margin:0 0 18px}}h3{{font-size:18px;margin-top:28px}}
.status{{border-left:5px solid #bd731d;background:#fff5df;padding:18px 22px;border-radius:6px}}.status strong{{display:block;letter-spacing:.05em;margin-bottom:5px}}
nav{{display:flex;flex-wrap:wrap;gap:18px;margin-top:24px}}nav a{{font-weight:bold;text-decoration:none}}main{{padding:0 28px}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:28px;margin:20px 0;scroll-margin-top:20px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}}.card{{background:#f1f6f8;padding:16px;border-radius:7px}}.card span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}}.card b{{display:block;font-size:19px;margin-top:5px}}
.muted{{color:var(--muted)}}svg{{width:100%;height:auto;display:block}}.legend{{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:12px;margin:8px 0 24px}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i{{display:inline-block;width:18px;height:3px}}
.table-wrap{{overflow-x:auto;max-height:680px;border:1px solid var(--line);border-radius:6px;margin:12px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;text-align:right}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}}thead{{background:#eaf1f5;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}tbody tr:nth-child(even){{background:#f8fafc}}pre{{overflow:auto;background:#f4f6f8;padding:18px;border-radius:6px;font-size:12px}}summary{{cursor:pointer;font-weight:bold;padding:12px 0}}.files{{columns:2;column-gap:32px}}footer{{padding:30px 28px;color:var(--muted);font-size:12px}}code{{background:#edf2f6;padding:2px 5px;border-radius:3px}}
@media(max-width:750px){{header{{padding:26px 16px 12px}}main{{padding:0 12px}}h1{{font-size:32px}}section{{padding:18px}}.cards{{grid-template-columns:1fr}}.files{{columns:1}}}}
@media print{{body{{background:white}}header,main,footer{{max-width:none}}section{{break-inside:avoid;border:0;padding:10px 0}}.table-wrap{{max-height:none}}details:not([open]){{display:none}}nav{{display:none}}}}
</style></head><body>
<header><div class="eyebrow">RiskLab / QARM II</div><h1>Inputs, portfolios and results</h1>
<div class="status"><strong>{label}</strong>{explanation}</div>
<nav><a href="#inputs">Input data</a><a href="#performance">Performance</a><a href="#risk">Risk diagnostics</a><a href="#execution">Execution &amp; solvers</a><a href="#provenance">Provenance &amp; files</a></nav></header>
<main><section id="inputs"><h2>1. Inspect the input</h2>
<div class="cards"><div class="card"><span>Cleaned price window</span><b>{start} — {end}</b></div><div class="card"><span>Usable price observations</span><b>{len(prices):,} × {len(prices.columns)} ETFs</b></div><div class="card"><span>Calibration window</span><b>{mandate.estimation_days} returns</b></div></div>
<p>Every input byte is preserved in <a href="raw_input.csv">raw_input.csv</a>. The analysis uses <a href="cleaned_prices.csv">cleaned_prices.csv</a>. Prices shown below are normalized only for this chart; the CSV retains their input units.</p>
{line_chart_svg(normalized, 'Input prices — first common observation = 100')}{_legend(prices)}
<h3>Cleaned price preview</h3>{_table(pd.concat([prices.head(5), prices.tail(5)]), index=True)}
<details><summary>View every cleaned input observation</summary>{_table(prices, index=True)}</details>
<details><summary>View original input bytes (first 12 lines)</summary><pre>{raw_preview}</pre></details>
<details><summary>View original cleaning and validation report</summary><pre>{quality}</pre></details>
</section>
<section id="performance"><h2>2. Performance of the evaluated strategies</h2><p><b>{label}.</b> The wealth index begins from one unit of capital; costs are included. Calendar-year tables show only observed out-of-sample return dates.</p>
{line_chart_svg(wealth, 'Net wealth index')}{_legend(wealth)}{_metrics_table(result.metrics)}
<p class="muted">Sharpe uses the configured constant annual risk-free assumption ({mandate.risk_free_annual:.2%}), not an observed historical rate series. Annualization and timing conventions are recorded in the run metadata and project methodology.</p>
{line_chart_svg(dd, 'Net drawdown from prior wealth peak', percent=True)}{_legend(dd)}
<h3>Calendar-year returns</h3>{''.join(year_notes)}{_table(yearly)}
</section>
<section id="risk"><h2>3. Risk-budget diagnostics</h2><p>Ex ante rows use the covariance available at each signal. Next-period rows are evaluation diagnostics using later observations. Fixed-target attribution and drifting holdings are distinct: the actual portfolio follows the weight histories exported with this run.</p>
<h3>Ex ante contribution tracking — latest rows</h3>{_table(risk.tail(18))}
<h3>Next-period risk contribution diagnostics — latest rows</h3>{_table(realised.tail(18))}
<p>All rows are retained in <a href="risk_tracking.csv">risk_tracking.csv</a> and <a href="next_quarter_realized_risk.csv">next_quarter_realized_risk.csv</a>. Equal risk may be unattainable under the mandate; report the residuals.</p>
</section>
<section id="execution"><h2>4. Execution and solver evidence</h2><p>{html.escape(result.execution_policy)}</p>
<p>Signal dates and execution dates are exported separately. Holdings remain unchanged while the signal waits for execution. Trade costs are charged on both purchases and sales; initial investment is recorded separately.</p>
<h3>Latest scheduled targets</h3>{_table(target_tail)}<h3>Latest actual trades</h3>{_table(result.trades.tail(18))}
<h3>Solver diagnostics — latest rows</h3>{_table(solver.tail(18))}
<p>Complete targets, trades and solver diagnostics are linked below. A successful run does not establish global optimality or empirical superiority.</p></section>
<section id="provenance"><h2>5. Provenance and reproducibility</h2><p>Input bytes, cleaning evidence, configuration, source-code fingerprints and runtime versions travel with this run. Source authenticity remains independently unverified.</p>
<p><b>Raw input SHA-256:</b> <code>{metadata['raw_input_sha256']}</code></p>
<p><b>Cleaned input SHA-256:</b> <code>{metadata['cleaned_prices_sha256']}</code></p>
<ul class="files">{links}</ul><details><summary>View full run metadata</summary><pre>{provenance}</pre></details></section></main>
<footer>Generated {html.escape(metadata['created_utc'])}. This report is self-contained and uses no remote scripts, fonts, data feeds or chart services. CSV and SVG links refer to sibling files in the exported run folder.</footer></body></html>'''


def export_run(result, prices, mandate, output_dir, raw_content: bytes, provenance: dict, *, _prepared=False) -> Path:
    """Export an already completed run and return its self-contained report path."""
    if not isinstance(raw_content, bytes):
        raise TypeError('raw_content must contain the exact original input bytes.')
    provenance = safe_provenance(provenance)
    kind = provenance.get('kind', prices.attrs.get('source_metadata', {}).get('kind', 'user_supplied'))
    reject_known_demo(raw_content)
    if kind != 'user_supplied':
        raise ValueError('Only user-supplied historical price inputs are supported.')
    folder = Path(output_dir).expanduser().resolve() if _prepared else _begin_run(
        output_dir, {**provenance, 'kind': kind}, raw_content)
    artifacts = []
    def csv(name, frame, *, index=False, index_label=None):
        frame.to_csv(folder / name, index=index, index_label=index_label)
        artifacts.append(name)
    (folder / 'raw_input.csv').write_bytes(raw_content)
    artifacts.append('raw_input.csv')
    before_cleaning = prices.attrs.get('analysis_input_content')
    if before_cleaning is not None:
        (folder / 'input_prices_before_cleaning.csv').write_bytes(before_cleaning)
        artifacts.append('input_prices_before_cleaning.csv')
    csv('cleaned_prices.csv', prices, index=True, index_label='Date')
    quality = prices.attrs.get('data_quality', getattr(result, 'data_quality', {}))
    _write_json(folder / 'cleaning_report.json', quality)
    artifacts.append('cleaning_report.json')
    csv('daily_returns.csv', result.daily_returns, index=True, index_label='Date')
    csv('wealth_index.csv', result.wealth, index=True, index_label='Date')
    csv('trades.csv', result.trades)
    metrics = result.metrics.copy()
    metrics.insert(0, 'dataset_kind', kind)
    csv('metrics.csv', metrics, index=True, index_label='strategy')
    csv('risk_tracking.csv', result.risk_tracking)
    csv('next_quarter_realized_risk.csv', result.realized_risk_tracking)
    csv('solver_diagnostics.csv', result.solver_diagnostics)
    _write_json(folder / 'solver_diagnostics.json', result.solver_diagnostics.to_dict(orient='records'))
    artifacts.append('solver_diagnostics.json')
    csv('target_weights.csv', result.target_weights)
    for name, frame in result.weights.items():
        csv(f'weights_{SLUGS[name]}.csv', frame, index=True, index_label='Date')
    annual = period_return_table(result.daily_returns)
    csv('calendar_year_returns.csv', annual)
    covariance_rows = []
    for date, estimators in result.cov_estimates.items():
        for estimator, matrix in estimators.items():
            for i, first in enumerate(prices.columns):
                for j, second in enumerate(prices.columns):
                    covariance_rows.append({'signal_date': date, 'estimator': estimator,
                                            'asset_i': first, 'asset_j': second, 'covariance': matrix[i,j]})
    csv('covariance_estimates.csv', pd.DataFrame(covariance_rows))
    normalized = prices.div(prices.iloc[0]).mul(100)
    drawdowns = result.wealth.div(result.wealth.cummax().clip(lower=1.0)).sub(1)
    chart_label = 'UNVERIFIED INPUT — '
    for name, frame, title, percent in (
        ('input_prices.svg', normalized, 'Input prices — normalized to 100', False),
        ('wealth_index.svg', result.wealth, 'Net wealth index', False),
        ('drawdowns.svg', drawdowns, 'Net drawdown', True),
    ):
        (folder / name).write_text(line_chart_svg(frame, chart_label+title, percent=percent), encoding='utf-8')
        artifacts.append(name)
    source = dict(prices.attrs.get('source_metadata', {}))
    source.update(provenance)
    source = safe_provenance(source)
    source['kind'] = kind
    source['independently_verified_market_data'] = False
    _write_json(folder / 'source_provenance.json', source)
    artifacts.append('source_provenance.json')
    metadata = {
        'schema_version': 1, 'status': 'complete', 'kind': kind,
        'application': 'RiskLab',
        'results_label': 'USER-SUPPLIED; NOT INDEPENDENTLY VERIFIED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': source, 'raw_input_sha256': _sha(raw_content),
        'raw_input_size_bytes': len(raw_content),
        'cleaned_prices_sha256': _sha((folder / 'cleaned_prices.csv').read_bytes()),
        'original_data_quality': quality, 'mandate': asdict(mandate),
        'first_out_of_sample_signal': str(result.initial_signal.date()),
        'input_coverage': {'first_cleaned_price_date': str(prices.index.min().date()),
                           'last_cleaned_price_date': str(prices.index.max().date()),
                           'cleaned_price_observations': len(prices)},
        'first_execution_date': str(pd.to_datetime(result.trades.loc[result.trades['trade_type'].eq('initial'), 'date']).min().date()),
        'first_recorded_return_date': str(result.daily_returns.index.min().date()),
        'last_recorded_return_date': str(result.daily_returns.index.max().date()),
        'execution_policy': result.execution_policy,
        'performance_conventions': {
            'CAGR': 'Annualized using elapsed calendar time from initial signal close to final observed close, divided by 365.25 days.',
            'volatility_and_Sharpe': '252 observations per year; sample standard deviation with ddof=1.',
            'risk_free': 'Constant configured annual assumption converted to a daily rate with 252 observations per year.',
            'initial_holdings': 'Cash until the first next-close execution; initial purchase cost is charged then.',
            'calendar_year_table': 'Product of net one-period returns within each observed calendar year; coverage dates identify partial years.',
        },
        'runtime_versions': _runtime_versions(), 'source_code_sha256': _source_hashes(),
        'artifact_hashes': {name: {'sha256': _sha((folder/name).read_bytes()), 'size_bytes': (folder/name).stat().st_size} for name in artifacts},
    }
    _write_json(folder / 'run_metadata.json', metadata)
    artifacts.append('run_metadata.json')
    context = metadata['results_label'] + '\n' + result.execution_policy + '\n'
    (folder / 'RUN_CONTEXT.txt').write_text(context, encoding='utf-8')
    artifacts.append('RUN_CONTEXT.txt')
    report = folder / 'report.html'
    report.write_text(_report_html(result, prices, mandate, metadata, raw_content, annual, artifacts), encoding='utf-8')
    _write_json(folder / 'run_status.json', {'application': 'RiskLab', 'status': 'complete', 'kind': kind, 'created_utc': metadata['created_utc'], 'report': report.name})
    old_failure = folder / 'failure_diagnostics.json'
    if old_failure.exists():
        old_failure.unlink()
    return report


def run_from_csv(input_path, output_dir, mandate=None, kind='user_supplied') -> Path:
    """Archive a selected input, run once and export. Missing input is an error."""
    from .core import backtest
    path = Path(input_path).expanduser().resolve()
    folder = Path(output_dir).expanduser().resolve()
    if kind != 'user_supplied':
        raise ValueError('Only user-supplied historical price inputs are supported.')
    metadata = {'kind': kind, 'input_filename': path.name, 'source': 'User-selected local CSV'}
    # Read first so selecting a previous generated raw_input.csv remains possible
    # even when that previous run is moved aside before this attempt starts.
    read_error = None
    try:
        content = path.read_bytes()
    except Exception as exc:
        read_error = exc
        content = None
    if content is not None:
        metadata.update(local_price_provenance(path, content))
    folder = _begin_run(folder, metadata, content)
    try:
        if read_error is not None:
            raise read_error
        prices = read_price_csv(content, archive_dir=folder, source_metadata=metadata)
        metadata.update(prices.attrs['source_metadata'])
        mandate = mandate or Mandate()
        result = backtest(prices, mandate)
        return export_run(result, prices, mandate, folder, content, metadata, _prepared=True)
    except Exception as exc:
        _failed_run(folder, exc, metadata)
        raise
