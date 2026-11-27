"""Run the complete historical RiskLab pipeline, or launch its dashboard."""
from __future__ import annotations
import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import webbrowser
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    try:
        from risklab.config import DATA_PATH, ESTIMATION_WINDOWS, Mandate
        from risklab.data import ensure_prices
        from risklab.export import run_from_csv
    except ModuleNotFoundError as exc:
        print(f'Missing dependency: {exc.name}. Run python -m pip install -r requirements.txt.', file=sys.stderr)
        return 2
    defaults = Mandate()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Explicit historical CSV; never downloaded or replaced')
    parser.add_argument('--output', type=Path, help='Custom run directory (requires a single --window)')
    parser.add_argument('--kind', choices=('user_supplied',), help=argparse.SUPPRESS)
    parser.add_argument('--window', type=int, help='Run one window instead of the configured pair')
    parser.add_argument('--refresh-data', action='store_true', help='Explicitly refresh the fixed-date dataset')
    parser.add_argument('--cost-bps', type=float, default=defaults.transaction_cost_bps)
    parser.add_argument('--shrinkage-intensity', '--shrinkage', type=float, default=defaults.shrinkage_intensity)
    parser.add_argument('--turnover-penalty', type=float, default=defaults.turnover_penalty)
    parser.add_argument('--risk-free-annual', type=float, default=defaults.risk_free_annual)
    parser.add_argument('--max-etf-weight', type=float, default=defaults.max_etf_weight)
    parser.add_argument('--max-equity-weight', type=float, default=defaults.max_equity_weight)
    parser.add_argument('--max-alternatives-weight', type=float, default=defaults.max_alternatives_weight)
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument('--open', action='store_true', help='Open the completed baseline report')
    browser.add_argument('--no-browser', action='store_true', help='Keep the browser closed (default)')
    parser.add_argument('--app', action='store_true', help='Launch Streamlit')
    args = parser.parse_args(argv)
    if args.input and args.refresh_data:
        parser.error('--refresh-data cannot be combined with --input.')
    if args.output and args.window is None:
        parser.error('--output requires --window to avoid mixing two sets of results.')
    if args.app:
        if args.input or args.refresh_data or args.output or args.window is not None:
            parser.error('Use the app sidebar for input and settings; run the analysis separately.')
        if importlib.util.find_spec('streamlit') is None:
            print('Install requirements.txt before launching Streamlit.', file=sys.stderr)
            return 2
        return subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(ROOT/'app.py')], cwd=ROOT)
    windows = (args.window,) if args.window is not None else ESTIMATION_WINDOWS
    try:
        mandates = [replace(defaults, estimation_days=window,
                    transaction_cost_bps=args.cost_bps, risk_free_annual=args.risk_free_annual,
                    shrinkage_intensity=args.shrinkage_intensity, turnover_penalty=args.turnover_penalty,
                    max_etf_weight=args.max_etf_weight, max_equity_weight=args.max_equity_weight,
                    max_alternatives_weight=args.max_alternatives_weight) for window in windows]
        if args.input is not None:
            selected_input = args.input.expanduser().resolve()
        else:
            print('Checking historical ETF prices...', flush=True)
            if args.refresh_data or not DATA_PATH.is_file():
                print('Downloading fixed-period Yahoo Finance history...', flush=True)
            selected_input = ensure_prices(DATA_PATH, refresh=args.refresh_data)
        print(f'Input: {selected_input}\nUSER-SUPPLIED DATA — NOT INDEPENDENTLY VERIFIED', flush=True)
        reports = []
        for i, mandate in enumerate(mandates, 1):
            window = mandate.estimation_days
            output = args.output or ROOT / (f'outputs_user_{window}' if args.input else f'outputs_real_{window}')
            print(f'[{i}/{len(mandates)}] Running {window}-day estimation window...', flush=True)
            report = run_from_csv(selected_input, output, mandate=mandate)
            reports.append(report)
            print(f'Complete: {report}', flush=True)
    except Exception as exc:
        print(f'RiskLab stopped: {exc}\nNo replacement data were substituted. Earlier completed runs may still exist; check run_status.json.', file=sys.stderr)
        return 1
    print('All requested backtests completed. Launch the app with: python run.py --app')
    if args.open:
        webbrowser.open(reports[0].as_uri())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
