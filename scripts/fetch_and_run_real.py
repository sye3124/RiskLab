"""Legacy explicit Yahoo download and single-window export; normal use is python run.py."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from risklab.config import START_DATE, END_DATE, Mandate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('yahoo',), default='yahoo')
    parser.add_argument('--start', default=START_DATE)
    parser.add_argument('--end', default=END_DATE)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs_downloaded_UNVERIFIED')
    parser.add_argument('--window', type=int, default=Mandate().estimation_days)
    parser.add_argument('--cost-bps', type=float, default=Mandate().transaction_cost_bps)
    args = parser.parse_args(argv)
    from risklab.config import Mandate
    from risklab.core import backtest
    from risklab.data import download_prices
    from risklab.export import export_run, _begin_run, _failed_run
    folder = args.output.expanduser().resolve()
    source = {'kind': 'user_supplied', 'source': 'Explicit Yahoo download', 'independently_verified_market_data': False}
    folder = _begin_run(folder, source)
    try:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        prices = download_prices(args.start, args.end, archive_dir=folder/f'source_archive_{stamp}')
        mandate = Mandate(estimation_days=args.window, transaction_cost_bps=args.cost_bps)
        result = backtest(prices, mandate)
        report = export_run(result, prices, mandate, folder, prices.attrs['raw_content'], source, _prepared=True)
    except Exception as exc:
        _failed_run(folder, exc, source)
        print(f'Download/backtest stopped: {exc}\nNo fallback was used. Diagnostics: {folder}', file=sys.stderr)
        return 1
    print(f'Completed a run on unverified downloaded data: {report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
