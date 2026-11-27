"""Explicit Yahoo download, with original response and validation archive."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from risklab.config import START_DATE, END_DATE, DATA_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default=START_DATE)
    parser.add_argument('--end', default=END_DATE)
    parser.add_argument('--out', type=Path, default=DATA_PATH)
    parser.add_argument('--archive', type=Path, help='Original provider-response and cleaning evidence folder')
    args = parser.parse_args(argv)
    from risklab.data import download_prices
    archive = args.archive or args.out.parent/(args.out.stem+'_source_archive')
    try:
        prices = download_prices(args.start, args.end, archive_dir=archive)
    except Exception as exc:
        print(f'Download stopped: {exc}\nNo fallback was used. Diagnostics: {archive.resolve()}', file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out, index_label='Date')
    print(f'Saved {len(prices)} cleaned observations to {args.out.resolve()}\nOriginal response and cleaning report: {archive.resolve()}\nDownloaded data remain independently unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
