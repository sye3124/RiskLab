"""Export a complete run from an explicitly supplied CSV; never downloads a replacement."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    if '--input' not in sys.argv[1:] and not any(a.startswith('--input=') for a in sys.argv[1:]):
        if not any(a in ('-h', '--help') for a in sys.argv[1:]):
            print('An explicit --input CSV is required. For the default historical pipeline, use python run.py.', file=sys.stderr)
            return 2
    from run import main as run_main
    return run_main(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
