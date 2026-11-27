#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python run.py --open
fi
exec python3 run.py --open
