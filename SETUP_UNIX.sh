#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
printf '%s\n' 'RiskLab numerical environment setup. Python 3.12 is recommended.'
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Setup complete. Run sh START_ANALYSIS.sh.'
