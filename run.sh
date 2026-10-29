#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
[ -f .venv/.deps_ok ] || { pip install -q -r requirements.txt && touch .venv/.deps_ok; }
if [ ! -f runtime/model.json ]; then
  pip install -q -r requirements-train.txt
  python train.py
  python export_runtime.py
fi
python app.py
