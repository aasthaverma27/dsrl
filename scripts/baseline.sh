#!/usr/bin/env bash
# The frozen GR00T policy without DSRL, on the same protocol as scripts/eval.sh.
#
#   GPU=0 scripts/baseline.sh --mode native --episodes 30 --layout 0 --max-chunks 120 \
#       --seed 456 --out native_s456.jsonl
#
# --mode native         GR00T samples its own noise (the stock policy)
# --mode stochastic     random steering actions (the exploration floor)
# --mode deterministic  one fixed noise draw per episode (the no-op)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"; exec "$PY" scripts/baseline.py "$@"
