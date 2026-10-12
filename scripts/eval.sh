#!/usr/bin/env bash
# Confirm a trained actor: n fresh episodes, stochastic sampling, layout 0.
#
#   GPU=0 scripts/eval.sh runs/<run>/ckpt_best_4000.pt --config runs/<run>/config.json \
#       --max-chunks 120 --episodes 30 --seed 123 --out confirm_s123.jsonl
#
# Every number in results/ is two such runs (seeds 123 and 456) pooled: n=60.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"; exec "$PY" scripts/eval_ckpt.py "$@"
