#!/usr/bin/env bash
# Train DSRL on the cube-sorting task (layout 0), in-process.
#
#   GPU=0 scripts/train.sh <run-name> <steps> [override ...]
#   GPU=0 scripts/train.sh r5-pixel 30000 sac.use_pixels=true groot.resample_unsteered=true
#
# Optional:  RESUME=<ckpt.pt>  RESUME_BUF=<buffer.npz>  (continue a run)
#            EVAL_CHUNKS=120   (episode length for in-training evals; default 120)
# Runs are written to runs/<timestamp>_<run-name>/. Overrides are not restored by
# --resume: restate every override when continuing a run.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ $# -ge 2 ] || { sed -n 2,11p "$0"; exit 2; }
NAME=$1; STEPS=$2; shift 2
OV=(); [ $# -gt 0 ] && OV=(--override "$@")
EXTRA=()
[ -n "${RESUME:-}" ] && EXTRA+=(--resume "$RESUME")
[ -n "${RESUME_BUF:-}" ] && EXTRA+=(--resume-buffer "$RESUME_BUF")
cd "$ROOT"
exec "$PY" -m dsrl.rl.train --env sort --steps "$STEPS" --layout 0 \
  --run-name "$NAME" --eval-every 2000 --eval-episodes 10 \
  --eval-chunks "${EVAL_CHUNKS:-120}" --save-every 5000 "${EXTRA[@]}" "${OV[@]}"
