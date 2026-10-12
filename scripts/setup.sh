#!/usr/bin/env bash
# One-time setup: external code, patches, Python environment, checkpoint.
#
#   bash scripts/setup.sh                 everything
#   bash scripts/setup.sh --no-checkpoint skip the 12.6 GB GR00T download
#   bash scripts/setup.sh --check         report what is present, change nothing
#
# Requires git, git-lfs and uv (https://docs.astral.sh/uv/). The GR00T base VLM
# (nvidia/Cosmos-Reason2-2B) is gated on Hugging Face: accept its licence and
# `huggingface-cli login` (or export HF_TOKEN) before the first run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# pypi.nvidia.com is slow; uv gives up after 30 s by default and torch then fails
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
CHECK=0; CKPT=1
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --no-checkpoint) CKPT=0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

GROOT=external/Isaac-GR00T
GROOT_URL=https://github.com/NVIDIA/Isaac-GR00T.git
GROOT_PIN=b9955401d50c92a29258732e3ad6ccd579f1bdc0
WBC=external/GR00T-WholeBodyControl
WBC_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
WBC_PIN=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
# The only Git LFS files the in-process stack loads: G1 model + meshes for the
# simulator and the controller, the two WBC ONNX policies, the Unitree SDK's .so.
WBC_LFS="gear_sonic/data/robot_model/model_data/g1/**,decoupled_wbc/control/robot_model/model_data/g1/**,decoupled_wbc/sim2mujoco/resources/robots/g1/policy/*,external_dependencies/unitree_sdk2_python/**"
HF_REPO=rahul-ai-01/red_green_blue_cube_sorting_in_sim
HF_REV=b879ba536732b57d1ed6d0006d8d16f375c65d92
CKPT_DIR=checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000

patched() { git -C "$1" apply --reverse --check "$ROOT/$2" >/dev/null 2>&1; }

if [ "$CHECK" = 1 ]; then
  for d in "$GROOT" "$WBC"; do
    [ -d "$d/.git" ] && echo "  $d @ $(git -C "$d" rev-parse --short HEAD)" || echo "  $d MISSING"
  done
  patched "$GROOT" patches/Isaac-GR00T_dsrl_noise.patch && echo "  Isaac-GR00T patch applied" || echo "  Isaac-GR00T patch NOT applied"
  patched "$WBC" patches/GR00T-WholeBodyControl_dsrl_sim.patch && echo "  WBC sim patch applied" || echo "  WBC sim patch NOT applied"
  [ -x .venv/bin/python ] && echo "  .venv present" || echo "  .venv MISSING"
  [ -f "$CKPT_DIR/config.json" ] && echo "  checkpoint present" || echo "  checkpoint MISSING"
  exit 0
fi

git lfs version >/dev/null 2>&1 || { echo "git-lfs is required (apt install git-lfs; git lfs install)" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

fetch() {   # fetch <dir> <url> <commit>
  [ -d "$1/.git" ] && return 0
  echo "[setup] $2 @ ${3:0:10}"
  if git submodule status "$1" >/dev/null 2>&1; then
    GIT_LFS_SKIP_SMUDGE=1 git submodule update --init "$1"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout "$2" "$1"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$1" checkout -q --detach "$3"
  fi
}
apply_once() {   # apply_once <repo> <patch>
  if patched "$1" "$2"; then echo "[setup] $(basename "$2") already applied"
  else git -C "$1" apply "$ROOT/$2" && echo "[setup] applied $(basename "$2")"; fi
}

# 1. external code at the tested commits, plus our patches
fetch "$GROOT" "$GROOT_URL" "$GROOT_PIN"
fetch "$WBC" "$WBC_URL" "$WBC_PIN"
echo "[setup] Git LFS files used by the simulator and controller (~110 MB)"
git -C "$WBC" lfs pull --include="$WBC_LFS"
apply_once "$GROOT" patches/Isaac-GR00T_dsrl_noise.patch
apply_once "$WBC" patches/GR00T-WholeBodyControl_dsrl_sim.patch

# 2. Python 3.10 environment with the pinned versions
if [ ! -x .venv/bin/python ]; then
  echo "[setup] creating .venv"
  uv venv -q --python 3.10 .venv
fi
uv pip install -q --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv/bin/python -r requirements.txt huggingface_hub

# 3. the fine-tuned GR00T N1.7 checkpoint (weights only, not the optimizer state)
if [ "$CKPT" = 1 ] && [ ! -f "$CKPT_DIR/config.json" ]; then
  echo "[setup] downloading $HF_REPO (checkpoint-22000, ~12.6 GB)"
  .venv/bin/python - "$HF_REPO" "$HF_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir="checkpoints/red_green_blue_cube_sorting_in_sim",
                  allow_patterns=["checkpoint-22000/*"], ignore_patterns=["*optimizer.pt"])
PY
fi

echo "[setup] done - check with:  bash scripts/setup.sh --check"
