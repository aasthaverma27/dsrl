# Sourced by the scripts in this directory: paths and runtime settings.
#   . scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WBC="$ROOT/external/GR00T-WholeBodyControl"
export DSRL_ROOT="$ROOT"
export DSRL_WBC_ROOT="${DSRL_WBC_ROOT:-$WBC}"
export DSRL_CHECKPOINT="${DSRL_CHECKPOINT:-$ROOT/checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000}"
# dsrl itself, Isaac-GR00T (patched), and the simulator / whole-body controller
# packages from GR00T-WholeBodyControl (gear_sonic, decoupled_wbc, unitree_sdk2py)
export PYTHONPATH="$ROOT:$ROOT/external/Isaac-GR00T:$WBC:$WBC/external_dependencies/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
export MUJOCO_GL=egl NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# EGL does not follow CUDA_VISIBLE_DEVICES: pin the render context to the same GPU.
if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" EGL_DEVICE_ID="$GPU"
fi
PY="${DSRL_PYTHON:-$ROOT/.venv/bin/python}"
