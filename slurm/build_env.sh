#!/bin/bash
# Build the DQL cluster environment (run once on a login/data-mover node WITH internet).
#   bash slurm/build_env.sh
# Assumes conda is installed. Override CONDA_ENV / OGBENCH_DATASET_DIR as needed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1   # keep ~/.local out of resolution -- it shadows env packages

CONDA_ENV=${CONDA_ENV:-rql}
PY=${PY:-3.11}
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"

# 1) conda env + deps ---------------------------------------------------------
if ! conda env list | grep -qE "^${CONDA_ENV}\b"; then
  echo "[build] creating conda env ${CONDA_ENV} (python ${PY})"
  conda create -n "${CONDA_ENV}" "python=${PY}" -y
fi
conda activate "${CONDA_ENV}"
echo "[build] installing requirements + JAX(CUDA) for A100"
pip install -r requirements.txt
# GPU JAX matched to this repo (jax 0.10.x). Adjust CUDA wheel to the cluster's CUDA if needed.
pip install --upgrade "jax[cuda12]==0.10.2" || echo "[build] set the jax cuda wheel to your cluster CUDA"
pip install ogbench

# 2) sanity: imports + GPU visible -------------------------------------------
python - <<'PY'
import jax, ogbench, flax
print("[build] jax", jax.__version__, "devices:", jax.devices())
PY

# 3) stage datasets to shared FS ---------------------------------------------
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}
export MUJOCO_GL=egl
echo "[build] staging datasets to ${OGBENCH_DATASET_DIR}"
python slurm/gen_tasks.py
python slurm/stage_datasets.py
echo "[build] DONE. Submit with:  sbatch slurm/dql_50.sbatch"
