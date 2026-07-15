#!/bin/bash
# Shared configuration for the DQL v11.1 50-task sweep (sourced by run_task.sh).
# Every value is overridable from the environment, e.g.:
#   AGENT=agents/dql_v11_2.py SEEDS=1 bash sbatch/submit_all.sh

# NOTE: not $HOME/miniconda3 -- $HOME has a 10G user quota; envs live in project space.
export CONDA_ROOT=${CONDA_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3}
export CONDA_ENV=${CONDA_ENV:-rql}
export PYTHONNOUSERSITE=1   # ~/.local has a broken cffi that shadows the conda env

export AGENT=${AGENT:-agents/dql_v11_1.py}
export STEPS=${STEPS:-1000000}
export SEEDS=${SEEDS:-3}
export SAVE_INTERVAL=${SAVE_INTERVAL:-100000}   # checkpoint cadence = resume granularity under the 4h walltime
export EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
export EVAL_EPISODES=${EVAL_EPISODES:-50}
export GROUP_PREFIX=${GROUP_PREFIX:-DQL111-50}

# API key lives OUTSIDE git: sbatch/.wandb_key (gitignored), or pre-set WANDB_API_KEY.
_KEY_FILE="$(dirname "${BASH_SOURCE[0]}")/.wandb_key"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_KEY_FILE" ]; then
  export WANDB_API_KEY="$(cat "$_KEY_FILE")"
fi
export WANDB_PROJECT=${WANDB_PROJECT:-rql-iclr2027-50tasks}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=300

export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-${REPO:-$PWD}/data/ogbench}   # project space, not $HOME (10G quota)
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9       # one job per GPU -> take most of the A100
