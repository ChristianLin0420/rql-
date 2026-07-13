#!/bin/bash
# =============================================================================
# DQL 50-task OGBench sweep — SLURM job array (1 A100 per task).
#   Array index -> (seed, task) via tasks.tsv. Submit from repo root:
#     sbatch slurm/dql_50.sbatch                            # 1 seed  -> 50 jobs
#     SEEDS=4 sbatch --array=0-199%32 slurm/dql_50.sbatch   # 4 seeds -> 200 jobs
#   Override AGENT / STEPS via env vars. Tune #SBATCH lines to your cluster.
# =============================================================================
#SBATCH --job-name=dql50
#SBATCH --partition=gpu                 # <-- set to your A100 partition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-49%16                 # 50 tasks, 1 seed; %16 = max concurrent. For 4 seeds: 0-199%32
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

AGENT=${AGENT:-agents/dql_v11_2.py}     # our best drift-based agent
STEPS=${STEPS:-1000000}
NTASKS=50
id=${SLURM_ARRAY_TASK_ID:-0}
task_idx=$(( id % NTASKS ))
seed=$(( id / NTASKS ))

# read row task_idx (skip header -> line task_idx+2)
row=$(sed -n "$((task_idx + 2))p" slurm/tasks.tsv)
IFS=$'\t' read -r ENV H EXP RHO DISC SPARSE <<< "$row"
SPARSE_FLAG=""; [ "$SPARSE" = "1" ] && SPARSE_FLAG="--sparse"

echo "[array $id] task=$task_idx seed=$seed env=$ENV h=$H e=$EXP rho=$RHO disc=$DISC sparse=$SPARSE"

source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-rql}"
export MUJOCO_GL=egl
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-rql-iclr2027-50tasks}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # one job per GPU -> take most of the A100
export OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}

srun python main.py \
  --agent="$AGENT" \
  --env_name="$ENV" \
  --agent.h="$H" --agent.expectile="$EXP" --agent.rho="$RHO" --agent.discount="$DISC" $SPARSE_FLAG \
  --offline_steps="$STEPS" \
  --eval_interval=50000 --eval_episodes=50 --log_interval=5000 --save_interval=500000 \
  --run_group="DQL-50/${ENV}" --seed="$seed"
