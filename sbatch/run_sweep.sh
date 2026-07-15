#!/bin/bash
# Run a single hyper-sweep variant (1 seed, 1 GPU) within a 4h window, with the same
# checkpoint-resume semantics as run_task.sh. All parameters arrive as RUN_* env vars
# exported by the generated sbatch file (see gen_sweep.py).
# Exit codes: 0 = run complete, 99 = work remaining (caller requeues).
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
source sbatch/env.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

: "${RUN_ENV:?}" "${RUN_H:?}" "${RUN_EXP:?}" "${RUN_RHO:?}" "${RUN_DISC:?}" "${RUN_NAME:?}" "${RUN_GROUP:?}"
RUN_STEPS=${RUN_STEPS:-500000}
RUN_EXTRA=${RUN_EXTRA:-}
SPARSE_FLAG=""; [ "${RUN_SPARSE:-0}" = "1" ] && SPARSE_FLAG="--sparse"

WINDOW=${WINDOW:-13800}
MIN_START=${MIN_START:-1500}

run_dir="exp/${WANDB_PROJECT}/${RUN_GROUP}/${RUN_NAME}"
if [ -f "$run_dir/params_${RUN_STEPS}.pkl" ]; then
  echo "[run_sweep] $RUN_NAME already complete"; exit 0
fi
left=$((WINDOW - SECONDS))
[ "$left" -lt "$MIN_START" ] && exit 99

echo "[run_sweep] $RUN_NAME env=$RUN_ENV h=$RUN_H e=$RUN_EXP rho=$RUN_RHO extra='$RUN_EXTRA'"
timeout -k 120 "$left" python main.py \
  --agent="$AGENT" \
  --env_name="$RUN_ENV" \
  --agent.h="$RUN_H" --agent.expectile="$RUN_EXP" --agent.rho="$RUN_RHO" --agent.discount="$RUN_DISC" $SPARSE_FLAG \
  $RUN_EXTRA \
  --offline_steps="$RUN_STEPS" \
  --eval_interval=25000 --eval_episodes=50 --log_interval=5000 --save_interval=100000 \
  --run_group="$RUN_GROUP" --seed=0 \
  --run_name="$RUN_NAME" --auto_resume

[ -f "$run_dir/params_${RUN_STEPS}.pkl" ] && exit 0 || exit 99
