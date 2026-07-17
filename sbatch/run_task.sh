#!/bin/bash
# Run one task of the 50-task sweep (all SEEDS seeds, sequentially, 1 GPU) inside
# a 4h SLURM window. Each seed auto-resumes from its latest checkpoint and is
# skipped once its final checkpoint exists, so all progress state lives in exp/
# on lustre -- the job can be requeued/killed/resubmitted at any time.
#   bash sbatch/run_task.sh <task_idx 0-49>
# Exit codes: 0 = every seed complete, 99 = work remaining (caller requeues).
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
source sbatch/env.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

idx=${1:?usage: run_task.sh <task_idx 0-49>}
row=$(sed -n "$((idx + 2))p" slurm/tasks.tsv)   # +2: skip header, 1-indexed
[ -n "$row" ] || { echo "[run_task] no tasks.tsv row for idx=$idx"; exit 1; }
IFS=$'\t' read -r ENV H EXP RHO DISC SPARSE BW <<< "$row"
SPARSE_FLAG=""; [ "$SPARSE" = "1" ] && SPARSE_FLAG="--sparse"
BW_FLAG=""; [ -n "$BW" ] && BW_FLAG="--agent.state_bw=$BW"   # per-family borrowing dial (v11.4+)

WINDOW=${WINDOW:-13800}       # use 3h50m of the 4h limit
MIN_START=${MIN_START:-1500}  # don't start a seed with <25min left; requeue instead

echo "[run_task] idx=$idx env=$ENV h=$H e=$EXP rho=$RHO disc=$DISC sparse=$SPARSE agent=$AGENT seeds=$SEEDS"

RUN_PREFIX=${RUN_PREFIX:-dql111}   # distinguishes run names / wandb ids across agent variants
pending=0
for seed in $(seq 0 $((SEEDS - 1))); do
  run_name="${RUN_PREFIX}__${ENV}__sd${seed}"
  run_dir="exp/${WANDB_PROJECT}/${GROUP_PREFIX}/${ENV}/${run_name}"  # mirrors main.py save_dir
  if [ -f "$run_dir/params_${STEPS}.pkl" ]; then
    echo "[run_task] seed $seed already complete -- skipping"
    continue
  fi
  left=$((WINDOW - SECONDS))
  if [ "$left" -lt "$MIN_START" ]; then
    echo "[run_task] only ${left}s left in window -> requeue before seed $seed"
    pending=1
    break
  fi
  echo "[run_task] seed $seed: running up to ${left}s"
  timeout -k 120 "$left" python main.py \
    --agent="$AGENT" \
    --env_name="$ENV" \
    --agent.h="$H" --agent.expectile="$EXP" --agent.rho="$RHO" --agent.discount="$DISC" $SPARSE_FLAG $BW_FLAG \
    --offline_steps="$STEPS" \
    --eval_interval="$EVAL_INTERVAL" --eval_episodes="$EVAL_EPISODES" \
    --log_interval=5000 --save_interval="$SAVE_INTERVAL" \
    --run_group="${GROUP_PREFIX}/${ENV}" --seed="$seed" \
    --run_name="$run_name" --auto_resume
  [ -f "$run_dir/params_${STEPS}.pkl" ] || pending=1
done

if [ "$pending" -eq 0 ]; then
  echo "[run_task] task $idx COMPLETE (all $SEEDS seeds finished ${STEPS} steps)"
  exit 0
fi
exit 99
