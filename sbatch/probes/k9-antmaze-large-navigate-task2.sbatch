#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch,backfill
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --requeue
#SBATCH --job-name=dql112pk9-antmaze-large-navigate-task2
#SBATCH --output=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/rql-/logs/%x.%j.log
#SBATCH --error=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/rql-/logs/%x.%j.log
#SBATCH --open-mode=append
# kappa=0.9 propagation probe (DQL112_FAILURE_ANALYSIS.md action #5): far-goal task, 1 seed, 500k
export REPO=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/rql-
cd $REPO
source sbatch/env.sh
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python main.py --agent=agents/dql_v11_2.py --env_name=antmaze-large-navigate-singletask-task2-v0 \
  --agent.h=1 --agent.expectile=0.9 --agent.rho=0.5 --agent.discount=0.99 \
  --offline_steps=500000 --eval_interval=50000 --eval_episodes=50 \
  --log_interval=5000 --save_interval=100000 \
  --run_group=DQL112-PROBE/antmaze-large-navigate-singletask-task2-v0 --seed=0 \
  --run_name=dql112pk9__antmaze-large-navigate-singletask-task2-v0__sd0 --auto_resume
