# DQL v11.1 — 50-task OGBench sweep (1 A100 per task, 3 seeds each)

50 jobs x 1 GPU = 50 GPUs total; each job trains its task's 3 seeds sequentially.
The cluster caps every partition at 4h, so each job requeues itself
(`--dependency=singleton`) and every run resumes from its latest checkpoint
(`main.py --run_name ... --auto_resume`). All progress state is the checkpoints in
`exp/` — jobs can be killed/resubmitted at any time, and completed seeds
(final `params_<STEPS>.pkl` exists) are never re-run.

## Files

| file | role |
|---|---|
| `env.sh` | single source of truth: conda env, wandb key/project, agent, seeds, steps |
| `run_task.sh <idx>` | runs one task's remaining seeds within a 4h window (exit 99 = requeue) |
| `gen_jobs.py` | generates the 50 job files in `jobs/` from `slurm/tasks.tsv` |
| `submit_all.sh [pat]` | submits all 50 jobs (or the subset matching `pat`) |
| `jobs/tNN-*.sbatch` | generated — do not edit by hand |

## Usage

```bash
bash slurm/build_env.sh          # once: conda env `rql` + stage the 10 datasets
bash sbatch/submit_all.sh        # launch all 50 tasks
bash sbatch/submit_all.sh puzzle # or a subset
squeue -u $USER                  # monitor; logs in logs/<jobname>.<jobid>.log
```

Everything in `env.sh` is overridable per-submission, e.g. rerun the sweep with
another agent without touching any file:

```bash
AGENT=agents/dql_v11_2.py GROUP_PREFIX=DQL112-50 bash sbatch/submit_all.sh
```

Wandb: project `rql-iclr2027-50tasks`, one run per (task, seed), grouped as
`DQL111-50/<env>`; requeued windows resume the same wandb run.

## Results

```bash
python report_dql111.py   # writes RESULTS_DQL111.md (DQL v11.1 vs RQL appendix Table 1)
python viz/build_data.py  # refreshes viz/report/index.html (HTML dashboard: benchmark,
                          # diagnostics scatters, paper-figure roadmap)
```

Both are safe to run any time -- incomplete runs are flagged / appear at their latest eval.
Dashboard figures: drop artifacts into viz/figs/<id>_*.png|gif|mp4 and rebuild.

## If a family underperforms: sweep -> promote -> rerun

```bash
# 1. coordinate sweep of the high-leverage levers (expectile, adv_temp, drift_step, rho)
#    on task1 of each failing family, 1 seed, 500k steps:
python sbatch/gen_sweep.py cube-double-play scene-play
JOBS_DIR=sbatch/jobs-sweep bash sbatch/submit_all.sh

# 2. promote winners: edit the family's hypers in slurm/gen_tasks.py, then
python slurm/gen_tasks.py && python sbatch/gen_jobs.py

# 3. rerun just those families under a fresh group (env vars propagate through sbatch):
GROUP_PREFIX=DQL111R2 bash sbatch/submit_all.sh cube-double
```

Edit the grid in `gen_sweep.py` (`LEVERS`); baseline values are skipped automatically.

To rerun everything with another agent, regenerate (knobs are baked into the job files
so requeues can't drift) and submit:

```bash
JOB_PREFIX=dql112 AGENT=agents/dql_v11_2.py GROUP_PREFIX=DQL112-50 python sbatch/gen_jobs.py
bash sbatch/submit_all.sh
RUN_PREFIX=dql112 GROUP_PREFIX=DQL112-50 python report_dql111.py   # -> RESULTS_DQL112.md
```
