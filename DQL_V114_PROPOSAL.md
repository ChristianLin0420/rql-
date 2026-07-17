# DQL v11.4 — Family-calibrated locality on the dataset-wide pool

Status: **50-task sweep RUNNING** (group `DQL114-50`, 3 seeds × 1M, launched 2026-07-17
~00:55). One change vs v11.3, calibrated on checkpoints before launch.

## S1 — Why (the v11.3 split verdict)

v11.3's dataset-wide pool + argmax-Q deployment transformed manipulation (scene 11.6→41.6,
puzzle-3x3 22.0→48.4) but its bandwidth statistic — `bw = 0.15 · mean(sq batch×pool)` —
made the locality penalty vanish: the 31 selected neighbors are the closest of 100k, so
`sq_sel/bw ≈ 0` and borrowing maxed out in every family (w_self 0.09–0.29). Forced
borrowing of phase-incoherent gait actions destroyed cyclic locomotion (humanoidmaze
61→0.2, antmaze-large 65→48, giant 15→1; overall 14.5 vs v11.2's 18.6). v11.2's high
humanoid w_self (0.63–0.69) was the correct regime for gaits — the borrowing level is task
physics, not a universal constant.

## S2 — The change (complete)

`bw = state_bw · mean(sq_sel nonself)` — the locality scale is anchored to the selected
neighbors' distances, making `state_bw` a direct borrowing dial. Checkpoint calibration
(`diagnostics/knn_probe_v114_bw.py`, 6 families × formula × c-grid) showed this normalizer
erases natural family differences (any single c gives all families the same w_self within
±0.06), so `state_bw` is a per-family run flag (new `state_bw` column in `slurm/tasks.tsv`,
passed by `sbatch/run_task.sh`):

| families | state_bw | predicted w_self | rationale |
|---|---|---|---|
| antmaze (large, giant) | 0.5 | ~0.33–0.35 | v11.2's healthy stitching regime (0.31–0.42) |
| humanoidmaze (medium, large) | 0.25 | ~0.64–0.69 | matches v11.2's gait-coherent regime exactly |
| scene, puzzle, cube | 1.0 | ~0.15–0.19 | v11.3's winning dense-borrowing regime |

Everything else — pool (n_pool=100k), argmax-Q_LCB deployment, critic, all v11.2 stability
machinery, per-family h/expectile/ρ/γ/sparse — identical to v11.3. Agent:
`agents/dql_v11_4.py` (config default state_bw=1.0, matching its manipulation-oriented
defaults). Smoke-tested: dial verified (state_bw 0.25 → w_self 0.54; 1.0 → 0.08 on
synthetic data), deployment + checkpoint round-trip clean.

## S3 — Pre-registered expectations

- **E1 (mechanism):** first train.csv rows show w_self ≈ per-family targets above —
  humanoid ≥0.5 AND scene/puzzle ≤0.3 in the same sweep (the property no prior version had).
- **E2 (manipulation):** holds v11.3's gains (scene ≥40, puzzle-3x3 ≥45); falsified if the
  bw change costs manipulation more than a few points.
- **E3 (humanoid recovery):** humanoidmaze-medium back to ≥ v11.2's 61; falsified if it
  stays ≤30 despite w_self ≈ 0.65 — that would mean something beyond locality broke it.
- **E4 (antmaze recovery):** antmaze-large ≥ v11.2's 65; giant ≥ 15.
- **E5 (headline):** overall > 25 (v11.2 18.6, v11.3 14.5) — the first version combining
  both regimes; stretch: locomotion at v11.2 levels + manipulation at v11.3 levels ≈ 28.
- **E6 (stability):** no post-peak decay at the 1M horizon (v11.3 property, expected to hold).

## S4 — Live observations

- 00:55 — 50 jobs submitted (1-min stagger), no name collisions.
- 01:46 — dual gate passed on MEANS: w_self humanoid 0.47→0.56, antmaze 0.32, manip 0.10–0.19.
- 03:50 — **INCIDENT (E1/E3 partial failure, caught mid-flight):** humanoidmaze evals flat 0
  through 550k while v11.2 had 44 by 250k. Selector A/B on the 500k checkpoint exonerated
  deployment (medoid 1/15 vs argmax 0/15 — both dead) and located the true fault: the
  batch-MEAN bw normalizer masks a 4× per-state neighbor-density spread, making w_self
  **bimodal** (p10 0.07 / p90 0.97; the dense gait-corridor states still over-borrow —
  exactly v11.3's failure, hidden by the mean). Secondary confirmed defect: with ρ=0,
  argmax-"LCB" is pure mean-Q noise and systematically picks geometric outliers (2× farther
  than medoid).
- 04:50 — **v11.4b hotfix, humanoidmaze only** (other families healthy and untouched):
  `bw_per_state=True` (per-state MEDIAN normalizer; calibration F2 c=0.25 → w_self 0.690
  with p10 0.362) + `deploy_medoid=True` (new tasks.tsv columns bw_ps/dm). The 10
  humanoidmaze jobs were restarted from scratch (contaminated 500k discarded). Lesson
  recorded: calibrate on distributions, not means. New humanoid ETA ~21:00; the health gate
  for it is w_self p10 > 0.3 and eval signal by 250k.

## S5 — Results

`RESULTS_DQL114.md` when all 150 seeds reach 1M (ETA 2026-07-17 late evening). Dashboard:
add `("dql114", "DQL114-50", "DQL v11.4")` to `VARIANTS` in `viz/build_data.py`.

## S6 — Sources

`DQL_V113_PROPOSAL.md` S5 (split verdict + root cause), `diagnostics/knn_probe_v114_bw.py`
(calibration tables), `agents/dql_v11_4.py`, `DQL112_FAILURE_ANALYSIS.md`.
