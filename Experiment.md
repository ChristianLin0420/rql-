# Experiment Tracker — DQL vs. RQL & all baselines

Reference: **RQL paper (arXiv 2606.17551), Table 1** — "Performance on 50 simulated robotic
manipulation tasks" (OGBench). 50 tasks = **10 categories × 5 tasks each**; numbers below are the
paper's **per-category aggregate** final success rate (%), averaged over 4 seeds. Our goal is to
**beat them all** — i.e. for each category, beat both RQL and the best baseline ("bar").

Legend for our column: `—` not run · `▶` running · number = final success (%) · **bold** = beats the bar.

---

## Scoreboard — the bar to beat (per category)

| Category                    | RQL | Best baseline (who)        | **Bar (max)** | DQL (ours) |
|-----------------------------|:---:|:---------------------------|:-------------:|:----------:|
| antmaze-large               | 83  | 94 (ReBRAC)                | **94**        | —          |
| antmaze-giant               | 37  | 57 (ReBRAC)                | **57**        | —          |
| humanoidmaze-medium         | 93  | 86 (IFQL)                  | **93 (RQL)**  | —          |
| humanoidmaze-large          | 39  | 24 (IFQL)                  | **39 (RQL)**  | —          |
| scene(-sparse)              | 89  | 99 (DSRL)                  | **99**        | —          |
| puzzle-3x3-sparse           | 100 | 100 (CGQL-M/IFQL/QAM/…)     | **100**       | —          |
| puzzle-4x4-100M-sparse      | 37  | 39 (QAM-E)                 | **39**        | —          |
| **cube-double** ★           | 23  | **74 (DSRL)**, 65 (QAM/-F/-E) | **74**     | ▶ (running) |
| **cube-triple** ★           | 4   | 8 (CGQL/QAM-E/TFQL)        | **8**         | —          |
| cube-quadruple-100M         | 51  | 19 (QSM)                   | **51 (RQL)**  | —          |
| **all — agg (50 tasks)**    | 56  | 46 (QAM-E)                 | **56 (RQL)**  | —          |

★ = **kernel-collapse categories**: RQL is *beaten by simple baselines* here (cube manipulation).
These are DQL's primary beat-RQL targets — where the one-step drift + interpolation-value is designed
to win. On `cube-double`, RQL (23) trails DSRL (74) and the QAM family (64–65) badly.

---

## Full Table 1 (per-category aggregate success %, all 19 methods)

Method order matches the paper header.

| Category | ReBRAC | FBRAC | BAM | FQL | FAWAC | CGQL | CGQL-M | CGQL-L | DAC | QSM | DSRL | FEdit | IFQL | QAM | QAM-F | QAM-E | BDPO | TFQL | **RQL** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| antmaze-large          | 94 | 2 | 84 | 76 | 17 | 76 | 71 | 65 | 88 | 90 | 61 | 58 | 36 | 81 | 83 | 83 | 83 | 0 | **83** |
| antmaze-giant          | 57 | 0 | 1  | 0  | 0  | 0  | 4  | 3  | 16 | 24 | 3  | 2  | 1  | 18 | 12 | 1  | 0  | 0 | **37** |
| humanoidmaze-medium    | 69 | 39 | 60 | 68 | 24 | 60 | 42 | 62 | 83 | 82 | 53 | 22 | 86 | 67 | 65 | 59 | 34 | 44 | **93** |
| humanoidmaze-large     | 17 | 0 | 5  | 9  | 0  | 5  | 6  | 6  | 0  | 6  | 3  | 3  | 24 | 11 | 12 | 2  | 1  | 1 | **39** |
| scene-sparse           | 65 | 50 | 98 | 78 | 38 | 38 | 74 | 88 | 68 | 78 | 99 | 62 | 84 | 97 | 95 | 97 | 94 | 61 | **89** |
| puzzle-3x3-sparse      | 79 | 0 | 56 | 70 | 3  | 48 | 100 | 90 | 68 | 57 | 87 | 99 | 100 | 100 | 99 | 100 | 82 | 100 | **100** |
| puzzle-4x4-100M-sparse | 0  | 15 | 0  | 5  | 0  | 24 | 0  | 0  | 0  | 0  | 0  | 34 | 0  | 0  | 6  | 39 | 0  | 36 | **37** |
| cube-double ★          | 9  | 0 | 47 | 46 | 2  | 38 | 41 | 45 | 35 | 33 | **74** | 40 | 11 | 64 | 65 | 65 | 32 | 48 | **23** |
| cube-triple ★          | 1  | 0 | 3  | 3  | 0  | 8  | 8  | 8  | 5  | 6  | 1  | 2  | 0  | 3  | 3  | 5  | 2  | 8 | **4**  |
| cube-quadruple-100M    | 9  | 0 | 0  | 2  | 0  | 0  | 1  | 0  | 3  | 19 | 2  | 5  | 2  | 3  | 14 | 6  | 0  | 0 | **51** |
| **all — agg (50)**     | 40 | 11 | 35 | 36 | 8  | 30 | 35 | 37 | 36 | 39 | 38 | 33 | 34 | 44 | 45 | 46 | 33 | 30 | **56** |

---

## Per-category reference hyperparameters

From RQL paper **Table 2 (common)** + **Table 3 (task-specific)**. `α` = BC/flow-BC regularization
weight, `κ` = expectile, `ρ` = ensemble-LCB pessimism, `h` = action-chunk horizon, `γ` = discount.
Common (all tasks): Adam lr 3e-4, batch 256, MLP [512×4] GELU, target-rate 0.005, flow-steps F=10,
ensemble K=10, **2M gradient steps**.

| Category | RQL α | RQL κ | ρ | h | γ | → DQL v10 (ours) |
|---|:--:|:--:|:--:|:--:|:--:|:--|
| antmaze-large          | 0.1 | 0.5 | 0.5 | 1 | 0.99  | e=0.5, ρ=0.5, h=1 |
| antmaze-giant          | 0.1 | 0.5 | 0.5 | 1 | 0.995 | e=0.5, ρ=0.5, h=1, γ=0.995 |
| humanoidmaze-medium    | 0.3 | 0.5 | 0   | 1 | 0.995 | e=0.5, ρ=0, h=1, γ=0.995 |
| humanoidmaze-large     | 0.3 | 0.5 | 0   | 1 | 0.995 | e=0.5, ρ=0, h=1, γ=0.995 |
| scene-sparse           | 3   | 0.7 | 0.5 | 5 | 0.99  | e=0.7, ρ=0.5, h=5, `--sparse` |
| puzzle-3x3-sparse      | 1   | 0.7 | 0.5 | 5 | 0.99  | e=0.7, ρ=0.5, h=5, `--sparse` |
| puzzle-4x4-100M-sparse | 1   | 0.9 | 0.5 | 5 | 0.99  | e=0.9, ρ=0.5, h=5, `--sparse`, 100M ds |
| **cube-double**        | 10  | 0.9 | 0.5 (ρ=0 often better) | 5 | 0.99 | **e=0.9, ρ=0.5, h=5** ← running |
| **cube-triple**        | 1   | 0.9 | 0.5 | 5 | 0.99  | **e=0.9, ρ=0.5, h=5** ← launching |
| **cube-quadruple-100M**| 1   | 0.7 | 0.5 | 5 | 0.99  | **e=0.9, ρ=0.5, h=5** (std play ds first) ← launching |

Notes:
- DQL has **no `α` (flow-BC) knob** — its trust-region analog is the value-selection drift
  (`drift_coef`, `state_bw`, `n_cand`) + `q_coef` (on-manifold improvement). `κ`→`expectile`, `ρ`,
  `h`, `γ` map directly.
- We keep a **single manipulation config (e=0.9, ρ=0.5, h=5)** across all cube tasks for a clean
  apples-to-apples "same DQL" story; RQL's per-task `α/κ` is the *bar* to beat, not our recipe.
- `cube-quadruple-100M` / `puzzle-4x4-100M`: paper uses the 100M-transition dataset (tens of GB via
  `--ogbench_dataset_dir`). We stage the standard `play` dataset first; 100M is a later exact-parity step.

---

## Our runs (fill as evals land)

| Run | Env | agent | key hypers | 250k | 500k | 750k | 1M (final) | vs bar | vs RQL |
|---|---|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| DQL-v10-cube-double    | cube-double-play-singletask-v0    | dql_v10 | h=5, e=0.9, q_pe primary | 0 | 0 | 0 | 0 (0.0 to 600k, killed) | bar=74 | RQL=23 |
| DQL-v10-cube-triple    | cube-triple-play-singletask-v0    | dql_v10 | h=5, e=0.9, q_pe primary | 0 | 0 | 0 | 0 (0.0 to 550k, killed) | bar=8  | RQL=4  |
| DQL-v10-cube-quadruple | cube-quadruple-play-singletask-v0 | dql_v10 | h=5, e=0.9, q_pe primary | 0 | 0 | 0 | 0 (0.0 to 550k, killed) | bar=51 | RQL=51 |
| **DQL-v11-cube-double** | cube-double-play-singletask-v0   | dql_v11 | action-sharp critic (IQL Q + contrast) | ▶ | — | — | — | bar=74 | RQL=23 |
| DQL-v11-cube-triple    | cube-triple-play-singletask-v0    | dql_v11 | action-sharp critic | gated@30k | — | — | — | bar=8 | RQL=4 |
| DQL-v11-cube-quadruple | cube-quadruple-play-singletask-v0 | dql_v11 | action-sharp critic | gated@30k | — | — | — | bar=51 | RQL=51 |

**v10 → killed (flat-value confirmed).** v11 = action-anchored Q + bounded in-support contrast (see `DQL_V11_PROPOSAL.md`).

### v11 findings (corrected)
- **Critic fix works:** rank_acc → 1.0 and holds (v10 was 0.57 and decayed); Vd−Vg calibrated ≈ 0.
- **cube-double = 0.0 through 500k** (real). Deploy ablation (medoid/argmax-Q/robust-Q) all 0.0 → not deploy.
  cube-double-play success density = 0.058% (baselines get 46–74 from it → v11 method gap on sparse manip).
- **antmaze-large validation (h=1, e=0.5): actor WORKS and climbs** — 0.0→0.10(100k)→0.30(175k)→**0.40(225k)**, rising.
  Underperforms RQL (0.76@50k, 0.94@100k) but functional. → actor is not broken; it's weaker/slower than RQL.

| antmaze-large (RQL 0.84 final) | 50k | 100k | 200k | 300k | 400k | 500k | mean≥250k |
|---|--:|--:|--:|--:|--:|--:|--:|
| **DQL-v11** | 0.0 | 0.10 | 0.20 | 0.50 | 0.37 | 0.17 | **0.30 (max 0.50)** |

**v11 FINAL VERDICT (antmaze 500k complete):** actor works but plateaus ~0.30 (max 0.50), **highly unstable**
(0.50→0.00→0.50 between consecutive evals = Q-ascent policy oscillation) — vs RQL 0.84 stable. cube-double 0.0.
→ v11 = real fix over v10 (flat critic solved) but **not competitive with RQL**; remaining weakness = actor
improvement is weak + unstable. The v10→v11 critic-flatness diagnosis stands as an independent finding.

### v11.1 (stability fix) → v11.2 (strength fix) — antmaze
- **v11.1** (advantage drift + damped ascent + eval EMA): stability solved (swing 0.16→0.08) but **capped
  at 0.38** (behavior level — used expectile 0.5).
- **v11.2** (sharper drift adv_temp 0.25 + drift_step 1.5 + **expectile 0.7**): the expectile correction
  broke the cap. Parallel 3-env validation:

| env (RQL) | 25k | 50k | 75k | 100k | vs v11.1 |
|---|--:|--:|--:|--:|--|
| antmaze-large (0.84) | 0.00 | 0.20 | 0.47 | **0.73** | v11.1 was 0.07@75k, 0.38 plateau → **cleared 0.5 gate** |
| antmaze-medium (~1.0) | 0.67 | 0.83 | 0.93 | **0.93** | near RQL-level |
| antmaze-giant (0.37) | 0.00 | 0.00 | 0.00 | 0.00 | no signal (killed @105k to free GPU) |

→ **v11.2 is competitive with RQL on antmaze** (large 0.73 climbing, medium 0.93). Root cause of v11.1's cap
= expectile 0.5 (no improvement signal), confirmed. **Gate cleared → cube-double launched** (h=5, e=0.9, ρ=0.5).

| cube-double (RQL 0.23, bar 74) | v11 | **v11.2** |
|---|--:|--:|
| status | 0.0 through 500k | ▶ running (does the working actor break 0.0?) |

**Notes**
- v10 = one-step drift actor + geometric interpolation-value critic (no learned flow / no reversal),
  marginalized boundary, noise-invariance reg, on-manifold `q_pe` improvement.
- `cube-double` chosen first: it's the sharpest kernel-collapse (RQL 23 ≪ DSRL 74). If v10 climbs and
  **holds ≥ 23 (beat RQL)** we've proven the thesis; **≥ 74 beats all** and takes the category.
- The paper's `cube-double` aggregate (23) is over 5 single-tasks; our singletask-v0 tracks one of them,
  so compare trend/plateau, not the exact aggregate, until we run all 5.
