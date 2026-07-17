# DQL v11.3 — Fix the measured extraction failures (spec + pre-registration)

Status: **50-task sweep RUNNING** (group `DQL113-50`, 3 seeds × 1M steps, launched
2026-07-16 09:40; commit `428cc85`). This version changes nothing about the critic or the
v11.2 stability/strength machinery — it repairs the two components that the v11.2
post-mortem (`DQL112_FAILURE_ANALYSIS.md`) proved broken by direct measurement on saved
checkpoints, plus one dataset-pipeline bug.

## S1 — What v11.2 taught us (evidence, condensed)

The v11.2 sweep scored 18.6 @1M / 17.3 @2M vs RQL 55.5 (W/T/L 1/5/44 at 1M). Failure
analysis over the 150 runs' diagnostics + checkpoint probes localized the loss to:

1. **Attraction starvation on manipulation.** The M=32 kNN candidates came from the
   256-state training batch; in manipulation state spaces that yields only **1–4 usable
   neighbors**, the locality softmax collapses onto the self candidate
   (w_self 0.67–0.86 vs antmaze 0.31–0.42), and the drift degenerates to single-positive
   BC — which scores ≈0 at 0.04–0.26% success density. Checkpoint probe: replacing the
   batch pool with a static **100k-state dataset-wide pool** drops cube-double w_self
   **0.671 → 0.173** with 31.8/32 usable, action-coherent neighbors. Pool density, not a
   representation problem: critic-feature kNN was strictly worse (0.266).
2. **Deployment averaging.** The medoid selector picks the sample nearest the candidate
   mean — a between-modes compromise on multimodal generators. Paired 100-episode
   ablation on 2M checkpoints: **argmax-Q_LCB 0.96 vs medoid 0.41** on puzzle-3x3-task1,
   scene 0.44 vs 0.33, never worse anywhere; kernel-mode ≈ medoid and single-sample worst
   (both refuted). Where the critic cannot separate candidates (cube: Q-spread ≈1% of
   Vd−Vr), rules tie — selector fixes extraction only where critic signal exists.
3. **puzzle-4x4 had no training signal at all** (pipeline bug, not algorithm): its
   datasets contain zero r=0 transitions, so `sparse=1`'s `(r != 0) * −1` produced a
   constant reward. Fixed: `slurm/gen_tasks.py` now runs puzzle-4x4 with `sparse=0`.

Not addressed in v11.3 (deliberately): the slow post-peak decay driven by the
never-saturating contrast hinge (landscape sharpening ×8 over training) — that is the
candidate v11.4 change (relative margin / negative-set redesign), kept out of v11.3 so the
two extraction fixes stay cleanly attributable. Mitigation: v11.3 trains 1M steps, before
the decay regime dominated v11.2.

## S2 — The changes (complete list)

| # | component | v11.2 | v11.3 |
|---|---|---|---|
| 1 | attraction candidates | batch-kNN (M=32 of 256) | **self + top-31 of a static 100k dataset-wide pool** (per-run draw through the training sampler, deterministic under the run seed; requeues rebuild it identically, checkpoints carry it) |
| 2 | locality bandwidth stat | mean batch-pairwise sq-dist | mean batch×pool sq-dist (measured 1.004–1.008× the old stat — tuning carries over) |
| 3 | deployment | medoid over K=32 EMA-actor samples | **argmax Q_LCB** over the same K=32 (clip before scoring) |
| 4 | run flags | puzzle-4x4 `sparse=1` | `sparse=0` (semi-dense −#unsolved reward) |

Everything else — critic (IQL h-step + expectile V + margin contrast), advantage-tilted
softmax (τ_adv 0.25), drift step η 1.5, damped ascent c_q 0.25, EMA λ 0.005, per-family
expectile/ρ/h/γ — is byte-identical to v11.2. New config key: `n_pool=100_000`.
Compute: one [256×100k] distance matmul + top-k per update (≈1% of the critic pass);
checkpoints grow ~25–70MB from the stored pool.

## S3 — Pre-registered expectations (written before results)

- **E1 (mechanism):** manipulation w_self < 0.4 from early training (probe predicts
  0.10–0.22). *Confirmed live at 09:57, first logs: scene 0.124, puzzle-3x3 0.118,
  puzzle-4x4 0.220, cube-double 0.129, cube-quad 0.199.*
- **E2 (scene/puzzle):** large gains over v11.2 (pool restores borrowing; selector alone
  was worth +55pts on puzzle-3x3-t1). Falsified if scene/puzzle-3x3 land ≤ v11.2
  (11.6 / 22.0 @1M) — that would mean multi-positivity is not the binding constraint.
- **E3 (cube):** bounded gains — the critic's on-manifold blind spot (candidate Q-spread
  ≈1% of Vd−Vr) is untouched; meaningful cube gains would imply the pool also sharpened
  on-manifold Q discrimination (log it as a bonus finding, don't claim it).
- **E4 (locomotion):** at least v11.2-level; the heavier borrowing (antmaze w_self
  0.42 → ~0.12) changes dynamics, so slower ramps are acceptable if the 500k+ level holds.
- **E5 (decay):** unchanged mechanism, but 1M horizon should end runs at/near peak; if
  strong decay appears *before* 1M on v11.3 where v11.2 held, the pool interacted with
  the contrast equilibrium — investigate before v11.4.
- **E6 (puzzle-4x4):** any nonzero learning validates the sparse fix (v11.2: exactly 0.0
  with a constant-reward critic).

## S4 — Live observations (updated as the sweep runs)

- 09:57 — all 50 jobs running, 0 errors, ~75 it/s on cube (no top_k slowdown); E1 confirmed.
- 12:00 — seed-0: puzzle-3x3-t1 **100 from 400k, flat** (v11.2 peaked 94@450k → 67@1M);
  antmaze-large-t1 climbing 88@750k with no decay (v11.2 peaked 88@400k then fell);
  scene-t1 62@600k rising (v11.2 falling through ~50 here); cube-double ~2–10 (per E3).
- 13:02 — no decay through 950k: antmaze-large-t1 ~80–86, scene-t1 ~74, puzzle-3x3-t1 100.
- 18:07 — 65/150 seeds at 1M, 0 errors, all requeue chains healthy.

## S5 — Results

`RESULTS_DQL113.md` (generated when all 150 seeds reach 1M; ETA 2026-07-17 morning).
Dashboard: `viz/report/index.html` (add `("dql113", "DQL113-50", "DQL v11.3")` to
`VARIANTS` in `viz/build_data.py` and rebuild).

## S6 — Sources

`DQL112_FAILURE_ANALYSIS.md` (five measured mechanisms + probe updates);
`diagnostics/knn_probe_action2.py` (pool probe), `diagnostics/qspread_probe.py`
(blind-spot probe), `viz/selector_ablation_v2.py` (paired selector grid);
`agents/dql_v11_3.py` (implementation, adversarially reviewed pre-launch).
