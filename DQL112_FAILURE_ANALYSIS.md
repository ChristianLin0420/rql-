# DQL v11.2 — Why we lost to RQL: a measured failure analysis

50 tasks × 3 seeds × 2M steps, final protocol score **17.3 vs RQL 55.5** (peak 18.6 at the
1M horizon; W/T/L 1/4/45 at ±3pt). Every claim below was computed from the run data
(`exp/rql-iclr2027-50tasks/DQL112-50/*/eval.csv|train.csv`, 150 runs), the dataset files, or
the maze geometry; each analysis was independently produced and then adversarially
spot-checked (headline numbers reproduced: overall 0.1863@1M / 0.1734@2M; correlations
re-derived within ±0.03). MEASURED vs INTERPRETATION is marked throughout.

**TL;DR — we did not lose to one bug. Five mechanisms, in descending evidence strength:**
the multi-positive attraction silently collapses to single-positive BC on manipulation
(raw-state kNN degeneracy); the contrast–ascent minimax has two confirmed failure modes
(the retracted-P6 blind spot, realized); locomotion fails as a cliff in goal distance
(value-propagation death, not actor pathology); successful tasks decay from early peaks
(equilibrium drift, not instability); and the medoid selector plausibly wastes solved tasks
at deployment. The pre-registered "one-step drift caps out" condition did **not** fire —
every measured failure localizes to a fixable, one-step-compatible component.

---

## 1. Manipulation ≈ 0: the core mechanism is inert there (STRONGEST finding)

The method's central claim — *multi-positive, state-local, advantage-tilted attraction
enables stitching* — is measurably inactive on exactly the families where the benchmark is
lost (scene −81 vs RQL, puzzle-3x3 −85, cube-quadruple −51).

MEASURED:
- `w_self` (attraction weight the state's own action keeps, uniform = 0.031): manipulation
  runs at **0.67–0.86** (puzzle-4x4: 0.857±0.004) vs antmaze **0.31–0.42**. ESS bound
  ≤ 1.4–2.2 of 32 candidates. Fraction of states with w_self > 0.5 ≈ 1.0 on all 20
  worst-family tasks. corr(w_self, success) across 50 tasks: r = −0.41 (p=0.003).
- Cause is geometry, not advantage: a simulation of the agent's exact kNN/kernel
  computation on the raw datasets reproduces the family ordering with **r = 0.966** —
  in high-D raw state space (obs 37–83D, h=5 chunks) the locality penalty hands the
  softmax to the self candidate. Usable neighbors (state-kernel > e⁻²): **1.1–3.8 of 32**
  in manipulation vs 24–29 in antmaze.
- The neighbors that remain are useless: mean pairwise distance between neighbor action
  chunks is **85–96% of random-pair distance** — the residual attraction pulls toward a
  blend of unrelated primitives.
- Consequence: the actor is a behavior-marginal sampler (gen_std/data_std = 1.01–1.06;
  Vd−Vg ≈ 0–0.4% of Vd−Vr) — i.e. BC. At cube-double's success density of **0.058%**,
  BC scores ≈ 0. Density is the amplifier, not the cause (RQL scores 23–100 on the same
  datasets, per its paper).
- rank_acc = 1.000 on all 30 manipulation tasks — saturated and uninformative; the critic
  ranks data-vs-random perfectly while having almost no discrimination *within* the
  data-like region the actor samples from (act_spread/|Q|: cubes 0.016–0.030; within-manip
  corr with success r = +0.68 — association, family-confounded).

INTERPRETATION: on manipulation, v11.2 degenerates to precisely DriftQL's single-positive
premise — our differentiating mechanism never engaged. The S7 review note "raw-state kNN is
the suspect representation" is the best-supported hypothesis in the entire analysis.
w_self also *rises* over training on manipulation (+0.08–0.15), consistent with (but not
proven to be) contrast-loss feedback.

**Decisive cheap test (no retraining):** recompute attraction weights on a 2M cube-double
checkpoint with (i) dataset-wide FAISS kNN instead of the 256-batch pool and (ii)
critic-representation features instead of raw state. If w_self falls to antmaze levels and
targets become mode-coherent, the fix is a representation swap, not a new algorithm.

## 2. The contrast–ascent minimax: retracted P6, now confirmed in two modes

The external review retracted P6 ("no degenerate equilibrium exists") on theoretical
grounds. The data instantiates both predicted failure modes.

MEASURED (margin m = 1.0; hinge is ~entirely driven by generator negatives — random
negatives sit 17–167 Q-units outside it):
- **Mode 1, blind-spot parking — puzzle-4x4 (success 0.000):** Vd−Vg = +1.14/+0.91/+1.00
  ≈ m on all 3 seeds with contrast ≈ 0.15–0.22 (hinge nearly satisfied, signal off) and
  state_spread 0.238 (no state-value gradient to propagate at all). The generator parks
  exactly one margin below data-Q and nothing moves.
- **Mode 2, standing tug-of-war — puzzle-3x3-t1 (worst regressor, family −7.4 1M→2M):**
  contrast never anneals (0.77–0.83 at 2M), Vd−Vg ≤ 0 (the v11 "generator gaming the
  critic" guardrail sign), step-to-step diff-corr(contrast, q_asc) = +0.36..+0.42, and the
  eval slope accelerates from −0.011 to −0.081 per M after 1M.
- Related: the hinge never saturates on regressing families, so the critic's action
  landscape sharpens without bound — Vd−Vr grows ×8 over training (puzzle-3x3-t1:
  21.8→172.3); act_spread/state_spread ("probe ratio") > 1 and growing marks every
  regressing family, ≤ 0.35 in the improving/flat ones (cross-env Spearman with the
  1M→2M delta: −0.87 level, −0.93 growth — **largely a between-family signal**; within
  a family it does not separate the failing sibling).

INTERPRETATION: fixes must be **regime-scoped** — puzzle-4x4 fails with the hinge OFF
(landscape too flat) while puzzle-3x3 fails with it ON (standing conflict), so a blanket
contrast anneal would push manipulation toward the 4x4 regime. Candidates: margin relative
to state_spread, excluding near-data generator samples from the negatives, or a margin on
advantage rather than Q.

## 3. Locomotion bimodality: a cliff in goal distance, not an actor problem

antmaze-large-t2 scores 11 while its sibling t3 scores 82; antmaze-giant t1–4 ≈ 0 while
t5 beats RQL (76 vs 69). Same maze, same dataset, same hypers — only the goal differs.

MEASURED:
- Within-family goal distance (BFS cells, from the ogbench maze maps) predicts success:
  Pearson −0.95 (antmaze-large), −0.92 (giant), −0.97 (humanoid-large); pooled z-scored
  ρ = −0.67 (p=0.0014, n=20 — effective n is closer to the family count).
- RQL degrades **gently** on the same axis (−2.1%/cell, its own per-task pattern has no
  hole); v11.2 turns it into a **cliff** (−10.7%/cell on antmaze-large).
- Failing siblings are internally indistinguishable from succeeding ones — rank_acc
  0.986–0.998, same w_self, same gen_std, same value scale. The failure is invisible to
  dataset-averaged probes: local ranking is fine, the *global* value chain to a far goal
  never forms. This is exactly the corrected P1 story: κ controls recursive propagation,
  and the κ=0.7 one-step expectile chain dies with distance.
- humanoidmaze-medium's post-1M improvement (+4.7) is its two longest-goal tasks still
  climbing at 2M (+1.1–1.4 per 100k) — propagation visibly in progress, ordered by
  measured time-to-goal (Spearman −1.00).
- humanoidmaze-large additionally hits the eval cap: at ~114 steps/cell its t2 needs
  ≈2160 steps against a 2000-step limit — partially infeasible regardless of policy.
- antmaze-giant fails earlier than pure distance predicts (t1–4 at 4.7·H vs the ~7.6·H
  cliff elsewhere); it also has the suite's lowest data density (11.6k transitions/cell)
  — plausible amplifier, but with no within-family variation to test it
  (INTERPRETATION, story-grade).
- Discount is NOT the binding variable (γ^T is the worst pooled predictor, ρ=+0.28 n.s.;
  giant already runs γ=0.995).

**Decisive cheap test:** κ = 0.9 on antmaze-giant-t2 + antmaze-large-t2, 1 seed, 500k
steps. (Note: κ=0.9 alone is not sufficient on manipulation — cube already trains there.)

## 4. Post-peak decay: equilibrium drift, not instability (and not "past 1M" per se)

MEASURED:
- The "1M→2M regression" is really **decay from an early peak** with family-specific
  onset: scene peaks at 0.1–0.3M (loses ~45 pts *before* 1M), puzzle-3x3 at 0.55–0.65M,
  antmaze-large at 1.0–2.0M. All three seeds decline in every regressing task
  (synchronized in sign → mechanism, not noise; fine-timing synchronized only in
  puzzle-3x3, corr 0.47–0.64).
- It is **not** P3's limit cycle returning: eval oscillation amplitude is flat
  (0.075→0.072 family-level), train-side q_asc oscillation *decreases*, |q_asc|/drift
  loss ratio moves +1.7–2.3%. The damped-ascent stability margin holds.
- What does drift: the (EMA) generator slowly slides down the critic's own ranking
  (q_data − q_asc and Vd−Vg grow monotonically) while the action landscape sharpens
  (see §2). Rejected as drivers, with numbers: critic rank degradation (rank_acc flat),
  generator collapse (gen_std flat to <1%), attraction drift (w_self flat on locomotion),
  value-scale inflation (q_data/v_mean constant to 0.3–2%).

INTERPRETATION: a slow monotone drift of the contrast–ascent equilibrium as the landscape
sharpens under a fixed q_coef — plausible transmission, but the chain is unproven (no
per-loss gradient norms are logged). Early stopping at per-task peak is cosmetic:
it buys ≈ +1.3 overall and none of the RQL gap.

## 5. Deployment selector: suggestive damage, cheaply decidable

MEASURED (`viz/figs/f6_selector_results.json`, 2M seed-0 checkpoints, 20 episodes):
argmax-Q_LCB beats the mean-proximal medoid on scene-t1 (0.60 vs 0.30) and
antmaze-large-t1 (0.70 vs 0.55); on cube-double-t1 medoid = kernel-mode = argmax = 0.05
while a **single raw sample scores 0.25**. No single rule wins everywhere, and n=20/seed-0
is not significant (p≈0.2) — unresolved, but selection-averaging plausibly harms
manipulation and the experiment to settle it is ~1 GPU-hour (all seeds, more episodes,
plus logging per-state candidate spread).

---

## Falsification verdict

The pre-registered kill condition (DQL_V11_PROPOSAL.md: "all three levers still ~v11.1 →
the one-step drift caps out; improvement must come from a multi-step generator") **did not
fire**: the Win branch's gate was met (antmaze-large mean(≥250k) = 0.653 ≥ 0.5; v11.1
plateau was 0.38). The pre-registration was locomotion-scoped and one-sided; it never
tested manipulation. Since every measured failure localizes to attraction representation
(§1), contrast coupling (§2), value propagation (§3), or the selector (§5) — all
one-step-compatible — **moving to a multi-step generator is not licensed by the evidence**.

## Prioritized next actions

| # | action | cost | decides |
|---|---|---|---|
| 1 | Count relabeled positive rewards per family at load time (pipeline sanity) | CPU minutes | rules out a data bug behind §1 before architectural conclusions |
| 2 | Recompute attraction weights on saved checkpoints with dataset-wide FAISS kNN and critic-representation features | GPU minutes | §1 root cause; the highest-leverage fix |
| 3 | Q-spread probe over the K=32 generator candidates per state | GPU minutes | §5/§2: if ≈0, medoid and argmax-Q are provably equivalent and inert |
| 4 | Selector ablation, all seeds × ≥100 episodes × 5 rules incl. single-sample | ~1 GPU-hr | §5 |
| 5 | κ=0.9 probe on antmaze-giant-t2 + antmaze-large-t2 (1 seed, 500k) | ~4 GPU-hr | §3 propagation hypothesis |
| 6 | Regime-scoped contrast redesign (margin ∝ state_spread; exclude near-data generator negatives) + per-state advantage normalization + ESS/entropy logging | retraining sweep | §2/§4 |

Sources: four independent analysis briefs + adversarial cross-check (2026-07-16), computed
over the DQL112-50 run data; proposal/gate text quoted from DQL_V11_PROPOSAL.md; selector
data from viz/figs/f6_selector_results.json; stitching evidence F1/F2 artifacts. RQL
reference numbers are from arXiv:2606.17551 appendix Table 1 (no RQL runs exist in this
repo; same-data equivalence is assumed, and its cube-quadruple 51 > cube-triple 4 anomaly
means per-family RQL anchors carry noise).
