# DQL Round 2 — Value-Selection Drift (IDQL-style, gradient-free)

## Problem (from Round 1)
The theory-faithful gradient-free drift is **inert for control**: it estimates the advantage
from the *single* data action, `A = Q(s,a_data) − V(s) ≈ 0.6` on a value scale `|Q| ≈ 184`
(~0.3%), so `exp(A/α)` barely tilts and the drift degenerates to behavior cloning → ~0 success.
The actor and critic are both fine; the **coupling throws away the usable signal**.

## Insight (IDQL / SfBC / EXPO)
The critic *can* rank actions relative to each other even when absolute advantages are tiny
(cube probe: argmax −161 > dataset −164 > random −165). So make the improvement signal
**relative and multi-candidate**: sample many actions, evaluate `Q` on all, reweight by their
*relative* rank. This is a **consistent, stronger estimator of the same** `π*∝μ·exp(A/α)`
(IDQL), and it stays **gradient-free** (only `Q`-evaluations, no `∇_aQ`) → the Wasserstein-flow
theory is preserved (arguably strengthened).

## Method — one contained change to `actor_loss`
Per state `s`, all at `s` (no cross-state kernel needed → simpler than Round 1):
- **Candidate pool** `= {G generator samples ã_g} ∪ {data action a}` (data action anchors in-support).
- Evaluate `q_i = Q(s, cand_i)`; **z-score per state** and `p = softmax(q_norm/α)` — scale-invariant
  implicit-actor weights (advantage normalization is automatic).
- **Attraction** `S_p`: normalized mean-shift toward the value-weighted pool (kernel × `p`) — keeps
  multimodality, pulls toward nearby high-`Q` in-support candidates.
- **Repulsion** `S_q`: normalized mean-shift over generator samples (unchanged) — coverage.
- Drift `V = S_p − S_q`; gradient-free regression `‖gen − sg(gen + drift_step·V)‖²`.

This **removes** the Round-1 machinery: `state_bw`, `adv_logmax`, `adv_wmax`, `k_state` all gone.
Net change is ≈ contained to `actor_loss`; critic (IQL) and eval (argmax-Q) unchanged.

## Plan
- **A (verify):** probe cube 50k — confirm pool weights `p` are non-degenerate and rollout > 0.
- **B (run):** full 1M — cube α-ablation + honest antmaze, vs RQL (auto-launch).
- **C (analyze):** curves vs RQL; win = cube climbs toward/past RQL's 0.38 without collapse.

## Success criteria & honest caveat
Success = cube-double meaningfully > 0 (target ≥ RQL's peak, no collapse). Expected to help
**manipulation strongly** (critic ranks actions there); **antmaze may still lag** (action-flat
critic → ranking is noise) — that's the intrinsic navigation limit, not this change's fault.

## Theory status
Still faithful: multi-candidate reweighting is a better Monte-Carlo estimate of `π*`; drift stays
gradient-free normalized-mean-shift → WGF-toward-`π*` holds.
