# DQL Round 4 — On-Manifold Deployment (avoid critic overestimation by construction)

## Motivation (from the calibration probe)
Rounds 1–3 all reached ~0 on cube. The calibration probe pinned the cause with hard numbers
(config = Round 3: `rho=0.5`, 10-ensemble):

| quantity | value |
|---|---|
| realized discounted return | −198.7 |
| predicted `Q(s0,a0)` | −158.3 |
| **overestimation gap** | **+40.4** |
| gap after LCB (`mean−ρ·std`) | +40.0 (barely moves) |
| ens-std at chosen action | 2.27 |
| ens-std at data action | 1.01 |
| `Q(chosen) − Q(data)` | +11.7 |

Two conclusions:
1. **The critic overestimates by ~40** at the actions our policy deploys — that is exactly why eval is 0.
2. **Ensemble-LCB pessimism cannot fix it** (closes only 0.5 of 40). The overestimation is a
   *shared, correlated bias* across ensemble members on OOD actions (they were all trained by the
   same Bellman backup that never saw those actions' true values), **not** epistemic disagreement
   an ensemble-std can detect.

## Reframing "how RQL avoids overestimation"
RQL's real defense is **not** its LCB — it is that it **keeps the critic on-manifold**: its value
lives on the reverse-flow of *data* actions, it improves *locally* along that flow, and it
**deploys the flow rollout with no argmax-over-samples**. So RQL never queries the critic where
the +40 shared bias lives. Our DQL leaked in exactly one place: the eval `argmax-Q` over free
generator samples, which *actively selected* the OOD action the critic overrates (the +11.7 gap,
std 2.27).

## Why on-manifold is natural for the drift setting
- The drift's **attraction targets ARE the dataset actions** → the policy is pulled onto the
  behavior support *by construction* (RQL must build its manifold; we get it for free).
- The critic is used only to **reweight/select among in-support data actions** — where it is
  reliable (ens-std 1.01, no meaningful bias). Training is already on-manifold.
- The generator is a **distribution-matcher, not a Q-maximizer** → no built-in incentive to
  exploit the critic. The only exploitation was the bolted-on eval `argmax-Q`.

## Method — one contained change (`sample_actions`)
Replace eval `argmax-Q` with **on-manifold deployment**: sample K generator actions and deploy the
**medoid** (the policy mode — minimum total distance to the other samples). No critic query at
deployment → the OOD, overestimated actions are never selected. Improvement is already baked into
the generator by the value-selection drift (which selects among in-support data actions during
training). This is RQL's "deploy the trained policy, don't re-select by Q" defense, adapted to a
one-step generator. Everything else (value-selection drift, IQL critic) is unchanged.

## Plan
- **Gate:** re-run the calibration probe with on-manifold deployment — expect the +40 gap to
  collapse (Q at the deployed medoid ≈ realized return), since the medoid is in-support.
- **Run:** full 1M — cube α∈{1, 0.5} + antmaze, vs RQL.
- **Success:** cube climbs toward/past RQL's 0.38 without collapse.

## Honest caveat
On-manifold caps performance at the *best in-support actions* (no OOD extrapolation) — which is
exactly what offline RL should do, and the RQL/IQL ceiling we target. The two failure causes were
separate: (a) inert selection signal — fixed in Round 2 (relative value-selection, verified
reliable on data actions); (b) the `argmax-Q` OOD leak — fixed here. This round tests them combined.
