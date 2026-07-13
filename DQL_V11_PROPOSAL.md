# DQL v11 — Action-sharp critic (anchor the value at the action, contrast in-support vs OOD)

![DQL v11](assets/dql_v11.svg)

*Figure. **Left:** the measured v10 defect — training `V(s, x_τ, τ) → y` for random `x` at every
`τ<1`, where the target `y` is independent of `x`, teaches the critic to **ignore the action**
(flat value; `rank_acc ≈ 0.55`, `ratio ≈ 0.02`). **Right:** v11 replaces the flat interpolation with
an **action-anchored `Q(s,a)`** (IQL transition target → action-informative by construction) plus a
**bounded in-support contrast** that lifts the data action above OOD/random by a margin `m`
(`rank_acc → 1`), a **noise-free state value `V(s)`**, and a **trust-region one-step drift** actor that
ascends the now-sharp `Q` while staying in-support.*

## Why (the diagnosis that forced this)
A 50k probe run (`probe/rank_acc`, `probe/ratio = act_spread/state_spread`) confirmed v10's critic is
**structurally action-flat and gets worse over training**:

| step | rank_acc | ratio | Vd−Vr |
|---:|---:|---:|---:|
| 17.5k | 0.63 (peak) | 0.086 | +0.65 |
| 50k | 0.57 | **0.022** | +0.19 |

`ratio → 0.02` means `V` varies ~45× more across **states** than **actions**. Root cause: v10's
geometric interpolation trains the value at random `x` for all `τ<1` toward an `x`-independent target,
which **actively regularizes the network to be action-invariant**. The τ=1 action signal is
measure-zero against it. A critic that can't rank actions gives the policy no gradient → **0.0 success
through 600k** on cube-double/triple/quadruple.

## What changes (v11)
Keep everything that was **healthy** in v10 (no overestimation, no mode-collapse, no actor drift,
one-step deploy) and fix the **one** broken thing (action-informativeness):

1. **Drop the interpolation coordinate.** The critic is a genuine action value `Q(s,a)` trained with the
   **IQL transition target** `Q(s,a) ← r + γ^h V(s')`. Because the target is transition-specific, `Q` is
   action-informative *by construction* (the same reason IQL/RQL critics are).
2. **Bounded in-support contrast (the anti-flatness lever).** A margin hinge
   `L_c = mean relu( Q(s, a⁻) − Q(s, a_data) + m )` over negatives `a⁻ ∈ {random, generator}` forces
   `Q(s, a_data) ≥ Q(s, a⁻) + m` → **directly lifts `rank_acc`** and simultaneously **pushes OOD `Q`
   down** (so the actor's Q-ascent cannot run away into OOD — the failure that made earlier CQL brittle).
   The hinge is bounded (caps at `m`), so no runaway/divergence.
3. **Noise-free state value `V(s)`** as its own head: `V(s) ← expectile_κ(Q(s, a_data))`. No
   marginalization trick needed (that existed only to extract a state value from the interpolation net).
4. **Trust-region one-step drift actor (unchanged from v10):** value-selection drift toward top-M
   in-support data + repulsion, plus bounded Q-ascent — now driven by the **action-sharp `Q`**. Deploy =
   one forward pass (medoid).

## Success criterion / falsification (measurable *before* the eval lands)
- **Leading indicator (new):** `probe/rank_acc` must climb toward **≥ 0.8** and `probe/ratio` must
  **grow** (critic uses the action). If the contrast can't lift `rank_acc` within ~30k without `Q`
  diverging, the flatness is not fixable by a bolt-on contrast and the deeper coupling (RQL-style
  flow) is required — we learn that in 30 min, not 12 h.
- **Outcome:** cube-double breaks off 0.0 and **holds ≥ 23 (beat RQL)**; stretch **≥ 74 (beat all)**.
- **Guardrail:** watch `Vd−Vg` (was negative in v10 = generator gaming a flat critic). It should go
  **positive** (data ≥ generator) once the contrast bites.

## Plan
- Relaunch cube-double first; **gate on `probe/rank_acc` at 30k** before committing cube-triple /
  cube-quadruple. If the gate passes, launch all three (GPU has room) to 1M with 50k evals.

---

# v11 results & the v11.1 actor fix

## What v11 actually showed (measured)
The gate passed decisively and the critic fix held all the way — **but the eval told a two-part story**:

| | critic (probe) | eval outcome |
|---|---|---|
| **v11 critic** | `rank_acc → 1.0` and **holds**; `Vd−Vg ≈ 0` (calibrated) | — |
| **cube-double** | sharp | **0.0 through 500k** (RQL 23). Deploy ablation (medoid / argmax-Q / robust-Q) **all 0.0** → not deploy. |
| **antmaze-large** | sharp | **climbs to ~0.30 mean (max 0.50) but wildly unstable** (0.50→0.00→0.50 between evals) vs RQL 0.84 stable. |

**Diagnosis:** the v10→v11 **critic** fix is real and validated. The remaining failure is entirely in the
**drift actor**: (a) it does *weak* improvement — the value-selection drift kernel-**averages** the top-M
candidates, blurring the best action into a mediocre mean; and (b) it is *unstable* — the raw Q-ascent
(`q_coef=1.0`) keeps pushing the generator into exploitable critic directions, and with **no eval EMA**
the deployed policy oscillates. It is **not** a critic problem and **not** a deploy problem.

## v11.1 — stronger, stable drift actor (still drift-based)

![DQL v11.1](assets/dql_v11_1.svg)

*Figure. The **critic is identical to v11**. Three changes to the **drift** actor, each targeting one measured
failure: **①** the drift target is **advantage-weighted** — attraction weights `w = softmax((Q−V)/τ)` with a
sharp `τ=0.5`, so the drift peaks toward the **argmax-advantage in-support action** instead of the blurred
Q-mean (real improvement, still a drift; multimodality preserved by the per-sample kernel + repulsion). **②**
the **Q-ascent is damped** (`q_coef 1.0 → 0.25`, on the lagged `target_q`) so the stable in-support drift
dominates and can't oscillate the policy. **③** deploy from a **slow EMA of the generator** (`actor_ema=0.005`)
— the RQL paper's own stability trick, which v11 lacked — removing the eval-to-eval swing.*

### Changes (concise)
1. **Advantage-weighted drift (badge ①).** `v_state = V(s)`; `adv = Q(s,a_cand) − v_state`; scale-free
   `adv_n = adv / mean|adv|`; `w = softmax(adv_n / adv_temp − sq_sel / bw)`, `adv_temp = 0.5`. Replaces v11's
   `softmax(q_normalized / alpha − …)`. Same drift machinery (per-sample kernel attraction + repulsion),
   just a **peaked, advantage-based** target.
2. **Damped Q-ascent (badge ②).** `actor_loss = drift_coef·drift_loss − q_coef·q_asc`, `q_coef = 0.25`
   (was 1.0), `q_asc` on `target_q` (stop-grad params, grad → gen only).
3. **Eval EMA (badge ③).** New `target_actor` module, updated each step by
   `θ̄ ← 0.005·θ + 0.995·θ̄`; `sample_actions` deploys `target_actor` (medoid of K).
4. **Critic unchanged** — IQL Q + bounded in-support contrast + noise-free `V(s)`.

### Success criterion / falsification (vs v11's own numbers)
- **Win:** antmaze **plateau ≳ 0.5 AND smooth** (consecutive-eval swing ≪ v11's ±0.5) — the fixes bought
  both strength and stability; then it earns a cube-double run.
- **Partial:** stability improves (EMA works) but plateau stays ~0.3 → the drift is *stable but still weak*;
  next lever is `adv_temp`↓ / `drift_step`↑ (sharper/stronger pull).
- **Falsified:** no better than v11 → the one-step drift actor caps out here and the improvement must come
  from a multi-step generator (i.e. the RQL-style capacity we set out to avoid).

### Plan
- **Validate on antmaze first** (h=1, expectile=0.5) against v11's 0.30/unstable baseline — cheap, ~decisive.
- Only if antmaze clears the win bar do we spend GPU on cube-double.

---

# v11.1 results & the v11.2 strength lever

## What v11.1 showed (measured, antmaze-large vs v11)
| step | v11 | **v11.1** |
|---:|---:|---:|
| 100k | 0.10 | 0.17 |
| 150k | 0.10 | 0.43 |
| 250k | 0.10 | 0.43 |
| 300k | 0.50 | 0.40 |
| 350k | 0.37 | 0.27 |
| **mean ≥250k** | 0.30 | **0.38** |
| **swing (std, ≥100k)** | 0.162 | **0.082** |

**Split result:** ✅ **stability fully solved** (swing halved, no crashes — the damped ascent + EMA worked)
but ⚠️ **strength only partial** — v11.1 plateaus **stable-but-weak at ~0.38** and does **not** clear 0.5
(RQL 0.84). The advantage drift converges to a modest in-support policy and stops improving.

## v11.2 — strength levers (still drift-based)

![DQL v11.2](assets/dql_v11_2.svg)

*Figure. v11.2 keeps every stability mechanism from v11.1 and pushes **strength** three ways. **Left:** a
**sharper** advantage weight (`adv_temp 0.5→0.25`, peaks ~0.9 on the argmax-advantage action vs ~0.7) and a
**stronger** pull (`drift_step 1.0→1.5`), so the generator is dragged decisively onto the best in-support
action instead of a soft blend. **Right-top — the bigger lever:** v11/v11.1 antmaze used **expectile 0.5**
(copied from RQL's flow-value regime) = a **behavior value** with no max, hence **no improvement signal** —
which is almost certainly why they capped at behavior level ~0.38. Our own **v10 ablation** proved
`κ=0.5→0.02` (fails) vs `κ=0.9→0.22` (unlocks). An IQL-style critic needs an **optimistic expectile**;
v11.2 antmaze uses **`κ=0.7`**. **Right-bottom:** validated on **3 antmaze envs in parallel**.*

### Changes (concise)
1. **Sharper advantage drift:** `adv_temp` 0.5 → **0.25** (code default).
2. **Stronger pull:** `drift_step` 1.0 → **1.5** (code default).
3. **Expectile correction:** antmaze runs at **`expectile 0.7`** (run flag; v11/v11.1 used 0.5) — the
   likely dominant fix, restoring an improvement signal to the IQL-style value.
4. **Stability machinery unchanged** — damped Q-ascent (`q_coef=0.25`) + eval EMA (`actor_ema=0.005`).

### Parallel 3-env validation
All `h=1, ρ=0.5, expectile=0.7, seed 0, 500k`:
- **antmaze-medium** — easier stitching (RQL high) → sanity that v11.2 can reach *strong*, not just 0.4.
- **antmaze-large** — the v11.1 head-to-head baseline (0.38): **does v11.2 clear 0.5?**
- **antmaze-giant** — hard long-horizon (RQL 0.37) → stress test of the stronger pull.

### Success / gate
- **Win:** antmaze-large **mean(≥250k) ≥ 0.5** (and medium clearly strong) → **launch cube-double next**.
- **Partial:** medium strong but large still ~0.4 → the cap is large-specific (harder stitching), not the recipe.
- **Falsified:** all three still ~v11.1 → strength doesn't come from these levers; the one-step drift caps out.
