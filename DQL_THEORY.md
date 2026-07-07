# Drift Q-Learning (DQL): a one-step generative policy as a Wasserstein gradient flow of the RL free energy

*Working theory note for the ICLR 2027 submission. Companion code: `agents/dql.py`,
`utils/drift_loss.py`, `diagnostics/toy_drift.py`.*

---

## 1. Setup and target policy

Offline RL with dataset `D = {(s,a,r,s')}` drawn from a behavior policy `μ(·|s)`.
We use the KL-regularized policy-improvement objective (per state `s`):

```
max_q   E_{a~q}[ Q(s,a) ]  −  α · KL( q(·|s) ‖ μ(·|s) )                     (1)
```

whose closed-form maximizer is the **value-tilted behavior policy**

```
π*(a|s)  =  (1/Z(s)) · μ(a|s) · exp( A(s,a) / α ),     A = Q − V.           (2)
```

Every offline-RL policy-extraction method (AWR/AWAC/CRR, IQL, Diffusion-QL, FQL, RQL)
is, at heart, an attempt to represent and sample from (2). The differences are only in
the **function class** for the policy and the **operator** used to push it toward `π*`.
DQL's thesis: use a **one-step generator** as the function class and a **gradient-free
drift** as the operator.

## 2. The policy class: a one-step generator

`a = π_θ(s, ε)`, `ε ~ N(0, I)` — a single forward pass. Its pushforward `q_θ(·|s)` is an
implicit distribution (no density, no time variable, no ODE). This class is as expressive
and multimodal as a diffusion/flow policy, but inference is O(1) network calls.

## 3. The improvement operator: drift toward `π*`

For a distribution `q` and target `p`, the **drift field** (kernel mean-shift; `drift_loss.py`)

```
V_{p,q}(x) = E_{y⁺~p}[ k(x,y⁺)(y⁺−x) ]  −  E_{y⁻~q}[ k(x,y⁻)(y⁻−x) ],   k(x,y)=exp(−‖x−y‖/τ)   (3)
```

is a consistent KDE estimator of the (kernel-smoothed) **score difference**

```
V_{p,q}(x) ≈ c · ∇_x log( p(x) / q(x) )    (as τ→0, up to normalization).   (4)
```

Take `p = π*(·|s)`. Substituting (2) and using `∇log π* = ∇log μ + ∇A/α`:

```
V_{π*,q}(x) ≈ c · ∇_x [ log μ(x|s) − log q(x|s) + A(s,x)/α ].              (5)
```

This is exactly the negative first variation of the free energy in (1):
with `F_s(q) = −E_q[Q] + α·KL(q‖μ)`, we have `δF_s/δq = −A + α·log(q/μ) + const`, so

```
V_{π*,q}  ∝  −∇_x ( δF_s/δq )(x).                                          (6)
```

**Therefore the drift dynamics `x ← x + V_{π*,q}(x)` is a (kernelized) Wasserstein
gradient flow that minimizes the RL free energy `F_s`, whose unique minimizer is `π*`.**

**Equilibrium.** `V_{p,q} = 0` for all `x` iff `q = p = π*` (drift anti-symmetry
`V_{p,q} = −V_{q,p}`; identifiability under the usual non-degeneracy of the kernel
Gram operator). So the only fixed point of DQL's actor update is the target policy (2).

## 4. Realizing the flow with a one-step net (no backprop through sampling)

We do not integrate (3) at inference. Instead we regress the generator to its own drifted
output — the **fixed-point / "drifting" objective**:

```
L_actor(θ) = E_{s,ε} ‖ π_θ(s,ε) − stopgrad( π_θ(s,ε) + V_{π*,q_θ}(π_θ(s,ε)) ) ‖²
           = E_{s,ε} ‖ V_{π*,q_θ}(π_θ(s,ε)) ‖².                            (7)
```

Each SGD step moves the *pushforward distribution* `q_θ` one drift-step toward `π*`; the
inference-time iteration of diffusion/flow is amortized into training-time iteration.
`∇_θ` never passes through a sampling chain (the target is `stopgrad`), so there is **no
backprop-through-sampling and no flow inversion**.

### Practical estimator of `π*` (what `agents/dql.py` computes)
`π*` is only implicit (we have `Q`, not samples from `π*`). We estimate the attraction
term of (3) by **advantage-weighting in-support data actions**, with a state kernel to
make it conditional:

```
weight_pos(s_b, a_j) = k_state(s_b, s_j) · exp( A(s_j, a_j) / α ),         (8)
```

renormalized to row-mean 1. Feeding (8) as `weight_pos` to `drift_loss` (whose repulsion
term is the generator's own samples) yields an estimator of `V_{π*,q}` **without ever
querying `∇_a Q`** — avoiding adversarial OOD action gradients. Because
`∇log(μ·e^{A/α}) = ∇log μ + ∇A/α`, (8) is a sample-based, gradient-free estimator of the
same field an explicit `∇_a Q` term would give (§3).

## 5. Critic (policy evaluation)

Standard IQL (implemented): `V(s)` expectile-regresses `Q_target(s,a)`; `Q(s,a)` regresses
the `h`-step target `Σγ^i r_i + γ^h V(s')`. No policy appears in the TD target ⇒ the critic
backup is a contraction independent of the (moving) actor — a clean base timescale.

## 6. Why this removes the RQL kernel pathology (empirically diagnosed)

RQL uses a **multi-step flow** actor, so to anchor its flow-value to data it must **invert**
the flow (reversal) and define a value over flow-time `f∈[0,1]`. We measured three broken
assumptions (worse ~15× for action chunks): (i) reverse-Euler ≠ true inverse (≈12–17%
reconstruction error); (ii) the `f=0` value stays noise-seed-dependent so the single-sample
bootstrap `V(s',x₀',0)` is mis-grounded (gap grows 7×); (iii) value is not flow-time
invariant (drift up to ~0.5). Behaviorally, RQL **learns then collapses** on chunked
manipulation (cube-double: peak ≈0.38 → ≈0.2).

DQL's actor is **one-step**, so *none of these objects exist*: no reversal, no flow-time,
no `f=0` boundary. Anchoring-to-data is just the attraction term (8); improvement is the
value-tilting; multimodality is the repulsion. The failure modes are structurally absent.

## 7. Convergence (two-timescale sketch)

Critic on the fast timescale (TD contraction, IQL) → `Q_k → Q^{π_k}`. Actor on the slow
timescale follows the WGF of `F_s` for the current `Q_k` (§3), a geodesically-convex
functional in `q` (KL + linear reward term) ⇒ each actor phase contracts toward `π*_k`.
Standard two-timescale stochastic-approximation conditions (`Σβ=∞, Σβ²<∞, β_actor/β_critic→0`)
give convergence to the fixed point `(Q^{π*}, π*)` of (2). Making the constants explicit
(kernel bandwidth `τ`, temperature `α`, state-kernel bias) is the theorem to prove.

## 8. Positioning: a new point in the design space

| Method | Expressive / multimodal | One-step inference | Improvement operator | In-support |
|---|---|---|---|---|
| AWR / AWAC / CRR | ✗ (Gaussian) | ✓ | weighted regression | ✓ |
| Diffusion-QL / IDQL | ✓ | ✗ (T steps) | backprop-through-sampling / resample | ✓ |
| FQL / RQL | ✓ | ✗ (flow; distill/invert) | flow-value + reversal | ✓ (breaks) |
| **DQL (ours)** | **✓** | **✓** | **gradient-free drift to `π*` (WGF)** | **✓** |

DQL is the first to be *expressive + one-step + gradient-free-improvement + in-support*
simultaneously, with a clean variational identity (`π*` = free-energy minimizer =
drift fixed point).

## 9. Assumptions, risks, and what the experiments must show
- **State-conditioning of the kernel** (8): in-batch `k_state × exp(A/α)`; escalate to
  NN-retrieval or a state-cluster memory bank if positives are too sparse (mirrors
  `memory_bank.py`). Bias of the state kernel is the main approximation.
- **Temperature `α`**: the toy (`toy_drift.py`) confirms `α` controls the
  improvement↔coverage tradeoff exactly as (2) predicts (moderate `α`: multimodal match,
  small `α`: greedy). Too-small `α` over-concentrates — a knob, not a bug.
- **Double moving target**: mitigated by EMA target critic + expectile (no policy in TD).
- **Headline empirical claim**: on the *same* OGBench tasks where RQL collapses
  (cube-double/-triple, puzzle), DQL should **not collapse** and match/exceed RQL, with
  antmaze as sanity — a direct, apples-to-apples test in this repo.
```
