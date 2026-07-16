"""ACTION 4: deployment-selector ablation at decisive scale on DQL112-50 2M seed-0 checkpoints.

Same weights, same K=32 target_actor candidates per state; only the selection rule differs.
Episodes are PAIRED across rules: rule r, episode e uses eval_env.reset(seed=BASE+e) and the
same candidate-noise key stream PRNGKey(777*(e+1)), so initial states (and step-0 candidate
sets) are identical across rules -> McNemar exact test vs medoid is valid.

Rules:
  medoid          : argmin_i sum_j d2_ij                (current deployment, = agent.sample_actions)
  argmax_qlcb     : argmax_i mean_E Q - rho*std_E Q     (rho = agent.config['rho'] = 0.5; NOTE this is
                                                         exactly "best-of-32-by-Q with LCB rho=0.5")
  argmax_qmean    : argmax_i mean_E Q                   (rho=0 variant, distinct from LCB)
  kernel_mode     : argmax_i sum_{j!=i} exp(-d2_ij/(2*tau2)), tau2 = tau_scale*mean_offdiag(d2)
                    (sigma from the agent's training tau2 recipe: tau2 = tau_scale * mean sq dist)
  single_sample   : cand[0] (one raw generator sample)

    JAX_PLATFORMS=cpu MUJOCO_GL=disabled python viz/selector_ablation_v2.py --episodes 100

Writes/updates viz/figs/f6b_selector_v2_results.json incrementally (per env x rule).
Success reported two ways: final-step info['success'] (repo eval.csv metric, PRIMARY) and
max-over-episode success (secondary).
"""
import argparse, json, math, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import jax, jax.numpy as jnp
from einops import rearrange, repeat

from viz.load_agent import load_run

RULES = ["medoid", "argmax_qlcb", "argmax_qmean", "kernel_mode", "single_sample"]
OUT = f"{REPO}/viz/figs/f6b_selector_v2_results.json"
BASE_SEED = 1000


def make_candidate_fn(agent, K=32):
    A = agent.config["action_dim"]

    @jax.jit
    def f(obs, key):
        o = jnp.atleast_2d(obs)[-1:]
        cand = agent.network.select("target_actor")(
            repeat(o, "1 o -> k o", k=K), jax.random.normal(key, (K, A)))
        cand = jnp.clip(cand, -1, 1)
        q = agent._Q("target_q", repeat(o, "1 o -> k o", k=K), cand)  # [E, K]
        d2 = jnp.sum((cand[:, None, :] - cand[None, :, :]) ** 2, -1)  # [K, K]
        return cand, q.mean(0), q.std(0), d2

    return f


def pick(cand, qm, qs, d2, rule, rho, tau_scale, K=32):
    if rule == "single_sample":
        i = 0
    elif rule == "medoid":
        i = int(np.argmin(d2.sum(-1)))
    elif rule == "argmax_qlcb":
        i = int(np.argmax(qm - rho * qs))
    elif rule == "argmax_qmean":
        i = int(np.argmax(qm))
    elif rule == "kernel_mode":
        off = d2[~np.eye(K, dtype=bool)]
        tau2 = tau_scale * (off.mean() + 1e-9)
        kern = np.exp(-d2 / (2 * tau2)) - np.eye(K)
        i = int(np.argmax(kern.sum(-1)))
    else:
        raise ValueError(rule)
    return cand[i]


def run_env(env_name, episodes, max_calls, results, rules, out):
    run = load_run(f"exp/rql-iclr2027-50tasks/DQL112-50/{env_name}/dql112__{env_name}__sd0")
    agent, eval_env = run.agent, run.eval_env
    rho = float(agent.config["rho"])
    tau_scale = float(agent.config["tau_scale"])
    h = int(agent.config["h"])
    cand_fn = make_candidate_fn(agent)
    res = results.setdefault(env_name, {"epoch": run.epoch, "rho": rho, "episodes": episodes})
    for rule in rules:
        if rule in res and len(res[rule]["final"]) >= episodes:
            print(f"[skip] {env_name} {rule} already done", flush=True)
            continue
        t0 = time.time()
        finals, maxes, lens = [], [], []
        for ep in range(episodes):
            o, _ = eval_env.reset(seed=BASE_SEED + ep)
            rng = jax.random.PRNGKey(777 * (ep + 1))
            done, calls, info, smax = False, 0, {}, 0.0
            while not done and calls < max_calls:
                rng, k = jax.random.split(rng)
                cand, qm, qs, d2 = cand_fn(jnp.asarray(o), k)
                a = pick(np.asarray(cand), np.asarray(qm), np.asarray(qs),
                         np.asarray(d2), rule, rho, tau_scale)
                a = rearrange(a, "(h d) -> h d", h=h)
                o, r, tm, tr, info = eval_env.step(np.asarray(a))
                smax = max(smax, float(info.get("success", 0.0)))
                done = tm or tr
                calls += 1
            finals.append(float(info.get("success", 0.0)))
            maxes.append(smax)
            lens.append(calls)
        res[rule] = {"final": finals, "max": maxes, "mean_calls": float(np.mean(lens)),
                     "wall_s": round(time.time() - t0, 1)}
        print(f"[f6b-v2] {env_name} · {rule}: final={np.mean(finals):.3f} "
              f"max={np.mean(maxes):.3f} ({res[rule]['wall_s']}s)", flush=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    del run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max_calls", type=int, default=1200)
    ap.add_argument("--envs", type=str, default="cube-double-play-singletask-task1-v0,scene-play-singletask-task1-v0")
    ap.add_argument("--rules", type=str, default=",".join(RULES))
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    for env_name in args.envs.split(","):
        run_env(env_name, args.episodes, args.max_calls, results,
                args.rules.split(","), args.out)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("[f6b-v2] done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
