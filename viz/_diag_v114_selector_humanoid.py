"""URGENT diagnosis: DQL v11.4 humanoidmaze-medium-task1 flat-0 eval @500k.
Selector ablation on one checkpoint: argmax_qlcb (deployed, rho=0.0 -> pure argmax mean-Q)
vs medoid (v11.2 rule) vs single_sample. Paired episodes (same reset seed + candidate key
stream per episode index across rules). Plus Q_LCB/geometry stats over dataset states.

Chunked: --mode stats | episodes (--ep_start/--ep_end/--rules). Appends to OUT json.
"""
import argparse, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import jax, jax.numpy as jnp
from einops import rearrange, repeat

from viz.load_agent import load_run

RUN = ("exp/rql-iclr2027-50tasks/DQL114-50/humanoidmaze-medium-navigate-singletask-task1-v0/"
       "dql114__humanoidmaze-medium-navigate-singletask-task1-v0__sd0")
OUT = f"{REPO}/viz/figs/diag_v114_selector_humanoid.json"
BASE_SEED = 1000
K = 32


def make_candidate_fn(agent):
    A = agent.config["action_dim"]

    @jax.jit
    def f(obs, key):
        o = jnp.atleast_2d(obs)[-1:]
        ok = repeat(o, "1 o -> k o", k=K)
        cand = jnp.clip(agent.network.select("target_actor")(ok, jax.random.normal(key, (K, A))), -1, 1)
        q = agent._Q("target_q", ok, cand)                      # [E, K]
        d2 = jnp.sum((cand[:, None, :] - cand[None, :, :]) ** 2, -1)  # [K, K]
        return cand, q.mean(0), q.std(0), d2

    return f


def pick_idx(qm, qs, d2, rule, rho):
    if rule == "single_sample":
        return 0
    if rule == "medoid":
        return int(np.argmin(d2.sum(-1)))
    if rule == "argmax_qlcb":  # deployed rule (sample_actions temperature=0); rho=0.0 here
        return int(np.argmax(qm - rho * qs))
    raise ValueError(rule)


def load_json():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save_json(d):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(d, f, indent=1)


def do_stats(run, cand_fn, rho, n_states=200):
    ds = run.train_dataset
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds["observations"]), n_states, replace=False)
    obs = np.asarray(ds["observations"][idx])
    rows = dict(qlcb_std=[], qlcb_mean=[], rank_medoid=[], agree=[],
                d2_argmax_to_rest=[], d2_medoid_to_rest=[], d2_argmax_vs_medoid=[],
                d2_mean_offdiag=[])
    for i in range(n_states):
        key = jax.random.PRNGKey(10_000 + i)
        cand, qm, qs, d2 = (np.asarray(x) for x in cand_fn(jnp.asarray(obs[i]), key))
        qlcb = qm - rho * qs
        i_max, i_med = int(np.argmax(qlcb)), int(np.argmin(d2.sum(-1)))
        rank = 1 + int((qlcb > qlcb[i_med]).sum())      # 1 = medoid is best under Q_LCB
        rows["qlcb_std"].append(float(qlcb.std()))
        rows["qlcb_mean"].append(float(qlcb.mean()))
        rows["rank_medoid"].append(rank)
        rows["agree"].append(int(i_max == i_med))
        rows["d2_argmax_to_rest"].append(float(d2[i_max].sum() / (K - 1)))
        rows["d2_medoid_to_rest"].append(float(d2[i_med].sum() / (K - 1)))
        rows["d2_argmax_vs_medoid"].append(float(d2[i_max, i_med]))
        rows["d2_mean_offdiag"].append(float(d2[~np.eye(K, dtype=bool)].mean()))
    s = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                 p90=float(np.percentile(v, 90))) for k, v in rows.items()}
    s["n_states"] = n_states
    s["rank_medoid_hist_1_2_4_8_16_32"] = [int((np.asarray(rows["rank_medoid"]) <= t).sum())
                                           for t in (1, 2, 4, 8, 16, 32)]
    res = load_json()
    res["stats"] = s
    save_json(res)
    print("[stats]", json.dumps(s), flush=True)


def do_episodes(run, cand_fn, rho, rules, ep_start, ep_end, max_calls=2000):
    agent, eval_env = run.agent, run.eval_env
    h = int(agent.config["h"])
    res = load_json()
    for ep in range(ep_start, ep_end):
        for rule in rules:
            r = res.setdefault(rule, {"final": {}, "max": {}, "len": {}})
            if str(ep) in r["final"]:
                continue
            t0 = time.time()
            o, _ = eval_env.reset(seed=BASE_SEED + ep)
            rng = jax.random.PRNGKey(777 * (ep + 1))
            done, calls, info, smax = False, 0, {}, 0.0
            while not done and calls < max_calls:
                rng, k = jax.random.split(rng)
                cand, qm, qs, d2 = cand_fn(jnp.asarray(o), k)
                a = np.asarray(cand)[pick_idx(np.asarray(qm), np.asarray(qs), np.asarray(d2), rule, rho)]
                o, rew, tm, tr, info = eval_env.step(np.asarray(rearrange(a, "(h d) -> h d", h=h)))
                smax = max(smax, float(info.get("success", 0.0)))
                done = tm or tr
                calls += 1
            r["final"][str(ep)] = float(info.get("success", 0.0))
            r["max"][str(ep)] = smax
            r["len"][str(ep)] = calls
            save_json(res)
            print(f"[ep {ep}] {rule}: final={r['final'][str(ep)]} max={smax} len={calls} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    for rule in rules:
        f = list(res.get(rule, {}).get("final", {}).values())
        if f:
            print(f"[agg] {rule}: n={len(f)} final={np.mean(f):.3f} "
                  f"max={np.mean(list(res[rule]['max'].values())):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stats", "episodes"], required=True)
    ap.add_argument("--rules", type=str, default="argmax_qlcb,medoid")
    ap.add_argument("--ep_start", type=int, default=0)
    ap.add_argument("--ep_end", type=int, default=5)
    ap.add_argument("--n_states", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    run = load_run(RUN, epoch=500000)
    rho = float(run.agent.config["rho"])
    print(f"[load] epoch={run.epoch} rho={rho} h={run.agent.config['h']} "
          f"eval_samples={run.agent.config['eval_samples']} ({time.time()-t0:.0f}s)", flush=True)
    cand_fn = make_candidate_fn(run.agent)
    if args.mode == "stats":
        do_stats(run, cand_fn, rho, args.n_states)
    else:
        do_episodes(run, cand_fn, rho, args.rules.split(","), args.ep_start, args.ep_end)


if __name__ == "__main__":
    main()
