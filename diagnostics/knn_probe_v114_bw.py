"""v11.4 locality-formula probe.

Reproduces dql_v11_3's candidate selection (self + top-31 raw-state neighbors from a
100k dataset-wide pool, exactly knn_probe_action2.py variant (b)) on DQL112-50 2M
checkpoints and computes the resulting attraction w_self under candidate bandwidth
formulas:
  F1: bw = c * mean(sq_sel_nonself)              [batch-level mean of 31 selected sq-dists]
  F2: bw = c * per-state median(sq_sel_nonself)  [row-wise normalizer]
  F3: bw = c * mean(sq_pool)                     [v11.3 form, tiny c, reference]
w = softmax(adv_n / adv_temp - sq_sel / bw),  adv_temp = 0.25 (training value).

Also reports per-family sq_sel distributions (mean/median/p90 nonself, mean sq_pool)
and neighbor-action coherence (top-8 weighted candidates pairwise chunk L2 / random).
"""
import json
import os
import sys
import time

import numpy as np

REPO = "/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/rql-"
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp

from ogbench.utils import load_dataset

from agents.dql_v11_2 import DQLv11_2Agent, get_config
from utils.flax_utils import restore_agent

N_PROBE_BLOCKS = 8      # 8 x 256 = 2048 probe states
BLOCK = 256
M = 32                  # n_cand (self + 31)
POOL_SIZE = 100_000
TOPW = 8
CKPT_STEP = 2_000_000

# per-family flags from slurm/tasks.tsv
ENVS = {
    "antmaze-large-navigate-singletask-task1-v0":      dict(h=1, expectile=0.7, rho=0.5, discount=0.99),
    "antmaze-giant-navigate-singletask-task1-v0":      dict(h=1, expectile=0.7, rho=0.5, discount=0.995),
    "humanoidmaze-medium-navigate-singletask-task1-v0": dict(h=1, expectile=0.7, rho=0.0, discount=0.995),
    "scene-play-singletask-task1-v0":                  dict(h=5, expectile=0.7, rho=0.5, discount=0.99),
    "puzzle-3x3-play-singletask-task1-v0":             dict(h=5, expectile=0.7, rho=0.5, discount=0.99),
    "cube-double-play-singletask-task1-v0":            dict(h=5, expectile=0.9, rho=0.5, discount=0.99),
}

C_F12 = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
C_F3 = [0.001, 0.003, 0.01, 0.03]
SETTINGS = [("F1", c) for c in C_F12] + [("F2", c) for c in C_F12] + [("F3", c) for c in C_F3]


def build_agent(env_name, overrides):
    """Load dataset without creating the mujoco env (same as knn_probe_action2.py)."""
    config = get_config()
    config.update(overrides)
    splits = env_name.split("-")
    pos = splits.index("singletask")
    dataset_name = "-".join(splits[:pos] + splits[-1:])
    path = os.path.join(os.environ["OGBENCH_DATASET_DIR"], f"{dataset_name}.npz")
    ds = load_dataset(path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False)
    ds["actions"] = np.clip(ds["actions"], -1 + 1e-5, 1 - 1e-5)
    h = config["h"]
    obs_dim, act_dim = ds["observations"].shape[1], ds["actions"].shape[1]
    ex_obs = jnp.zeros((h + 1, 1, obs_dim))
    ex_act = jnp.zeros((h + 1, 1, act_dim))
    agent = DQLv11_2Agent.create(0, ex_obs, ex_act, config)
    ckpt_dir = f"{REPO}/exp/rql-iclr2027-50tasks/DQL112-50/{env_name}/dql112__{env_name}__sd0"
    agent = restore_agent(agent, ckpt_dir, CKPT_STEP)
    return agent, ds


def make_fns(agent):
    rho = agent.config["rho"]

    @jax.jit
    def qlcb_target(s, a):
        q = agent.network.select("target_q")(jnp.concatenate([s, a], -1))
        return q.mean(0) - rho * q.std(0)

    @jax.jit
    def v_mean(s):
        return agent.network.select("v")(s).mean(0)

    return qlcb_target, v_mean


def batched(fn, *arrays, bs=8192):
    outs = []
    n = arrays[0].shape[0]
    for i in range(0, n, bs):
        outs.append(np.asarray(fn(*[a[i:i + bs] for a in arrays])))
    return np.concatenate(outs, 0)


def chunk_actions(actions, idxs, term_of, h):
    cols = [actions[np.minimum(idxs + i, term_of)] for i in range(h)]
    return np.concatenate(cols, -1)


def sqdist(X, Y, bs=256):
    y2 = (Y ** 2).sum(-1)
    out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs]
        out[i:i + bs] = (xb ** 2).sum(-1)[:, None] + y2[None] - 2.0 * xb @ Y.T
    return np.maximum(out, 0.0)


def softmax(logits):
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def run_env(env_name, overrides):
    print(f"\n=== {env_name} (h={overrides['h']}, rho={overrides['rho']}) ===", flush=True)
    t0 = time.time()
    agent, ds = build_agent(env_name, overrides)
    qlcb_target, v_mean = make_fns(agent)
    cfg = agent.config
    h, adv_temp = cfg["h"], cfg["adv_temp"]

    obs = np.asarray(ds["observations"], dtype=np.float32)
    acts = np.asarray(ds["actions"], dtype=np.float32)
    terminals = np.asarray(ds["terminals"])
    term_locs = np.nonzero(terminals > 0)[0]
    initial_locs = np.concatenate([[0], term_locs[:-1] + 1])
    term_of_all = term_locs[np.searchsorted(term_locs, np.arange(len(obs)), side="left")]
    print(f"dataset: {len(obs)} transitions, obs_dim={obs.shape[1]}, act_dim={acts.shape[1]} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(0)
    valid_traj = np.flatnonzero(term_locs - initial_locs)
    n_probe = N_PROBE_BLOCKS * BLOCK
    tr = rng.choice(valid_traj, n_probe, replace=True)
    starts = rng.randint(initial_locs[tr], term_locs[tr])
    s_probe = obs[starts]
    c_probe = chunk_actions(acts, starts, term_of_all[starts], h)

    valid_mask = np.arange(len(obs)) < term_of_all
    valid_idx = np.flatnonzero(valid_mask)
    valid_idx = np.setdiff1d(valid_idx, starts)
    pool_idx = rng.choice(valid_idx, POOL_SIZE, replace=False)
    s_pool = obs[pool_idx]
    c_pool = chunk_actions(acts, pool_idx, term_of_all[pool_idx], h)

    ri = rng.choice(len(c_pool), (16384, 2))
    rand_ref = float(np.sqrt(((c_pool[ri[:, 0]] - c_pool[ri[:, 1]]) ** 2).sum(-1)).mean())

    v_probe = batched(v_mean, s_probe)

    iu = np.triu_indices(TOPW, 1)
    acc = {key: {"w": [], "coh": [], "use": []} for key in SETTINGS}
    dsel_all, pool_means = [], []

    for b in range(N_PROBE_BLOCKS):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        s0, c0, v0 = s_probe[sl], c_probe[sl], v_probe[sl]
        # ---- variant (b) candidate selection: self + top-31 of the 100k pool
        dq = sqdist(s0, s_pool)                               # [256, 100k]
        nn = np.argpartition(dq, M - 1, -1)[:, :M - 1]
        dsel = np.take_along_axis(dq, nn, -1)
        order = np.argsort(dsel, -1)
        nn = np.take_along_axis(nn, order, -1)
        dsel = np.take_along_axis(dsel, order, -1)            # [256, 31] nonself sq dists
        sq_sel = np.concatenate([np.zeros((BLOCK, 1), np.float32), dsel], -1)  # [256, 32]
        chunks_cand = np.concatenate([c0[:, None, :], c_pool[nn]], 1)          # [256, 32, hA]
        dsel_all.append(dsel)
        pool_means.append(dq.mean())

        # ---- advantage term (bw-independent), exact training computation
        q_im = batched(qlcb_target,
                       np.repeat(s0, M, 0),
                       chunks_cand.reshape(BLOCK * M, -1)).reshape(BLOCK, M)
        adv = q_im - v0[:, None]
        adv_n = adv / (np.abs(adv).mean() + 1e-6)
        adv_term = adv_n / adv_temp

        # ---- full pairwise candidate chunk L2 (for coherence, computed once)
        pdall = np.sqrt(np.maximum(
            (chunks_cand[:, :, None, :] - chunks_cand[:, None, :, :]) ** 2, 0.0).sum(-1))  # [256,32,32]

        for (form, c) in SETTINGS:
            if form == "F1":
                bw = np.float32(c * dsel.mean()) + 1e-8                    # scalar per block
            elif form == "F2":
                bw = c * np.median(dsel, -1, keepdims=True) + 1e-8         # [256,1] row-wise
            else:
                bw = np.float32(c * dq.mean()) + 1e-8                      # scalar per block
            w = softmax(adv_term - sq_sel / bw)
            top = np.argsort(-w, -1)[:, :TOPW]
            sub = np.take_along_axis(
                np.take_along_axis(pdall, top[:, :, None], 1), top[:, None, :], 2)  # [256,8,8]
            coher = sub[:, iu[0], iu[1]].mean(-1) / rand_ref
            usable = (sq_sel < 2.0 * np.broadcast_to(bw, (BLOCK, 1))).sum(-1)
            acc[(form, c)]["w"].append(w[:, 0])
            acc[(form, c)]["coh"].append(coher)
            acc[(form, c)]["use"].append(usable)

    dsel_all = np.concatenate(dsel_all, 0)                    # [2048, 31]
    dist = dict(
        sq_nonself_mean=float(dsel_all.mean()),
        sq_nonself_median=float(np.median(dsel_all)),
        sq_nonself_p90=float(np.percentile(dsel_all, 90)),
        sq_row_median_mean=float(np.median(dsel_all, -1).mean()),
        mean_sq_pool=float(np.mean(pool_means)),
        rand_ref_chunkL2=rand_ref,
    )
    print(f"dist stats: nonself sq mean={dist['sq_nonself_mean']:.4g} "
          f"med={dist['sq_nonself_median']:.4g} p90={dist['sq_nonself_p90']:.4g} | "
          f"mean(sq_pool)={dist['mean_sq_pool']:.4g} | "
          f"ratio mean_sel/mean_pool={dist['sq_nonself_mean']/dist['mean_sq_pool']:.5f}", flush=True)

    res = {}
    for (form, c) in SETTINGS:
        ws = np.concatenate(acc[(form, c)]["w"])
        ch = np.concatenate(acc[(form, c)]["coh"])
        us = np.concatenate(acc[(form, c)]["use"])
        p = np.percentile(ws, [10, 50, 90])
        res[f"{form}_c{c}"] = dict(
            w_self_mean=float(ws.mean()), w_self_p10=float(p[0]), w_self_p50=float(p[1]),
            w_self_p90=float(p[2]), coherence=float(np.nanmean(ch)), usable=float(us.mean()))
        print(f"  [{form} c={c:<6}] w_self mean={ws.mean():.3f} "
              f"p10/50/90={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f} | coher={np.nanmean(ch):.3f} | "
              f"usable={us.mean():.1f}", flush=True)

    print(f"env total: {time.time()-t0:.0f}s", flush=True)
    return dict(dist=dist, settings=res)


if __name__ == "__main__":
    out = {}
    for env_name, ov in ENVS.items():
        out[env_name] = run_env(env_name, ov)
    print("\nJSON:", json.dumps(out))
