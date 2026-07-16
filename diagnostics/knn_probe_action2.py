"""ACTION 2 (DQL112_FAILURE_ANALYSIS): decisive kNN probe.

Reproduces dql_v11_2's attraction-weight computation
    w = softmax(adv_n / adv_temp - sq_sel / (state_bw * mean(sq)))
for ~2k probe states under three neighbor pools:
  (a) BASELINE   : 256-state uniform batch, M=32 raw-state kNN incl self (training pool)
  (b) DATASET    : top-M=32 raw-state neighbors from a 100k-state dataset subsample (self prepended)
  (c) CRITIC-REP : top-M=32 neighbors by distance in the V-network penultimate features
                   (2-ensemble 512-d post-GELU+LN features, concatenated -> 1024-d);
                   the state-kernel term sq_sel/bw is also computed in feature space.

Per variant: w_self mean/percentiles, usable neighbors (state-kernel factor > e^-2),
neighbor-action coherence (mean pairwise chunk L2 among top-8 weighted candidates /
random-pair chunk L2).
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
BLOCK = 256             # training batch size (adv/bw statistics computed per block, as in training)
M = 32                  # n_cand
POOL_SIZE = 100_000
TOPW = 8                # top-8 weighted candidates for coherence
CKPT_STEP = 2_000_000

ENVS = {
    "cube-double-play-singletask-task1-v0": dict(h=5, expectile=0.9, rho=0.5, discount=0.99),
    "antmaze-large-navigate-singletask-task1-v0": dict(h=1, expectile=0.7, rho=0.5, discount=0.99),
}


def build_agent(env_name, overrides):
    """Load dataset without creating the mujoco env (no EGL on login node).

    load_dataset(compact_dataset=False) yields exactly the observations/actions/terminals
    used in training; relabel_dataset (which needs the env) only adds rewards/masks, and
    the attraction-weight computation never touches those. Action clip mirrors
    envs/env_utils.make_env_and_datasets (clip to +-(1 - 1e-5)).
    """
    config = get_config()
    config.update(overrides)
    splits = env_name.split("-")
    pos = splits.index("singletask")
    dataset_name = "-".join(splits[:pos] + splits[-1:])  # e.g. cube-double-play-v0
    path = os.path.join(os.environ["OGBENCH_DATASET_DIR"], f"{dataset_name}.npz")
    ds = load_dataset(path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False)
    ds["actions"] = np.clip(ds["actions"], -1 + 1e-5, 1 - 1e-5)
    h = config["h"]
    obs_dim, act_dim = ds["observations"].shape[1], ds["actions"].shape[1]
    ex_obs = jnp.zeros((h + 1, 1, obs_dim))   # same shapes main.py's sample_traj ex_batch gives
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

    @jax.jit
    def v_feat(s):
        _, st = agent.network.apply_fn(
            {"params": agent.network.params}, s, name="v", mutable=["intermediates"]
        )
        f = st["intermediates"]["modules_v"]["value_net"]["feature"][0]  # [2, N, 512]
        return jnp.concatenate([f[0], f[1]], -1)  # [N, 1024]

    return qlcb_target, v_mean, v_feat


def batched(fn, *arrays, bs=8192):
    outs = []
    n = arrays[0].shape[0]
    for i in range(0, n, bs):
        outs.append(np.asarray(fn(*[a[i:i + bs] for a in arrays])))
    return np.concatenate(outs, 0)


def chunk_actions(actions, idxs, term_of, h):
    """action chunk at idxs with terminal clamping, exactly mirroring sample_traj."""
    cols = [actions[np.minimum(idxs + i, term_of)] for i in range(h)]
    return np.concatenate(cols, -1)  # [N, h*A]


def sqdist(X, Y, bs=256):
    """squared euclidean distances [len(X), len(Y)], chunked over X."""
    y2 = (Y ** 2).sum(-1)
    out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs]
        out[i:i + bs] = (xb ** 2).sum(-1)[:, None] + y2[None] - 2.0 * xb @ Y.T
    return np.maximum(out, 0.0)


def weight_stats(w, sq_sel, bw, chunks_cand, rand_ref):
    """w:[B,M] softmax weights (col 0 = self), sq_sel:[B,M], chunks_cand:[B,M,hA]."""
    w_self = w[:, 0]
    usable = (sq_sel < 2.0 * bw).sum(-1)  # state-kernel factor > e^-2
    top = np.argsort(-w, -1)[:, :TOPW]
    c = np.take_along_axis(chunks_cand, top[..., None], 1)  # [B,8,hA]
    pd = np.sqrt(np.maximum(
        ((c[:, :, None, :] - c[:, None, :, :]) ** 2).sum(-1), 0.0))
    iu = np.triu_indices(TOPW, 1)
    coher = pd[:, iu[0], iu[1]].mean(-1) / rand_ref
    return w_self, usable, coher


def summarize(name, w_self, usable, coher, extra=""):
    p = np.percentile(w_self, [10, 25, 50, 75, 90])
    print(f"  [{name}] w_self mean={w_self.mean():.3f} "
          f"p10/25/50/75/90={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}/{p[4]:.3f} | "
          f"usable_nbrs(of {M}) mean={usable.mean():.1f} med={np.median(usable):.0f} | "
          f"coherence={np.nanmean(coher):.3f} {extra}", flush=True)
    return dict(w_self_mean=float(w_self.mean()),
                w_self_p=[float(x) for x in p],
                usable_mean=float(usable.mean()),
                coherence=float(np.nanmean(coher)))


def run_env(env_name, overrides):
    print(f"\n=== {env_name} (h={overrides['h']}, expectile={overrides['expectile']}) ===", flush=True)
    t0 = time.time()
    agent, ds = build_agent(env_name, overrides)
    qlcb_target, v_mean, v_feat = make_fns(agent)
    cfg = agent.config
    h, adv_temp, state_bw = cfg["h"], cfg["adv_temp"], cfg["state_bw"]

    obs = np.asarray(ds["observations"], dtype=np.float32)
    acts = np.asarray(ds["actions"], dtype=np.float32)
    terminals = np.asarray(ds["terminals"])  # 1 at last transition of each trajectory
    term_locs = np.nonzero(terminals > 0)[0]
    initial_locs = np.concatenate([[0], term_locs[:-1] + 1])
    term_of_all = term_locs[np.searchsorted(term_locs, np.arange(len(obs)), side="left")]
    print(f"dataset: {len(obs)} transitions, obs_dim={obs.shape[1]}, act_dim={acts.shape[1]}, "
          f"{len(term_locs)} trajs  ({time.time()-t0:.0f}s)", flush=True)

    # ---- probe states: replicate sample_traj start-state distribution, keep dataset indices
    rng = np.random.RandomState(0)
    valid_traj = np.flatnonzero(term_locs - initial_locs)
    n_probe = N_PROBE_BLOCKS * BLOCK
    tr = rng.choice(valid_traj, n_probe, replace=True)
    starts = rng.randint(initial_locs[tr], term_locs[tr])  # start in [initial, terminal)
    s_probe = obs[starts]                                   # [P, O]
    c_probe = chunk_actions(acts, starts, term_of_all[starts], h)  # [P, hA]

    # ---- dataset-wide pool: 100k valid start indices, excluding probe indices
    valid_mask = np.arange(len(obs)) < term_of_all          # idx strictly before its terminal
    valid_idx = np.flatnonzero(valid_mask)
    valid_idx = np.setdiff1d(valid_idx, starts)
    pool_idx = rng.choice(valid_idx, POOL_SIZE, replace=False)
    s_pool = obs[pool_idx]
    c_pool = chunk_actions(acts, pool_idx, term_of_all[pool_idx], h)

    # ---- random-pair chunk distance reference
    ri = rng.choice(len(c_pool), (16384, 2))
    rand_ref = float(np.sqrt(((c_pool[ri[:, 0]] - c_pool[ri[:, 1]]) ** 2).sum(-1)).mean())
    print(f"random-pair chunk L2 reference: {rand_ref:.3f}", flush=True)

    # ---- shared per-probe quantities
    v_probe = batched(v_mean, s_probe)                      # [P]

    def attraction(s0, v0, sq_sel, chunks_cand, mean_sq):
        """exact training computation for one block: adv_n/adv_temp - sq_sel/bw softmax."""
        B_, M_ = sq_sel.shape
        q_im = batched(qlcb_target,
                       np.repeat(s0, M_, 0),
                       chunks_cand.reshape(B_ * M_, -1)).reshape(B_, M_)
        adv = q_im - v0[:, None]
        adv_n = adv / (np.abs(adv).mean() + 1e-6)
        bw = state_bw * (mean_sq + 1e-8)
        logits = adv_n / adv_temp - sq_sel / bw
        z = logits - logits.max(-1, keepdims=True)
        w = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        return w, bw

    results = {}
    # ================= (a) BASELINE: within-block 256-batch kNN =================
    accs = {k: [] for k in ("w", "u", "c")}
    for b in range(N_PROBE_BLOCKS):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        s0, c0, v0 = s_probe[sl], c_probe[sl], v_probe[sl]
        sq = sqdist(s0, s0)
        nn = np.argsort(sq, -1)[:, :M]                       # self at col 0 (dist 0)
        sq_sel = np.take_along_axis(sq, nn, -1)
        chunks_cand = c0[nn]
        w, bw = attraction(s0, v0, sq_sel, chunks_cand, sq.mean())
        ws, us, ch = weight_stats(w, sq_sel, bw, chunks_cand, rand_ref)
        accs["w"].append(ws); accs["u"].append(us); accs["c"].append(ch)
    results["a_baseline"] = summarize(
        "a BASELINE 256-batch", *[np.concatenate(accs[k]) for k in ("w", "u", "c")])

    # ================= (b) DATASET-WIDE raw-state pool =================
    accs = {k: [] for k in ("w", "u", "c")}
    nbr_d = []
    for b in range(N_PROBE_BLOCKS):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        s0, c0, v0 = s_probe[sl], c_probe[sl], v_probe[sl]
        dq = sqdist(s0, s_pool)                              # [256, 100k]
        nn = np.argpartition(dq, M - 1, -1)[:, :M - 1]
        dsel = np.take_along_axis(dq, nn, -1)
        order = np.argsort(dsel, -1)
        nn = np.take_along_axis(nn, order, -1)
        dsel = np.take_along_axis(dsel, order, -1)
        sq_sel = np.concatenate([np.zeros((BLOCK, 1), np.float32), dsel], -1)  # self first
        chunks_cand = np.concatenate([c0[:, None, :], c_pool[nn]], 1)
        w, bw = attraction(s0, v0, sq_sel, chunks_cand, dq.mean())
        ws, us, ch = weight_stats(w, sq_sel, bw, chunks_cand, rand_ref)
        accs["w"].append(ws); accs["u"].append(us); accs["c"].append(ch)
        nbr_d.append(np.sqrt(dsel).mean())
    results["b_dataset"] = summarize(
        "b DATASET 100k pool", *[np.concatenate(accs[k]) for k in ("w", "u", "c")],
        extra=f"| mean nbr raw-dist={np.mean(nbr_d):.3f}")

    # ================= (c) CRITIC-REPRESENTATION pool =================
    t1 = time.time()
    f_probe = batched(v_feat, s_probe).astype(np.float32)    # [P, 1024]
    f_pool = batched(v_feat, s_pool).astype(np.float32)      # [100k, 1024]
    print(f"V penultimate features extracted ({time.time()-t1:.0f}s), dim={f_pool.shape[1]}", flush=True)
    accs = {k: [] for k in ("w", "u", "c")}
    raw_d = []
    for b in range(N_PROBE_BLOCKS):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        s0, c0, v0 = s_probe[sl], c_probe[sl], v_probe[sl]
        dq = sqdist(f_probe[sl], f_pool)
        nn = np.argpartition(dq, M - 1, -1)[:, :M - 1]
        dsel = np.take_along_axis(dq, nn, -1)
        order = np.argsort(dsel, -1)
        nn = np.take_along_axis(nn, order, -1)
        dsel = np.take_along_axis(dsel, order, -1)
        sq_sel = np.concatenate([np.zeros((BLOCK, 1), np.float32), dsel], -1)  # feature-space kernel
        chunks_cand = np.concatenate([c0[:, None, :], c_pool[nn]], 1)
        w, bw = attraction(s0, v0, sq_sel, chunks_cand, dq.mean())
        ws, us, ch = weight_stats(w, sq_sel, bw, chunks_cand, rand_ref)
        accs["w"].append(ws); accs["u"].append(us); accs["c"].append(ch)
        # raw-state distance of the feature neighbors (context)
        rd = np.sqrt(np.maximum(((s0[:, None, :] - s_pool[nn]) ** 2).sum(-1), 0))
        raw_d.append(rd.mean())
    results["c_critic_rep"] = summarize(
        "c CRITIC-REP pool", *[np.concatenate(accs[k]) for k in ("w", "u", "c")],
        extra=f"| mean nbr raw-dist={np.mean(raw_d):.3f}")

    print(f"env total: {time.time()-t0:.0f}s", flush=True)
    return results


if __name__ == "__main__":
    out = {}
    for env_name, ov in ENVS.items():
        out[env_name] = run_env(env_name, ov)
    print("\nJSON:", json.dumps(out))
