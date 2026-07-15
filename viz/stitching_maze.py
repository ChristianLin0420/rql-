"""F1 + F2 maze figures: stitching evidence and w_self heatmap (antmaze/humanoidmaze).

F1: policy rollouts over dataset trajectories, colored by action *provenance* --
the dataset episode whose action currently dominates the attraction weights.
Provenance switches (stitch points) are marked.
F2: hexbin of training-style w_self (attraction weight retained by a state's own
action among its kNN candidates) over the maze -- low = borrowing regions.

    JAX_PLATFORMS=cpu MUJOCO_GL=disabled python viz/stitching_maze.py \
        [--run exp/.../<run_dir>] [--episodes 4]
"""
import argparse, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp

from viz.load_agent import load_run

PAL = ["#76B900", "#3B7BBF", "#C87A2E", "#B04A73"]
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list("nv", ["#eef3e2", "#76B900", "#33520a"])

DEFAULT_RUN = ("exp/rql-iclr2027-50tasks/DQL111-50/antmaze-large-navigate-singletask-task1-v0/"
               "dql111__antmaze-large-navigate-singletask-task1-v0__sd0")


def attraction(agent, bank_obs, bank_act, bank_ep, bw_scale, s, M=32):
    """Training-style attraction weights of state s over the dataset bank."""
    sq = ((bank_obs - s[None]) ** 2).sum(-1)
    nn = np.argpartition(sq, M)[:M]
    q = np.asarray(agent._qlcb("target_q", jnp.tile(jnp.asarray(s)[None], (M, 1)),
                               jnp.asarray(bank_act[nn])))
    v = float(np.asarray(agent._V("v", jnp.asarray(s)[None]).mean(0))[0])
    adv = q - v
    advn = adv / (np.abs(adv).mean() + 1e-6)
    logits = advn / agent.config["adv_temp"] - sq[nn] / bw_scale
    w = np.exp(logits - logits.max()); w /= w.sum()
    return nn, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--bank", type=int, default=100_000)
    ap.add_argument("--wself_states", type=int, default=12_000)
    ap.add_argument("--skip_wself", action="store_true")
    ap.add_argument("--debounce", type=int, default=15,
                    help="steps a new provenance episode must persist to count as a stitch")
    args = ap.parse_args()

    run = load_run(args.run)
    short = run.env_name.replace("-singletask", "").replace("-v0", "")
    ds = run.train_dataset
    obs, act, term = (np.asarray(ds["observations"]), np.asarray(ds["actions"]),
                      np.asarray(ds["terminals"]).astype(int))
    ep_id = np.concatenate([[0], np.cumsum(term)[:-1]])
    stride = max(1, len(obs) // args.bank)
    bank = slice(0, None, stride)
    bank_obs, bank_act, bank_ep = obs[bank], act[bank], ep_id[bank]
    bw_scale = run.config["state_bw"] * float(((bank_obs[:2000, None] - bank_obs[None, :200]) ** 2)
                                              .sum(-1).mean())
    print(f"[maze] {short}: bank={len(bank_obs)} bw={bw_scale:.1f}", flush=True)

    # ---------------- F1: rollouts with provenance ----------------
    rng = jax.random.PRNGKey(0)
    rollouts = []
    for ep in range(args.episodes):
        o, _ = run.eval_env.reset()
        xy, prov, done, steps = [o[:2].copy()], [], False, 0
        while not done and steps < 2000:
            rng, k = jax.random.split(rng)
            a = np.clip(np.asarray(run.agent.sample_actions(obs=o, temperature=0, seed=k)), -1, 1)
            nn, w = attraction(run.agent, bank_obs, bank_act, bank_ep, bw_scale, o)
            prov.append(int(bank_ep[nn[np.argmax(w)]]))
            o, r, tm, tr, info = run.eval_env.step(a)
            done = tm or tr
            xy.append(o[:2].copy()); steps += 1
        # debounce: provenance only switches when the new episode persists
        prov = np.array(prov)
        smooth, cur, streak = [], prov[0], 0
        for t in range(len(prov)):
            if prov[t] != cur:
                streak += 1
                if streak >= args.debounce:
                    cur, streak = prov[t], 0
            else:
                streak = 0
            smooth.append(cur)
        rollouts.append((np.array(xy), np.array(smooth), float(info.get("success", r > 0))))
        print(f"[maze] rollout {ep}: {steps} steps success={rollouts[-1][2]}", flush=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=150)
    shown = np.unique(ep_id)[:: max(1, ep_id.max() // 120)]
    for e in shown:
        m = ep_id == e
        ax.plot(obs[m, 0], obs[m, 1], color="#cfcfcc", lw=0.5, alpha=0.5, zorder=1)
    n_stitch = 0
    for xy, prov, success in rollouts:
        seg_start, ci = 0, 0
        for t in range(1, len(prov)):
            if prov[t] != prov[t - 1]:
                ax.plot(xy[seg_start:t + 1, 0], xy[seg_start:t + 1, 1],
                        color=PAL[ci % 4], lw=2.2, zorder=3)
                ax.scatter(*xy[t], marker="^", s=30, color="#1a1a1a", zorder=5)
                seg_start, ci, n_stitch = t, ci + 1, n_stitch + 1
        ax.plot(xy[seg_start:, 0], xy[seg_start:, 1], color=PAL[ci % 4], lw=2.2, zorder=3)
        ax.scatter(*xy[0], marker="o", s=60, facecolor="#fff", edgecolor="#1a1a1a", lw=1.6, zorder=6)
    ax.set_title(f"F1 · {short}: rollout action provenance over dataset trajectories\n"
                 f"color changes + ▲ = provenance switches (stitches): {n_stitch} across "
                 f"{args.episodes} episodes · success {np.mean([r[2] for r in rollouts]):.2f}",
                 fontsize=9.5, loc="left")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    p1 = f"{REPO}/viz/figs/f1_stitching_{short}.png"
    fig.savefig(p1, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[maze] wrote {p1}", flush=True)

    if args.skip_wself:
        return
    # ---------------- F2: w_self heatmap ----------------
    idx = np.random.default_rng(0).choice(len(bank_obs), size=min(args.wself_states, len(bank_obs)),
                                          replace=False)
    wself = np.zeros(len(idx))
    M = 32
    for i, bi in enumerate(idx):
        s = bank_obs[bi]
        sq = ((bank_obs - s[None]) ** 2).sum(-1)
        nn = np.argpartition(sq, M + 1)[:M + 1]
        nn = np.concatenate([[bi], nn[nn != bi][:M - 1]])  # self is candidate 0
        q = np.asarray(run.agent._qlcb("target_q", jnp.tile(jnp.asarray(s)[None], (len(nn), 1)),
                                       jnp.asarray(bank_act[nn])))
        v = float(np.asarray(run.agent._V("v", jnp.asarray(s)[None]).mean(0))[0])
        adv = q - v; advn = adv / (np.abs(adv).mean() + 1e-6)
        logits = advn / run.config["adv_temp"] - sq[nn] / bw_scale
        w = np.exp(logits - logits.max()); w /= w.sum()
        wself[i] = w[0]
        if i % 2000 == 0:
            print(f"[maze] w_self {i}/{len(idx)}", flush=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=150)
    hb = ax.hexbin(bank_obs[idx, 0], bank_obs[idx, 1], C=wself, gridsize=42, cmap=GREENS,
                   reduce_C_function=np.mean)
    fig.colorbar(hb, ax=ax, label="mean w_self (weight kept by own action)", shrink=0.8)
    ax.set_title(f"F2 · {short}: where the policy borrows neighbors' actions\n"
                 f"dark = own action dominates · light = attraction mass flows to neighboring states",
                 fontsize=9.5, loc="left")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    p2 = f"{REPO}/viz/figs/f2_wself_maze_{short}.png"
    fig.savefig(p2, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[maze] wrote {p2}", flush=True)


if __name__ == "__main__":
    main()
