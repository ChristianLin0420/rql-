"""F6a: the deployment-selector flaw, one state at a time.

Draw K generator samples for a manipulation state, project to 2D (PCA), color by
Q_LCB, and mark what each selector deploys: the current 'medoid' (provably the
sample nearest the sample MEAN) vs a kernel-density mode vs argmax-Q_LCB.

    JAX_PLATFORMS=cpu MUJOCO_GL=disabled python viz/selector_scatter.py
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

DEFAULT_RUN = ("exp/rql-iclr2027-50tasks/DQL111-50/cube-double-play-singletask-task1-v0/"
               "dql111__cube-double-play-singletask-task1-v0__sd0")
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list("nv", ["#dfe8cc", "#76B900", "#33520a"])


def selectors(cand, q):
    d2 = ((cand[:, None] - cand[None]) ** 2).sum(-1)
    mean_prox = int(np.argmin(d2.sum(-1)))              # current: nearest the mean
    sigma2 = np.median(d2[d2 > 0])
    mode = int(np.argmax((np.exp(-d2 / (2 * sigma2)) - np.eye(len(cand))).sum(-1)))
    return {"mean-proximal (current)": mean_prox, "kernel mode": mode,
            "argmax Q_LCB": int(np.argmax(q))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--n_states", type=int, default=200, help="states scanned to find the most multimodal")
    args = ap.parse_args()

    run = load_run(args.run)
    short = run.env_name.replace("-singletask", "").replace("-v0", "")
    obs = np.asarray(run.train_dataset["observations"])
    A = run.agent.config["action_dim"]
    rng = jax.random.PRNGKey(0)

    # scan states, keep the one whose sample set is most spread (most multimodal)
    best = None
    idxs = np.random.default_rng(0).choice(len(obs), size=args.n_states, replace=False)
    for i in idxs:
        rng, k = jax.random.split(rng)
        eps = jax.random.normal(k, (args.K, A))
        cand = np.asarray(run.agent.network.select("target_actor")(
            jnp.tile(jnp.asarray(obs[i])[None], (args.K, 1)), eps))
        q = np.asarray(run.agent._qlcb("target_q",
                                       jnp.tile(jnp.asarray(obs[i])[None], (args.K, 1)),
                                       jnp.asarray(cand)))
        pk = selectors(cand, q)
        # rank states by how far apart the selectors' picks are (disagreement)
        disagree = ((cand[pk["mean-proximal (current)"]] - cand[pk["kernel mode"]]) ** 2).sum() \
                 + ((cand[pk["mean-proximal (current)"]] - cand[pk["argmax Q_LCB"]]) ** 2).sum()
        if best is None or disagree > best[0]:
            best = (disagree, i, cand, q)
    _, si, cand, q = best

    # PCA to 2D
    c = cand - cand.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:2].T
    picks = selectors(cand, q)
    mean_xy = xy.mean(0)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=q, cmap=GREENS, s=90, edgecolor="white",
                    linewidth=1.6, zorder=3)
    fig.colorbar(sc, ax=ax, label="Q_LCB", shrink=0.8)
    ax.scatter(*mean_xy, marker="+", s=140, color="#8a8a8a", zorder=4)
    ax.annotate("sample mean", mean_xy, textcoords="offset points", xytext=(8, 6),
                fontsize=9, color="#8a8a8a")
    marks = {"mean-proximal (current)": ("#C0392B", "v"), "kernel mode": ("#3B7BBF", "s"),
             "argmax Q_LCB": ("#1a1a1a", "*")}
    for name, ki in picks.items():
        col, m = marks[name]
        ax.scatter(*xy[ki], marker=m, s=210, facecolor="none", edgecolor=col, linewidth=2.4,
                   zorder=5)
        off = {"mean-proximal (current)": (10, -16), "kernel mode": (10, 12),
               "argmax Q_LCB": (-10, -18)}[name]
        ax.annotate(name, xy[ki], textcoords="offset points", xytext=off, fontsize=9,
                    color=col, fontweight="bold",
                    ha="right" if off[0] < 0 else "left")
    ax.set_title(f"F6a · {short}: {args.K} generator samples for one state (PCA of action chunks)\n"
                 "current selector deploys the sample nearest the MEAN -- not a mode, "
                 "not the highest value", fontsize=9.5, loc="left")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = f"{REPO}/viz/figs/f6_selector_scatter_{short}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[f6a] state idx {si}, picks {picks}\n[f6a] wrote {out}")


if __name__ == "__main__":
    main()
