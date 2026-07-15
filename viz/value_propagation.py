"""F3: improvement propagation vs expectile kappa -- V(s) heatmaps over the maze.

Renders one V(s) panel per available kappa variant on antmaze-large-task1:
  kappa=0.7  -> the finished baseline checkpoint (always available)
  kappa=0.5 / 0.9 -> DQL112-ABL sweep runs, added automatically when trained.
A behavior-shaped V map at kappa=0.5 vs a distance-to-go-shaped map at 0.7+ is
the visual signature of recursive improvement propagation.

    JAX_PLATFORMS=cpu MUJOCO_GL=disabled python viz/value_propagation.py
"""
import glob, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

from viz.load_agent import load_run

ENV = "antmaze-large-navigate-singletask-task1-v0"
GREENS = matplotlib.colors.LinearSegmentedColormap.from_list("nv", ["#f2f4ea", "#76B900", "#26400a"])


def variants():
    out = [("0.7 (baseline)", f"exp/rql-iclr2027-50tasks/DQL111-50/{ENV}/dql111__{ENV}__sd0")]
    for k in ("0.5", "0.9"):
        d = f"{REPO}/exp/rql-iclr2027-50tasks/DQL112-ABL/{ENV}/sweep__{ENV}__expectile{k}"
        if glob.glob(f"{d}/params_*.pkl"):
            out.append((f"{k} (ablation)", os.path.relpath(d, REPO)))
    return out


def main():
    vs = variants()
    fig, axes = plt.subplots(1, len(vs), figsize=(6.4 * len(vs), 4.6), dpi=150, squeeze=False)
    for ax, (label, run_dir) in zip(axes[0], vs):
        run = load_run(run_dir)
        obs = np.asarray(run.train_dataset["observations"])[:: max(1, len(run.train_dataset["observations"]) // 40000)]
        V = []
        for c in range(0, len(obs), 4096):
            V.append(np.asarray(run.agent._V("v", jnp.asarray(obs[c:c + 4096])).mean(0)))
        V = np.concatenate(V)
        hb = ax.hexbin(obs[:, 0], obs[:, 1], C=V, gridsize=46, cmap=GREENS,
                       reduce_C_function=np.mean)
        fig.colorbar(hb, ax=ax, label="V(s)", shrink=0.8)
        ax.set_title(f"kappa = {label}  (step {run.epoch:,})", fontsize=10, loc="left")
        ax.set_aspect("equal"); ax.axis("off")
        del run
    missing = 3 - len(vs)
    note = f"  ·  {missing} kappa variant(s) pending DQL112-ABL training" if missing else ""
    fig.suptitle("F3 · value propagation vs expectile: behavior-shaped V (kappa=0.5) vs "
                 f"distance-to-go-shaped V (kappa>=0.7) -- antmaze-large{note}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{REPO}/viz/figs/f3_value_propagation.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[f3] wrote {out} ({len(vs)} variant(s))")


if __name__ == "__main__":
    main()
