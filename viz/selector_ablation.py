"""F6b: deployment-selector ablation on SAVED checkpoints (same weights, zero training confound).

For each task's seed-0 checkpoint, roll out with four action selectors over the same
K=32 generator candidates:
  mean-proximal (current), kernel-density mode, argmax Q_LCB, single sample.
Writes viz/figs/f6_selector_bars.png + viz/figs/f6_selector_results.json.

    JAX_PLATFORMS=cpu MUJOCO_GL=disabled python viz/selector_ablation.py --episodes 20
"""
import argparse, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from einops import rearrange, repeat

from viz.load_agent import load_run

PAL = ["#76B900", "#3B7BBF", "#C87A2E", "#B04A73"]
TASKS = [
    "cube-double-play-singletask-task1-v0",
    "scene-play-singletask-task1-v0",
    "antmaze-large-navigate-singletask-task1-v0",
]
SELECTORS = ["mean-proximal (current)", "kernel mode", "argmax Q_LCB", "single sample"]


def pick(agent, obs, key, selector, K=32):
    A = agent.config["action_dim"]
    o = jnp.atleast_2d(jnp.asarray(obs))[-1:]
    cand = np.asarray(agent.network.select("target_actor")(
        repeat(o, "1 o -> k o", k=K), jax.random.normal(key, (K, A))))
    if selector == "single sample":
        a = cand[0]
    elif selector == "mean-proximal (current)":
        d2 = ((cand[:, None] - cand[None]) ** 2).sum(-1)
        a = cand[int(np.argmin(d2.sum(-1)))]
    elif selector == "kernel mode":
        d2 = ((cand[:, None] - cand[None]) ** 2).sum(-1)
        s2 = np.median(d2[d2 > 0]) + 1e-9
        a = cand[int(np.argmax((np.exp(-d2 / (2 * s2)) - np.eye(K)).sum(-1)))]
    elif selector == "argmax Q_LCB":
        q = np.asarray(agent._qlcb("target_q", repeat(o, "1 o -> k o", k=K), jnp.asarray(cand)))
        a = cand[int(np.argmax(q))]
    a = np.clip(a, -1, 1)
    return rearrange(a, "(h d) -> h d", h=agent.config["h"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=1200)
    args = ap.parse_args()

    results = {}
    for env_name in TASKS:
        run = load_run(f"exp/rql-iclr2027-50tasks/DQL111-50/{env_name}/dql111__{env_name}__sd0")
        short = env_name.replace("-singletask", "").replace("-v0", "")
        results[short] = {}
        rng = jax.random.PRNGKey(0)
        for sel in SELECTORS:
            succ = []
            for ep in range(args.episodes):
                o, _ = run.eval_env.reset()
                done, steps, info = False, 0, {}
                while not done and steps < args.max_steps:
                    rng, k = jax.random.split(rng)
                    a = pick(run.agent, o, k, sel)
                    o, r, tm, tr, info = run.eval_env.step(np.asarray(a))
                    done = tm or tr
                    steps += 1
                succ.append(float(info.get("success", 0.0)))
            results[short][sel] = float(np.mean(succ))
            print(f"[f6b] {short} · {sel}: {results[short][sel]:.2f}", flush=True)
        del run

    with open(f"{REPO}/viz/figs/f6_selector_results.json", "w") as f:
        json.dump(results, f, indent=1)

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    tasks = list(results)
    xpos = np.arange(len(tasks))
    bw = 0.19
    for si, sel in enumerate(SELECTORS):
        vals = [100 * results[t][sel] for t in tasks]
        bars = ax.bar(xpos + (si - 1.5) * bw, vals, width=bw - 0.02, color=PAL[si], label=sel)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                    fontsize=8.5, color="#333")
    ax.set_xticks(xpos, tasks, fontsize=9)
    ax.set_ylabel("success % (20 episodes, seed-0 checkpoint)")
    ax.legend(fontsize=8.5, frameon=False, ncol=2)
    ax.set_title("F6b · deployment selector ablation on identical weights\n"
                 "(only the choice among the same 32 generator samples differs)",
                 fontsize=9.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = f"{REPO}/viz/figs/f6_selector_bars.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[f6b] wrote {out}")


if __name__ == "__main__":
    main()
