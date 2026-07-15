"""F7 artifact: critic ranking accuracy vs task success across all 50 tasks (seed 0).

Static (paper) version of the dashboard's live scatter. Data comes from
viz/report/data.js (run `python viz/build_data.py` first).

    python viz/critic_health.py
"""
import json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"antmaze": "#76B900", "humanoid": "#3B7BBF", "scene/puzzle": "#C87A2E", "cube": "#B04A73"}

data = json.loads(open(f"{REPO}/viz/report/data.js").read().split("=", 1)[1].rstrip(";\n"))
tasks = [t for t in data["variants"][0]["tasks"] if t["rank_acc"] is not None and t["ours"] is not None]

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
for dom, col in PAL.items():
    pts = [(t["rank_acc"], t["ours"]) for t in tasks if t["domain"] == dom]
    if pts:
        x, y = zip(*pts)
        ax.scatter(x, y, s=52, color=col, edgecolor="white", linewidth=1.4, label=dom, zorder=3)
ax.axvline(0.95, color="#dddddd", lw=1, zorder=1)
ax.annotate("critic ranks near-perfectly\nyet success ~0\n(extraction bottleneck)",
            xy=(0.985, 4), xytext=(0.62, 22), fontsize=9, color="#555",
            arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=1.2))
ax.set_xlabel("critic ranking accuracy (final, probe/rank_acc)")
ax.set_ylabel("task success % (1M steps)")
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.set_title("F7 · critic health is necessary but not sufficient (50 tasks, seed 0)\n"
             "bottom-right cluster: healthy critic, zero success -- policy extraction is the bottleneck",
             fontsize=9.5, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = f"{REPO}/viz/figs/f7_critic_health.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"[f7] {len(tasks)} tasks -> {out}")
