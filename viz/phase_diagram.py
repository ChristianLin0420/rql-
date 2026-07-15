"""F5: stability slices over (c_q, eta) from the DQL112-ABL runs.

For each q_coef / drift_step variant on antmaze-large-task1: final success and
eval-to-eval swing (std of consecutive eval diffs). Renders once at least two
variants have >= 4 evals; extends automatically as more runs finish.

    python viz/phase_diagram.py
"""
import csv, glob, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#76B900", "#3B7BBF", "#C87A2E", "#B04A73"]
ENV = "antmaze-large-navigate-singletask-task1-v0"
BASE = {"q_coef": 0.25, "drift_step": 1.5}


def read_curve(d):
    f = os.path.join(d, "eval.csv")
    if not os.path.exists(f):
        return None
    rows = sorted((int(float(r["step"])), float(r["evaluation/success"]) * 100)
                  for r in csv.DictReader(open(f)) if r.get("evaluation/success"))
    return rows if len(rows) >= 4 else None


def collect(lever):
    pts = {}
    for d in glob.glob(f"{REPO}/exp/rql-iclr2027-50tasks/DQL112-ABL/{ENV}/sweep__{ENV}__{lever}*"):
        val = float(d.rsplit(lever, 1)[1])
        rows = read_curve(d)
        if rows:
            succ = np.mean([v for _, v in rows[-3:]])
            swing = np.std(np.diff([v for _, v in rows]))
            pts[val] = (succ, swing, rows[-1][0])
    # v11.2 baseline supplies the recipe point
    b = read_curve(f"{REPO}/exp/rql-iclr2027-50tasks/DQL112-50/{ENV}/dql112__{ENV}__sd0")
    if b:
        succ = np.mean([v for _, v in b[-3:]])
        pts[BASE[lever]] = (succ, np.std(np.diff([v for _, v in b])), b[-1][0])
    return pts


def main():
    panels = [("q_coef", "c_q (Q-ascent weight)"), ("drift_step", "eta (drift step)")]
    data = {lv: collect(lv) for lv, _ in panels}
    if sum(len(v) for v in data.values()) < 3:
        print("[f5] not enough DQL112-ABL runs with >=4 evals yet -- skipping render")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
    for ax, (lever, xlabel) in zip(axes, panels):
        pts = data[lever]
        if not pts:
            ax.set_axis_off(); continue
        xs = sorted(pts)
        ax.plot(xs, [pts[x][0] for x in xs], "o-", color=PAL[0], lw=2, ms=6, label="success %")
        ax.plot(xs, [pts[x][1] for x in xs], "s--", color=PAL[3], lw=2, ms=6, label="eval swing (std of diffs)")
        ax.axvline(BASE[lever], color="#dddddd", lw=1)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8.5, frameon=False)
        step = min(p[2] for p in pts.values())
        ax.set_title(f"{lever} slice (others at recipe; at step >= {step:,})", fontsize=9.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("F5 · stability slices through the (c_q, eta) plane -- antmaze-large, DQL112-ABL",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{REPO}/viz/figs/f5_stability_slices.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[f5] wrote {out}  ({ {lv: sorted(d) for lv, d in data.items()} })")


if __name__ == "__main__":
    main()
