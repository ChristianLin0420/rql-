"""Build viz/report/data.js for the HTML research dashboard.

Extracts everything the dashboard shows from what is already on disk (eval.csv,
train.csv, queue state, figure files) -- no training-time hooks. Rerun any time:

    python viz/build_data.py          # then open viz/report/index.html

Extend by adding entries to FIGURES (a new card appears automatically) or a new
variant to VARIANTS (a new series appears in the benchmark chart).
"""
import csv, glob, json, os, subprocess, datetime
from statistics import mean

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = "rql-iclr2027-50tasks"
STEPS = int(os.environ.get("STEPS", 1_000_000))   # 2M campaign: STEPS=2000000 python viz/build_data.py
SEEDS = 3

# (run_prefix, group, label) -- add promoted variants here as they are launched.
VARIANTS = [
    ("dql111", "DQL111-50", "DQL v11.1"),
    ("dql112", "DQL112-50", "DQL v11.2"),
]

RQL = {  # arxiv 2606.17551 appendix Table 1, RQL column, task1..task5
    "antmaze-large-navigate":       [84, 80, 95, 81, 76],
    "antmaze-giant-navigate":       [15, 44, 21, 35, 69],
    "humanoidmaze-medium-navigate": [96, 99, 99, 72, 99],
    "humanoidmaze-large-navigate":  [76,  4, 36, 42, 37],
    "scene-play":                   [100, 72, 96, 100, 79],
    "puzzle-3x3-play":              [100, 100, 100, 100, 100],
    "puzzle-4x4-play":              [64, 26, 32, 40, 21],
    "cube-double-play":             [51, 25, 19,  6, 12],
    "cube-triple-play":             [11,  1,  1,  0,  5],
    "cube-quadruple-play":          [87, 81, 62, 25,  0],
}

DOMAIN = lambda cat: ("antmaze" if cat.startswith("antmaze")
                      else "humanoid" if cat.startswith("humanoid")
                      else "cube" if cat.startswith("cube")
                      else "scene/puzzle")

# Paper-figure roadmap: each entry renders as a card; files matching `pattern`
# under viz/figs/ are embedded (png/svg/gif/mp4) once they exist.
FIGURES = [
    dict(id="f1", title="F1 · Stitching evidence (maze rollouts + action provenance)",
         claim="Multi-positive local attraction recombines dataset trajectories: rollouts switch action provenance between episodes at junctions.",
         why="Canonical stitching visualization (IQL/TT/HIQL/OGBench idiom); qualitative twin of the M=1 vs M=32 ablation.",
         regen="python viz/stitching_maze.py --env antmaze-large-navigate-singletask-task1-v0"),
    dict(id="f2", title="F2 · w_self: where the policy borrows neighbors' actions",
         claim="Attraction mass flows to neighboring states' actions (low w_self) exactly on stitching-heavy families / regions.",
         why="Turns an already-logged internal diagnostic into falsifiable spatial evidence.",
         regen="python viz/stitching_maze.py (heatmap artifact) + live scatter below"),
    dict(id="f3", title="F3 · Improvement propagation vs expectile κ",
         claim="κ=0.5 improves only near the goal (one-step tilting); κ≥0.7 extends success outward (recursive propagation).",
         why="Converts the reviewer's P1 correction into a mechanism figure a success rate cannot show.",
         regen="training RUNNING (DQL112-ABL: kappa 0.5/0.9 on antmaze-large-t1); auto-figure via viz/value_propagation.py when done"),
    dict(id="f4", title="F4 · τ_adv bias-variance U-curve",
         claim="Performance vs τ_adv is a U-curve whose components (blur bias, weight ESS/entropy) we measure directly.",
         why="Theory-predicts-experiment overlay; rebuts 'smaller temperature is a hack'.",
         regen="training RUNNING (DQL112-ABL: adv_temp 0.125/0.5/1.0, drift_step 0.5/1/2); auto-figure via viz/blur_curve.py when done"),
    dict(id="f5", title="F5 · Stability phase diagram (c_q × η)",
         claim="Damped ascent has a measurable stability boundary; theory overlay must match the empirical grid.",
         why="Standard form for stability claims; we already own two anchor points (c_q=1.0 vs 0.25).",
         regen="training RUNNING (DQL112-ABL: q_coef 0/0.1/0.5/1.0 + drift_step slices); auto-figure via viz/phase_diagram.py when done"),
    dict(id="f6", title="F6 · Deployment selector (mean-proximal vs kernel-mode)",
         claim="The current selector picks the sample nearest the MEAN (provably); kernel-mode/argmax-Q on the same checkpoints changes sparse-manip success.",
         why="Same-weights comparison = zero training confound; honest found-and-fixed narrative.",
         regen="python viz/selector_scatter.py + python viz/selector_ablation.py --episodes 20"),
    dict(id="f7", title="F7 · Critic health vs success (live below)",
         claim="rank_acc→1 while sparse-manip success stays 0: critic quality is necessary but not sufficient; extraction is the bottleneck.",
         why="One scatter carries the diagnostic narrative across all 50 tasks; data already logged.",
         regen="python viz/build_data.py (rendered live in the Diagnostics block)"),
    dict(id="f8", title="F8 · Toy exactness panel (2D, ground truth known)",
         claim="Drift field, blur bias, and stability boundary match theory exactly where p_w is computable in closed form.",
         why="Pre-empts the estimator-vs-theory objection (review P4) with ground-truth verification.",
         regen="python viz/toy_exactness.py"),
]


def read_last(path, cols):
    """Last-row values for `cols` from a csv (None if missing)."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for last in csv.DictReader(f):
            pass
    if last is None:
        return None
    out = {}
    for c in cols:
        try:
            out[c] = float(last[c])
        except (KeyError, TypeError, ValueError):
            out[c] = None
    return out


def read_eval(prefix, group, env, seed):
    p = f"{REPO}/exp/{PROJECT}/{group}/{env}/{prefix}__{env}__sd{seed}/eval.csv"
    if not os.path.exists(p):
        return None
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            rows.append((int(float(r["step"])), float(r["evaluation/success"]) * 100))
        except (KeyError, ValueError):
            continue
    if not rows:
        return None
    rows.sort()
    return dict(step=rows[-1][0], last3=mean(v for _, v in rows[-3:]),
                curve=[[s, round(v, 1)] for s, v in rows])


def snap(seed_curves, horizon):
    """Mean over seeds of (last-3 evals at or before `horizon`), seeds that reached it only."""
    vals = [mean(v for _, v in [p for p in c if p[0] <= horizon][-3:])
            for c in seed_curves if c and c[-1][0] >= horizon]
    return round(mean(vals), 1) if vals else None


def mean_curve(seed_curves):
    """Per-step mean success across the seeds that logged that step."""
    by_step = {}
    for c in seed_curves:
        for s, v in c or []:
            by_step.setdefault(s, []).append(v)
    return [[s, round(mean(vs), 1)] for s, vs in sorted(by_step.items())]


def collect(prefix, group):
    cats, tasks = {}, []
    for cat, rql5 in RQL.items():
        vals = []
        for t in range(1, 6):
            env = f"{cat}-singletask-task{t}-v0"
            per_seed, seed_curves, diag, done = [], [], None, 0
            for s in range(SEEDS):
                ev = read_eval(prefix, group, env, s)
                seed_curves.append(ev["curve"] if ev else None)
                if ev is None:
                    continue
                if ev["step"] >= STEPS:
                    per_seed.append(ev["last3"]); done += 1
                if s == 0:
                    diag = read_last(
                        f"{REPO}/exp/{PROJECT}/{group}/{env}/{prefix}__{env}__sd0/train.csv",
                        ["training/probe/rank_acc", "training/w_self", "training/probe/ratio"])
            ours = round(mean(per_seed), 1) if per_seed else None
            tasks.append(dict(env=env, cat=cat, domain=DOMAIN(cat), task=t,
                              ours=ours, n=done, rql=rql5[t - 1],
                              seeds=[round(v, 1) for v in per_seed],
                              ours_1m=snap(seed_curves, 1_000_000),
                              ours_2m=snap(seed_curves, 2_000_000),
                              curve=mean_curve(seed_curves),
                              rank_acc=(diag or {}).get("training/probe/rank_acc"),
                              w_self=(diag or {}).get("training/w_self")))
            if ours is not None:
                vals.append(ours)
        cats[cat] = round(mean(vals), 1) if vals else None
    return dict(categories=cats, tasks=tasks,
                done=sum(t["n"] for t in tasks),
                overall=round(mean(v for v in cats.values() if v is not None), 1)
                        if any(v is not None for v in cats.values()) else None)


def queue_status():
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%t"], capture_output=True,
                             text=True, timeout=20).stdout.split()
        return dict(running=out.count("R"), pending=out.count("PD"))
    except Exception:
        return dict(running=None, pending=None)


def figures():
    out = []
    for f in FIGURES:
        media = sorted(glob.glob(f"{REPO}/viz/figs/{f['id']}_*"))
        out.append({**f, "media": [os.path.relpath(m, f"{REPO}/viz/report") for m in media]})
    return out


def with_fallback(prefix, group, label):
    """Live collect; if a variant has no local runs (e.g. trained on another cluster),
    fall back to the committed snapshot in viz/<prefix>_static.json."""
    v = dict(prefix=prefix, group=group, label=label, **collect(prefix, group))
    fb = f"{REPO}/viz/{prefix.replace('dql', 'v')}_static.json"
    if v["overall"] is None and os.path.exists(fb):
        v.update(json.load(open(fb)))
    return v


data = dict(
    generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    queue=queue_status(),
    rql_overall=round(mean(mean(v) for v in RQL.values()), 1),
    rql_cats={k: round(mean(v), 1) for k, v in RQL.items()},
    variants=[with_fallback(p, g, l) for p, g, l in VARIANTS],
    figures=figures(),
)

os.makedirs(f"{REPO}/viz/report", exist_ok=True)
os.makedirs(f"{REPO}/viz/figs", exist_ok=True)
with open(f"{REPO}/viz/report/data.js", "w") as f:
    f.write("window.REPORT_DATA = " + json.dumps(data) + ";\n")
print(f"wrote viz/report/data.js  ({sum(v['done'] for v in data['variants'])} finished runs, "
      f"{len(data['figures'])} figure cards)")
