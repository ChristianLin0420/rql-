"""Generate RESULTS_DQL111.md: DQL v11.1 50-task sweep vs RQL (paper appendix Table 1).

Aggregates eval.csv from all (task, seed) runs of the sweep launched by sbatch/submit_all.sh
and compares against the RQL column of arxiv 2606.17551 appendix Table 1. Safe to run at
any time -- incomplete runs are reported at their latest eval step and flagged.

    python report_dql111.py            # writes RESULTS_DQL111.md
"""
import csv, glob, os
from statistics import mean, stdev

PROJECT = os.environ.get("WANDB_PROJECT", "rql-iclr2027-50tasks")
GROUP_PREFIX = os.environ.get("GROUP_PREFIX", "DQL111-50")
RUN_PREFIX = os.environ.get("RUN_PREFIX", "dql111")   # e.g. RUN_PREFIX=dql112 GROUP_PREFIX=DQL112-50 for the v11.2 sweep
SEEDS = int(os.environ.get("SEEDS", 3))
STEPS = int(os.environ.get("STEPS", 1_000_000))
OUT = f"RESULTS_{RUN_PREFIX.upper()}.md"
LABEL = {"dql111": "DQL v11.1", "dql112": "DQL v11.2"}.get(RUN_PREFIX, RUN_PREFIX)

# RQL per-task success (%) from arxiv 2606.17551, appendix Table 1 (last column),
# categories in tasks.tsv order; entries are task1..task5.
RQL = {
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
# Caveats vs our setup, stated in the report:
#  - RQL trains 2M gradient steps (paper Table 2); we train 1M.
#  - Paper's puzzle-4x4 / cube-quadruple use the 100M-transition dataset variants; we use standard v0.


def read_eval(env, seed):
    """Return (success_pct_final, success_pct_last3, step) from a run's eval.csv, or None."""
    f = f"exp/{PROJECT}/{GROUP_PREFIX}/{env}/{RUN_PREFIX}__{env}__sd{seed}/eval.csv"
    if not os.path.exists(f):
        return None
    rows = []
    for r in csv.DictReader(open(f)):
        try:
            rows.append((int(float(r["step"])), float(r["evaluation/success"]) * 100))
        except (KeyError, ValueError):
            continue
    if not rows:
        return None
    rows.sort()
    last3 = [v for _, v in rows[-3:]]
    return rows[-1][1], mean(last3), rows[-1][0]


def fmt(x):
    return f"{x:.0f}" if x is not None else "--"


def main():
    lines, incomplete = [], []
    cat_ours, cat_rql = {}, {}
    per_task = []

    for cat, rql_tasks in RQL.items():
        for t in range(1, 6):
            env = f"{cat}-singletask-task{t}-v0"
            per_seed, step_min = [], None
            for s in range(SEEDS):
                r = read_eval(env, s)
                if r is None:
                    incomplete.append(f"{env} sd{s}: no eval yet")
                    continue
                final, last3, step = r
                per_seed.append(last3)
                step_min = step if step_min is None else min(step_min, step)
                if step < STEPS:
                    incomplete.append(f"{env} sd{s}: at step {step}")
            ours = mean(per_seed) if per_seed else None
            sd = stdev(per_seed) if len(per_seed) > 1 else 0.0
            rql = rql_tasks[t - 1]
            per_task.append((env, ours, sd, len(per_seed), step_min, rql))
            if ours is not None:
                cat_ours.setdefault(cat, []).append(ours)
            cat_rql.setdefault(cat, []).append(rql)

    done = not incomplete
    n_runs = sum(n for *_ , n, _, _ in [(0, 0, 0, p[3], p[4], p[5]) for p in per_task])

    lines.append(f"# {LABEL} -- 50-task OGBench results vs RQL\n")
    lines.append(f"Runs `{GROUP_PREFIX}` ({LABEL}), {SEEDS} seeds x {STEPS/1e6:g}M offline steps, success rate (%) "
                 "averaged over the last 3 evals (50 episodes each). RQL reference: "
                 "[arxiv 2606.17551](https://arxiv.org/abs/2606.17551) appendix Table 1.\n")
    if not done:
        lines.append(f"> **PARTIAL RESULTS** -- {len(incomplete)} run(s) not yet at {STEPS:,} steps; "
                     "numbers below use each run's latest eval.\n")

    lines.append("## Category aggregates\n")
    lines.append(f"| category | {LABEL} | RQL | delta |")
    lines.append("|---|---|---|---|")
    ours_all, rql_all = [], []
    for cat in RQL:
        o = mean(cat_ours[cat]) if cat_ours.get(cat) else None
        r = mean(cat_rql[cat])
        d = f"{o - r:+.0f}" if o is not None else "--"
        lines.append(f"| {cat} | {fmt(o)} | {r:.0f} | {d} |")
        if o is not None:
            ours_all.append(o)
        rql_all.append(r)
    o_all = mean(ours_all) if ours_all else None
    lines.append(f"| **all (50 tasks)** | **{fmt(o_all)}** | **{mean(rql_all):.0f}** | "
                 f"**{f'{o_all - mean(rql_all):+.0f}' if o_all is not None else '--'}** |\n")

    if ours_all:
        wins = sum(1 for _, o, _, n, _, r in per_task if o is not None and o > r + 3)
        ties = sum(1 for _, o, _, n, _, r in per_task if o is not None and abs(o - r) <= 3)
        losses = sum(1 for _, o, _, n, _, r in per_task if o is not None and o < r - 3)
        lines.append(f"Per-task (+-3pt band): **{wins} wins / {ties} ties / {losses} losses** vs RQL.\n")

    lines.append("## Per-task results\n")
    lines.append(f"| task | {LABEL} (mean+-std, n) | RQL | delta | min step |")
    lines.append("|---|---|---|---|---|")
    for env, o, sd, n, step, r in per_task:
        val = f"{o:.0f} +- {sd:.0f} (n={n})" if o is not None else "--"
        d = f"{o - r:+.0f}" if o is not None else "--"
        st = f"{step:,}" if step else "--"
        lines.append(f"| {env.replace('-singletask', '').replace('-v0', '')} | {val} | {r} | {d} | {st} |")

    lines.append("\n## Comparison caveats\n")
    lines.append(f"- RQL trains **2M** gradient steps (paper Table 2); these runs use **{STEPS/1e6:g}M**.")
    lines.append("- Paper's `puzzle-4x4` and `cube-quadruple` rows use the **100M-transition** dataset "
                 "variants; we use the standard `-v0` datasets.")
    lines.append(f"- Ours: {SEEDS} seeds, mean of last 3 evals; paper: bootstrap CI over its own seeds/protocol.")

    if incomplete:
        lines.append("\n## Incomplete runs\n")
        for s in incomplete[:30]:
            lines.append(f"- {s}")
        if len(incomplete) > 30:
            lines.append(f"- ... and {len(incomplete) - 30} more")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}  (complete={done})")


if __name__ == "__main__":
    main()
