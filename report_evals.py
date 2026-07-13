"""Report DQL v10 eval success at milestones across the 3 cube runs vs RQL/best-baseline bars."""
import csv, glob, sys

RUNS = {
    "cube-double":    ("DQL-v11-cube-double",    23, 74),   # (name, RQL, bar)
    "cube-triple":    ("DQL-v11-cube-triple",     4,  8),
    "cube-quadruple": ("DQL-v11-cube-quadruple", 51, 51),
}
MILES = [50000, 100000, 200000, 300000]
TOL = 5000  # match a milestone within +-TOL steps


def latest_eval_csv(group):
    fs = sorted(glob.glob(f"exp/*/{group}/*/eval.csv"))
    return fs[-1] if fs else None


def load(group):
    f = latest_eval_csv(group)
    if not f:
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        try:
            step = int(float(r["step"]))
            sr = float(r["evaluation/success"]) * 100
        except (KeyError, ValueError):
            continue
        out[step] = sr
    return out


def at(evals, m):
    cand = [(abs(s - m), s) for s in evals if abs(s - m) <= TOL]
    if not cand:
        return None
    return evals[min(cand)[1]]


def main():
    print(f"{'task':<16}{'RQL':>5}{'bar':>5} | " + "".join(f"{m//1000:>6}k" for m in MILES) + "   (DQL v10 success %)")
    print("-" * 78)
    for task, (group, rql, bar) in RUNS.items():
        ev = load(group)
        cells = []
        for m in MILES:
            v = at(ev, m)
            if v is None:
                cells.append(f"{'—':>7}")
            else:
                flag = "*" if v >= bar else ("+" if v > rql else "")
                cells.append(f"{v:>6.0f}{flag}")
        maxstep = max(ev) if ev else 0
        print(f"{task:<16}{rql:>5}{bar:>5} | " + "".join(cells) + f"   [@{maxstep//1000}k]")
    print("\n* = beats bar (all methods)   + = beats RQL")


if __name__ == "__main__":
    main()
