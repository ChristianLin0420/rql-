"""Aggregate f6b-v2 shard JSONs: Wilson 95% CI per rule + exact McNemar vs medoid (paired by episode seed)."""
import glob, json, math, sys
import numpy as np

def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - hw), min(1.0, c + hw)

def mcnemar_exact(rule_s, med_s):
    b = int(sum(1 for r, m in zip(rule_s, med_s) if r > 0.5 and m < 0.5))  # rule wins
    c = int(sum(1 for r, m in zip(rule_s, med_s) if r < 0.5 and m > 0.5))  # medoid wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial p under H0: b ~ Bin(n, 0.5)
    p = sum(math.comb(n, i) for i in range(0, n + 1)
            if abs(i - n / 2) >= abs(b - n / 2) - 1e-12) * 0.5 ** n
    return b, c, min(1.0, p)

def main(pattern):
    merged = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for env, res in d.items():
            merged.setdefault(env, {}).update(res)
    for env, res in merged.items():
        rules = [k for k in res if isinstance(res[k], dict) and "final" in res[k]]
        print(f"\n=== {env} (epoch {res.get('epoch')}, n={len(res[rules[0]]['final'])}/rule, paired seeds) ===")
        med = res.get("medoid", {}).get("final")
        for metric in ("final", "max"):
            print(f"-- success metric: {metric}-of-episode --")
            for rule in ["medoid", "argmax_qlcb", "argmax_qmean", "kernel_mode", "single_sample"]:
                if rule not in res:
                    continue
                s = res[rule][metric]
                n = len(s)
                k = int(sum(1 for x in s if x > 0.5))
                p, lo, hi = wilson(k, n)
                line = f"  {rule:>14}: {k:3d}/{n} = {p:.3f}  Wilson95=[{lo:.3f},{hi:.3f}]"
                if rule != "medoid" and med is not None:
                    ms = res["medoid"][metric]
                    b, c, pv = mcnemar_exact(s, ms)
                    line += f"  vs medoid: +{b}/-{c} McNemar p={pv:.4f}"
                print(line)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/f6bv2/*.json")
