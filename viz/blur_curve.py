"""F4: tau_adv bias-variance decomposition.

Components (weight entropy, effective sample size, blur value-gap) are measured
DIRECTLY on a trained checkpoint by replaying the agent's exact attraction
computation at different tau_adv -- no retraining. The success-vs-tau panel is
added automatically once the DQL112-ABL adv_temp runs finish.

    JAX_PLATFORMS=cpu MUJOCO_GL=disabled python viz/blur_curve.py
"""
import csv, glob, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

from viz.load_agent import load_run

PAL = ["#76B900", "#3B7BBF", "#C87A2E", "#B04A73"]
ENV = "antmaze-large-navigate-singletask-task1-v0"
RUN = f"exp/rql-iclr2027-50tasks/DQL111-50/{ENV}/dql111__{ENV}__sd0"
TAUS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
BASE_TAU = 0.25  # v11.2 recipe


def components(run, taus, n_batches=30, B=256, M=32):
    obs = np.asarray(run.train_dataset["observations"])
    act = np.asarray(run.train_dataset["actions"])
    rng = np.random.default_rng(0)
    out = {t: dict(ent=[], ess=[], gap=[]) for t in taus}
    for b in range(n_batches):
        idx = rng.choice(len(obs), B, replace=False)
        s, a = obs[idx], act[idx]
        sq = ((s[:, None] - s[None]) ** 2).sum(-1)
        nn = np.argsort(sq, -1)[:, :M]
        sq_sel = np.take_along_axis(sq, nn, -1)
        q = np.asarray(run.agent._qlcb(
            "target_q",
            jnp.asarray(np.repeat(s, M, 0)),
            jnp.asarray(a[nn].reshape(B * M, -1)))).reshape(B, M)
        v = np.asarray(run.agent._V("v", jnp.asarray(s)).mean(0))
        adv = q - v[:, None]
        advn = adv / (np.abs(adv).mean() + 1e-6)
        bw = run.config["state_bw"] * (sq.mean() + 1e-8)
        for t in taus:
            logits = advn / t - sq_sel / bw
            w = np.exp(logits - logits.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            out[t]["ent"].append(-(w * np.log(w + 1e-12)).sum(-1).mean())
            out[t]["ess"].append((1.0 / (w ** 2).sum(-1)).mean())
            out[t]["gap"].append((q.max(-1) - (w * q).sum(-1)).mean())
    return {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in out.items()}


def ablation_success():
    """success at latest eval for DQL112-ABL adv_temp runs (plus the v11.2 baseline at 0.25)."""
    pts = {}
    for t in TAUS:
        pat = f"{REPO}/exp/rql-iclr2027-50tasks/DQL112-ABL/{ENV}/sweep__{ENV}__adv_temp{t}/eval.csv"
        for f in glob.glob(pat):
            rows = [(int(float(r["step"])), float(r["evaluation/success"]) * 100)
                    for r in csv.DictReader(open(f)) if r.get("evaluation/success")]
            if rows:
                pts[t] = (max(rows)[1], max(rows)[0])
    b = f"{REPO}/exp/rql-iclr2027-50tasks/DQL112-50/{ENV}/dql112__{ENV}__sd0/eval.csv"
    if os.path.exists(b):
        rows = [(int(float(r["step"])), float(r["evaluation/success"]) * 100)
                for r in csv.DictReader(open(b)) if r.get("evaluation/success")]
        if rows:
            pts[BASE_TAU] = (max(rows)[1], max(rows)[0])
    return pts


def main():
    run = load_run(RUN)
    comp = components(run, TAUS)
    succ = ablation_success()
    n_panels = 3 if succ else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.0), dpi=150)

    ax = axes[0]
    ax.plot(TAUS, [comp[t]["ess"] for t in TAUS], "o-", color=PAL[0], lw=2, ms=6)
    ax.axvline(BASE_TAU, color="#dddddd", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("tau_adv")
    ax.set_ylabel("effective sample size (of 32 candidates)")
    ax.set_title("(a) variance side: ESS collapses as tau_adv -> 0", fontsize=9.5, loc="left")

    ax = axes[1]
    ax.plot(TAUS, [comp[t]["gap"] for t in TAUS], "o-", color=PAL[2], lw=2, ms=6)
    ax.axvline(BASE_TAU, color="#dddddd", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("tau_adv")
    ax.set_ylabel("blur value-gap  E[max Q - sum(w Q)]")
    ax.set_title("(b) bias side: blur gap grows with tau_adv", fontsize=9.5, loc="left")

    if succ:
        ax = axes[2]
        ts = sorted(succ)
        ax.plot(ts, [succ[t][0] for t in ts], "o-", color=PAL[1], lw=2, ms=6)
        ax.axvline(BASE_TAU, color="#dddddd", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("tau_adv"); ax.set_ylabel("success %")
        step = min(s for _, s in succ.values())
        ax.set_title(f"(c) performance (DQL112-ABL, at step >= {step:,})", fontsize=9.5, loc="left")

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    note = "" if succ else "   (performance panel appears when DQL112-ABL adv_temp runs finish)"
    fig.suptitle(f"F4 · tau_adv is a bias-variance trade-off -- components measured on the trained "
                 f"critic ({ENV.split('-singletask')[0]}){note}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = f"{REPO}/viz/figs/f4_blur_biasvariance.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[f4] wrote {out} (success points: {sorted(succ) if succ else 'pending training'})")


if __name__ == "__main__":
    main()
