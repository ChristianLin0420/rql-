"""F8: toy exactness panel -- verify the drift machinery where ground truth is computable.

2D action space, bimodal behavior mu (two Gaussians), quadratic Q. Three panels:
 (a) the advantage-tilted target p_w (exact, contours) + drift field V_drift (quiver)
     + samples from a generator trained with our exact actor loss;
 (b) blur bias vs tau_adv: value gap between the softmax-blended attraction target
     and the argmax-advantage candidate (measured; ~linear in tau_adv);
 (c) stability: oscillation amplitude of the trained generator vs c_q for two eta --
     the damped-ascent stability boundary, measured.

    JAX_PLATFORMS=cpu python viz/toy_exactness.py
"""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
import optax

PAL = ["#76B900", "#3B7BBF", "#C87A2E", "#B04A73"]
M1, M2, SIG = jnp.array([-0.5, -0.3]), jnp.array([0.5, 0.4]), 0.12
QPEAK = jnp.array([0.62, 0.50])
ALPHA = 0.15  # tilt temperature for the exact target


def mu_logpdf(a):
    d1 = ((a - M1) ** 2).sum(-1) / (2 * SIG**2)
    d2 = ((a - M2) ** 2).sum(-1) / (2 * SIG**2)
    return jnp.logaddexp(-d1, -d2) - jnp.log(2.0)


BUMP_AT, BUMP_W = jnp.array([-0.75, 0.75]), 0.08


def Q(a, ood_bump=False):
    base = -((a - QPEAK) ** 2).sum(-1)
    if ood_bump:  # deceptive off-support Q artifact (positive curvature away from data)
        base = base + 4.0 * jnp.exp(-((a - BUMP_AT) ** 2).sum(-1) / (2 * BUMP_W**2))
    return base


def sample_mu(key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    pick = jax.random.bernoulli(k1, 0.5, (n, 1))
    return jnp.where(pick, M1, M2) + SIG * jax.random.normal(k3, (n, 2))


def drift_field(g, cand, w, tau2):
    """V_drift at points g given weighted candidates (attraction) and g itself (repulsion)."""
    d_pc = ((g[:, None] - cand[None]) ** 2).sum(-1)
    kp = jnp.exp(-d_pc / (2 * tau2)) * w[None]
    m_p = (kp[..., None] * cand[None]).sum(1) / (kp.sum(-1, keepdims=True) + 1e-12)
    d_gg = ((g[:, None] - g[None]) ** 2).sum(-1)
    kq = jnp.exp(-d_gg / (2 * tau2)) * (1 - jnp.eye(len(g)))
    m_q = (kq[..., None] * g[None]).sum(1) / (kq.sum(-1, keepdims=True) + 1e-12)
    return m_p - m_q


def weights(cand, tau_adv):
    adv = Q(cand) - Q(sample_mu(jax.random.PRNGKey(7), 4096)).mean()  # A = Q - V_mu
    advn = adv / (jnp.abs(adv).mean() + 1e-6)
    w = jax.nn.softmax(advn / tau_adv)
    return w


def init_mlp(key, hidden=64):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(w1=jax.random.normal(k1, (2, hidden)) * 0.5, b1=jnp.zeros(hidden),
                w2=jax.random.normal(k2, (hidden, hidden)) * 0.2, b2=jnp.zeros(hidden),
                w3=jax.random.normal(k3, (hidden, 2)) * 0.2, b3=jnp.zeros(2))


def mlp(p, eps):
    h = jax.nn.gelu(eps @ p["w1"] + p["b1"])
    h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    return jnp.tanh(h @ p["w3"] + p["b3"])


def train_generator(tau_adv=0.25, eta=1.5, c_q=0.25, steps=2500, G=64, M=128, seed=0,
                    track=False, moving_q=False, period=400, radius=0.3):
    """moving_q simulates the nonstationary (EMA-lagged) critic of the real system:
    Q's peak slowly orbits; the ascent term chases it with gain c_q."""
    key = jax.random.PRNGKey(seed)
    params = init_mlp(key)
    opt = optax.adam(3e-3); ost = opt.init(params)
    tau2 = 0.02
    traj = []

    @jax.jit
    def step(params, ost, key, i):
        km, ke = jax.random.split(key)
        cand = sample_mu(km, M)
        w = weights(cand, tau_adv)
        eps = jax.random.normal(ke, (G, 2))
        ang = 2 * jnp.pi * i / period
        peak = QPEAK + (radius * jnp.array([jnp.cos(ang), jnp.sin(ang)]) if moving_q else 0.0)

        def loss(p):
            g = mlp(p, eps)
            V = drift_field(jax.lax.stop_gradient(g), cand, w, tau2)
            goal = jax.lax.stop_gradient(g + eta * V)
            q_asc = -((g - peak) ** 2).sum(-1).mean()
            return ((g - goal) ** 2).sum(-1).mean() - c_q * q_asc

        gr = jax.grad(loss)(params)
        up, ost2 = opt.update(gr, ost)
        return optax.apply_updates(params, up), ost2, mlp(params, eps).mean(0)

    for i in range(steps):
        key, k = jax.random.split(key)
        params, ost, gm = step(params, ost, k, i)
        if track and i > steps // 2:
            traj.append(np.asarray(gm))
    return params, np.array(traj) if track else None


def main():
    os.makedirs(f"{REPO}/viz/figs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=150)

    # (a) exact target + drift field + trained samples
    ax = axes[0]
    lin = np.linspace(-1, 1, 120)
    Gx, Gy = np.meshgrid(lin, lin)
    grid = jnp.array(np.stack([Gx.ravel(), Gy.ravel()], -1))
    logp = mu_logpdf(grid) + (Q(grid) - Q(sample_mu(jax.random.PRNGKey(7), 4096)).mean()) / ALPHA
    p = np.exp(np.asarray(logp - logp.max())).reshape(120, 120)
    ax.contour(Gx, Gy, p, levels=6, colors="#3B7BBF", linewidths=1.0, alpha=0.8)
    key = jax.random.PRNGKey(1)
    cand = sample_mu(key, 256); w = weights(cand, 0.25)
    ql = np.linspace(-0.95, 0.95, 16)
    qg = jnp.array(np.stack(np.meshgrid(ql, ql), -1).reshape(-1, 2))
    V = np.asarray(drift_field(qg, cand, w, 0.02))
    ax.quiver(np.asarray(qg)[:, 0], np.asarray(qg)[:, 1], V[:, 0], V[:, 1],
              color="#b9b9b4", width=0.0035, scale=6)
    params, _ = train_generator()
    smp = np.asarray(mlp(params, jax.random.normal(jax.random.PRNGKey(3), (400, 2))))
    ax.scatter(smp[:, 0], smp[:, 1], s=7, color=PAL[0], alpha=0.7, label="trained generator")
    ax.scatter(*np.asarray(QPEAK), marker="*", s=160, color="#1a1a1a", zorder=5, label="argmax Q")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("(a) exact tilted target p_w (contours), drift field, generator", fontsize=9.5, loc="left")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

    # (b) blur bias vs tau_adv
    ax = axes[1]
    taus = np.array([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
    gaps_blend, gaps_exp = [], []
    cand = sample_mu(jax.random.PRNGKey(11), 512)
    for t in taus:
        w = weights(cand, float(t))
        blend = (w[:, None] * cand).sum(0)
        gaps_blend.append(float(Q(cand).max() - Q(blend[None])[0]))
        gaps_exp.append(float(Q(cand).max() - (w * Q(cand)).sum()))
    ax.plot(taus, gaps_exp, "o-", color=PAL[0], lw=2, ms=6, label="E_w[Q] gap")
    ax.plot(taus, gaps_blend, "s--", color=PAL[1], lw=2, ms=6, label="Q(blended action) gap")
    ax.set_xscale("log"); ax.set_xlabel("tau_adv"); ax.set_ylabel("value gap to argmax candidate")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) softmax blur bias grows with tau_adv (measured)", fontsize=9.5, loc="left")

    # (c) stability: oscillation amplitude vs c_q for two eta
    ax = axes[2]
    cqs = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for li, (eta, mv) in enumerate([(0.5, False), (1.5, False), (0.5, True), (1.5, True)]):
        amp = []
        for cq in cqs:
            _, traj = train_generator(eta=eta, c_q=cq, steps=1600, track=True, moving_q=mv)
            amp.append(traj.std(0).mean())
        ax.plot(cqs, amp, "o-" if mv else "o--", color=PAL[li % 2], lw=2, ms=5,
                label=f"eta={eta}, {'moving critic' if mv else 'static Q'}")
    ax.set_yscale("log")
    ax.set_xlabel("c_q (Q-ascent weight)"); ax.set_ylabel("E[g] oscillation amplitude (log)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("(c) chasing a moving critic: oscillation grows with c_q (solid);\n"
                 "static concave Q is stabilizing (dashed) -- damping is about nonstationarity",
                 fontsize=9, loc="left")

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("F8 · toy exactness: 2D bimodal mu, quadratic Q -- every quantity has ground truth",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{REPO}/viz/figs/f8_toy_exactness.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[f8] wrote {out}")


if __name__ == "__main__":
    main()
