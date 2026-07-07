"""
Phase 1 toy: does value-tilted drift recover the KL-optimal policy  pi* ~ mu * exp(A/alpha)
while STAYING multimodal?

Setup (no state; a pure policy-improvement operator test):
  - behavior mu = 4 Gaussian modes at (+/-2, +/-2)
  - advantage A(a) = a_x + a_y  (linear tilt -> favors top-right mode, kills bottom-left)
  - target pi* ~ mu * exp(A/alpha); per-mode target mass ~ exp(A_mode/alpha)

We train a one-step generator eps->a with drift_loss (positives = mu samples weighted by
exp(A/alpha)), and compare to an advantage-weighted Gaussian (AWR) baseline.

Claim to prove: drift matches the tilted multimodal target (low L1 on mode masses),
AWR-Gaussian cannot (it is unimodal -> collapses toward the mean).
Logs a figure + scalars to wandb.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from absl import app, flags

from utils.drift_loss import drift_loss
from utils.log_utils import setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_float("alpha", 2.0, "AWR temperature")
flags.DEFINE_integer("steps", 4000, "training steps")
flags.DEFINE_integer("seed", 0, "seed")

CENTERS = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]], dtype=np.float32)
SIGMA = 0.3


def adv_fn(a):
    return a[..., 0] + a[..., 1]


def sample_mu(key, n):
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, 4)
    return jnp.asarray(CENTERS)[idx] + SIGMA * jax.random.normal(k2, (n, 2))


class Gen(nn.Module):
    @nn.compact
    def __call__(self, eps):
        x = eps
        for _ in range(3):
            x = nn.gelu(nn.Dense(128)(x))
        return nn.Dense(2)(x)


def mode_masses(samples):
    d = np.linalg.norm(samples[:, None, :] - CENTERS[None], axis=-1)
    lbl = d.argmin(1)
    on_mode = (d.min(1) < 0.9)  # within 3*sigma of SOME center
    masses = np.array([((lbl == m) & on_mode).mean() for m in range(4)])
    return masses, float(on_mode.mean())


def target_masses(alpha):
    w = np.exp(adv_fn(CENTERS) / alpha)
    return w / w.sum()


def train_drift(alpha, steps, seed, n_gen=256, n_pos=256):
    key = jax.random.PRNGKey(seed)
    gen = Gen()
    params = gen.init(key, jnp.zeros((1, 2)))["params"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)

    @jax.jit
    def step(params, opt, key):
        k1, k2 = jax.random.split(key)
        pos = sample_mu(k1, n_pos)                            # [P, 2]
        w = jnp.exp(adv_fn(pos) / alpha)                      # [P]
        w = n_pos * w / w.sum()                               # mean 1

        def loss_fn(p):
            eps = jax.random.normal(k2, (n_gen, 2))
            g = gen.apply({"params": p}, eps)                # [G, 2]
            loss, info = drift_loss(
                gen=g[None], fixed_pos=pos[None], weight_pos=w[None]
            )
            return loss.mean(), info

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt = tx.update(grads, opt)
        params = optax.apply_updates(params, updates)
        return params, opt, loss

    for i in range(steps):
        key, k = jax.random.split(key)
        params, opt, loss = step(params, opt, k)
    eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (4000, 2))
    return np.array(gen.apply({"params": params}, eps))


def train_awr_gaussian(alpha, seed):
    """Advantage-weighted MLE of a single Gaussian (AWR baseline)."""
    key = jax.random.PRNGKey(seed + 7)
    pos = np.array(sample_mu(key, 20000))
    w = np.exp(adv_fn(pos) / alpha)
    w = w / w.sum()
    mean = (w[:, None] * pos).sum(0)
    cov = (w[:, None, None] * ((pos - mean)[:, :, None] * (pos - mean)[:, None, :])).sum(0)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=4000).astype(np.float32)


def main(_):
    setup_wandb(
        project=os.environ.get("WANDB_PROJECT", "rql-iclr2027-kernel-analysis"),
        group="dql-phase1-toy",
        name=f"toy_drift_sweep_sd{FLAGS.seed}",
    )
    # alpha=1e6 ~ unweighted (sanity: must recover all 4 modes equally)
    alphas = [1e6, 8.0, 4.0, 2.0]
    fig, ax = plt.subplots(2, len(alphas), figsize=(4 * len(alphas), 8))
    q_data = float(adv_fn(np.array(sample_mu(jax.random.PRNGKey(9), 4000))).mean())

    for ci, alpha in enumerate(alphas):
        tgt = target_masses(alpha)
        drift_s = train_drift(alpha, FLAGS.steps, FLAGS.seed)
        awr_s = train_awr_gaussian(alpha, FLAGS.seed)
        m_drift, on_drift = mode_masses(drift_s)
        m_awr, on_awr = mode_masses(awr_s)
        l1_drift = float(np.abs(m_drift - tgt).sum())
        l1_awr = float(np.abs(m_awr - tgt).sum())
        q_drift = float(adv_fn(drift_s).mean())
        q_awr = float(adv_fn(awr_s).mean())
        print(f"alpha={alpha:>7g} target={np.round(tgt,3)}")
        print(f"   drift masses={np.round(m_drift,3)} on={on_drift:.2f} L1={l1_drift:.3f} A={q_drift:.2f}")
        print(f"   awr-G masses={np.round(m_awr,3)} on={on_awr:.2f} L1={l1_awr:.3f} A={q_awr:.2f}")
        wandb.log({
            f"toy/l1_drift": l1_drift, f"toy/l1_awr": l1_awr,
            f"toy/onmode_drift": on_drift, f"toy/onmode_awr": on_awr,
            f"toy/meanA_drift": q_drift, f"toy/meanA_awr": q_awr,
            "alpha": alpha,
        }, step=ci)
        for row, s, name, on in [(0, drift_s, "DQL drift", on_drift), (1, awr_s, "AWR-Gauss", on_awr)]:
            a = ax[row, ci]
            a.scatter(s[:, 0], s[:, 1], s=3, alpha=0.25)
            a.scatter(CENTERS[:, 0], CENTERS[:, 1], c="r", marker="x", s=120)
            a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_aspect("equal")
            a.set_title(f"{name} a={alpha:g}\ntgt={np.round(tgt,2)} on={on:.2f}", fontsize=8)
    fig.suptitle(f"mean A: data={q_data:.2f}  (top: DQL drift, bottom: AWR-Gaussian)")
    fig.tight_layout()
    wandb.log({"toy/samples": wandb.Image(fig)})
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
