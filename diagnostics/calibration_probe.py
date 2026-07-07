"""Overestimation calibration probe.

Trains a DQL agent briefly, then measures whether the critic OVERESTIMATES:
  - gold standard: predicted Q(s0,a0) vs the realized discounted return the policy achieves.
      gap = Q_pred - realized_return.  ~0 => calibrated; >>0 => overestimating.
    Reported for BOTH Q_mean and Q_lcb (mean - rho*std) so we see if pessimism calibrates.
  - proxies (dataset batch): ensemble std and Q at the policy's CHOSEN action vs DATA actions.
"""
import os, numpy as np, jax, jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags
from einops import rearrange, repeat
from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.log_utils import setup_wandb
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "cube-double-play-singletask-v0", "env")
flags.DEFINE_integer("steps", 60000, "train steps")
flags.DEFINE_integer("n_episodes", 30, "calibration rollout episodes")
config_flags.DEFINE_config_file("agent", "agents/dql.py", lock_config=False)


def main(_):
    cfg = FLAGS.agent
    setup_wandb(project=os.environ.get("WANDB_PROJECT", "rql-iclr2027-kernel-analysis"),
                group="calibration", name=f"calib__{FLAGS.env_name}__rho{cfg['rho']}")
    env, eval_env, ds, _ = make_env_and_datasets(FLAGS.env_name, agent_config=cfg)
    ds = Dataset.create(**ds); ds.config = cfg
    np.random.seed(0)
    ex = ds.sample(4)
    agent = agents[cfg["agent_name"]].create(0, ex["observations"][0], ex["actions"][0], cfg)
    for i in range(1, FLAGS.steps + 1):
        agent, info = agent.update(ds.sample(cfg["batch_size"]))
        if i % 20000 == 0:
            print(f"[{i}] critic={float(info['critic_loss']):.2f} q={float(info['q_mean']):.1f}", flush=True)

    rho, gamma = cfg["rho"], cfg["discount"]

    def q_ens(s, a):  # s:[N,O] a:[N,Aflat] -> [ens,N]
        return np.array(agent.network.select("critic")(jnp.asarray(s), jnp.asarray(a)))

    # ---- gold standard: Q(s0,a0) vs realized discounted return ----
    rng = jax.random.PRNGKey(123)
    Qm, Ql, R, S0 = [], [], [], []
    for ep in range(FLAGS.n_episodes):
        ob, _ = eval_env.reset()
        rng, k = jax.random.split(rng)
        a0 = np.array(agent.sample_actions(obs=ob, temperature=0, seed=k))          # [h, ad]
        q = q_ens(np.atleast_2d(ob), a0.reshape(1, -1))[:, 0]                        # [ens]
        Qm.append(q.mean()); Ql.append(q.mean() - rho * q.std()); S0.append(q.std())
        G, disc, done = 0.0, 1.0, False
        while not done:
            rng, k = jax.random.split(rng)
            a = np.array(agent.sample_actions(obs=ob, temperature=0, seed=k))
            ob, r, term, trunc, _ = eval_env.step(a.copy())
            G += disc * r; disc *= gamma; done = bool(term or trunc)
        R.append(G)
    Qm, Ql, R, S0 = map(np.array, [Qm, Ql, R, S0])

    # ---- proxies on a dataset batch: chosen vs data ----
    b = ds.sample(256); s = np.asarray(b["observations"][0])
    a_data = np.asarray(rearrange(jnp.asarray(b["actions"][: cfg["h"]]), "h b d -> b (h d)"))
    A = a_data.shape[-1]; K = 32
    eps = np.array(jax.random.normal(jax.random.PRNGKey(7), (256, K, A)))
    cand = np.array(agent.network.select("actor")(jnp.asarray(repeat(s, "b o -> b k o", k=K)), jnp.asarray(eps)))
    qc = q_ens(repeat(s, "b o -> (b k) o", k=K), cand.reshape(256 * K, A)).reshape(-1, 256, K)
    lcb_c = qc.mean(0) - rho * qc.std(0)                                             # [256,K]
    ci = lcb_c.argmax(1)
    a_ch = cand[np.arange(256), ci]
    qd = q_ens(s, a_data); qh = q_ens(s, a_ch)                                       # [ens,256]

    out = {
        "realized_return": float(R.mean()),
        "Q_mean_s0a0": float(Qm.mean()), "Q_lcb_s0a0": float(Ql.mean()),
        "gap_mean(overest)": float((Qm - R).mean()), "gap_lcb(overest)": float((Ql - R).mean()),
        "ens_std_chosen": float(qh.std(0).mean()), "ens_std_data": float(qd.std(0).mean()),
        "Qmean_chosen_minus_data": float((qh.mean(0) - qd.mean(0)).mean()),
    }
    print("\n=== CALIBRATION (rho=%.2f) ===" % rho)
    for k, v in out.items():
        print(f"  {k:26s} {v:.2f}")
    print("  interpret: gap>>0 => overestimating; gap_lcb<<gap_mean => pessimism helps; "
          "ens_std_chosen>>ens_std_data => chosen actions are OOD")
    wandb.log(out); wandb.finish()


if __name__ == "__main__":
    app.run(main)
