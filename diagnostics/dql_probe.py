"""Quick mechanism probe for DQL on a single env: is the actor state-dependent,
does the critic rank actions, and does the argmax-Q policy actually reach goals?"""
import os, numpy as np, jax, jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags
from einops import rearrange, repeat
from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.evaluation import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "antmaze-large-navigate-singletask-v0", "env")
flags.DEFINE_integer("steps", 50000, "train steps")
config_flags.DEFINE_config_file("agent", "agents/dql.py", lock_config=False)


def main(_):
    cfg = FLAGS.agent
    env, eval_env, ds, _ = make_env_and_datasets(FLAGS.env_name, agent_config=cfg)
    ds = Dataset.create(**ds); ds.config = cfg
    np.random.seed(0)
    ex = ds.sample(4)
    agent = agents[cfg["agent_name"]].create(0, ex["observations"][0], ex["actions"][0], cfg)

    for i in range(1, FLAGS.steps + 1):
        agent, info = agent.update(ds.sample(cfg["batch_size"]))
        if i % 10000 == 0:
            g = lambda k: float(info[k]) if k in info else float('nan')
            print(f"[{i}] critic={g('critic_loss'):.3f} adv={g('adv_mean'):.3f} "
                  f"gen_std={g('gen_std'):.3f} gen_q={g('gen_q_mean'):.2f}", flush=True)

    # --- actor state-dependence + critic ranking (module names differ per agent) ---
    b = ds.sample(256)
    s = jnp.asarray(b["observations"][0]); A = cfg["action_dim"]
    pol = "generator" if cfg["agent_name"] == "dql_hybrid" else "actor"
    def qfn(ss, aa):
        if cfg["agent_name"] == "dql_hybrid":
            st = jnp.concatenate([ss, aa, jnp.ones((*aa.shape[:-1], 1))], -1)
            return agent.network.select("value")(st).mean(0)
        return agent.network.select("critic")(ss, aa).min(0)
    try:
        a_fixed = agent.network.select(pol)(s, jnp.zeros((256, A)))
        s0 = repeat(s[:1], "1 o -> k o", k=256)
        a_noise = agent.network.select(pol)(s0, jax.random.normal(jax.random.PRNGKey(1), (256, A)))
        print(f"actor std across STATES (fixed noise) = {float(a_fixed.std()):.3f}  "
              f"across NOISE (fixed state) = {float(a_noise.std()):.3f}")
        acts = rearrange(jnp.asarray(b["actions"][: cfg['h']]), "h b d -> b (h d)")
        q_data = qfn(s, acts)
        K = 32
        s_rep = repeat(s, "b o -> (b k) o", k=K)
        cand = agent.network.select(pol)(s_rep, jax.random.normal(jax.random.PRNGKey(2), (256 * K, A)))
        q_argmax = qfn(s_rep, cand).reshape(256, K).max(1)
        q_rand = qfn(s, jax.random.uniform(jax.random.PRNGKey(3), (256, A), minval=-1, maxval=1))
        print(f"Q  argmax-cand={float(q_argmax.mean()):.2f}  dataset={float(q_data.mean()):.2f}  random={float(q_rand.mean()):.2f}")
    except Exception as e:
        print("actor/critic probe skipped:", e)

    # --- rollouts with argmax-Q eval policy ---
    info, _, _ = evaluate(agent=agent, env=eval_env, env_name=FLAGS.env_name, config=cfg,
                          num_eval_episodes=20, num_video_episodes=0, video_frame_skip=3)
    print(f"ROLLOUT success={info.get('success'):.3f}  return={info.get('episode.return'):.1f}")


if __name__ == "__main__":
    app.run(main)
