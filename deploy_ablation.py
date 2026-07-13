"""Deploy-rule ablation on a trained DQL v11 checkpoint.

The v11 critic is action-sharp (rank_acc=1.0) but eval success is 0.0. Question: is the CRITIC-AGNOSTIC
medoid deploy wasting it? Load a checkpoint and roll out the SAME policy net under 3 deploy rules:
  - medoid   : geometric median of K generator samples (current; ignores the critic)
  - argmaxQ  : pick the candidate with highest Q_lcb  (uses the sharp critic)
  - robustQ  : medoid of the top-k highest-Q candidates (critic-guided but robust to a single outlier)
"""
import os, sys, glob
import numpy as np
import jax, jax.numpy as jnp
from einops import repeat, rearrange

sys.path.insert(0, "/localhome/local-chrislin/rql-")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

from agents import agents
from agents.dql_v11 import get_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent

ENV = "cube-double-play-singletask-v0"
GROUP = "DQL-v11-cube-double"
EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 250000
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
K = 32
TOPK = 8

env, eval_env, train_ds, _ = make_env_and_datasets(ENV, frame_stack=None, agent_config={"h": 5})
train_ds = Dataset.create(**train_ds)
train_ds = ReplayBuffer.create_from_initial_dataset(dict(train_ds), size=train_ds.size + 1)
train_ds.frame_stack = None

cfg = get_config()
cfg["h"] = 5; cfg["expectile"] = 0.9; cfg["rho"] = 0.5
ex = train_ds.sample(1)
agent = agents["dql_v11"].create(0, ex["observations"], ex["actions"], cfg)

ckpt_dir = sorted(glob.glob(f"exp/*/{GROUP}/*"))[-1]
agent = restore_agent(agent, ckpt_dir, EPOCH)
print(f"restored {ckpt_dir}  epoch {EPOCH}", flush=True)
A = agent.config["action_dim"]; H = agent.config["h"]


@jax.jit
def candidates(obs, key):
    obs = jnp.atleast_2d(obs)[-1:]
    cand = agent.network.select("actor")(repeat(obs, "1 o -> k o", k=K), jax.random.normal(key, (K, A)))
    q = agent._qlcb("q", repeat(obs, "1 o -> k o", k=K), cand)   # [K]
    return jnp.clip(cand, -1, 1), q


def pick(cand, q, rule):
    if rule == "medoid":
        d = jnp.sum((cand[:, None, :] - cand[None, :, :]) ** 2, -1)
        a = cand[jnp.argmin(d.sum(-1))]
    elif rule == "argmaxQ":
        a = cand[jnp.argmax(q)]
    else:  # robustQ: medoid among top-k by Q
        idx = jnp.argsort(q)[-TOPK:]
        top = cand[idx]
        d = jnp.sum((top[:, None, :] - top[None, :, :]) ** 2, -1)
        a = top[jnp.argmin(d.sum(-1))]
    return np.asarray(rearrange(a, "(h d) -> h d", h=H))


def rollout(rule, seed0):
    succ = []
    key = jax.random.PRNGKey(seed0)
    for ep in range(N_EP):
        obs, info = eval_env.reset()
        done = False; s = 0.0
        while not done:
            key, k2 = jax.random.split(key)
            cand, q = candidates(jnp.asarray(obs), k2)
            a = pick(cand, q, rule)
            obs, r, term, trunc, info = eval_env.step(a)
            s = max(s, float(info.get("success", 0.0)))
            done = term or trunc
        succ.append(s)
    return float(np.mean(succ))


print(f"{'rule':>10} | success over {N_EP} ep")
for rule in ["medoid", "argmaxQ", "robustQ"]:
    sr = rollout(rule, seed0=42)
    print(f"{rule:>10} | {sr*100:5.1f}%", flush=True)
