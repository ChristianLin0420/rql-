"""Rebuild a trained agent (and optionally its env/dataset) from a run directory.

    from viz.load_agent import load_run
    run = load_run("exp/rql-iclr2027-50tasks/DQL111-50/<env>/<run_name>")
    run.agent, run.env, run.train_dataset, run.config

Used by all viz/ figure scripts -- no retraining, works on CPU (set
JAX_PLATFORMS=cpu MUJOCO_GL=disabled before importing).
"""
import glob, json, os, sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents import agents                       # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.flax_utils import restore_agent      # noqa: E402


def load_run(run_dir, epoch=None, with_env=True):
    run_dir = os.path.join(REPO, run_dir) if not os.path.isabs(run_dir) else run_dir
    flags = json.load(open(os.path.join(run_dir, "flags.json")))
    config, env_name = dict(flags["agent"]), flags["env_name"]

    ckpts = sorted(int(os.path.basename(p)[len("params_"):-len(".pkl")])
                   for p in glob.glob(os.path.join(run_dir, "params_*.pkl")))
    if not ckpts:
        raise FileNotFoundError(f"no params_*.pkl under {run_dir}")
    epoch = epoch or ckpts[-1]

    env, eval_env, train_dataset, _ = make_env_and_datasets(env_name, agent_config=config) \
        if with_env else (None, None, None, None)
    if train_dataset is None:  # need example shapes even without env
        raise ValueError("with_env=False unsupported: dataset provides example shapes")

    ex = train_dataset if isinstance(train_dataset, dict) else dict(train_dataset)
    agent_class = agents[config["agent_name"]]
    create_kwargs = {}
    if getattr(agent_class, "needs_pool", False):
        # Placeholder pool of the right shapes: pool_obs/pool_act are pytree leaves, so
        # restore_agent overwrites them with the checkpointed pool (byte-exact).
        import numpy as np
        obs_dim = ex["observations"].shape[-1]
        act_dim = ex["actions"].shape[-1] * config["h"]
        create_kwargs["pool"] = (np.zeros((config["n_pool"], obs_dim), np.float32),
                                 np.zeros((config["n_pool"], act_dim), np.float32))
    agent = agent_class.create(
        flags.get("seed", 0), ex["observations"][:1], ex["actions"][:1], config, **create_kwargs)
    agent = restore_agent(agent, run_dir, epoch)
    return SimpleNamespace(agent=agent, env=env, eval_env=eval_env,
                           train_dataset=ex, config=config, flags=flags,
                           env_name=env_name, epoch=epoch, run_dir=run_dir)
