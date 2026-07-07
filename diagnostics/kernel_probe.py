"""
Kernel diagnostic for Reversal Q-Learning (RQL).

Goal: empirically characterize the RQL "reversal kernel" -- the flow value
function V_phi(s, x, f) that is trained by reverse-integrating the actor's
flow field from a data action back to a random flow-time f, and bootstrapped
at f=0 with FRESH Gaussian noise.

We probe four properties on a partially trained agent:

  A. f=0 noise-dependence.  The TD target uses V_target(s', x0', f=0) with a
     single fresh noise x0'. For that to be a valid state value V(s'), the
     network's f=0 slice must be (nearly) constant in the noise argument.
     We measure std_over_noise vs std_over_states.  ratio ~ 0 => valid V(s);
     ratio ~ 1 => the bootstrap target is a high-variance value-OF-noise.

  B. Reversal invertibility.  reverse(a) -> x0_hat ; forward(x0_hat) -> a_hat.
     If the reversal kernel is a true inverse of the sampler, a_hat ~ a.
     Large error => the value inputs x_f live off the flow-matching manifold.

  C. Bootstrap grounding mismatch.  At d=1 the reversal trains
     V(s, x0_from_a, 0) -> Q(s,a). The bootstrap instead reads
     V(s, fresh_noise, 0). We compare the two on the SAME states.

  D. Flow-time invariance.  RQL regresses V(s, x_f, 1-d) to the SAME target
     for every reversal depth d. We measure how far V actually drifts along
     the reversal path (std over d) -- the residual the critic cannot fit.

Results are logged to wandb (project rql-iclr2027-kernel-analysis).
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import wandb
from ml_collections import config_flags
from absl import app, flags

from agents import agents
from agents.rql import get_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.log_utils import setup_wandb, get_exp_name

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-navigate-singletask-v0', 'env')
flags.DEFINE_integer('train_steps', 50000, 'quick training steps before/along probes')
flags.DEFINE_integer('probe_interval', 5000, 'probe every N steps')
flags.DEFINE_integer('seed', 0, 'seed')
config_flags.DEFINE_config_file('agent', 'agents/rql.py', lock_config=False)


def reverse_to_f(agent, obs, actions, d):
    """Reverse-integrate the actor flow from f=1 back to f=1-d (rql.py:52-61)."""
    cfg = agent.config
    bs = actions.shape[0]
    d_b = (d / cfg['flow_steps'])
    x_f = jnp.asarray(actions)
    f = jnp.ones((bs, 1))
    for _ in range(cfg['flow_steps']):
        inp = jnp.concatenate([obs, x_f, f], -1)
        out = agent.network.select('actor')(inp).mode()
        x_f = x_f - out * d_b
        f = f - d_b
    return x_f, f


def forward_flow(agent, obs, x0):
    """Forward Euler sampler from an arbitrary x0 at t=0 (rql.py:159-165)."""
    cfg = agent.config
    actions = jnp.asarray(x0)
    for i in range(cfg['flow_steps']):
        t = jnp.full((actions.shape[0], 1), i / cfg['flow_steps'])
        inp = jnp.concatenate([obs, actions, t], -1)
        out = agent.network.select('actor')(inp).mode()
        actions = actions + out / cfg['flow_steps']
    return actions


def value_ens(agent, obs, x, f):
    inp = jnp.concatenate([obs, x, f], -1)
    return agent.network.select('value')(inp)  # (ensemble, B)


def probe(agent, obs, actions, key):
    """Return a dict of scalar diagnostics."""
    cfg = agent.config
    B = obs.shape[0]
    out = {}

    # ---- A. f=0 noise-dependence ----
    S = min(64, B)             # states
    K = 32                     # noise samples per state
    s_rep = jnp.repeat(obs[:S], K, axis=0)               # (S*K, obs)
    noise = jax.random.normal(key, (S * K, cfg['action_dim']))
    zeros = jnp.zeros((S * K, 1))
    v = value_ens(agent, s_rep, noise, zeros).mean(0)    # ensemble-mean, (S*K,)
    v = v.reshape(S, K)
    std_over_noise = v.std(axis=1).mean()                # avg within-state spread over noise
    std_over_states = v.mean(axis=1).std()               # spread of per-state means
    out['A/std_over_noise'] = float(std_over_noise)
    out['A/std_over_states'] = float(std_over_states)
    out['A/noise_to_state_ratio'] = float(std_over_noise / (std_over_states + 1e-8))

    # ---- B. reversal invertibility (d=1: reverse all the way to f=0) ----
    x0_hat, f0 = reverse_to_f(agent, obs, actions, d=1.0)
    a_hat = forward_flow(agent, obs, x0_hat)
    recon_err = jnp.sqrt(jnp.mean((a_hat - actions) ** 2))
    act_scale = jnp.sqrt(jnp.mean(actions ** 2))
    out['B/reversal_recon_rmse'] = float(recon_err)
    out['B/reversal_recon_rel'] = float(recon_err / (act_scale + 1e-8))
    out['B/x0_hat_std'] = float(x0_hat.std())   # should be ~1 if it maps to N(0,1)
    out['B/final_f'] = float(jnp.mean(f0))       # should be ~0

    # ---- C. bootstrap grounding mismatch at f=0 ----
    v_rev0 = value_ens(agent, obs, x0_hat, jnp.zeros((B, 1))).mean(0)      # trained toward Q(s,a)
    fresh = jax.random.normal(jax.random.fold_in(key, 1), (B, cfg['action_dim']))
    v_fresh = value_ens(agent, obs, fresh, jnp.zeros((B, 1))).mean(0)      # what bootstrap uses
    out['C/V_reversed_a_mean'] = float(v_rev0.mean())
    out['C/V_fresh_noise_mean'] = float(v_fresh.mean())
    out['C/grounding_gap_mean'] = float(jnp.mean(v_rev0 - v_fresh))
    out['C/grounding_gap_absmean'] = float(jnp.mean(jnp.abs(v_rev0 - v_fresh)))

    # ---- D. flow-time invariance along the reversal path ----
    vs = []
    for d in [0.25, 0.5, 0.75, 1.0]:
        x_f, f = reverse_to_f(agent, obs, actions, d=d)
        vs.append(value_ens(agent, obs, x_f, f).mean(0))   # (B,)
    vs = jnp.stack(vs, 0)                                    # (4, B)
    out['D/V_std_over_depth'] = float(vs.std(axis=0).mean())
    out['D/V_range_over_depth'] = float((vs.max(0) - vs.min(0)).mean())
    out['D/V_scale'] = float(jnp.abs(vs).mean())
    return out


def main(_):
    cfg = FLAGS.agent
    setup_wandb(
        project=os.environ.get('WANDB_PROJECT', 'rql-iclr2027-kernel-analysis'),
        group='kernel-diagnostics',
        name=f'kernel-probe__{FLAGS.env_name}__{get_exp_name(FLAGS.seed)}',
    )

    env, _, train_dataset, _ = make_env_and_datasets(FLAGS.env_name, agent_config=cfg)
    train_dataset = Dataset.create(**train_dataset)
    train_dataset.config = cfg

    np.random.seed(FLAGS.seed)
    ex = train_dataset.sample(4)
    agent = agents[cfg['agent_name']].create(
        FLAGS.seed, ex['observations'][0], ex['actions'][0], cfg,
    )

    key = jax.random.PRNGKey(FLAGS.seed + 100)
    for step in range(1, FLAGS.train_steps + 1):
        batch = train_dataset.sample(cfg['batch_size'])
        agent, info = agent.update(batch)
        if step % FLAGS.probe_interval == 0 or step == 1:
            key, pk = jax.random.split(key)
            pb = train_dataset.sample(512)
            obs = jnp.asarray(pb['observations'][0])
            from einops import rearrange
            acts = rearrange(jnp.asarray(pb['actions'][:cfg['h']]), 'h b d -> b (h d)')
            diag = probe(agent, obs, acts, pk)
            diag['training/critic_loss'] = float(info['critic_loss'])
            diag['training/bc_loss'] = float(info['bc_loss'])
            diag['training/q_mean'] = float(info['q_mean'])
            wandb.log(diag, step=step)
            print(f"[{step}] "
                  f"A.noise/state={diag['A/noise_to_state_ratio']:.3f} "
                  f"B.recon_rel={diag['B/reversal_recon_rel']:.3f} "
                  f"C.gap={diag['C/grounding_gap_absmean']:.3f} "
                  f"D.std_depth={diag['D/V_std_over_depth']:.3f} "
                  f"q={diag['training/q_mean']:.2f}", flush=True)
    wandb.finish()


if __name__ == '__main__':
    app.run(main)
