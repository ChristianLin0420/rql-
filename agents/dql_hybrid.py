"""
DQL-actor + RQL-critic hybrid.

Motivation: RQL's flow-value critic V(s,x,f) produces a usable ACTION-DEPENDENT value
Q(s,a)=V(s,a,f=1) even in the dense-reward / one-action-per-state regime (where a plain
IQL Q(s,a) is under-determined and a sample-based one-step policy has no signal). We keep
RQL's critic + flow actor training EXACTLY as-is (the flow actor is needed for the reversal
that trains the value), and add a ONE-STEP drift generator as a parallel policy head:

  generator loss = drift( gen -> advantage-weighted data actions )        # in-support, multimodal
                 + q_coef * ( -normalized Q(s, gen) ),  Q(s,a)=V(s,a,1)    # one-step Q-ascent

The one-step generator is what we DEPLOY (argmax over K candidates by RQL value). This
isolates the question: given RQL's good value, does a one-step drifting actor work?
"""
import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange, repeat

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, OneStepGenerator
from utils.drift_loss import drift_loss


class DQLHybridAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _rql_value_q(self, s, a):
        """RQL value as an action-dependent Q: Q(s,a) = V(s, a, f=1)."""
        st = jnp.concatenate([s, a, jnp.ones((*a.shape[:-1], 1))], axis=-1)
        return self.network.select("value")(st)  # [ens, ...]

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        bs, action_dim = self.config["batch_size"], self.config["action_dim"]
        rng, n_rng, u_rng, r_rng = jax.random.split(rng, 4)
        s0 = batch["observations"][0]

        # ============================ RQL critic + flow actor (unchanged) ============================
        next_state = jnp.concatenate([batch["observations"][-1], jax.random.normal(n_rng, (bs, action_dim)), jnp.zeros((bs, 1))], axis=-1)
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        d = jnp.concatenate([
            jax.random.uniform(u_rng, (bs // 2,)),
            jax.random.randint(r_rng, (bs // 2,), 0, self.config["flow_steps"] + 1) / self.config["flow_steps"],
        ], 0)
        d_b = d / self.config["flow_steps"]
        actions = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")

        x_f = jnp.copy(actions)
        f = jnp.ones((bs, 1))
        for _ in range(self.config["flow_steps"]):
            fm = jnp.concatenate([s0, x_f, f], -1)
            out = self.network.select("actor")(fm).mode()
            x_f = x_f - out * d_b[..., None]
            f = f - d_b[..., None]
        state = jnp.concatenate([s0, jax.lax.stop_gradient(x_f), f], axis=-1)
        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate([jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], axis=0)
        n_rews = (batch["rewards"] * self.config["discount_mul"][..., None] * (1 - rs_terminals)).sum(0)
        tqt_q = n_rews + (self.config["discount"] ** self.config["h"]) * next_q * batch["masks"][-2]
        valids = (rs_terminals.sum(0) <= 1).astype(tqt_q.dtype)
        critic_loss = (self.expectile_loss(tqt_q - q, tqt_q - q, self.config["expectile"]) * valids).mean()

        # RQL flow BC loss + flow-actor Q improvement
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (bs, action_dim))
        x_1 = actions
        t = jax.random.uniform(t_rng, (bs, 1))
        x_t = (1 - t) * x_0 + t * x_1
        tgt = x_1 - x_0
        pred = self.network.select("actor")(jnp.concatenate([s0, x_t, t], axis=-1), params=grad_params).mode()
        q_pe = self.network.select("value")(
            jnp.concatenate([s0, x_t + pred * jnp.minimum(1 / self.config["flow_steps"], 1 - t),
                             jnp.clip(t + 1 / self.config["flow_steps"], max=1)], axis=-1)
        ).mean(axis=0)
        ac_mask = repeat(1 - rs_terminals[:-1], "h b -> b (h r)", r=self.config["action_dim"] // self.config["h"])
        bc_loss = (jnp.square(pred - tgt) * ac_mask).mean()
        flow_actor_loss = -(q_pe * valids).mean()

        # ============================ one-step drift generator (new) ============================
        rng, g_rng = jax.random.split(rng)
        G = self.config["n_gen"]
        # advantage from RQL value: A = Q(s,a) - V(s);  V(s)=value(s, noise, f=0)
        q_data = self._rql_value_q(s0, actions).mean(axis=0)                       # [B]
        v_s = self.network.select("value")(
            jnp.concatenate([s0, jax.random.normal(g_rng, (bs, action_dim)), jnp.zeros((bs, 1))], -1)
        ).mean(axis=0)
        adv = jax.lax.stop_gradient(q_data - v_s)
        w_adv = jnp.clip(jnp.exp(jnp.clip(adv / self.config["awr_alpha"], max=6.0)), max=100.0)  # [B]

        sq = jnp.sum((s0[:, None, :] - s0[None, :, :]) ** 2, axis=-1)
        bw = self.config["state_bw"] * (jax.lax.stop_gradient(sq.mean()) + 1e-8)
        w_state = jax.nn.softmax(-sq / bw, axis=-1)
        w_pos = w_state * w_adv[None, :]
        w_pos = jax.lax.stop_gradient(bs * w_pos / (w_pos.sum(-1, keepdims=True) + 1e-8))

        eps = jax.random.normal(rng, (bs, G, action_dim))
        s_rep = repeat(s0, "b o -> b g o", g=G)
        gen = self.network.select("generator")(s_rep, eps, params=grad_params)      # [B,G,A]
        fixed_pos = repeat(actions, "j a -> b j a", b=bs)
        drift, dinfo = drift_loss(gen=gen, fixed_pos=fixed_pos, weight_pos=w_pos)

        gen_flat = rearrange(gen, "b g a -> (b g) a")
        s_flat = repeat(s0, "b o -> (b g) o", g=G)
        q_gen = self._rql_value_q(s_flat, gen_flat).mean(axis=0)                     # grad -> generator only (value has no grad_params here)
        q_norm = jax.lax.stop_gradient(jnp.abs(q_gen).mean() + 1e-6)
        gen_loss = self.config["gen_drift_coef"] * drift.mean() + self.config["q_coef"] * (-(q_gen / q_norm).mean())

        total_loss = flow_actor_loss + bc_loss * self.config["alpha"] + critic_loss + gen_loss
        return total_loss, {
            "critic_loss": critic_loss, "bc_loss": bc_loss, "flow_actor_loss": flow_actor_loss,
            "gen_drift": drift.mean(), "gen_q_mean": q_gen.mean(), "q": q.mean(),
            "adv_mean": adv.mean(), "w_adv_max": w_adv.max(), "gen_std": gen.std(),
        }

    def target_update(self, network, module_name, d):
        new_tp = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_tp

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))
        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1 - self.config["ema"])
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        obs = jnp.atleast_2d(obs)[-1:]
        A = self.config["action_dim"]
        if temperature > 0:
            a = self.network.select("generator")(obs, jax.random.normal(seed, (1, A)))[0]
        else:
            K = self.config["eval_samples"]
            eps = jax.random.normal(seed, (K, A))
            obs_rep = repeat(obs, "1 o -> k o", k=K)
            cand = self.network.select("generator")(obs_rep, eps)                 # [K, A]
            qv = self._rql_value_q(obs_rep, cand).mean(axis=0)                    # [K]
            a = cand[jnp.argmax(qv)]
        a = jnp.clip(a, -1, 1)
        return rearrange(a, "(h d) -> h d", h=self.config["h"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = ex_actions[..., :1]
        ex_in = jnp.concatenate([ex_observations, ex_actions, ex_times], -1)
        action_dim = ex_actions.shape[-1]

        value_def = Value(hidden_dims=config["value_hidden_dims"], layer_norm=config["layer_norm"], num_ensembles=config["ensemble_ct"])
        actor_def = Actor(hidden_dims=config["actor_hidden_dims"], action_dim=action_dim,
                          layer_norm=config["actor_layer_norm"], tanh_squash=False,
                          state_dependent_std=True, const_std=False, final_fc_init_scale=1)
        gen_def = OneStepGenerator(hidden_dims=config["actor_hidden_dims"], action_dim=action_dim,
                                   layer_norm=config["actor_layer_norm"], tanh_squash=True)

        network_info = dict(
            value=(value_def, (ex_in,)),
            target_value=(copy.deepcopy(value_def), (ex_in,)),
            actor=(actor_def, (ex_in,)),
            target_actor=(copy.deepcopy(actor_def), (ex_in,)),
            generator=(gen_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_actor"] = params["modules_actor"]
        config["action_dim"] = action_dim
        config["discount_mul"] = jnp.array(config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf]))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(dict(
        agent_name="dql_hybrid",
        h=1, alpha=0.1, expectile=0.5, ensemble_ct=10, rho=0.5,  # RQL critic/flow hyperparams
        lr=3e-4, discount=0.99, batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
        layer_norm=True, actor_layer_norm=False, tau=0.005, ema=0.999, flow_steps=10, q_agg="mean",
        # one-step drift generator head:
        n_gen=8, awr_alpha=0.5, state_bw=0.05, gen_drift_coef=1.0, q_coef=1.0, eval_samples=32,
    ))
    return config
