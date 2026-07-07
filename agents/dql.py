"""
Drift Q-Learning (DQL).

A one-step generative policy trained by *drifting* toward the value-tilted behavior
distribution, combined with an IQL-style critic.

Actor  a = pi_theta(s, eps)  -- a single forward pass (no flow-time, no reversal,
       no backprop-through-sampling). The policy distribution q(.|s) is shaped during
       training so that its equilibrium is the KL-regularized optimal policy
            pi*(a|s)  ~  mu(a|s) * exp(A(s,a) / alpha).
Critic IQL:  V(s) expectile-regresses target Q(s,a);  Q(s,a) regresses r + gamma^h V(s').

Policy improvement = a gradient-free drift (utils/drift_loss.drift_loss):
  For each state s in the batch we generate G actions and attract them toward the
  batch's dataset actions, each positive weighted by
        w(s, a_j) = k_state(s, s_j) * exp(A(s_j, a_j) / alpha),
  while the generated samples repel each other (built into drift_loss -> multimodal).
This is a nonparametric, one-step generalization of advantage-weighted regression.
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
from utils.networks import OneStepGenerator, Value


class DQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(diff, expectile):
        weight = jnp.where(diff >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _chunk_actions(self, batch):
        return rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")

    def _q_lcb(self, module, *args):
        """Ensemble lower-confidence-bound Q: mean - rho*std (pessimism vs overestimation)."""
        q = self.network.select(module)(*args)              # [ensemble, ...]
        return q.mean(axis=0) - self.config["rho"] * q.std(axis=0)

    def _nstep_target(self, batch, next_v):
        """h-step discounted reward + gamma^h V(s') with terminal masking (mirrors RQL)."""
        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], axis=0
        )
        n_rews = (
            batch["rewards"] * self.config["discount_mul"][..., None] * (1 - rs_terminals)
        ).sum(0)
        tgt = n_rews + (self.config["discount"] ** self.config["h"]) * next_v * batch["masks"][-2]
        valids = (rs_terminals.sum(0) <= 1).astype(tgt.dtype)
        return tgt, valids

    # ------------------------------------------------------------------ critic
    def critic_loss(self, batch, grad_params, rng):
        s = batch["observations"][0]
        s_next = batch["observations"][-1]
        actions = self._chunk_actions(batch)
        B = s.shape[0]

        q_tgt = self._q_lcb("target_critic", s, actions)
        v = self.network.select("value")(s, params=grad_params)
        _, valids = self._nstep_target(batch, jnp.zeros_like(v))
        # V(s) <- expectile of target Q(s, a)  (defines the advantage for the drift)
        value_loss = (self.expectile_loss(q_tgt - v, self.config["expectile"]) * valids).mean()

        if self.config["critic_mode"] == "sac":
            # Action-DEPENDENT bootstrap: target uses policy actions at s' (best of K),
            # so Q(s,a) becomes action-sensitive even with 1 data action/state.
            K = self.config["critic_next_samples"]
            eps = jax.random.normal(rng, (B, K, self.config["action_dim"]))
            s_next_rep = repeat(s_next, "b o -> b k o", k=K)
            a_next = self.network.select("actor")(s_next_rep, eps)              # [B,K,A]
            an = rearrange(a_next, "b k a -> (b k) a")
            sn = rearrange(s_next_rep, "b k o -> (b k) o")
            q_next = self._q_lcb("target_critic", sn, an).reshape(B, K)
            next_val = q_next.max(axis=1)                                       # in-support max over policy samples
        else:
            next_val = self.network.select("value")(s_next)                    # IQL: V(s')

        target_q, valids = self._nstep_target(batch, next_val)
        q = self.network.select("critic")(s, actions, params=grad_params)  # [ens, B]
        critic_loss = (((q - target_q[None]) ** 2).mean(axis=0) * valids).mean()

        adv = (q_tgt - v).mean()
        return value_loss, critic_loss, {
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "v_mean": v.mean(),
            "q_mean": q.mean(),
            "adv_mean": adv,
            "target_q_mean": target_q.mean(),
        }

    # ------------------------------------------------------------------ actor (value-selection drift)
    def actor_loss(self, batch, grad_params, rng):
        """GRADIENT-FREE value-selection drift toward pi* (Round 2).

        Attraction targets are the batch's IN-SUPPORT data actions {a_j}, reweighted per state by
        RELATIVE value at the current state and state proximity:
            w_ij = softmax_j( zscore_j[Q(s_i, a_j)] / alpha  -  ||s_i-s_j||^2 / bw ).
        This is a strong, scale-free estimate of pi*(.|s_i) supported on data actions (value
        selection + stitching), avoiding the collapse of pooling the generator's own samples.
        Drift = normalized mean-shift score  V = S_p - S_q = mean_p - mean_q  (attraction / repulsion).
        Only Q-evaluations are used (no grad_a Q) -> WGF-toward-pi* stays intact.
        """
        s = batch["observations"][0]                       # [B, obs]
        a_data = self._chunk_actions(batch)                # [B, A]
        B, A = a_data.shape
        G = self.config["n_gen"]

        # Q(s_i, a_j) for every (state, data-action) pair -> [B, B]
        si = repeat(s, "i o -> (i j) o", j=B)
        aj = repeat(a_data, "j a -> (i j) a", i=B)
        q_ij = self._q_lcb("critic", si, aj).reshape(B, B)
        q_ij = jax.lax.stop_gradient(q_ij)
        q_n = (q_ij - q_ij.mean(-1, keepdims=True)) / (q_ij.std(-1, keepdims=True) + 1e-6)
        sq_s = jnp.sum((s[:, None, :] - s[None, :, :]) ** 2, axis=-1)          # [B, B]
        bw_s = self.config["state_bw"] * (jax.lax.stop_gradient(sq_s.mean()) + 1e-8)
        w = jax.lax.stop_gradient(jax.nn.softmax(q_n / self.config["alpha"] - sq_s / bw_s, axis=-1))  # [B,B]

        eps = jax.random.normal(rng, (B, G, A))
        s_rep = repeat(s, "b o -> b g o", g=G)
        gen = self.network.select("actor")(s_rep, eps, params=grad_params)     # [B, G, A]
        gsg = jax.lax.stop_gradient(gen)
        a_sq = jnp.sum(a_data ** 2, axis=-1)                                   # [B(j)]
        g_sq = jnp.sum(gsg ** 2, axis=-1)                                      # [B, G]

        # attraction S_p: normalized mean-shift toward the value-selected data actions (kernel x w)
        dist_pa = g_sq[..., None] + a_sq[None, None, :] - 2 * jnp.einsum("bga,ja->bgj", gsg, a_data)  # [B,G,B]
        tau2 = self.config["tau_scale"] * (jax.lax.stop_gradient(dist_pa.mean()) + 1e-8)
        kern_p = jnp.exp(-dist_pa / (2 * tau2)) * w[:, None, :]
        mean_p = jnp.einsum("bgj,ja->bga", kern_p, a_data) / (kern_p.sum(-1, keepdims=True) + 1e-12)

        # repulsion S_q: normalized mean-shift over generator samples (exclude self)
        dist_gg = g_sq[..., None] + g_sq[:, None, :] - 2 * jnp.einsum("bga,bha->bgh", gsg, gsg)   # [B,G,G]
        kern_q = jnp.exp(-dist_gg / (2 * tau2)) * (1.0 - jnp.eye(G))[None]
        mean_q = jnp.einsum("bgh,bha->bga", kern_q, gsg) / (kern_q.sum(-1, keepdims=True) + 1e-12)

        V = mean_p - mean_q                                                   # [B, G, A]
        goal = jax.lax.stop_gradient(gen + self.config["drift_step"] * V)
        actor_loss = ((gen - goal) ** 2).sum(-1).mean()                      # gradient-free drift regression

        info = {
            "actor_drift_loss": actor_loss,
            "drift_norm": jnp.sqrt((V ** 2).sum(-1)).mean(),
            "tau2": tau2,
            "w_self": jnp.diag(w).mean(),     # weight the policy puts on its own data action
            "w_max": w.max(-1).mean(),        # peakedness of the value+proximity selection
            "gen_std": gen.std(),
        }
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng):
        rng, a_rng, c_rng = jax.random.split(rng, 3)
        value_loss, crit_loss, cinfo = self.critic_loss(batch, grad_params, c_rng)
        act_loss, ainfo = self.actor_loss(batch, grad_params, a_rng)
        total = value_loss + crit_loss + act_loss
        info = {"total_loss": total, **cinfo, **ainfo}
        return total, info

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

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic", d=self.config["tau"])
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        obs = jnp.atleast_2d(obs)[-1:]                      # [1, obs]
        A = self.config["action_dim"]
        if temperature > 0:
            # exploration: single stochastic sample
            eps = jax.random.normal(seed, (1, A)) * self.config["expl_temp"]
            a = self.network.select("actor")(obs, eps)[0]
        else:
            # ON-MANIFOLD deterministic eval: deploy the policy MODE (medoid of K samples).
            # No critic query -> never selects the OOD actions the critic overestimates
            # (RQL's "no argmax-over-samples" defense; improvement is baked into the drift-trained generator).
            K = self.config["eval_samples"]
            eps = jax.random.normal(seed, (K, A))
            obs_rep = repeat(obs, "1 o -> k o", k=K)
            cand = self.network.select("actor")(obs_rep, eps)          # [K, A]
            d = jnp.sum((cand[:, None, :] - cand[None, :, :]) ** 2, axis=-1)  # [K, K]
            a = cand[jnp.argmin(d.sum(-1))]                            # medoid = in-support policy mode
        a = jnp.clip(a, -1, 1)
        return rearrange(a, "(h d) -> h d", h=self.config["h"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        action_dim = ex_actions.shape[-1]
        ex_obs = ex_observations
        ex_noise = ex_actions

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=1,
        )
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        actor_def = OneStepGenerator(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
        )

        network_info = dict(
            value=(value_def, (ex_obs,)),
            critic=(critic_def, (ex_obs, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_obs, ex_actions)),
            actor=(actor_def, (ex_obs, ex_noise)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["action_dim"] = action_dim
        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf])
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="dql",
            h=1,
            alpha=1.0,          # softmax temperature for value selection: p = softmax(z(Q)/alpha)
            expectile=0.9,      # IQL expectile for V
            critic_mode="iql",  # "iql" (V-bootstrap) or "sac" (action-dependent policy-bootstrap)
            critic_next_samples=4,  # K policy samples at s' for sac-mode target
            ensemble_ct=10,     # Q ensemble (for LCB uncertainty)
            rho=0.5,            # LCB pessimism: Q_lcb = mean - rho*std (anti-overestimation)
            n_gen=16,           # generator samples per state (repulsion set)
            state_bw=0.15,      # state-proximity bandwidth (x mean sq dist) in the value+proximity selection
            tau_scale=0.5,      # action-kernel bandwidth tau^2 = tau_scale * mean pairwise sq dist
            drift_step=1.0,     # step size for the drift regression target x + drift_step * V
            eval_samples=32,    # candidates at eval; pick argmax_a Q(s,a)
            expl_temp=1.0,      # noise scale during online exploration
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            tau=0.005,          # target critic EMA
        )
    )
    return config
