"""
DQL (v10): one-step drift actor + GEOMETRIC interpolation-value critic.

Recovers RQL's action-informative, on-manifold value WITHOUT the multi-step flow:
  - value V(s, x, tau) is trained on the straight line x_tau = (1-tau)*eps + tau*a  (a from data).
    No learned flow -> no reversal -> no kernel bug; coordinate is stationary; V(s,a,1)=Q(s,a).
  - MARGINALIZED boundary V(s') = mean_k V(s', eps_k, 0)  (fixes RQL's single-sample noise-dependent f=0).
  - NOISE-INVARIANCE reg at tau=0 -> a proper state value.
Actor: one-step generator, trust-region value-selection drift (multimodal via repulsion) + PRIMARY
on-manifold improvement q_pe (RQL-style, but on the interpolation -> bounded, no OOD drift/decay).
Deploy: one forward pass (medoid). Beats RQL where its kernel collapses (manipulation).
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


class DQLv10Agent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        w = jnp.where(adv >= 0, expectile, 1 - expectile)
        return w * diff ** 2

    def _chunk(self, batch):
        return rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")

    def _nstep(self, batch, next_v):
        rs = jnp.concatenate([jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], 0)
        n_rews = (batch["rewards"] * self.config["discount_mul"][..., None] * (1 - rs)).sum(0)
        tgt = n_rews + (self.config["discount"] ** self.config["h"]) * next_v * batch["masks"][-2]
        valids = (rs.sum(0) <= 1).astype(tgt.dtype)
        return tgt, valids

    def _V(self, module, s, x, tau, params=None):
        return self.network.select(module)(jnp.concatenate([s, x, tau], -1), params=params)

    def _lcb(self, v):
        return v.mean(0) - self.config["rho"] * v.std(0)

    def _Qlcb(self, s, a):  # value at tau=1 = action-informative Q
        return self._lcb(self._V("value", s, a, jnp.ones((*a.shape[:-1], 1))))

    @jax.jit
    def total_loss(self, batch, grad_params, rng):
        B, A = self.config["batch_size"], self.config["action_dim"]
        s0 = batch["observations"][0]
        s_next = batch["observations"][-1]
        a_data = self._chunk(batch)
        rng, e1, b1, ea, ta, pr = jax.random.split(rng, 6)

        # ===================== interpolation-value critic =====================
        eps = jax.random.normal(e1, (B, A))
        tau = jax.random.uniform(jax.random.fold_in(e1, 1), (B, 1))
        x_tau = (1 - tau) * eps + tau * a_data
        V_pred = self._V("value", s0, x_tau, tau, params=grad_params)                 # [ens, B]

        K = self.config["boundary_k"]
        epsb = jax.random.normal(b1, (B, K, A))
        sn = repeat(s_next, "b o -> (b k) o", k=K)
        Vb = self._V("target_value", sn, rearrange(epsb, "b k a -> (b k) a"), jnp.zeros((B * K, 1)))
        Vb = self._lcb(Vb).reshape(B, K).mean(-1)                                      # marginalized boundary V(s')
        y, valids = self._nstep(batch, jax.lax.stop_gradient(Vb))
        critic_v = (self.expectile_loss(y[None] - V_pred, y[None] - V_pred, self.config["expectile"]) * valids).mean()

        # noise-invariance at tau=0 (a proper state value)
        s0k = repeat(s0, "b o -> (b k) o", k=K)
        Vinv = self._V("value", s0k, rearrange(epsb, "b k a -> (b k) a"), jnp.zeros((B * K, 1)),
                       params=grad_params).mean(0).reshape(B, K)
        L_inv = Vinv.var(-1).mean()
        critic_loss = critic_v + self.config["inv_coef"] * L_inv

        # ===================== one-step drift actor =====================
        G = self.config["n_gen"]
        eps_g = jax.random.normal(ea, (B, G, A))
        s_rep = repeat(s0, "b o -> b g o", g=G)
        gen = self.network.select("actor")(s_rep, eps_g, params=grad_params)           # [B, G, A]
        gsg = jax.lax.stop_gradient(gen)

        # (a) trust region: value-selection drift toward top-M in-support data actions + repulsion
        M = min(self.config["n_cand"], B)
        sq = jnp.sum((s0[:, None, :] - s0[None, :, :]) ** 2, -1)
        nn = jnp.argsort(sq, -1)[:, :M]
        a_cand = a_data[nn]                                                            # [B, M, A]
        sq_sel = jnp.take_along_axis(sq, nn, -1)
        q_im = jax.lax.stop_gradient(
            self._Qlcb(repeat(s0, "i o -> (i m) o", m=M), rearrange(a_cand, "b m a -> (b m) a")).reshape(B, M))
        q_n = (q_im - q_im.mean(-1, keepdims=True)) / (q_im.std(-1, keepdims=True) + 1e-6)
        bw = self.config["state_bw"] * (jax.lax.stop_gradient(sq.mean()) + 1e-8)
        w = jax.lax.stop_gradient(jax.nn.softmax(q_n / self.config["alpha"] - sq_sel / bw, -1))
        c_sq = jnp.sum(a_cand ** 2, -1)
        g_sq = jnp.sum(gsg ** 2, -1)
        dist_pc = g_sq[..., None] + c_sq[:, None, :] - 2 * jnp.einsum("bga,bma->bgm", gsg, a_cand)
        tau2 = self.config["tau_scale"] * (jax.lax.stop_gradient(dist_pc.mean()) + 1e-8)
        kern_p = jnp.exp(-dist_pc / (2 * tau2)) * w[:, None, :]
        mean_p = jnp.einsum("bgm,bma->bga", kern_p, a_cand) / (kern_p.sum(-1, keepdims=True) + 1e-12)
        dist_gg = g_sq[..., None] + g_sq[:, None, :] - 2 * jnp.einsum("bga,bha->bgh", gsg, gsg)
        kern_q = jnp.exp(-dist_gg / (2 * tau2)) * (1.0 - jnp.eye(G))[None]
        mean_q = jnp.einsum("bgh,bha->bga", kern_q, gsg) / (kern_q.sum(-1, keepdims=True) + 1e-12)
        V_drift = mean_p - mean_q
        goal = jax.lax.stop_gradient(gen + self.config["drift_step"] * V_drift)
        drift_loss = ((gen - goal) ** 2).sum(-1).mean()

        # (b) PRIMARY improvement: on-manifold q_pe along the generator's own interpolation
        tau_a = jax.random.uniform(ta, (B, G, 1))
        step = jnp.minimum(self.config["interp_step"], 1.0 - tau_a)                     # bounded, no overstep
        x2 = (1 - tau_a) * eps_g + tau_a * gen
        x2_ahead = x2 + step * (gen - eps_g)                                           # one interp-step ahead (on the line)
        tau_ahead = tau_a + step
        qpe = self._V("value", rearrange(s_rep, "b g o -> (b g) o"),
                      rearrange(x2_ahead, "b g a -> (b g) a"),
                      rearrange(tau_ahead, "b g a -> (b g) a"))                        # value stored params -> grad to gen
        q_pe = self._lcb(qpe).mean()
        actor_loss = self.config["drift_coef"] * drift_loss - self.config["q_coef"] * q_pe

        total = critic_loss + actor_loss

        # ===================== action-informativeness probe (diagnostic) =====================
        # Is V(s, x, tau=1) discriminative in the ACTION coordinate, or flat (state-dominated)?
        Kp = 16
        a_rand = jax.random.uniform(pr, (B, Kp, A), minval=-1.0, maxval=1.0)
        Vd = jax.lax.stop_gradient(self._Qlcb(s0, a_data))                                  # [B] V at data action
        sK = repeat(s0, "b o -> (b k) o", k=Kp)
        Vr = jax.lax.stop_gradient(self._Qlcb(sK, rearrange(a_rand, "b k a -> (b k) a")).reshape(B, Kp))
        Vg = jax.lax.stop_gradient(self._Qlcb(s0, jax.lax.stop_gradient(gen[:, 0, :])))      # [B] V at a generator action
        rank_acc = (Vd[:, None] > Vr).mean()                # P(V[data] > V[random]); 0.5 = uninformative
        act_spread = Vr.std(-1).mean()                      # V variation ACROSS actions (per fixed state)
        state_spread = Vd.std()                             # V variation ACROSS states
        probe_ratio = act_spread / (state_spread + 1e-8)    # <<1 => value ignores action (flat)

        return total, {
            "critic_loss": critic_v, "inv_loss": L_inv, "drift_loss": drift_loss,
            "q_pe": q_pe, "v_mean": V_pred.mean(), "Vb_mean": Vb.mean(),
            "gen_std": gen.std(), "w_self": w[:, 0].mean(),
            "probe/rank_acc": rank_acc, "probe/act_spread": act_spread,
            "probe/state_spread": state_spread, "probe/ratio": probe_ratio,
            "probe/Vd_minus_Vr": Vd.mean() - Vr.mean(), "probe/Vd_minus_Vg": Vd.mean() - Vg.mean(),
        }

    def target_update(self, network, module, d):
        new_tp = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module}"], self.network.params[f"modules_target_{module}"])
        network.params[f"modules_target_{module}"] = new_tp

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))
        self.target_update(new_network, "value", d=self.config["tau"])
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        obs = jnp.atleast_2d(obs)[-1:]
        A = self.config["action_dim"]
        if temperature > 0:
            a = self.network.select("actor")(obs, jax.random.normal(seed, (1, A)) * self.config["expl_temp"])[0]
        else:
            K = self.config["eval_samples"]
            cand = self.network.select("actor")(repeat(obs, "1 o -> k o", k=K), jax.random.normal(seed, (K, A)))
            d = jnp.sum((cand[:, None, :] - cand[None, :, :]) ** 2, -1)
            a = cand[jnp.argmin(d.sum(-1))]
        a = jnp.clip(a, -1, 1)
        return rearrange(a, "(h d) -> h d", h=self.config["h"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        A = ex_actions.shape[-1]
        ex_vin = jnp.concatenate([ex_observations, ex_actions, ex_actions[..., :1]], -1)

        value_def = Value(hidden_dims=config["value_hidden_dims"], layer_norm=config["layer_norm"],
                          num_ensembles=config["ensemble_ct"])
        actor_def = OneStepGenerator(hidden_dims=config["actor_hidden_dims"], action_dim=A,
                                     layer_norm=config["actor_layer_norm"], tanh_squash=True)
        network_info = dict(
            value=(value_def, (ex_vin,)),
            target_value=(copy.deepcopy(value_def), (ex_vin,)),
            actor=(actor_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network = TrainState.create(network_def, network_def.init(init_rng, **network_args)["params"],
                                    tx=optax.adam(learning_rate=config["lr"]))
        params = network.params
        params["modules_target_value"] = params["modules_value"]
        config["action_dim"] = A
        config["discount_mul"] = jnp.array(config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf]))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return mlc.ConfigDict(dict(
        agent_name="dql_v10",
        h=5, expectile=0.9, ensemble_ct=10, rho=0.5,
        boundary_k=4,        # marginalized boundary samples
        inv_coef=0.5,        # noise-invariance reg weight
        n_gen=16, n_cand=32, state_bw=0.15, alpha=1.0, tau_scale=0.5, drift_step=1.0,
        drift_coef=1.0,      # trust-region drift weight
        q_coef=1.0,          # PRIMARY on-manifold improvement (q_pe)
        interp_step=0.1,     # q_pe step along the interpolation
        eval_samples=32, expl_temp=1.0,
        lr=3e-4, discount=0.99, batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
        layer_norm=True, actor_layer_norm=False, tau=0.005,
    ))
