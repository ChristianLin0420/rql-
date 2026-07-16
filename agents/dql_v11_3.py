"""
DQL (v11.3): fix the two measured extraction failures of v11.2 (DQL112_FAILURE_ANALYSIS.md),
leaving the critic and every v11.2 stability/strength mechanism untouched.

  (1) DATASET-WIDE attraction pool. v11.2 drew the M=32 kNN candidates from the 256-state
      batch; in manipulation state spaces that yields only ~1-4 usable neighbors, the softmax
      collapses onto the self candidate (w_self 0.67-0.86) and the drift degenerates to
      single-positive BC — the checkpoint probe showed a static 100k-state pool restores full
      multi-positivity (cube-double w_self 0.671 -> 0.173, 31.8/32 usable, coherent actions).
      Candidates = self + top-(M-1) pool neighbors, so w_self keeps its meaning.
  (2) argmax-Q_LCB deployment (replaces the mean-proximal medoid). Paired 100-episode
      ablation on 2M checkpoints: puzzle-3x3-t1 0.96 vs medoid 0.41, scene-t1 0.44 vs 0.33,
      never worse; kernel-mode and single-sample refuted.
Expectations (pre-registered): manipulation w_self ~< 0.4 from early training; scene/puzzle
success well above v11.2; cube limited by the critic's candidate blind spot (Q-spread ~1% of
Vd-Vr) — gains there imply the pool also sharpened on-manifold Q discrimination.
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


class DQLv11_3Agent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    pool_obs: Any   # [P, obs]   static dataset-wide attraction pool (v11.3)
    pool_act: Any   # [P, h*d]   the pool states' action chunks
    config: Any = nonpytree_field()

    needs_pool = True   # main.py samples the pool from the training dataset at create time

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

    def _Q(self, module, s, a, params=None):
        return self.network.select(module)(jnp.concatenate([s, a], -1), params=params)   # [E, ...]

    def _V(self, module, s, params=None):
        return self.network.select(module)(s, params=params)                             # [Ev, ...]

    def _lcb(self, q):
        return q.mean(0) - self.config["rho"] * q.std(0)

    def _qlcb(self, module, s, a, params=None):
        return self._lcb(self._Q(module, s, a, params=params))

    @jax.jit
    def total_loss(self, batch, grad_params, rng):
        B, A = self.config["batch_size"], self.config["action_dim"]
        s0 = batch["observations"][0]
        s_next = batch["observations"][-1]
        a_data = self._chunk(batch)
        rng, ea, an, pr = jax.random.split(rng, 4)

        # ===================== action-sharp critic (unchanged from v11.2) =====================
        q_pred = self._Q("q", s0, a_data, params=grad_params)                            # [E, B]
        q_data_m = q_pred.mean(0)                                                         # [B]

        # V(s) <- expectile of target Q(s, a_data)  (noise-free state value)
        q_for_v = self._Q("target_q", s0, a_data).mean(0)                                # [B] stop-grad
        v_pred = self._V("v", s0, params=grad_params).mean(0)                            # [B]
        adv = q_for_v - v_pred
        L_v = self.expectile_loss(adv, adv, self.config["expectile"]).mean()

        # Q(s, a_data) <- r + gamma^h V(s')   (transition target -> action-informative)
        v_next = self._V("v", s_next).mean(0)                                            # [B] stop-grad
        y, valids = self._nstep(batch, v_next)
        L_q = (((q_pred - y[None]) ** 2) * valids[None]).mean()

        # generator samples (also used as OOD negatives + actor)
        G = self.config["n_gen"]
        eps_g = jax.random.normal(ea, (B, G, A))
        s_rep = repeat(s0, "b o -> b g o", g=G)
        gen = self.network.select("actor")(s_rep, eps_g, params=grad_params)             # [B, G, A]
        gsg = jax.lax.stop_gradient(gen)

        # bounded in-support contrast: Q(data) >= Q(neg) + margin  (neg = random + generator)
        Nr = self.config["n_rand"]
        a_rand = jax.random.uniform(an, (B, Nr, A), minval=-1.0, maxval=1.0)
        a_neg = jnp.concatenate([a_rand, gsg], axis=1)                                    # [B, Nr+G, A]
        Nn = Nr + G
        q_neg = self._Q("q", repeat(s0, "b o -> (b n) o", n=Nn),
                        rearrange(a_neg, "b n a -> (b n) a"), params=grad_params).mean(0).reshape(B, Nn)
        L_c = jax.nn.relu(q_neg - q_data_m[:, None] + self.config["margin"]).mean()

        critic_loss = L_q + self.config["v_coef"] * L_v + self.config["cql_coef"] * L_c

        # ===================== one-step drift actor =====================
        # (a) trust region: value-selection drift toward top-M in-support data + repulsion.
        # v11.3: candidates = self + top-(M-1) neighbors from the DATASET-WIDE pool (v11.2 used
        # the 256-state batch, which starves manipulation of neighbors -> single-positive collapse).
        # Squared distances via the quadratic expansion (a [B, P] matrix; forming [B, P, obs] would
        # not fit); a batch state that also lives in the pool just duplicates the self candidate.
        M = self.config["n_cand"]
        sq_pool = jnp.maximum(
            jnp.sum(s0 ** 2, -1)[:, None] + jnp.sum(self.pool_obs ** 2, -1)[None, :]
            - 2.0 * s0 @ self.pool_obs.T, 0.0)                                            # [B, P]
        neg_sq_nn, nn = jax.lax.top_k(-sq_pool, M - 1)                                    # [B, M-1]
        a_cand = jnp.concatenate([a_data[:, None, :], self.pool_act[nn]], 1)             # [B, M, A] self at 0
        sq_sel = jnp.concatenate([jnp.zeros((B, 1)), -neg_sq_nn], 1)                      # [B, M]
        q_im = jax.lax.stop_gradient(
            self._qlcb("target_q", repeat(s0, "i o -> (i m) o", m=M),
                       rearrange(a_cand, "b m a -> (b m) a")).reshape(B, M))
        # ADVANTAGE-weighted drift: peak the attraction toward the argmax-advantage in-support action
        v_state = jax.lax.stop_gradient(self._V("v", s0).mean(0))                        # [B]
        adv_cand = q_im - v_state[:, None]                                               # [B, M] advantage
        adv_n = adv_cand / (jax.lax.stop_gradient(jnp.abs(adv_cand).mean()) + 1e-6)       # scale-free
        bw = self.config["state_bw"] * (jax.lax.stop_gradient(sq_pool.mean()) + 1e-8)
        w = jax.lax.stop_gradient(jax.nn.softmax(adv_n / self.config["adv_temp"] - sq_sel / bw, -1))
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

        # (b) bounded Q-ascent on the sharp Q (grad -> gen only; target_q params frozen)
        q_asc = self._qlcb("target_q", rearrange(s_rep, "b g o -> (b g) o"),
                           rearrange(gen, "b g a -> (b g) a")).mean()
        actor_loss = self.config["drift_coef"] * drift_loss - self.config["q_coef"] * q_asc

        total = critic_loss + actor_loss

        # ===================== action-informativeness probe (diagnostic) =====================
        Kp = 16
        a_rp = jax.random.uniform(pr, (B, Kp, A), minval=-1.0, maxval=1.0)
        Vd = jax.lax.stop_gradient(self._qlcb("q", s0, a_data))
        Vr = jax.lax.stop_gradient(self._qlcb("q", repeat(s0, "b o -> (b k) o", k=Kp),
                                              rearrange(a_rp, "b k a -> (b k) a")).reshape(B, Kp))
        Vg = jax.lax.stop_gradient(self._qlcb("q", s0, gsg[:, 0, :]))
        rank_acc = (Vd[:, None] > Vr).mean()
        act_spread = Vr.std(-1).mean()
        state_spread = Vd.std()

        return total, {
            "critic_loss": L_q, "v_loss": L_v, "contrast": L_c, "drift_loss": drift_loss,
            "q_asc": q_asc, "q_data": q_data_m.mean(), "v_mean": v_pred.mean(),
            "gen_std": gen.std(), "w_self": w[:, 0].mean(),
            "probe/rank_acc": rank_acc, "probe/act_spread": act_spread,
            "probe/state_spread": state_spread, "probe/ratio": act_spread / (state_spread + 1e-8),
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
        self.target_update(new_network, "q", d=self.config["tau"])
        self.target_update(new_network, "actor", d=self.config["actor_ema"])   # slow EMA of generator for eval
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        obs = jnp.atleast_2d(obs)[-1:]
        A = self.config["action_dim"]
        if temperature > 0:
            a = self.network.select("target_actor")(obs, jax.random.normal(seed, (1, A)) * self.config["expl_temp"])[0]
        else:
            # v11.3: argmax-Q_LCB over the K candidates (medoid averaged across modes; refuted by
            # the paired selector ablation -- puzzle-3x3-t1 0.96 vs 0.41, never worse anywhere)
            K = self.config["eval_samples"]
            obs_k = repeat(obs, "1 o -> k o", k=K)
            cand = jnp.clip(self.network.select("target_actor")(obs_k, jax.random.normal(seed, (K, A))), -1, 1)
            a = cand[jnp.argmax(self._qlcb("target_q", obs_k, cand))]
        a = jnp.clip(a, -1, 1)
        return rearrange(a, "(h d) -> h d", h=self.config["h"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, pool=None):
        assert pool is not None, "dql_v11_3 needs the dataset-wide attraction pool (main.py samples it)"
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        A = ex_actions.shape[-1]
        ex_qin = jnp.concatenate([ex_observations, ex_actions], -1)

        q_def = Value(hidden_dims=config["value_hidden_dims"], layer_norm=config["layer_norm"],
                      num_ensembles=config["ensemble_ct"])
        v_def = Value(hidden_dims=config["value_hidden_dims"], layer_norm=config["layer_norm"],
                      num_ensembles=2)
        actor_def = OneStepGenerator(hidden_dims=config["actor_hidden_dims"], action_dim=A,
                                     layer_norm=config["actor_layer_norm"], tanh_squash=True)
        network_info = dict(
            q=(q_def, (ex_qin,)),
            target_q=(copy.deepcopy(q_def), (ex_qin,)),
            v=(v_def, (ex_observations,)),
            actor=(actor_def, (ex_observations, ex_actions)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network = TrainState.create(network_def, network_def.init(init_rng, **network_args)["params"],
                                    tx=optax.adam(learning_rate=config["lr"]))
        params = network.params
        params["modules_target_q"] = params["modules_q"]
        params["modules_target_actor"] = params["modules_actor"]
        config["action_dim"] = A
        config["discount_mul"] = jnp.array(config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf]))
        pool_obs, pool_act = (jnp.asarray(p, dtype=jnp.float32) for p in pool)
        assert pool_act.shape[-1] == A, f"pool action chunks must be {A}-d, got {pool_act.shape}"
        return cls(rng, network=network, pool_obs=pool_obs, pool_act=pool_act,
                   config=flax.core.FrozenDict(**config))


def get_config():
    return mlc.ConfigDict(dict(
        agent_name="dql_v11_3",
        h=5, expectile=0.9, ensemble_ct=10, rho=0.5,
        n_pool=100_000,      # dataset-wide attraction pool size (v11.3)
        margin=1.0,          # in-support contrast margin (anti-flatness lever)
        cql_coef=1.0,        # contrast weight
        v_coef=1.0,          # state-value expectile weight
        n_rand=8,            # random OOD negatives for contrast
        n_gen=16, n_cand=32, state_bw=0.15, tau_scale=0.5, drift_step=1.5,
        adv_temp=0.25,       # peak harder on argmax-advantage action
        drift_coef=1.0,      # advantage-weighted drift weight
        q_coef=0.25,         # DAMPED Q-ascent weight
        actor_ema=0.005,     # slow EMA of generator for eval stability
        eval_samples=32, expl_temp=1.0,
        lr=3e-4, discount=0.99, batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
        layer_norm=True, actor_layer_norm=False, tau=0.005,
    ))
