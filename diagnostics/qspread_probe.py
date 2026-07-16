"""Action 3: Q-spread over deployment candidates (blind-spot probe).
For N dataset states: draw K=32 candidates from target_actor (as deployment does),
measure std of Q_LCB over candidates vs Vd-Vr scale, and per-state action std.
CPU-runnable: JAX_PLATFORMS=cpu python diagnostics/qspread_probe.py <env> <ckpt_dir>
"""
import sys, os, glob, pickle
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_collections import ConfigDict
import ogbench.utils as ogu

env_name, run_dir = sys.argv[1], sys.argv[2]
h = int(sys.argv[3]); expectile = float(sys.argv[4]); rho = float(sys.argv[5])

from agents.dql_v11_2 import DQLv11_2Agent, get_config
cfg = get_config(); cfg.h = h; cfg.expectile = expectile; cfg.rho = rho
ddir = os.environ.get('OGBENCH_DATASET_DIR')
dsname = env_name.split('-singletask')[0] + '-v0'
train, _ = ogu.load_dataset(os.path.join(ddir, dsname + '.npz')), None
obs, acts, term = train['observations'], train['actions'], train['terminals']
# terminal-clamped h-chunks (mirror sampler)
N = len(obs); idx = np.random.default_rng(0).choice(N - h - 1, 500, replace=False)
def chunk(i):
    a = []
    j = i
    for _ in range(h):
        a.append(acts[j])
        if not term[j]: j += 1
    return np.concatenate(a)
A = np.stack([chunk(i) for i in idx]); S = obs[idx]
agent = DQLv11_2Agent.create(0, S[:1], acts[:1], cfg)  # atomic action: create() multiplies by h
ck = os.path.join(run_dir, 'params_2000000.pkl')
with open(ck, 'rb') as f: sd = pickle.load(f)
import flax
agent = flax.serialization.from_state_dict(agent, sd['agent'])

net = agent.network
K = 32
rng = jax.random.PRNGKey(1)
def qlcb(s, a):
    q = net.select('target_q')(s, a)   # try target; fallback online below
    return q.mean(0) - rho * q.std(0)
outs = {'qstd': [], 'astd': [], 'vdvr': []}
for i in range(0, 500, 50):
    s = jnp.asarray(S[i:i+50]); a_d = jnp.asarray(A[i:i+50])
    B = s.shape[0]
    rng, k = jax.random.split(rng)
    eps = jax.random.normal(k, (B, K, A.shape[1]))
    s_rep = jnp.repeat(s[:, None], K, 1).reshape(B*K, -1)
    cand = net.select('target_actor')(s_rep, eps.reshape(B*K, -1))
    cand = jnp.clip(cand, -1, 1)
    q = net.select('q')(s_rep, cand)           # online q, LCB
    ql = (q.mean(0) - rho * q.std(0)).reshape(B, K)
    outs['qstd'].append(np.asarray(ql.std(1)))
    outs['astd'].append(np.asarray(cand.reshape(B, K, -1).std(1).mean(-1)))
    rng, k2 = jax.random.split(rng)
    ar = jax.random.uniform(k2, (B*16, A.shape[1]), minval=-1, maxval=1)
    qr = net.select('q')(jnp.repeat(s[:, None], 16, 1).reshape(B*16, -1), ar)
    qrl = (qr.mean(0) - rho*qr.std(0)).reshape(B, 16).mean(1)
    qd = net.select('q')(s, a_d); qdl = qd.mean(0) - rho*qd.std(0)
    outs['vdvr'].append(np.asarray(qdl - qrl))
qs = np.concatenate(outs['qstd']); asd = np.concatenate(outs['astd']); vd = np.concatenate(outs['vdvr'])
pct = lambda x: (np.mean(x), *np.percentile(x, [10, 50, 90]))
print(f"{env_name}")
print(f"  Q-spread over 32 candidates: mean {pct(qs)[0]:.4f} p10/p50/p90 {pct(qs)[1]:.4f}/{pct(qs)[2]:.4f}/{pct(qs)[3]:.4f}")
print(f"  Vd-Vr scale:                 mean {pct(vd)[0]:.3f}")
print(f"  ratio qstd/VdVr:             {np.mean(qs)/max(np.mean(vd),1e-9):.5f}")
print(f"  per-state action std (mean/dim): mean {pct(asd)[0]:.4f} p10/p50/p90 {pct(asd)[1]:.4f}/{pct(asd)[2]:.4f}/{pct(asd)[3]:.4f}")
