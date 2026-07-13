"""Pre-stage all OGBench datasets for the 50-task sweep (one-time, run on a node WITH internet).
Deduplicates by category so only the 10 underlying datasets download (not 50). Point
OGBENCH_DATASET_DIR at your shared filesystem before running."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.env_utils import make_env_and_datasets

# one representative task per category (task1) triggers that category's dataset download
CATS = [
    "antmaze-large-navigate", "antmaze-giant-navigate",
    "humanoidmaze-medium-navigate", "humanoidmaze-large-navigate",
    "scene-play", "puzzle-3x3-play", "puzzle-4x4-play",
    "cube-double-play", "cube-triple-play", "cube-quadruple-play",
]
H = {"antmaze": 1, "humanoidmaze": 1}  # h only affects the loader's chunk wrapper, not the dataset
for c in CATS:
    env = f"{c}-singletask-task1-v0"
    h = 1 if c.startswith(("antmaze", "humanoidmaze")) else 5
    t = time.time()
    print(f"[stage] {env} ...", flush=True)
    try:
        _, _, tr, _ = make_env_and_datasets(env, agent_config={"h": h})
        print(f"[stage] OK {c}: obs={tr['observations'].shape} ({time.time()-t:.0f}s)", flush=True)
    except Exception as e:
        print(f"[stage] FAIL {c}: {type(e).__name__}: {e}", flush=True)
print("[stage] DONE", flush=True)
