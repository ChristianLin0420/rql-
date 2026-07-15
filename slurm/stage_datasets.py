"""Pre-stage all OGBench datasets for the 50-task sweep (one-time, run on a node WITH internet).
Uses ogbench.download_datasets directly -- no env creation, so it works on login nodes
without a GPU/EGL. Deduplicates by category: only the 10 underlying datasets download
(not 50). Point OGBENCH_DATASET_DIR at your shared filesystem before running."""
import os, time
import ogbench

CATS = [
    "antmaze-large-navigate", "antmaze-giant-navigate",
    "humanoidmaze-medium-navigate", "humanoidmaze-large-navigate",
    "scene-play", "puzzle-3x3-play", "puzzle-4x4-play",
    "cube-double-play", "cube-triple-play", "cube-quadruple-play",
]
dataset_dir = os.environ.get("OGBENCH_DATASET_DIR", "~/.ogbench/data")
print(f"[stage] dataset_dir = {dataset_dir}", flush=True)
for c in CATS:
    name = f"{c}-v0"
    t = time.time()
    print(f"[stage] {name} ...", flush=True)
    try:
        ogbench.download_datasets([name], dataset_dir=dataset_dir)
        print(f"[stage] OK {name} ({time.time() - t:.0f}s)", flush=True)
    except Exception as e:
        print(f"[stage] FAIL {name}: {type(e).__name__}: {e}", flush=True)
print("[stage] DONE", flush=True)
