"""Generate the 50-task table (10 OGBench categories x 5 tasks) with per-category hypers.
Hypers = RQL Table 2/3 with our validated v11.2 corrections (IQL-style critic needs an optimistic
expectile: locomotion 0.5->0.7; manipulation 0.7-0.9). Columns consumed by the sbatch array."""
CATS = [
    # (category, token, h, expectile, rho, discount, sparse, state_bw)
    # state_bw (v11.4+): borrowing dial on the selected-neighbor scale, calibrated on
    # checkpoints (diagnostics/knn_probe_v114_bw.py) -- antmaze 0.5 -> w_self ~0.33,
    # humanoidmaze 0.25 -> ~0.65 (cyclic gaits need self-dominant attraction),
    # manipulation 1.0 -> ~0.15-0.19 (dense borrowing regime that fixed scene/puzzle).
    ("antmaze-large",        "navigate", 1, 0.7, 0.5, 0.99,  0, 0.5),
    ("antmaze-giant",        "navigate", 1, 0.7, 0.5, 0.995, 0, 0.5),
    ("humanoidmaze-medium",  "navigate", 1, 0.7, 0.0, 0.995, 0, 0.25),
    ("humanoidmaze-large",   "navigate", 1, 0.7, 0.0, 0.995, 0, 0.25),
    ("scene",                "play",     5, 0.7, 0.5, 0.99,  1, 1.0),
    ("puzzle-3x3",           "play",     5, 0.7, 0.5, 0.99,  1, 1.0),
    # puzzle-4x4 MUST NOT sparsify: its datasets contain no r=0 transitions (raw max -1 of
    # -15..0), so (r != 0) * -1 yields a CONSTANT reward and zero training signal
    # (see DQL112_FAILURE_ANALYSIS.md, pipeline probe 2026-07-16).
    ("puzzle-4x4",           "play",     5, 0.9, 0.5, 0.99,  0, 1.0),
    ("cube-double",          "play",     5, 0.9, 0.5, 0.99,  0, 1.0),
    ("cube-triple",          "play",     5, 0.9, 0.5, 0.99,  0, 1.0),
    ("cube-quadruple",       "play",     5, 0.9, 0.5, 0.99,  0, 1.0),
]
rows=[]
for cat,tok,h,e,rho,disc,sp,bw in CATS:
    for t in range(1,6):
        env=f"{cat}-{tok}-singletask-task{t}-v0"
        rows.append((env,h,e,rho,disc,sp,bw))
with open("slurm/tasks.tsv","w") as f:
    f.write("env_name\th\texpectile\trho\tdiscount\tsparse\tstate_bw\n")
    for r in rows: f.write("\t".join(map(str,r))+"\n")
print(f"wrote slurm/tasks.tsv  ({len(rows)} tasks)")
for i,r in enumerate(rows): print(f"{i:2d}  {r[0]:<46} h={r[1]} e={r[2]} rho={r[3]} disc={r[4]} sparse={r[5]}")
