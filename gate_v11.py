import csv, glob, time, subprocess, os
os.chdir("/localhome/local-chrislin/rql-")
def rows():
    fs=sorted(glob.glob("exp/*/DQL-v11-cube-double/*/train.csv"))
    return list(csv.DictReader(open(fs[-1]))) if fs else []
# wait for step >= 30000 (or process death)
while True:
    rs=rows(); step=int(float(rs[-1]["step"])) if rs else 0
    alive=subprocess.run(["pgrep","-f","DQL-v11-cube-double"],capture_output=True).returncode==0
    if step>=30000: break
    if not alive and step<30000: print(f"FAIL: cube-double died at {step}"); raise SystemExit
    time.sleep(30)
rs=rows()
ra=[float(r["training/probe/rank_acc"]) for r in rs if float(r["step"])>=20000]
rt=[float(r["training/probe/ratio"]) for r in rs if float(r["step"])>=20000]
mra=sum(ra)/len(ra); mrt=sum(rt)/len(rt)
print(f"GATE @30k: mean rank_acc(20-30k)={mra:.3f}  mean ratio={mrt:.3f}")
if mra>=0.75:
    print("PASS -> launching cube-triple + cube-quadruple")
    env=dict(os.environ, WANDB_MODE="online", MUJOCO_GL="egl",
             XLA_PYTHON_CLIENT_MEM_FRACTION="0.25", WANDB_PROJECT="rql-iclr2027-kernel-analysis")
    for T in ["triple","quadruple"]:
        log=open(f"logs/v11_cube_{T}.log","w")
        subprocess.Popen(["python","main.py","--agent=agents/dql_v11.py",
            f"--env_name=cube-{T}-play-singletask-v0","--agent.h=5","--agent.expectile=0.9","--agent.rho=0.5",
            "--offline_steps=1000000","--eval_interval=50000","--eval_episodes=20","--log_interval=2500",
            "--save_interval=250000",f"--run_group=DQL-v11-cube-{T}","--seed=0"],
            stdout=log, stderr=subprocess.STDOUT, env=env)
        print(f"  launched cube-{T}"); time.sleep(8)
else:
    print("FAIL: rank_acc did not hold >=0.75 -> NOT launching others; needs iteration")
