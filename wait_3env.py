import csv, glob, time, subprocess, os
os.chdir("/localhome/local-chrislin/rql-")
target=150000
envs=["large","medium","giant"]
while True:
    ready=0; dead=[]
    for E in envs:
        fs=sorted(glob.glob(f"exp/*/DQL-v11_2-antmaze-{E}/*/eval.csv"))
        ms=0
        if fs:
            for r in csv.DictReader(open(fs[-1])):
                try: ms=max(ms,int(float(r["step"])))
                except: pass
        if ms>=target: ready+=1
        if subprocess.run(["pgrep","-f",f"DQL-v11_2-antmaze-{E}"],capture_output=True).returncode!=0: dead.append(E)
    if ready>=3: print(f"ALL3_AT_{target}"); break
    if dead: print(f"DIED:{','.join(dead)} (ready={ready})"); break
    time.sleep(60)
