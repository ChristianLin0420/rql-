import csv, glob, time, subprocess, os
os.chdir("/localhome/local-chrislin/rql-")
target=250000
while True:
    fs=sorted(glob.glob("exp/*/DQL-v11_2-cube-double/*/eval.csv"))
    ms=0; best=0.0
    if fs:
        for r in csv.DictReader(open(fs[-1])):
            try: ms=max(ms,int(float(r["step"]))); best=max(best,float(r["evaluation/success"]))
            except: pass
    alive=subprocess.run(["pgrep","-f","DQL-v11_2-cube-double"],capture_output=True).returncode==0
    if best>=0.23: print(f"CUBE_BEAT_RQL best={best:.2f} at<={ms}"); break   # hit RQL's 0.23
    if ms>=target: print(f"CUBE_AT_{target} best={best:.2f}"); break
    if not alive: print(f"CUBE_DIED_AT_{ms} best={best:.2f}"); break
    time.sleep(120)
