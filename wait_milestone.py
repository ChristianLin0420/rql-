import csv, glob, sys, time
target = int(sys.argv[1])
groups = ["DQL-v11-cube-double", "DQL-v11-cube-triple", "DQL-v11-cube-quadruple"]
import subprocess
def alive(t):
    return subprocess.run(["pgrep","-f",f"cube-{t}-play"],capture_output=True).returncode==0
tags={"DQL-v11-cube-double":"double","DQL-v11-cube-triple":"triple","DQL-v11-cube-quadruple":"quadruple"}
while True:
    ready=0; dead=[]
    for g in groups:
        fs=sorted(glob.glob(f"exp/*/{g}/*/eval.csv"))
        ms=0
        if fs:
            for r in csv.DictReader(open(fs[-1])):
                try: ms=max(ms,int(float(r["step"])))
                except: pass
        if ms>=target: ready+=1
        if not alive(tags[g]): dead.append(tags[g])
    if ready>=3: print("ALL_AT_%d"%target); break
    if dead: print("RUN_DIED:%s (ready=%d)"%(",".join(dead),ready)); break
    time.sleep(60)
