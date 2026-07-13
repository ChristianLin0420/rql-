import csv, glob, time, subprocess, sys, os
os.chdir("/localhome/local-chrislin/rql-")
group=sys.argv[1]; target=int(sys.argv[2]); pgrep_key=sys.argv[3]
while True:
    fs=sorted(glob.glob(f"exp/*/{group}/*/eval.csv"))
    ms=0
    if fs:
        for r in csv.DictReader(open(fs[-1])):
            try: ms=max(ms,int(float(r["step"])))
            except: pass
    alive=subprocess.run(["pgrep","-f",pgrep_key],capture_output=True).returncode==0
    if ms>=target: print(f"{group}_AT_{target}"); break
    if not alive: print(f"{group}_DIED_AT_{ms}"); break
    time.sleep(60)
