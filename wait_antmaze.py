import csv, glob, time, subprocess, os
os.chdir("/localhome/local-chrislin/rql-")
target=100000
while True:
    fs=sorted(glob.glob("exp/*/DQL-v11-antmaze-large/*/eval.csv"))
    ms=0
    if fs:
        for r in csv.DictReader(open(fs[-1])):
            try: ms=max(ms,int(float(r["step"])))
            except: pass
    alive=subprocess.run(["pgrep","-f","DQL-v11-antmaze"],capture_output=True).returncode==0
    if ms>=target: print(f"ANTMAZE_AT_{target}"); break
    if not alive: print(f"ANTMAZE_DIED_AT_{ms}"); break
    time.sleep(45)
