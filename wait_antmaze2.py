import csv, glob, time, subprocess, os
os.chdir("/localhome/local-chrislin/rql-")
target=350000
def succ_at(step_target):
    fs=sorted(glob.glob("exp/*/DQL-v11-antmaze-large/*/eval.csv"))
    if not fs: return 0,-1
    ms=0
    for r in csv.DictReader(open(fs[-1])):
        try: ms=max(ms,int(float(r["step"])))
        except: pass
    return ms,0
while True:
    ms,_=succ_at(target)
    alive=subprocess.run(["pgrep","-f","DQL-v11-antmaze"],capture_output=True).returncode==0
    if ms>=target: print(f"ANTMAZE_AT_{target}"); break
    if not alive: print(f"ANTMAZE_DONE_OR_DIED_AT_{ms}"); break
    time.sleep(60)
