#!/bin/bash
# Submit the DQL v11.1 50-task sweep: 50 jobs x 1 A100, each running 3 seeds.
#   bash sbatch/submit_all.sh            # all 50 tasks
#   bash sbatch/submit_all.sh antmaze    # only jobs whose name matches a pattern
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

pat=${1:-}
n=0
for f in ${JOBS_DIR:-sbatch/jobs}/*.sbatch; do
  [[ -z "$pat" || "$f" == *"$pat"* ]] || continue
  sbatch "$f"
  n=$((n + 1))
done
echo "submitted $n jobs; monitor with: squeue -u $USER"
