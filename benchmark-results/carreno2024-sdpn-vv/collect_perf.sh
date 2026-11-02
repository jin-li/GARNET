#!/usr/bin/env bash
# Copy the GPU/CPU performance records of slurm/stage_and_submit_carreno_perf.sh
# into data/perf/<job>-<plan>/: every run's JSON, the job's progress.log and
# plan, and the log of each failed run (make_figures.py reads the failure
# reason from it).
#
#   bash benchmark-results/carreno2024-sdpn-vv/collect_perf.sh [results-root ...]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
roots=("$@")
[[ ${#roots[@]} -eq 0 ]] && roots=(/data/scratch/shared/poncet_m/garnet-carreno-perf/results
                                   /data/scratch/shared/poncet_m/garnet-carreno-perf3d/results)
for root in "${roots[@]}"; do
    for d in "$root"/*/; do
        name=$(basename "$d")
        out="$HERE/data/perf/$name"
        mkdir -p "$out"
        cp "$d"/*.json "$d"/progress.log "$d"/*.plan "$out"/ 2>/dev/null || true
        grep '^FAIL' "$d/progress.log" 2>/dev/null | awk '{print $2}' | while read -r tag; do
            cp "$d/$tag.log" "$out"/ 2>/dev/null || true
        done
    done
done
echo "collected into $HERE/data/perf: $(ls -d "$HERE"/data/perf/*/ | wc -l) jobs"
