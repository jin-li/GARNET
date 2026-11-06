#!/usr/bin/env bash
# Copy finished Slurm results of the GARNET vs FEMFFUSION report into data/perf/.
# Each job directory keeps its JSON records, progress.log, plan, PROVENANCE.txt
# and the logs of failed runs; run logs of successful runs and FEMFFUSION work
# directories stay on scratch.
#   bash collect.sh [JOBID ...]     (default: every job directory found)
set -euo pipefail
src="${GARNET_PERF_RESULTS:-/data/scratch/shared/poncet_m/garnet-femffusion-perf/results}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/data/perf"
if [[ $# -gt 0 ]]; then
    dirs=(); for id in "$@"; do dirs+=("$src/$id"-*); done
else
    dirs=("$src"/*/)
fi
for d in "${dirs[@]}"; do
    d="${d%/}"; name="$(basename "$d")"
    [[ -f "$d/progress.log" ]] || continue
    mkdir -p "$here/data/perf/$name"
    cp "$d"/*.json "$d/progress.log" "$d"/*.plan "$here/data/perf/$name/" 2>/dev/null || true
    [[ -f "$d/PROVENANCE.txt" ]] && cp "$d/PROVENANCE.txt" "$here/data/perf/$name/"
    while read -r _ tag _; do
        cp "$d/$tag.log" "$here/data/perf/$name/" 2>/dev/null || true
    done < <(grep '^FAIL' "$d/progress.log" || true)
    echo "$name: $(grep -c '^DONE' "$d/progress.log") done, $(grep -c '^FAIL' "$d/progress.log" || true) failed"
done
