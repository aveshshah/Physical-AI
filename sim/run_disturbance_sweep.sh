#!/usr/bin/env bash
# One Mcity lap of the CARLA tadpole NMPC per road-load scenario (src/talk2drive/disturbances.py).
# The undisturbed lap runs first; the others are compared with it at the same distance along the route.
#   bash sim/run_disturbance_sweep.sh [scenario ...]      (default: all), 2 in parallel
#   ROBOT=magna_susp OUT=runs/disturbances_susp bash sim/run_disturbance_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/isaac-env/bin/python}
OUT=${OUT:-runs/disturbances}
JOBS=${JOBS:-2}
ROBOT=${ROBOT:-magna}
export OMNI_KIT_ACCEPT_EULA=YES
scenarios=("$@")
[ ${#scenarios[@]} -eq 0 ] && scenarios=(wet low_mu ice_patches grade wind storm drag combined)
mkdir -p "$OUT"
run() {
  "$PY" sim/m2_mcity_tadpole_nmpc.py --headless --robot "$ROBOT" --disturbance "$1" --out "$OUT/$1" \
    --baseline "$OUT/nominal/log.npz" > "$OUT/$1.log" 2>&1 || echo "[sweep] $1 exited with $?"
  grep -E "FAILED|lap_time_s" "$OUT/$1.log" | head -2 | sed "s/^/[sweep] $1: /"
}
export -f run; export PY OUT ROBOT
[ -f "$OUT/nominal/log.npz" ] || run nominal
printf '%s\n' "${scenarios[@]}" | xargs -P "$JOBS" -I{} bash -c 'run "$@"' _ {}
"$PY" sim/summarise_disturbances.py "$OUT"
