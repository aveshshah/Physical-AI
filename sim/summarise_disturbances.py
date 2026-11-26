"""Table and overlay plot for a road-load sweep (sim/run_disturbance_sweep.sh).

    python sim/summarise_disturbances.py runs/disturbances
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["nominal", "wet", "low_mu", "ice_patches", "grade", "wind", "storm", "drag", "combined"]


def main(root: Path) -> None:
    runs = {}
    for d in sorted(root.iterdir()):
        if (d / "metrics.json").exists():
            runs[d.name] = (json.loads((d / "metrics.json").read_text()), np.load(d / "log.npz"))
    names = sorted(runs, key=lambda n: ORDER.index(n) if n in ORDER else 99)
    rows = ["| Scenario | What the plant gets | Lap | Speed error vs NMPC ref (RMSE / p95 / worst below) | vs undisturbed lap (RMSE / lap time) "
            "| Plan restarts | Observer error | Rules 1B / 3C | Max CTE | Max roll |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        m, _ = runs[n]
        st = m["speed_tracking"]
        r = st["vs_nmpc_reference"]
        u = st.get("vs_undisturbed_lap")
        lap = f"{m['lap_time_s']:.1f} s" if m["lap_completed"] else "**did not finish**"
        und = f"{u['rmse']:.2f} m/s / {u['lap_time_delta_s']:+.1f} s" if u and n != "nominal" else "-"
        ob = st["observer"]
        rules = ("pass" if m["rule_1B_long_accel"]["pass"] else f"**fail** ({m['rule_1B_long_accel']['max_accel']:.2f}/{m['rule_1B_long_accel']['max_decel']:.2f})") \
            + " / " + ("pass" if m["rule_3C_lane_keeping"]["pass"] else "**fail**")
        roll = float(np.degrees(np.abs(runs[n][1]["roll"]).max()))
        rows.append(f"| {n} | {m['disturbance']['description']} | {lap} | {r['rmse']:.2f} / {r['p95_abs']:.2f} / {r['max_below']:.2f} m/s "
                    f"| {und} | {st['plan_restarts']['count']} ({st['plan_restarts']['time_off_plan_s']:.1f} s) "
                    f"| {ob['rmse_N']:.0f} N | {rules} | {100 * m['rule_3C_lane_keeping']['cte_max']:.0f} cm | {roll:.1f} deg |")
    table = "\n".join(rows)
    (root / "summary.md").write_text(table + "\n")
    print(table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for n in names:
        _, lg = runs[n]
        lw, c = (2.0, "k") if n == "nominal" else (0.9, None)
        ax[0].plot(lg["s"], lg["vx"], lw=lw, color=c, label=n)
        ax[1].plot(lg["s"], lg["vx"] - lg["v_ref"], lw=lw * 0.7, color=c, label=n)
    ax[0].set_ylabel("speed [m/s]")
    ax[1].set_ylabel("speed - NMPC reference [m/s]")
    ax[1].set_xlabel("distance along route [m]")
    ax[0].legend(ncol=5, fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("CARLA tadpole NMPC, Mcity lap, road-load uncertainty the controller is not told about")
    fig.tight_layout()
    fig.savefig(root / "summary.png", dpi=110)
    print(f"[summary] {root / 'summary.md'}, {root / 'summary.png'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/disturbances"))
