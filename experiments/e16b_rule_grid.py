"""E16b - Settings of the shift rule, evaluated on the stored replay (no new s-IDK^2 runs).

Rule, exactly as backend/machine_status.py applies it in every 30-minute cycle: a channel is in shift when its
last p windows all have similarity below median - k * 1.4826 * MAD of the current 3-day population. The system
raises a shift when any vibration or noise channel is in shift.

Protocol (the load event is not used to choose k or p):
  * calibration: 1-20 April 2026, excluding the conveyor stop of 10-14 April and the 3 days after restart
    (10 April 12:00 to 17 April 12:00), in which the population itself contains the stop;
  * hold-out normal operation: 21 April to 5 May 08:00;
  * event: 5 May 08:00 to 7 May 12:00 (detection delay, share of load steps in shift).
Output: experiments/results/e16b_grid.csv
"""
import itertools
import os

import numpy as np
import pandas as pd

from common import EVENT_END, EVENT_START, ensure_results_dir

SHIFT = ["z_rms", "x_rms", "z_peak", "x_peak", "noise"]
CAL_END = pd.Timestamp("2026-04-21 00:00")
STOP = (pd.Timestamp("2026-04-10 12:00"), pd.Timestamp("2026-04-17 12:00"))


def runs(flag: pd.Series) -> int:
    f = flag.astype(int).to_numpy()
    return int(f[0] + np.sum((f[1:] == 1) & (f[:-1] == 0))) if len(f) else 0


def week_rate(flag: pd.Series) -> float:
    return round(runs(flag) / (len(flag) / 336), 2) if len(flag) else float("nan")


def main():
    d = pd.read_csv(os.path.join(ensure_results_dir(), "e16_steps.csv"), parse_dates=["datetime"], index_col="datetime")
    in_stop = (d.index >= STOP[0]) & (d.index < STOP[1])
    rows = []
    for mode, k, p in itertools.product(["windows", "cycles"], [3, 4, 5, 6], [1, 2, 4, 6, 8]):
        ch_flags = {}
        for ch in SHIFT:
            thr = d[f"{ch}_median"] - k * 1.4826 * d[f"{ch}_mad"]
            if mode == "windows":  # last p windows of the current population below its threshold
                last = pd.concat([d[f"{ch}_s{j}"] < thr for j in range(1, p + 1)], axis=1)
                ch_flags[ch] = last.all(axis=1)
            else:  # latest window flagged in p consecutive 30-minute cycles (threshold of each cycle)
                ch_flags[ch] = (d[f"{ch}_score"] < thr).astype(int).rolling(p, min_periods=p).sum().eq(p)
        anyf = pd.concat(ch_flags, axis=1).any(axis=1)
        cal = anyf[(anyf.index < CAL_END) & ~in_stop]
        stop = anyf[in_stop]
        hold = anyf[(anyf.index >= CAL_END) & (anyf.index < EVENT_START)]
        load = anyf[(anyf.index >= EVENT_START) & (anyf.index < EVENT_END)]
        first = load[load].index.min() if load.any() else pd.NaT
        row = {"persistence_mode": mode, "k": k, "persistence": p,
               "cal_alarm_runs_per_week": week_rate(cal), "stop_period_share_in_shift": round(float(stop.mean()), 3),
               "holdout_alarm_runs_per_week": week_rate(hold), "holdout_share_in_shift": round(float(hold.mean()), 3),
               "event_delay_h": round((first - EVENT_START).total_seconds() / 3600, 1) if pd.notna(first) else None,
               "event_share_in_shift": round(float(load.mean()), 3)}
        for ch in SHIFT:
            lf = ch_flags[ch][(ch_flags[ch].index >= EVENT_START) & (ch_flags[ch].index < EVENT_END)]
            row[f"delay_{ch}_h"] = round((lf[lf].index.min() - EVENT_START).total_seconds() / 3600, 1) if lf.any() else None
        rows.append(row)
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(ensure_results_dir(), "e16b_grid.csv"), index=False)
    pd.set_option("display.width", 260)
    print(g.to_string(index=False))
    print(f"\ncalibration steps: {int(((d.index < CAL_END) & ~in_stop).sum())}, hold-out steps: "
          f"{int(((d.index >= CAL_END) & (d.index < EVENT_START)).sum())}")


if __name__ == "__main__":
    main()
