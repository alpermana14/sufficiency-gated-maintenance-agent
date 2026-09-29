"""E16c - Two alternatives to the MAD rule, evaluated on the stored replay (same protocol as E16b).

A. ratio rule:     latest similarity < r * median of its 3-day population, for p consecutive steps
B. quantile rule:  latest similarity < q-quantile of the latest-window scores in the calibration period
                   (1-20 April 2026, normal operation), for p consecutive steps
The load event (5-7 May) is not used to choose r, q or p; it is only reported.
Output: experiments/results/e16c_alternatives.csv
"""
import itertools
import os

import numpy as np
import pandas as pd

from common import EVENT_END, EVENT_START, ensure_results_dir

SHIFT = ["z_rms", "x_rms", "z_peak", "x_peak", "noise"]
CAL_END = pd.Timestamp("2026-04-21 00:00")


def runs(flag):
    f = flag.astype(int).to_numpy()
    return int(f[0] + np.sum((f[1:] == 1) & (f[:-1] == 0))) if len(f) else 0


def evaluate(d, raw_flags, label, p):
    ch = {c: raw_flags[c].astype(int).rolling(p, min_periods=p).sum().eq(p) for c in SHIFT}
    anyf = pd.concat(ch, axis=1).any(axis=1)
    cal, hold = anyf[anyf.index < CAL_END], anyf[(anyf.index >= CAL_END) & (anyf.index < EVENT_START)]
    load = anyf[(anyf.index >= EVENT_START) & (anyf.index < EVENT_END)]
    first = load[load].index.min() if load.any() else pd.NaT
    out = {"rule": label, "persistence_steps": p,
           "cal_alarm_runs_per_week": round(runs(cal) / (len(cal) / 336), 2),
           "holdout_alarm_runs_per_week": round(runs(hold) / (len(hold) / 336), 2),
           "holdout_share_in_shift": round(float(hold.mean()), 3),
           "event_delay_h": round((first - EVENT_START).total_seconds() / 3600, 1) if pd.notna(first) else None,
           "event_share_in_shift": round(float(load.mean()), 3)}
    for c in SHIFT:
        lf = ch[c][(ch[c].index >= EVENT_START) & (ch[c].index < EVENT_END)]
        out[f"delay_{c}_h"] = round((lf[lf].index.min() - EVENT_START).total_seconds() / 3600, 1) if lf.any() else None
    return out


def main():
    d = pd.read_csv(os.path.join(ensure_results_dir(), "e16_steps.csv"), parse_dates=["datetime"], index_col="datetime")
    rows = []
    for r, p in itertools.product([0.3, 0.4, 0.5, 0.6], [1, 2, 4]):
        raw = {c: d[f"{c}_score"] < r * d[f"{c}_median"] for c in SHIFT}
        rows.append(evaluate(d, raw, f"ratio r={r}", p))
    cal = d[d.index < CAL_END]
    for q, p in itertools.product([0.005, 0.01, 0.02], [1, 2, 4]):
        raw = {c: d[f"{c}_score"] < cal[f"{c}_score"].quantile(q) for c in SHIFT}
        rows.append(evaluate(d, raw, f"quantile q={q}", p))
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(ensure_results_dir(), "e16c_alternatives.csv"), index=False)
    pd.set_option("display.width", 250)
    print(g.to_string(index=False))
    print("\ncalibration quantiles (q=0.01):", {c: round(cal[f'{c}_score'].quantile(0.01), 3) for c in SHIFT})


if __name__ == "__main__":
    main()
