"""E16 - Label-free replay of the deployed distribution-shift rule (CAEE decision B1).

At every 30-minute step the deployed system scores the last 144 records (3 days) of each channel with
s-IDK^2 (psi = 2, omega = 20, t = 100) and flags the latest window when its similarity is below
median - 3 * 1.4826 * MAD of that population (backend/machine_status.py). This script replays exactly that
on the recorded data, without any tuning, and reports:
  * the alarm rate on normal operation (records before the 40 kg load) and on the hold-out period
    from 21 April to the load,
  * the detection delay after the load was added (5 May 08:00),
  * the share of flagged steps during the load and after its removal.
The load labels are used only to report these numbers, never to set the rule.

s-IDK^2 draws its subsamples with the standard-library `random` module (backend/IDK.py) and the deployed system
does not seed it, so flags vary from run to run. The replay is repeated for several seeds; the generator is
seeded per seed, step and channel (random.seed("e16-<seed>-<timestamp>-<channel>")), so any single step can be
reproduced without replaying the steps before it (E17 uses this).

Usage:
  PM_EVENT_START="2026-05-05 08:00:00" PM_EVENT_END="2026-05-07 12:00:00" python experiments/e16_shift_rule_replay.py
Env: E16_START (default 2026-04-01 00:00), E16_END (default 2026-05-12 00:00), E16_SEEDS (default 10),
     E16_WORKERS (default 4)
Outputs: experiments/results/e16_steps.csv (seed 0, read by E16b, E16c, E17, E18), e16_seeds.csv (one row per
         seed and signal), e16_summary.csv (median, minimum and maximum over seeds)
"""
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import EVENT_END, EVENT_START, REPO_ROOT, ensure_results_dir, load_data  # noqa: F401

sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
import machine_status as R  # noqa: E402
import ml_engine  # noqa: E402
from IDK_square_sliding import IDK_square_sliding  # noqa: E402

START = pd.Timestamp(os.environ.get("E16_START", "2026-04-01 00:00"))
END = pd.Timestamp(os.environ.get("E16_END", "2026-05-12 00:00"))
HOLDOUT_START = pd.Timestamp(os.environ.get("E16_HOLDOUT_START", "2026-04-21 00:00"))
SEEDS = int(os.environ.get("E16_SEEDS", "10"))
WORKERS = int(os.environ.get("E16_WORKERS", "4"))
CHANNELS = ["temperature"] + R.SHIFT_CHANNELS


def seed_idk(seed, ts, ch):
    random.seed(f"e16-{seed}-{pd.Timestamp(ts)}-{ch}")


def window_scores(X, seed, ts, ch):
    seed_idk(seed, ts, ch)
    return np.asarray(IDK_square_sliding(X, t=ml_engine.IDK_T, psi1=ml_engine.IDK_PSI,
                                         width=ml_engine.IDK_WIDTH, psi2=ml_engine.IDK_PSI)).ravel()


def replay(seed):
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    steps = df.index[(df.index >= START) & (df.index < END)]
    pos = {ts: i for i, ts in enumerate(df.index)}
    rows, t0 = [], time.time()
    for k, ts in enumerate(steps):
        i = pos[ts]
        if i + 1 < ml_engine.IDK_POPULATION:
            continue
        row = {"datetime": ts}
        for ch in CHANNELS:
            X = df[ch].to_numpy(dtype=float)[i + 1 - ml_engine.IDK_POPULATION:i + 1].reshape(-1, 1)
            s = window_scores(X, seed, ts, ch)
            thr = R.shift_threshold(s)
            med = float(np.median(s))
            row[f"{ch}_score"] = float(s[-1])
            row[f"{ch}_median"] = med
            row[f"{ch}_mad"] = float(np.median(np.abs(s - med)))
            row[f"{ch}_threshold"] = thr
            row[f"{ch}_flag"] = bool(s[-1] < thr)
            below = s < thr
            run = 0
            for b in below[::-1]:  # consecutive flagged windows ending at the latest one
                if not b:
                    break
                run += 1
            row[f"{ch}_run"] = run
            for j in range(1, 9):  # last 8 window scores, for other rule settings (E16b)
                row[f"{ch}_s{j}"] = float(s[-j])
        rows.append(row)
        if seed == 0 and k % 400 == 0:
            print(f"  seed 0: step {k}/{len(steps)} {ts} ({time.time() - t0:.0f} s)", flush=True)
    out = pd.DataFrame(rows).set_index("datetime")
    out["any_shift_channel"] = out[[f"{c}_flag" for c in R.SHIFT_CHANNELS]].any(axis=1)
    out["phase"] = np.where(out.index < EVENT_START, "normal_before_load",
                            np.where(out.index < EVENT_END, "load", "after_removal"))
    return seed, out


def new_flag_runs(flag: pd.Series) -> int:
    f = flag.astype(int).to_numpy()
    return int(f[0] + np.sum((f[1:] == 1) & (f[:-1] == 0))) if len(f) else 0


def summarise(seed, out):
    rows = []
    for sig in CHANNELS + ["any_shift_channel"]:
        col = sig if sig == "any_shift_channel" else f"{sig}_flag"
        f = out[col]
        normal = f[out.phase == "normal_before_load"]
        hold = normal[normal.index >= HOLDOUT_START]
        load = f[out.phase == "load"]
        after = f[out.phase == "after_removal"]
        first = load[load].index.min() if load.any() else pd.NaT
        rows.append({
            "seed": seed, "signal": sig,
            "normal_steps": len(normal), "normal_flag_rate": round(float(normal.mean()), 4) if len(normal) else None,
            "normal_alarm_runs_per_week": round(new_flag_runs(normal) / (len(normal) / 336), 2) if len(normal) else None,
            "holdout_steps": len(hold), "holdout_flag_rate": round(float(hold.mean()), 4) if len(hold) else None,
            "holdout_alarm_runs_per_week": round(new_flag_runs(hold) / (len(hold) / 336), 2) if len(hold) else None,
            "detection_delay_h": round((first - EVENT_START).total_seconds() / 3600, 1) if pd.notna(first) else None,
            "load_flag_rate": round(float(load.mean()), 4) if len(load) else None,
            "after_removal_flag_rate": round(float(after.mean()), 4) if len(after) else None,
        })
    return rows


def main():
    t0 = time.time()
    res_dir = ensure_results_dir()
    per_seed = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for seed, out in pool.map(replay, range(SEEDS)):
            if seed == 0:
                out.to_csv(os.path.join(res_dir, "e16_steps.csv"))
            per_seed.extend(summarise(seed, out))
            print(f"[INFO] seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    seeds = pd.DataFrame(per_seed)
    seeds.to_csv(os.path.join(res_dir, "e16_seeds.csv"), index=False)

    metrics = ["normal_flag_rate", "normal_alarm_runs_per_week", "holdout_flag_rate", "holdout_alarm_runs_per_week",
               "detection_delay_h", "load_flag_rate", "after_removal_flag_rate"]
    rows = []
    for sig, g in seeds.groupby("signal", sort=False):
        row = {"signal": sig, "seeds": len(g), "seeds_detecting_load": int(g["detection_delay_h"].notna().sum())}
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_median"] = round(float(v.median()), 4) if len(v) else None
            row[f"{m}_min"] = round(float(v.min()), 4) if len(v) else None
            row[f"{m}_max"] = round(float(v.max()), 4) if len(v) else None
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(res_dir, "e16_summary.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(seeds.pivot(index="seed", columns="signal", values="detection_delay_h").to_string())
    print(summary.to_string(index=False))
    print(f"[OK] {SEEDS} seeds; wrote e16_steps.csv (seed 0), e16_seeds.csv, e16_summary.csv to {res_dir} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
