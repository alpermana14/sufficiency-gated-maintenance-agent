"""E23 - How long can the detector keep flagging a change that stays? (CAEE R2-10)

The deployed rule scores the last 144 records (3 days) of each channel as one population and
flags the latest window when its similarity falls below median - 3 * 1.4826 * MAD. Reviewer 2
asks what happens when those three days already contain the changed condition. They do, after
the change has lasted long enough: the changed windows stop being a minority of the population,
the median moves with them, and the flag stops. This script measures when that happens, for
trailing populations of 1, 3, 7 and 14 days.

Nothing else is varied: same rule, same psi and omega, same data, same seeds.

Reported per population size:
  * alarm rate on normal operation before the load,
  * detection delay after the load was added,
  * for how many hours after the load the rule keeps flagging, which is the contamination point.

Usage:  python experiments/e23_population.py
Env:    E23_SEEDS (default 3), E23_WORKERS (default 4)
Output: experiments/results/e23_population.csv, e23_population_steps.csv
"""
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENT_END, EVENT_START, REPO_ROOT, ensure_results_dir, load_data  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
import machine_status as R  # noqa: E402
import ml_engine  # noqa: E402
from IDK_square_sliding import IDK_square_sliding  # noqa: E402

POPULATIONS = {"1 day": 48, "3 days": 144, "7 days": 336, "14 days": 672}
CHANNELS = list(R.SHIFT_CHANNELS)
START = pd.Timestamp(os.getenv("E23_START", "2026-04-01 00:00:00"))
END = pd.Timestamp(os.getenv("E23_END", "2026-05-12 00:00:00"))
SEEDS = int(os.getenv("E23_SEEDS", "3"))
WORKERS = int(os.getenv("E23_WORKERS", "4"))


def window_scores(X, seed, ts, ch, pop):
    random.seed(f"e23-{pop}-{seed}-{ts}-{ch}")
    return np.asarray(IDK_square_sliding(X, t=ml_engine.IDK_T, psi1=ml_engine.IDK_PSI,
                                         width=ml_engine.IDK_WIDTH, psi2=ml_engine.IDK_PSI)).ravel()


def replay(args):
    name, pop, seed = args
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    steps = df.index[(df.index >= START) & (df.index < END)]
    pos = {ts: i for i, ts in enumerate(df.index)}
    rows, t0 = [], time.time()
    for k, ts in enumerate(steps):
        i = pos[ts]
        if i + 1 < pop:
            continue
        row = {"datetime": ts, "population": name, "pop_records": pop, "seed": seed}
        for ch in CHANNELS:
            X = df[ch].to_numpy(dtype=float)[i + 1 - pop:i + 1].reshape(-1, 1)
            s = window_scores(X, seed, ts, ch, pop)
            row[f"{ch}_flag"] = bool(s[-1] < R.shift_threshold(s))
        rows.append(row)
        if seed == 0 and k % 600 == 0:
            print(f"  {name} seed 0: step {k}/{len(steps)} ({time.time() - t0:.0f} s)", flush=True)
    out = pd.DataFrame(rows).set_index("datetime")
    out["any"] = out[[f"{c}_flag" for c in CHANNELS]].any(axis=1)
    return out


def summarise(out):
    name = out["population"].iloc[0]
    seed = int(out["seed"].iloc[0])
    before = out[out.index < EVENT_START]["any"]
    during = out[(out.index >= EVENT_START) & (out.index < EVENT_END)]["any"]
    after = out[out.index >= EVENT_END]["any"]
    # detection delay: first flagged step at or after the load was added
    first = during[during].index.min() if during.any() else pd.NaT
    delay = (first - EVENT_START).total_seconds() / 3600 if pd.notna(first) else float("nan")
    # contamination: the last flagged step of the load period, measured from the load
    last = during[during].index.max() if during.any() else pd.NaT
    held = (last - EVENT_START).total_seconds() / 3600 if pd.notna(last) else float("nan")
    # the longest run of consecutive flagged steps inside the load period
    run = best = 0
    for v in during.to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    return {"population": name, "pop_records": int(out["pop_records"].iloc[0]), "seed": seed,
            "alarm_rate_normal_%": round(100 * before.mean(), 2) if len(before) else float("nan"),
            "detection_delay_h": round(delay, 1),
            "last_flag_after_load_h": round(held, 1),
            "longest_flagged_run_h": round(best * 0.5, 1),
            "flagged_share_during_load_%": round(100 * during.mean(), 1) if len(during) else float("nan"),
            "flagged_share_after_removal_%": round(100 * after.mean(), 1) if len(after) else float("nan")}


def main():
    results_dir = ensure_results_dir()
    jobs = [(name, pop, seed) for name, pop in POPULATIONS.items() for seed in range(SEEDS)]
    print(f"[INFO] {len(jobs)} replays: {len(POPULATIONS)} population sizes x {SEEDS} seeds, "
          f"channels {CHANNELS}", flush=True)
    t0 = time.time()
    frames, rows = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for out in ex.map(replay, jobs):
            frames.append(out)
            rows.append(summarise(out))
            print(f"  done {rows[-1]['population']} seed {rows[-1]['seed']} ({time.time() - t0:.0f} s)",
                  flush=True)
    steps = pd.concat(frames)
    steps.to_csv(f"{results_dir}/e23_population_steps.csv")
    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/e23_population.csv", index=False)
    agg = df.groupby("population").median(numeric_only=True).reindex(list(POPULATIONS))
    pd.set_option("display.width", 200)
    print("\n=== median over seeds ===")
    print(agg.to_string())
    print(f"\n[OK] wrote e23_population.csv and e23_population_steps.csv in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
