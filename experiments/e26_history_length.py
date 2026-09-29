"""E26 - What a longer history does to forecasting accuracy and to its cost (R2-7).

R2-7 asks how the system behaves when the record grows to millions, and whether the
reported scalability analysis represents long-term industrial deployment. The cost half
of that question is answered by e10_compute_profile.py. This script answers the other
half on the forecasting layer: does a longer training history make the forecasts better,
and what does it cost?

Protocol (identical to the paper's, e01b_forecasting_modelcomp.py / model_comparison.py):
  * test windows  = several consecutive 7-day blocks at the end of the record, so that
                    the curve does not rest on a single week;
  * training set  = the most recent K days BEFORE that window, K varied;
  * features      = absolute-value lag features, LAG_STEPS lags;
  * evaluation    = recursive FORECAST_HORIZON-step forecast from N_ORIGINS evenly
                    spaced origins inside the test window, pooled over the six channels;
  * model         = LightGBM with the settings of ml_engine (PM_LGB_CONFIG), NOT retuned
                    per arm, so the only thing that varies is how much history is used.

Reported per arm: pooled MAE and MSE, training time, peak resident memory.

Output: experiments/results/e26_history_length.csv
Usage:  python experiments/e26_history_length.py
  env:  E26_DAYS     comma-separated training lengths in days (default 30,60,90,120,180)
        E26_WINDOWS  number of consecutive 7-day test blocks (default 4)
        PM_LGB_CONFIG  tuned (default) | deployed
"""

import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, ensure_results_dir  # noqa: E402
import ml_engine  # noqa: E402
from ml_engine import TARGETS, LAG_STEPS, FORECAST_HORIZON  # noqa: E402
from e01b_forecasting_modelcomp import (  # noqa: E402
    build_supervised, recursive_forecast_one_origin)

STEPS_PER_DAY = 48
TEST_DAYS = 7
N_ORIGINS = 20
SEED = 42


class PeakRSS:
    """Sample process RSS in a background thread; report the peak over the baseline."""

    def __init__(self, interval=0.03):
        self.interval = interval
        self.proc = psutil.Process()
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    @property
    def delta_mb(self):
        return (self.peak - self.baseline) / 1e6


def days_list():
    raw = os.environ.get("E26_DAYS")
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return [30, 60, 90, 120, 180]


def evaluate(models, X_cols, train_df, test_df):
    """Pooled MAE and MSE of the recursive forecast over N_ORIGINS origins."""
    max_origin = len(test_df) - FORECAST_HORIZON - 1
    origins = np.linspace(LAG_STEPS, max_origin, num=N_ORIGINS, dtype=int).tolist()
    full = pd.concat([train_df, test_df])
    offset = len(train_df)
    pred = {t: [] for t in TARGETS}
    true = {t: [] for t in TARGETS}
    for o in origins:
        history = full.iloc[: offset + o]
        fc = recursive_forecast_one_origin(models, history, X_cols)
        actual = test_df.iloc[o: o + FORECAST_HORIZON]
        if len(actual) < FORECAST_HORIZON:
            continue
        for t in TARGETS:
            pred[t].extend(fc[t].tolist())
            true[t].extend(actual[t].tolist())
    p = np.concatenate([np.asarray(pred[t]) for t in TARGETS])
    y = np.concatenate([np.asarray(true[t]) for t in TARGETS])
    return mean_absolute_error(y, p), mean_squared_error(y, p)


def main():
    results_dir = ensure_results_dir()
    cfg = ml_engine.LGB_CONFIG
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    n_test = TEST_DAYS * STEPS_PER_DAY
    n_win = int(os.environ.get("E26_WINDOWS", "4"))
    print(f"[INFO] LightGBM config: {cfg}; {len(df)} records; "
          f"{n_win} test windows of {TEST_DAYS} days")

    rows = []
    for w in range(n_win):
        hi = len(df) - n_test * w
        test_df = df.iloc[hi - n_test: hi]
        train_all = df.iloc[: hi - n_test]
        print(f"\n[window {w}] test {test_df.index[0]} to {test_df.index[-1]}; "
              f"history before it: {len(train_all)} records "
              f"({len(train_all) / STEPS_PER_DAY:.0f} days)")
        for days in days_list():
            n = days * STEPS_PER_DAY
            if n > len(train_all):
                print(f"  {days:>4} days  skipped (only {len(train_all)} records available)")
                continue
            train_df = train_all.iloc[-n:]
            X, y = build_supervised(train_df)
            t0 = time.time()
            with PeakRSS() as mem:
                models = ml_engine.train_models(X, y)
            fit_s = time.time() - t0
            mae, mse = evaluate(models, list(X.columns), train_df, test_df)
            rows.append({"lgb_config": cfg, "window": w, "train_days": days,
                         "train_records": len(train_df), "fit_s": round(fit_s, 2),
                         "peak_rss_delta_MB": round(mem.delta_mb, 1),
                         "MAE": round(float(mae), 5), "MSE": round(float(mse), 5)})
            print(f"  {days:>4} days  {len(train_df):>6} records  fit={fit_s:6.2f}s  "
                  f"MAE={mae:.5f}  MSE={mse:.5f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(results_dir, "e26_history_length.csv"), index=False)
    agg = out.groupby("train_days").agg(
        windows=("window", "nunique"), train_records=("train_records", "max"),
        fit_s=("fit_s", "median"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        MSE=("MSE", "mean")).round(5).reset_index()
    agg.to_csv(os.path.join(results_dir, "e26_history_length_mean.csv"), index=False)
    print("\n=== mean over test windows: accuracy versus length of training history ===")
    print(agg.to_string(index=False))
    best = agg.loc[agg["MAE"].idxmin()]
    worst_long = agg.iloc[-1]
    print(f"\n[OK] lowest mean MAE at {int(best['train_days'])} days; "
          f"longest arm ({int(worst_long['train_days'])} days) gives "
          f"{(worst_long['MAE'] / best['MAE'] - 1) * 100:.1f}% higher MAE "
          f"at {worst_long['fit_s'] / best['fit_s']:.1f}x the training time")


if __name__ == "__main__":
    main()
