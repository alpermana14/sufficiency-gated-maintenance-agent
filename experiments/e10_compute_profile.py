"""E10 - Computational profiling (answers R4-17).

Reviewer 4: "training time is reported, but also evaluate memory consumption,
inference latency, and scalability to larger datasets."

This profiles the ACTUAL deployed retrain pipeline (ml_engine.run_pipeline,
which the 5-minute scheduler invokes: feature engineering -> train 7 LightGBM
models -> recursive forecast -> s-IDK^2 anomaly scan) under CPU-only settings:

  1. Retrain scaling: wall-clock time and PEAK RSS memory as a function of
     dataset size (most-recent 25/50/100% of records, plus 200/500/1000%
     stress cases built by tiling the record set with jitter), to characterise
     scalability. Sizes above 100% are synthetic and measure resource cost as a
     function of data volume only; they say nothing about model accuracy.
     At the 30-minute sampling interval, 1000% is roughly seven years of
     continuous operation.
  2. Stage breakdown at full size: feature engineering, training, forecast,
     anomaly detection.
  3. Inference latency: one forecast call and one anomaly-detection call in
     isolation (the per-cycle cost once models are trained).

Peak RSS is sampled by a background thread (psutil) so it captures native
allocations from numpy/pandas/lightgbm, not just Python-level objects.

NOTE: "latency under concurrent chat sessions" (R4-17) is a property of the
live FastAPI + GPT-4o service and is not covered here (it would incur LLM API
calls); it is listed as a follow-up. What this script measures is the
CPU-bound analytical retrain/inference cost, which is the scalability concern.

The LightGBM configuration under test is selected by PM_LGB_CONFIG (deployed |
tuned, see ml_engine). Outputs are suffixed with it so both configurations can
be profiled on the same machine without overwriting each other.

Outputs (experiments/results/): e10_scaling_<config>.csv,
e10_breakdown_<config>.csv, e10_hardware_<config>.csv

Usage:  python experiments/e10_compute_profile.py

Environment:
  PM_LGB_CONFIG  tuned (default, Table 2) | deployed (earlier generic setting)
  E10_THREADS    pin OpenMP/BLAS threads, e.g. 2 to emulate a small cloud box
  E10_FRACS      comma-separated dataset fractions (default 0.25,0.5,1.0,2.0);
                 values above 1 are synthetic stress cases
"""

import os
import platform
import sys
import threading
import time

# Thread pinning must happen BEFORE numpy, pandas and LightGBM are imported,
# because the OpenMP runtime reads OMP_NUM_THREADS at import time. Setting it
# later has no effect. Use E10_THREADS to emulate a small cloud instance.
_THREADS = os.environ.get("E10_THREADS")
if _THREADS:
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = _THREADS

import numpy as np
import pandas as pd
import psutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

import ml_engine
from common import load_data, ensure_results_dir


class PeakRSS:
    """Sample process RSS in a background thread; report peak delta over baseline."""
    def __init__(self, interval=0.03):
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    @property
    def peak_mb(self):
        return self.peak / 1e6

    @property
    def delta_mb(self):
        return (self.peak - self.baseline) / 1e6


def make_sized(df, frac):
    """Most-recent frac of rows; frac>1 stress-tests by tiling with jitter."""
    n = len(df)
    if frac <= 1.0:
        return df.iloc[-int(n * frac):].copy()
    reps = int(np.ceil(frac))
    tiled = pd.concat([df] * reps, ignore_index=False)
    out = tiled.iloc[-int(n * frac):].copy()
    out.index = pd.date_range(end=df.index[-1], periods=len(out), freq="30min")
    return out


def cpu_model():
    """Best-effort CPU model string, so the profiling platform can be reported."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_note():
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "cpu_model": cpu_model(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "cpu_max_MHz": round(freq.max) if freq else None,
        "total_RAM_GB": round(vm.total / 1e9, 1),
        "thread_limit": os.environ.get("E10_THREADS", "none"),
        "lgb_config": ml_engine.LGB_CONFIG,
    }


def fractions():
    """Dataset fractions to profile; E10_FRACS overrides the default sweep."""
    raw = os.environ.get("E10_FRACS")
    if not raw:
        return [0.25, 0.5, 1.0, 2.0]
    return [float(x) for x in raw.split(",") if x.strip()]


def main():
    results_dir = ensure_results_dir()
    cfg = ml_engine.LGB_CONFIG
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    hw = hardware_note()
    # Absolute timings belong to whatever machine this runs on, so report the
    # hardware alongside them. Running on a small 2-vCPU cloud instance gives
    # figures representative of the deployment server; running on a developer
    # workstation does not. Thread pinning is applied at import time above.
    if _THREADS:
        print(f"[INFO] thread-limited to {_THREADS} to emulate a small cloud instance")
    pd.Series(hw).to_csv(f"{results_dir}/e10_hardware_{cfg}.csv")
    print(f"[INFO] LIGHTGBM CONFIG: {cfg}")
    print(f"[INFO] PROFILING HARDWARE: {hw}")
    print(f"[INFO] full dataset: {len(df)} records")

    # ---- 1. retrain scaling ----
    rows = []
    for frac in fractions():
        d = make_sized(df, frac)
        t0 = time.time()
        with PeakRSS() as mem:
            ml_engine.run_pipeline(d)
        dt = time.time() - t0
        rows.append({"lgb_config": cfg, "fraction": frac, "records": len(d),
                     "retrain_s": round(dt, 2),
                     "peak_rss_delta_MB": round(mem.delta_mb, 1),
                     "peak_rss_total_MB": round(mem.peak_mb, 1)})
        print(f"  frac={frac:<4} records={len(d):<6} retrain={dt:6.2f}s "
              f"peak_delta={mem.delta_mb:6.1f}MB total={mem.peak_mb:6.1f}MB")
    scaling = pd.DataFrame(rows)
    scaling.to_csv(f"{results_dir}/e10_scaling_{cfg}.csv", index=False)

    # ---- 2. stage breakdown + 3. inference latency at full size ----
    print("\n[INFO] stage breakdown at full size...")
    d = df
    tgt = ml_engine.TARGETS
    t0 = time.time()
    X, y = ml_engine.build_supervised(d)
    t_feat = time.time() - t0

    # same training path as ml_engine.run_pipeline
    t0 = time.time()
    if ml_engine._LGB_SETTINGS[cfg]["early_stopping_rounds"]:
        split = int(len(X) * 0.9)
        models = ml_engine.train_models(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
    else:
        models = ml_engine.train_models(X, y)
    t_train = time.time() - t0

    # inference latencies (median of a few calls)
    fc_times, an_times = [], []
    for _ in range(5):
        t0 = time.time(); ml_engine.generate_forecast(models, d, X.columns); fc_times.append(time.time() - t0)
    for _ in range(3):
        t0 = time.time(); ml_engine.detect_anomalies(d); an_times.append(time.time() - t0)

    breakdown = pd.DataFrame([
        {"lgb_config": cfg, "stage": "feature_engineering", "seconds": round(t_feat, 3)},
        {"lgb_config": cfg, "stage": f"train_{len(tgt)}_models", "seconds": round(t_train, 3)},
        {"lgb_config": cfg, "stage": "forecast_inference (median)", "seconds": round(float(np.median(fc_times)), 3)},
        {"lgb_config": cfg, "stage": "anomaly_inference (median)", "seconds": round(float(np.median(an_times)), 3)},
    ])
    breakdown.to_csv(f"{results_dir}/e10_breakdown_{cfg}.csv", index=False)

    print(f"\n=== Retrain scaling ({cfg}) ===")
    print(scaling.to_string(index=False))
    print(f"\n=== Stage breakdown, full dataset ({cfg}) ===")
    print(breakdown.to_string(index=False))
    print(f"\n[OK] wrote e10_scaling_{cfg}.csv, e10_breakdown_{cfg}.csv, "
          f"e10_hardware_{cfg}.csv to {results_dir}")


if __name__ == "__main__":
    main()
