"""E24 - Measured cost of the four distribution-shift detectors (CAEE EiC-10).

All four are timed on the same data, the same machine and the same settings as Table 5: the
evaluation range of the load experiment, the six measured channels and the multivariate signal,
with the settings fixed before evaluation. The point detectors are fitted on the normal-condition
records and then score every point; s-IDK squared has no training stage and scores every sliding
window against the population.

Reported per detector: fitting time, scoring time for the whole range, scoring time per point,
and the peak resident memory added by the call.

Usage: python experiments/e24_detector_timing.py
Output: experiments/results/e24_detector_timing.csv
"""
import os
import random
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EVENT_START, TARGETS, ensure_results_dir, idk_window_scores,  # noqa: E402
                    load_data, slice_eval_range)

CHANNELS = [c for c in TARGETS if c != "current"]
A_PRIORI = {
    "Isolation Forest": dict(n_estimators=100, max_samples=256),
    "One-Class SVM": dict(nu=0.5, gamma="scale"),
    "Local Outlier Factor": dict(n_neighbors=20),
    "s-IDK2": dict(psi=2, width=20),
}
REPEATS = int(os.getenv("E24_REPEATS", "5"))


def timed(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return dt, peak


def main():
    results_dir = ensure_results_dir()
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)
    normal = df.index < EVENT_START
    rows = []
    for sig, cols in [("multivariate", CHANNELS)] + [(c, [c]) for c in CHANNELS]:
        raw = df[cols].to_numpy(dtype=float)
        scaler = StandardScaler().fit(raw[normal])
        Xs = scaler.transform(raw)
        Xtr = Xs[normal]
        for name, cfg in A_PRIORI.items():
            fits, scores, peaks = [], [], []
            for r in range(REPEATS):
                if name == "s-IDK2":
                    random.seed(f"e24-{sig}-{r}")
                    inp = raw if len(cols) == 1 else Xs
                    dt, pk = timed(lambda: idk_window_scores(inp, width=cfg["width"],
                                                             psi1=cfg["psi"], psi2=cfg["psi"]))
                    fits.append(0.0)
                    scores.append(dt)
                    peaks.append(pk)
                else:
                    if name == "Isolation Forest":
                        det = IsolationForest(contamination="auto", random_state=r, n_jobs=1, **cfg)
                    elif name == "One-Class SVM":
                        det = OneClassSVM(kernel="rbf", **cfg)
                    else:
                        det = LocalOutlierFactor(novelty=True, **cfg)
                    dt_fit, pk1 = timed(lambda: det.fit(Xtr))
                    dt_sc, pk2 = timed(lambda: det.decision_function(Xs))
                    fits.append(dt_fit)
                    scores.append(dt_sc)
                    peaks.append(max(pk1, pk2))
            rows.append({"signal": sig, "detector": name, "n_train": int(normal.sum()),
                         "n_score": len(Xs), "dim": len(cols),
                         "fit_s": round(float(np.median(fits)), 4),
                         "score_s": round(float(np.median(scores)), 4),
                         "score_ms_per_point": round(1000 * float(np.median(scores)) / len(Xs), 3),
                         "peak_MB": round(float(np.median(peaks)), 2)})
        print(f"  {sig} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{results_dir}/e24_detector_timing.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== multivariate signal, median of "
          f"{REPEATS} repeats ===")
    print(out[out.signal == "multivariate"].to_string(index=False))
    print("\n=== median over the six single channels ===")
    print(out[out.signal != "multivariate"].groupby("detector")[
        ["fit_s", "score_s", "score_ms_per_point", "peak_MB"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
