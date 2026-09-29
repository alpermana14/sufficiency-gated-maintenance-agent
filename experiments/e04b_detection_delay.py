"""E4b - Detection delay and false-alarm rate at matched thresholds.

AUROC (E4) measures ranking quality but says nothing about what an operator
actually experiences: how long after a fault starts does the system alarm,
and how often does it cry wolf during normal operation? This script answers
both, for s-IDK^2 and the same three sklearn baselines from E4.

Protocol (train -> calibrate -> test, no leakage):
  1. TRAIN   = normal period, oldest days up to (EVENT_START - HOLDOUT_DAYS)
  2. HOLDOUT = normal period, the HOLDOUT_DAYS immediately before the event
               (unseen during fitting/calibration - measures REALIZED false
               alarm rate, not the in-sample rate the threshold was built on)
  3. EVENT   = the labelled anomaly window
  Detectors are fit on TRAIN only. Alarm thresholds are quantiles of TRAIN
  scores at target FPRs (1%, 5%). Thresholds are then applied, frozen, to
  HOLDOUT (-> realized FPR) and EVENT (-> detection delay).

Delay is reported two ways:
  - single-crossing: first window where score > threshold
  - persistent (k=3): first window starting a run of >=3 consecutive
    exceedances (filters single-window noise spikes an operator would
    reasonably ignore)

*** METHODOLOGY CAVEAT (read before citing these numbers in the paper) ***
Like E3/E4, this script scores s-IDK^2 by calling IDK_square_sliding ONCE
over the full TRAIN+HOLDOUT+EVENT slice, i.e. the level-2 "population" each
window is compared against (eq. 14 in the manuscript) is pooled across all
~488 windows, including the ~104 event windows themselves. This is an
OFFLINE/FORENSIC evaluation, not a simulation of the deployed algorithm:
production `ml_engine.detect_anomalies()` recomputes the population from
only the TRAILING 144 points (~3 days) ending at "now", every 5-minute
cycle, and never looks ahead. A true online replay (rolling population,
no future leakage) is the correct experiment for a delay claim that will
appear in the paper as "the deployed system detects the fault within X
minutes" - track that as a follow-up before finalizing the manuscript
numbers. What this script DOES validly show: relative delay/false-alarm
ranking between s-IDK^2 and the baselines under identical (if idealized)
conditions, and a rough magnitude.

Usage:  python experiments/e04b_detection_delay.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from common import (
    EVENT_END,
    EVENT_START,
    TARGETS,
    align_window_labels,
    ensure_results_dir,
    idk_window_scores,
    load_data,
    set_seed,
    slice_eval_range,
)

N_SEEDS = 5
IDK_WIDTH = 20
IDK_PSI = 4
HOLDOUT_DAYS = 2.0
TARGET_FPRS = [0.01, 0.05]
PERSIST_K = 3
CHANNELS_OF_INTEREST = ["z_rms", "noise"]  # measured responding channels (current removed)


def sk_detector(name: str, seed: int):
    if name == "IsolationForest":
        return IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    if name == "OneClassSVM":
        return OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    if name == "LOF":
        return LocalOutlierFactor(n_neighbors=20, novelty=True)
    raise ValueError(name)


def first_crossing_hours(scores: np.ndarray, times: pd.DatetimeIndex,
                          threshold: float, k: int) -> float:
    """Hours from EVENT_START to the first run of >=k consecutive
    exceedances starting inside the event. NaN if never detected."""
    above = scores > threshold
    run = 0
    for i, is_above in enumerate(above):
        run = run + 1 if is_above else 0
        if run >= k:
            onset_idx = i - k + 1
            return (times[onset_idx] - EVENT_START).total_seconds() / 3600.0
    return np.nan


def realized_fpr(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores > threshold))


def main() -> None:
    results_dir = ensure_results_dir()
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)

    aligned_times = df.index[IDK_WIDTH - 1 :]
    train_mask = aligned_times < (EVENT_START - pd.Timedelta(days=HOLDOUT_DAYS))
    holdout_mask = (aligned_times >= (EVENT_START - pd.Timedelta(days=HOLDOUT_DAYS))) & (
        aligned_times < EVENT_START
    )
    event_mask = (aligned_times >= EVENT_START) & (aligned_times < EVENT_END)
    print(
        f"[INFO] windows: train={train_mask.sum()} holdout={holdout_mask.sum()} "
        f"event={event_mask.sum()} (aligned range {aligned_times[0]} .. {aligned_times[-1]})"
    )

    feature_sets = {ch: [ch] for ch in CHANNELS_OF_INTEREST}
    feature_sets["multivariate"] = list(TARGETS)

    rows = []
    traces = {}  # (features, detector) -> (aligned_times, mean_score) for plotting

    for feat_name, cols in feature_sets.items():
        X_all = df[cols].to_numpy(dtype=float)
        train_raw_mask = df.index < (EVENT_START - pd.Timedelta(days=HOLDOUT_DAYS))
        scaler = StandardScaler().fit(X_all[train_raw_mask])
        Xs = scaler.transform(X_all)

        # ---- sklearn baselines: fit on TRAIN only, higher = more anomalous ----
        for det_name in ["IsolationForest", "OneClassSVM", "LOF"]:
            seeds = range(N_SEEDS) if det_name == "IsolationForest" else [0]
            per_seed_scores = []
            for seed in seeds:
                det = sk_detector(det_name, seed)
                det.fit(Xs[train_raw_mask])
                point_scores = -det.decision_function(Xs)
                per_seed_scores.append(point_scores[IDK_WIDTH - 1 :])
            rows += _summarize(feat_name, det_name, per_seed_scores, aligned_times,
                                train_mask, holdout_mask, event_mask)
            traces[(feat_name, det_name)] = (aligned_times, np.mean(per_seed_scores, axis=0))

        # ---- s-IDK^2 ----
        idk_input = df[cols].to_numpy(dtype=float) if len(cols) == 1 else Xs
        per_seed_scores = []
        for seed in range(N_SEEDS):
            set_seed(seed)
            sim = idk_window_scores(idk_input, width=IDK_WIDTH, psi1=IDK_PSI, psi2=IDK_PSI)
            per_seed_scores.append(-sim)  # higher = more anomalous
        rows += _summarize(feat_name, "s-IDK2", per_seed_scores, aligned_times,
                            train_mask, holdout_mask, event_mask)
        traces[(feat_name, "s-IDK2")] = (aligned_times, np.mean(per_seed_scores, axis=0))

        print(f"[done] {feat_name}")

    res = pd.DataFrame(rows)
    out_csv = f"{results_dir}/e04b_detection_delay.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}\n")
    show_cols = ["features", "detector", "target_fpr", "realized_fpr_mean",
                 "delay_single_h_mean", "delay_persist3_h_mean", "ever_detected_frac"]
    print(res[show_cols].round(3).to_string(index=False))

    for feat_name in feature_sets:
        _plot_traces(feat_name, traces, aligned_times, event_mask, results_dir)
    print(f"\n[OK] trace plots written to {results_dir}")


def _summarize(feat_name, det_name, per_seed_scores, aligned_times,
               train_mask, holdout_mask, event_mask) -> list:
    out = []
    for target_fpr in TARGET_FPRS:
        fprs, delays1, delaysK, ever = [], [], [], []
        for scores in per_seed_scores:
            threshold = np.quantile(scores[train_mask], 1 - target_fpr)
            fprs.append(realized_fpr(scores[holdout_mask], threshold))
            ev_scores = scores[event_mask]
            ev_times = aligned_times[event_mask]
            delays1.append(first_crossing_hours(ev_scores, ev_times, threshold, k=1))
            delaysK.append(first_crossing_hours(ev_scores, ev_times, threshold, k=PERSIST_K))
            ever.append(float(np.any(ev_scores > threshold)))
        out.append({
            "features": feat_name,
            "detector": det_name,
            "target_fpr": target_fpr,
            "realized_fpr_mean": float(np.mean(fprs)),
            "realized_fpr_std": float(np.std(fprs)),
            "delay_single_h_mean": float(np.nanmean(delays1)),
            "delay_single_h_std": float(np.nanstd(delays1)),
            "delay_persist3_h_mean": float(np.nanmean(delaysK)),
            "delay_persist3_h_std": float(np.nanstd(delaysK)),
            "ever_detected_frac": float(np.mean(ever)),
            "n_runs": len(per_seed_scores),
        })
    return out


def _plot_traces(feat_name, traces, aligned_times, event_mask, results_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    for det_name in ["IsolationForest", "OneClassSVM", "LOF", "s-IDK2"]:
        t, scores = traces[(feat_name, det_name)]
        z = (scores - np.mean(scores)) / (np.std(scores) + 1e-9)  # z-score for common y-axis
        ax.plot(t, z, label=det_name, linewidth=1.2, alpha=0.85)
    ax.axvspan(aligned_times[event_mask][0], aligned_times[event_mask][-1],
               color="red", alpha=0.08, label="labelled event")
    ax.set_title(f"Anomaly score traces (z-scored) — {feat_name}")
    ax.set_ylabel("z-scored anomaly score")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{results_dir}/e04b_trace_{feat_name}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
