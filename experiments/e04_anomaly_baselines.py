"""E4 — s-IDK^2 vs unsupervised baselines on the conveyor data (answers R4-9, R1).

Protocol (train-on-normal / score-everything):
  * Training set: all points BEFORE the labelled event window (assumed normal).
  * Scoring set:  every point in the evaluation range.
  * Detectors:    Isolation Forest, One-Class SVM, Local Outlier Factor
                  (novelty mode) — per channel and multivariate (all 7 channels)
                  — vs s-IDK^2 (paper parameters psi1=psi2=2, width=20;
                  override psi with PM_IDK_PSI).
  * Metrics:      AUROC and Average Precision (PR-AUC), higher = better.
                  Stochastic detectors averaged over N_SEEDS runs.

s-IDK^2 scores are window-level; they are aligned to the window-end timestamp
and all detectors are evaluated on the common index range [width-1:] so the
comparison is index-for-index fair.

Outputs: experiments/results/e04_baselines.csv (+ printed summary table)

Usage:  python experiments/e04_anomaly_baselines.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from common import (
    EVENT_END,
    EVENT_START,
    TARGETS,
    align_window_labels,
    ensure_results_dir,
    event_labels,
    idk_window_scores,
    load_data,
    set_seed,
    slice_eval_range,
)

N_SEEDS = 5
IDK_WIDTH = 20
# Paper operating point (E3: smallest psi in the search space, best AUROC).
IDK_PSI = int(os.environ.get("PM_IDK_PSI", "2"))


def sk_detectors(seed: int) -> dict:
    return {
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True),
    }


def evaluate(y: np.ndarray, anomaly_score: np.ndarray) -> dict:
    return {
        "auroc": roc_auc_score(y, anomaly_score),
        "ap": average_precision_score(y, anomaly_score),
    }


def main() -> None:
    results_dir = ensure_results_dir()
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)
    labels = event_labels(df.index)
    if labels.sum() == 0:
        raise RuntimeError("No labelled anomaly points in range; check PM_EVENT_*.")

    normal_mask = df.index < EVENT_START
    print(
        f"[INFO] {len(df)} points | train-normal={normal_mask.sum()} "
        f"| anomalous={labels.sum()}"
    )
    print(
        f"[INFO] event window {EVENT_START} .. {EVENT_END} | "
        f"s-IDK2 psi={IDK_PSI}, width={IDK_WIDTH}, seeds={N_SEEDS}"
    )

    # Common evaluation range: indices >= IDK_WIDTH - 1 (where IDK has scores)
    y_common = align_window_labels(labels, IDK_WIDTH)
    rows = []

    feature_sets = {ch: [ch] for ch in TARGETS}
    feature_sets["multivariate"] = list(TARGETS)

    for name, cols in feature_sets.items():
        X_all = df[cols].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X_all[normal_mask])
        Xs = scaler.transform(X_all)

        # --- sklearn baselines (point-level scores, sliced to common range) ---
        for det_name in ["IsolationForest", "OneClassSVM", "LOF"]:
            per_seed = []
            seeds = range(N_SEEDS) if det_name == "IsolationForest" else [0]
            for seed in seeds:
                det = sk_detectors(seed)[det_name]
                det.fit(Xs[normal_mask])
                # decision_function: higher = more normal -> negate
                point_scores = -det.decision_function(Xs)
                m = evaluate(y_common, point_scores[IDK_WIDTH - 1 :])
                per_seed.append(m)
            rows.append(_agg(name, det_name, per_seed))

        # --- s-IDK^2 (window-level scores) ---
        # Raw values per channel (paper protocol). For the multivariate set,
        # standardize first: raw scales differ by orders of magnitude (noise
        # ~65 dB vs current ~2 A) and would dominate the isolation kernel's
        # Euclidean distances.
        idk_input = df[cols].to_numpy(dtype=float) if len(cols) == 1 else Xs
        per_seed = []
        for seed in range(N_SEEDS):
            set_seed(seed)
            sim = idk_window_scores(
                idk_input,
                width=IDK_WIDTH, psi1=IDK_PSI, psi2=IDK_PSI,
            )
            per_seed.append(evaluate(y_common, -sim))
        rows.append(_agg(name, "s-IDK2", per_seed))

    res = pd.DataFrame(rows)
    out_csv = f"{results_dir}/e04_baselines.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}\n")
    print(
        res.pivot(index="features", columns="detector", values="auroc_mean")
        .round(3)
        .to_string()
    )


def _agg(features: str, detector: str, per_seed: list) -> dict:
    aurocs = [m["auroc"] for m in per_seed]
    aps = [m["ap"] for m in per_seed]
    return {
        "features": features,
        "detector": detector,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "ap_mean": float(np.mean(aps)),
        "ap_std": float(np.std(aps)),
        "n_runs": len(per_seed),
    }


if __name__ == "__main__":
    main()
