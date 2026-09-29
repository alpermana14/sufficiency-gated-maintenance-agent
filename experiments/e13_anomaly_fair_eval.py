"""E13 - Fair re-evaluation of distribution-shift detection on the six measured channels.

Answers CAEE EiC-8 (tune all methods fairly, report tuning steps and values),
EiC-9 (statistical tests), R2-11 (channel dependence) and R2-12 (confidence
intervals and significance of AUROC differences).

Changes with respect to E3/E4:
  * The motor-current channel is excluded (not a valid measurement; CAEE
    decision D7). The multivariate signal uses the six measured channels.
  * Every detector gets the same grid budget (25 configurations) and is
    reported under two settings:
      (a) a-priori  - settings fixed without access to labels: scikit-learn
                      defaults for the point detectors, and for s-IDK^2 the
                      configuration justified in Section 4.3 before evaluation
                      (psi = 2, the smallest value of the search space in [24];
                      omega = 20, one 10-hour operating cycle), also deployed;
      (b) grid-best - the configuration with the highest mean AUROC on the
                      evaluation labels (an optimistic upper bound applied
                      identically to all methods).
    The grid median AUROC is reported as a hyperparameter-robustness figure.
  * All methods and all configurations are scored on ONE common index range
    (starting at the largest window width - 1), so different omega values do
    not change the evaluated windows.
  * 95% confidence intervals from a moving-block bootstrap (block = 20 points)
    of the seed-averaged AUROC, and paired bootstrap tests of the AUROC
    difference between s-IDK^2 and each baseline (same resampled indices).

Protocol otherwise identical to E4: point detectors are fitted on the
normal-condition points before the event; s-IDK^2 scores the whole evaluation
range as one population; window scores are assigned to the window-end point.

Usage (same labelled window as E3/E4):
  PM_EVENT_START="2026-05-05 08:00:00" PM_EVENT_END="2026-05-07 12:00:00" \
      python experiments/e13_anomaly_fair_eval.py

Outputs: experiments/results/e13_grid.csv, e13_summary.csv, e13_paired.csv
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from common import (
    EVENT_END,
    EVENT_START,
    TARGETS,
    ensure_results_dir,
    event_labels,
    idk_window_scores,
    load_data,
    set_seed,
    slice_eval_range,
)

CHANNELS = [c for c in TARGETS if c != "current"]
N_SEEDS = 5
N_BOOT = 2000
BLOCK = 20
BOOT_SEED = 12345

GRIDS = {
    "IsolationForest": [
        {"n_estimators": n, "max_samples": m}
        for n, m in itertools.product([50, 100, 200, 300, 500], [16, 32, 64, 128, 256])
    ],
    "OneClassSVM": [
        {"nu": nu, "gamma": g}
        for nu, g in itertools.product([0.01, 0.05, 0.1, 0.25, 0.5], ["scale", 0.01, 0.1, 1.0, 10.0])
    ],
    "LOF": [{"n_neighbors": k} for k in range(5, 130, 5)],
    "s-IDK2": [
        {"psi": p, "width": w}
        for p, w in itertools.product([2, 4, 8, 16, 32], [8, 12, 20, 32, 48])
    ],
}
A_PRIORI = {
    "IsolationForest": {"n_estimators": 100, "max_samples": 256},  # sklearn default ('auto' = min(256, n))
    "OneClassSVM": {"nu": 0.5, "gamma": "scale"},                  # sklearn default
    "LOF": {"n_neighbors": 20},                                    # sklearn default
    "s-IDK2": {"psi": 2, "width": 20},                             # a-priori choice of Section 4.3 (decision D16)
}
STOCHASTIC = {"IsolationForest", "s-IDK2"}
START = max(cfg["width"] for cfg in GRIDS["s-IDK2"]) - 1  # common evaluation start index


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def point_scores(method: str, cfg: dict, seed: int, X_train: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    if method == "IsolationForest":
        det = IsolationForest(contamination="auto", random_state=seed, n_jobs=-1, **cfg)
    elif method == "OneClassSVM":
        det = OneClassSVM(kernel="rbf", **cfg)
    else:
        det = LocalOutlierFactor(novelty=True, **cfg)
    det.fit(X_train)
    return -det.decision_function(X_all)  # higher = more anomalous


def idk_scores(cfg: dict, seed: int, X_all: np.ndarray) -> np.ndarray:
    set_seed(seed)
    sim = idk_window_scores(X_all, width=cfg["width"], psi1=cfg["psi"], psi2=cfg["psi"])
    # window j ends at point j + width - 1; negate so higher = more anomalous
    aligned = np.full(len(X_all), np.nan)
    aligned[cfg["width"] - 1:] = -sim
    return aligned


def block_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)
    return idx[:, :n]


def boot_auroc(y: np.ndarray, seed_scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Seed-averaged AUROC for each bootstrap replicate (NaN if one class is missing)."""
    out = np.empty(len(idx))
    for b, ib in enumerate(idx):
        yb = y[ib]
        out[b] = np.mean([fast_auroc(yb, s[ib]) for s in seed_scores])
    return out


def main() -> None:
    results_dir = ensure_results_dir()
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)
    labels = event_labels(df.index)
    normal_mask = df.index < EVENT_START
    y = labels[START:]
    print(f"[INFO] event {EVENT_START} .. {EVENT_END} | channels={CHANNELS}")
    print(f"[INFO] evaluation points={len(y)} (from index {START}) | shifted={int(y.sum())} "
          f"| baseline={int(len(y) - y.sum())} | normal-fit points={int(normal_mask.sum())}")

    signals = {ch: [ch] for ch in CHANNELS}
    signals["multivariate"] = list(CHANNELS)

    grid_rows, summary_rows, paired_rows = [], [], []
    rng = np.random.default_rng(BOOT_SEED)
    idx = block_indices(len(y), rng)  # same replicates for every method -> paired tests

    for sig, cols in signals.items():
        raw = df[cols].to_numpy(dtype=float)
        scaler = StandardScaler().fit(raw[normal_mask])
        Xs = scaler.transform(raw)
        idk_input = raw if len(cols) == 1 else Xs
        store = {}  # (method, cfg_key) -> array (n_seeds, n_eval)

        for method, grid in GRIDS.items():
            seeds = range(N_SEEDS) if method in STOCHASTIC else [0]
            for cfg in grid:
                key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
                per_seed = []
                for seed in seeds:
                    if method == "s-IDK2":
                        s = idk_scores(cfg, seed, idk_input)
                    else:
                        s = point_scores(method, cfg, seed, Xs[normal_mask], Xs)
                    s = s[START:]
                    per_seed.append(s)
                    grid_rows.append({"signal": sig, "method": method, "config": str(cfg), "seed": seed,
                                      "auroc": fast_auroc(y, s), "ap": average_precision_score(y, s)})
                store[(method, key)] = np.vstack(per_seed)
            print(f"  {sig:12s} {method:16s} grid done")

        g = pd.DataFrame([r for r in grid_rows if r["signal"] == sig])
        chosen = {}
        for method in GRIDS:
            gm = g[g.method == method].groupby("config")[["auroc", "ap"]].mean()
            best_cfg = gm["auroc"].idxmax()
            for setting, cfg_str in [("a_priori", str(A_PRIORI[method])), ("grid_best", best_cfg)]:
                cfg = eval(cfg_str)  # configs are our own literal dicts
                key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
                scores = store[(method, key)]
                boot = boot_auroc(y, scores, idx)
                chosen[(method, setting)] = boot
                summary_rows.append({
                    "signal": sig, "method": method, "setting": setting, "config": cfg_str,
                    "auroc_mean": gm.loc[cfg_str, "auroc"],
                    "auroc_seed_std": g[(g.method == method) & (g.config == cfg_str)]["auroc"].std(ddof=0),
                    "ci_lo": np.nanpercentile(boot, 2.5), "ci_hi": np.nanpercentile(boot, 97.5),
                    "ap_mean": gm.loc[cfg_str, "ap"],
                    "grid_median_auroc": gm["auroc"].median(),
                    "grid_min_auroc": gm["auroc"].min(), "grid_max_auroc": gm["auroc"].max(),
                })

        for setting in ["a_priori", "grid_best"]:
            ref = chosen[("s-IDK2", setting)]
            for base in ["IsolationForest", "OneClassSVM", "LOF"]:
                d = ref - chosen[(base, setting)]
                d = d[~np.isnan(d)]
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                paired_rows.append({"signal": sig, "setting": setting, "baseline": base,
                                    "delta_ci_lo": np.percentile(d, 2.5), "delta_ci_hi": np.percentile(d, 97.5),
                                    "p_boot": min(1.0, p), "n_valid_boot": len(d)})

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    # point estimate of the difference from the summary table (mean AUROC over seeds)
    lut = summary.set_index(["signal", "method", "setting"])["auroc_mean"]
    paired["delta_auroc"] = [
        lut[(r.signal, "s-IDK2", r.setting)] - lut[(r.signal, r.baseline, r.setting)] for r in paired.itertuples()
    ]
    paired = paired[["signal", "setting", "baseline", "delta_auroc", "delta_ci_lo", "delta_ci_hi", "p_boot", "n_valid_boot"]]

    pd.DataFrame(grid_rows).to_csv(f"{results_dir}/e13_grid.csv", index=False)
    summary.to_csv(f"{results_dir}/e13_summary.csv", index=False)
    paired.to_csv(f"{results_dir}/e13_paired.csv", index=False)

    pd.set_option("display.width", 200)
    for setting in ["a_priori", "grid_best"]:
        print(f"\n=== AUROC ({setting}) mean [95% block-bootstrap CI] ===")
        sub = summary[summary.setting == setting].copy()
        sub["cell"] = sub.apply(lambda r: f"{r.auroc_mean:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
        print(sub.pivot(index="signal", columns="method", values="cell").loc[list(signals)].to_string())
    print("\n=== grid-best configurations ===")
    print(summary[summary.setting == "grid_best"].pivot(index="signal", columns="method", values="config")
          .loc[list(signals)].to_string())
    print("\n=== paired differences s-IDK2 - baseline ===")
    print(paired.round(3).to_string(index=False))
    print(f"\n[OK] wrote e13_grid.csv, e13_summary.csv, e13_paired.csv to {results_dir}")


if __name__ == "__main__":
    main()
