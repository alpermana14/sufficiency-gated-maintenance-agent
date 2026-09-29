"""E28 - Sensitivity of the four detectors to added sensor noise (R1-6).

R1-6 asks for the sensitivity of s-IDK^2 to noise. The source paper of the method
(Ting et al., VLDB J. 33:753-780, Sect. 7.3) studies noise for a different task, a dataset
of many short series, so the question is answered here on the conveyor data.

Protocol identical to E13: the same evaluation range, the same labels, the same settings
fixed before evaluation, and the same evaluation start index. Gaussian noise with a
standard deviation of k times each channel's baseline standard deviation is added to every
record of the evaluation range before scoring, for k in NOISE_LEVELS. The baseline
standard deviation is computed on the records before the load, so the noise scale does not
depend on the shifted condition.

Output: experiments/results/e28_noise.csv
Usage:  PM_EVENT_START="2026-05-05 08:00:00" PM_EVENT_END="2026-05-07 12:00:00" \
            python experiments/e28_noise_sensitivity.py
  env:  E28_LEVELS  comma-separated noise levels (default 0,0.25,0.5,1.0)
        E28_SEEDS   number of noise draws per level (default 5)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EVENT_START, ensure_results_dir, event_labels, load_data,  # noqa: E402
                    slice_eval_range)
from e13_anomaly_fair_eval import (  # noqa: E402
    A_PRIORI, CHANNELS, START, STOCHASTIC, fast_auroc, idk_scores, point_scores)

N_SEEDS = int(os.environ.get("E28_SEEDS", "5"))


def levels():
    raw = os.environ.get("E28_LEVELS")
    return [float(x) for x in raw.split(",")] if raw else [0.0, 0.25, 0.5, 1.0]


def main():
    results_dir = ensure_results_dir()
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)
    labels = event_labels(df.index)
    normal_mask = df.index < EVENT_START
    print(f"[INFO] evaluation points={len(df)} (from index {START}) | "
          f"shifted={int(labels[START:].sum())} | baseline={int((1 - labels[START:]).sum())}")

    rows = []
    for signal in CHANNELS + ["multivariate"]:
        cols = CHANNELS if signal == "multivariate" else [signal]
        base_sd = df.loc[normal_mask, cols].std().to_numpy()
        X0 = df[cols].to_numpy(dtype=float)
        y = labels[START:]
        for lvl in levels():
            for seed in range(N_SEEDS if lvl > 0 else 1):
                rng = np.random.default_rng(1000 * seed + 7)
                X = X0 + rng.normal(0.0, lvl * base_sd, size=X0.shape) if lvl > 0 else X0
                Xtr = X[normal_mask]
                for method, cfg in A_PRIORI.items():
                    seeds = range(N_SEEDS) if method in STOCHASTIC else [0]
                    aucs = []
                    for s in seeds:
                        if method == "s-IDK2":
                            sc = idk_scores(cfg, s, X)
                        else:
                            sc = point_scores(method, cfg, s, Xtr, X)
                        aucs.append(fast_auroc(y, sc[START:]))
                    rows.append({"signal": signal, "noise_level": lvl, "noise_seed": seed,
                                 "method": method, "auroc": float(np.mean(aucs))})
            print(f"  {signal:12} noise x{lvl:<5} done")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(results_dir, "e28_noise.csv"), index=False)
    agg = (out.groupby(["signal", "method", "noise_level"]).auroc.mean()
           .round(3).reset_index())
    agg.to_csv(os.path.join(results_dir, "e28_noise_mean.csv"), index=False)
    print("\n=== AUROC against added noise, responding channels ===")
    for signal in ["z_rms", "noise", "multivariate"]:
        p = agg[agg.signal == signal].pivot(index="method", columns="noise_level", values="auroc")
        print(f"\n{signal}:")
        print(p.to_string())
    print("\n[OK] wrote e28_noise.csv and e28_noise_mean.csv")


if __name__ == "__main__":
    main()
