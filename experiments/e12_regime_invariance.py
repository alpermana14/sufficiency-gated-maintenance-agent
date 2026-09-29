"""E12 - Regime-invariant forecasting ablation.

The submitted manuscript describes plain lag features predicting absolute
values (eq. 8-9). The DEPLOYED code (backend/ml_engine.py) does something
different and, we argue, better: it predicts one-step DELTAS over
LEVEL-INVARIANT CENTRED lag features, trained with RECENCY-weighted samples.
This is why the multi-step forecast tracks a shifted operating regime instead
of reverting toward the historical mean. That mechanism was never written up
and is the forecasting-side novelty of the revision.

This script isolates the three ingredients with an ablation:

  A  absolute     : raw lag features, absolute target, uniform weights
                    (== the method the paper currently DESCRIBES)
  B  centred+delta: level-invariant centred lags, one-step delta target,
                    uniform weights   (isolates the centring/delta idea)
  C  full         : B + recency-weighted training (== the code ACTUALLY runs)

Protocol: train each variant ONCE on all data strictly before the load event
(so the elevated-load regime is genuinely UNSEEN), then run recursive
multi-step forecasts (H=12 = 6 h) from many origins stepping through the
event window. Report per-horizon RMSE/MAE and, crucially, the SIGNED level
bias on the channels that shift under load (z_rms, current, noise): variant A
is expected to under-predict (pull toward the low-load mean), variants B/C to
track.

Outputs (experiments/results/):
  e12_metrics.csv           per-variant/per-horizon RMSE, MAE, signed bias
  e12_bias_summary.csv      mean signed level bias per variant x channel
  e12_trajectory_zrms.png   actual vs A/B/C forecast paths at one elevated origin

Usage:  python experiments/e12_regime_invariance.py
"""

import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    EVENT_END,
    EVENT_START,
    TARGETS,
    ensure_results_dir,
    load_data,
    slice_eval_range,
)

LAG = 48                 # 24 h of history at 30-min sampling (repo LAG_STEPS)
HORIZON = int(os.environ.get("PM_HORIZON", "12"))  # 12 steps = 6 h (repo default)
N_ORIGINS = 24           # evenly spaced forecast origins across the event window
STEPS_PER_DAY = 48
HALF_LIFE_DAYS = 30      # repo RECENCY_HALF_LIFE_DAYS
SHIFT_CHANNELS = ["z_rms", "noise"]  # measured channels that move under load (current removed)

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbosity": -1,
    "seed": 42,
}
NUM_ROUNDS = 300


def build_supervised(df: pd.DataFrame, variant: str):
    """Return (X, y_by_target, sample_weight) for a training frame."""
    base = df[TARGETS].copy()
    lagged = {}
    for tgt in TARGETS:
        for k in range(1, LAG + 1):
            lagged[f"{tgt}_lag{k}"] = base[tgt].shift(k)
    lag_df = pd.DataFrame(lagged, index=base.index)
    full = pd.concat([base, lag_df], axis=1).dropna()

    if variant == "absolute":
        X = full[[f"{t}_lag{k}" for t in TARGETS for k in range(1, LAG + 1)]]
        y = {t: full[t].to_numpy() for t in TARGETS}
        w = np.ones(len(full))
    else:  # centred + delta (variants B and C share features/target)
        cols = {}
        for t in TARGETS:
            ref = full[f"{t}_lag1"]
            for k in range(2, LAG + 1):  # lag1 becomes all-zero, drop it
                cols[f"{t}_lag{k}"] = full[f"{t}_lag{k}"] - ref
        X = pd.DataFrame(cols, index=full.index)
        y = {t: (full[t] - full[f"{t}_lag1"]).to_numpy() for t in TARGETS}
        if variant == "full":
            age = np.arange(len(full) - 1, -1, -1)
            w = 0.5 ** (age / (HALF_LIFE_DAYS * STEPS_PER_DAY))
        else:
            w = np.ones(len(full))
    return X, y, w


def train(df_train: pd.DataFrame, variant: str):
    X, y, w = build_supervised(df_train, variant)
    models = {}
    for tgt in TARGETS:
        ds = lgb.Dataset(X, y[tgt], weight=w)
        models[tgt] = lgb.train(LGB_PARAMS, ds, num_boost_round=NUM_ROUNDS)
    return models, list(X.columns)


def recursive_forecast(models, x_cols, history: pd.DataFrame, variant: str):
    """H-step recursive forecast from the end of `history` (raw values)."""
    buf = history[TARGETS].iloc[-LAG:].to_numpy(dtype=float)  # (LAG, n_targets)
    col_idx = {t: i for i, t in enumerate(TARGETS)}
    out = {t: [] for t in TARGETS}

    for _ in range(HORIZON):
        row = {}
        for t in TARGETS:
            series = buf[:, col_idx[t]]        # oldest..newest
            ref = series[-1]
            for k in range(1, LAG + 1):
                val = series[-k]               # lag k
                if variant == "absolute":
                    row[f"{t}_lag{k}"] = val
                elif k >= 2:
                    row[f"{t}_lag{k}"] = val - ref
        X_pred = pd.DataFrame([row]).reindex(columns=x_cols).fillna(0)

        new = np.empty(len(TARGETS))
        for t in TARGETS:
            pred = float(models[t].predict(X_pred)[0])
            if variant == "absolute":
                new[col_idx[t]] = pred
            else:
                new[col_idx[t]] = buf[-1, col_idx[t]] + pred  # last value + delta
            out[t].append(new[col_idx[t]])
        buf = np.vstack([buf, new])[-LAG:]
    return pd.DataFrame(out)


def main() -> None:
    results_dir = ensure_results_dir()
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()

    train_df = df.loc[df.index < EVENT_START]
    # test frame: enough context (LAG) before the event through a bit past it
    test_frame = slice_eval_range(df, days_before=1.5, days_after=1.0)
    print(f"[INFO] train rows (pre-event): {len(train_df)} "
          f"({train_df.index[0]} .. {train_df.index[-1]})")
    print(f"[INFO] evaluating {N_ORIGINS} origins across "
          f"{EVENT_START} .. {EVENT_END}")

    variants = ["absolute", "centred_delta", "full"]
    trained = {}
    for v in variants:
        t0 = time.time()
        models, cols = train(train_df, v)
        trained[v] = (models, cols)
        print(f"[train] {v:14s} {time.time() - t0:5.1f}s")

    # forecast origins: timestamps inside the event window with room for H ahead
    event_times = test_frame.loc[
        (test_frame.index >= EVENT_START) & (test_frame.index < EVENT_END)
    ].index
    origins = event_times[:: max(1, len(event_times) // N_ORIGINS)][:N_ORIGINS]

    metric_rows, bias_rows = [], []
    traj = None
    for origin in origins:
        hist = df.loc[df.index <= origin]
        future = df.loc[df.index > origin].iloc[:HORIZON]
        if len(hist) < LAG or len(future) < HORIZON:
            continue
        actual = future[TARGETS].reset_index(drop=True)

        for v in variants:
            models, cols = trained[v]
            fc = recursive_forecast(models, cols, hist, v)
            for h in range(HORIZON):
                for t in TARGETS:
                    err = fc.loc[h, t] - actual.loc[h, t]
                    metric_rows.append(
                        {"variant": v, "origin": origin, "h": h + 1,
                         "target": t, "err": err, "abs_err": abs(err),
                         "sq_err": err ** 2}
                    )
            for t in SHIFT_CHANNELS:
                bias_rows.append({
                    "variant": v, "origin": origin, "target": t,
                    "signed_bias": float((fc[t].to_numpy()
                                          - actual[t].to_numpy()).mean()),
                })
        # keep one mid-event origin's z_rms trajectory for the figure
        if traj is None and origin >= EVENT_START + pd.Timedelta(hours=12):
            traj = (origin, actual["z_rms"].to_numpy(),
                    {v: recursive_forecast(trained[v][0], trained[v][1],
                                           hist, v)["z_rms"].to_numpy()
                     for v in variants})

    md = pd.DataFrame(metric_rows)
    summary = (md.groupby(["variant", "h"])
               .agg(rmse=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
                    mae=("abs_err", "mean")).reset_index())
    summary.to_csv(f"{results_dir}/e12_metrics.csv", index=False)

    bias = pd.DataFrame(bias_rows)
    bias_summary = (bias.groupby(["variant", "target"])["signed_bias"]
                    .agg(["mean", "std"]).reset_index())
    bias_summary.to_csv(f"{results_dir}/e12_bias_summary.csv", index=False)

    # ---- reports ----
    print("\n=== Aggregate RMSE/MAE across horizon (mean over 1..12) ===")
    agg = (md.groupby("variant")
           .agg(rmse=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
                mae=("abs_err", "mean")).reset_index())
    print(agg.round(4).to_string(index=False))

    print("\n=== Signed level bias on shifting channels "
          "(negative = under-predicts the elevated regime) ===")
    piv = bias_summary.pivot(index="target", columns="variant", values="mean")
    print(piv.round(4).to_string())

    print("\n=== RMSE by horizon step ===")
    print(summary.pivot(index="h", columns="variant", values="rmse")
          .round(4).to_string())

    if traj is not None:
        _plot_trajectory(traj, results_dir)
        print(f"\n[OK] trajectory figure written to {results_dir}")
    print(f"[OK] metrics -> {results_dir}/e12_metrics.csv, "
          f"e12_bias_summary.csv")


def _plot_trajectory(traj, results_dir):
    origin, actual, fcs = traj
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(1, HORIZON + 1)
    ax.plot(steps, actual, "k-o", label="actual", linewidth=2)
    labels = {"absolute": "A: absolute lags (paper as-written)",
              "centred_delta": "B: centred + delta",
              "full": "C: full (centred+delta+recency)"}
    for v, path in fcs.items():
        ax.plot(steps, path, "--", marker="s", markersize=3, label=labels[v])
    ax.set_title(f"z_rms 6-h recursive forecast during elevated load\n"
                 f"origin {origin}")
    ax.set_xlabel("horizon step (30 min each)")
    ax.set_ylabel("z_rms (mm/s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/e12_trajectory_zrms.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
