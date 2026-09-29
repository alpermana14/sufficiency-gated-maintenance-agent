"""E27 - Sensitivity of the forecast to the lag-window length (R2-4).

R2-4 asks how the lag window, horizon, targets and sampling were chosen "rather than
empirically fixed". The lag window w = 48 (24 h at the 30-minute sampling interval) was
chosen because it covers one full daily cycle of the recorded signals. This script checks
that choice against the alternatives.

It is run on VALIDATION windows, consecutive seven-day blocks ending immediately before
the test week, and never on the test week itself, so the choice is not made on the data used to report
forecasting accuracy in Section 5.1.

Protocol (identical to the paper's, e01b_forecasting_modelcomp.py):
  * training      = every record before the validation window;
  * evaluation    = recursive FORECAST_HORIZON-step forecast from N_ORIGINS evenly spaced
                    origins inside the validation window, pooled over the six channels;
  * model         = LightGBM with the settings of ml_engine, not retuned per arm, so the
                    only thing that varies is the lag window.

Output: experiments/results/e27_lag_window.csv
Usage:  python experiments/e27_lag_window.py
  env:  E27_LAGS     comma-separated lag windows in steps (default 12,24,48,96)
        E27_WINDOWS  number of consecutive 7-day validation windows (default 4)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, ensure_results_dir  # noqa: E402
import ml_engine  # noqa: E402
from ml_engine import TARGETS, FORECAST_HORIZON, make_lag_features  # noqa: E402

STEPS_PER_DAY = 48
WINDOW_DAYS = 7
N_ORIGINS = 20


def build_supervised(df, lag):
    data = make_lag_features(df[TARGETS].copy(), lag)
    return data.drop(columns=TARGETS), data[TARGETS]


def recursive_forecast(models, history_df, X_cols, lag, horizon=FORECAST_HORIZON):
    """Recursive multi-step forecast, the protocol of e01b with the lag as a parameter."""
    buffer = history_df.iloc[-lag:].copy()
    out = {t: [] for t in TARGETS}
    base_cols = list({c.split("_lag")[0] for c in X_cols if "_lag" in c})
    for _ in range(horizon):
        row = {}
        for col in base_cols:
            if col in buffer.columns:
                for k in range(1, lag + 1):
                    lc = f"{col}_lag{k}"
                    if lc in X_cols:
                        row[lc] = buffer.iloc[-k][col]
        X_pred = pd.DataFrame([row]).reindex(columns=X_cols).fillna(0)
        new_row = buffer.iloc[-1].copy()
        for t in TARGETS:
            v = float(models[t].predict(X_pred)[0])
            out[t].append(v)
            new_row[t] = v
        buffer = pd.concat([buffer, new_row.to_frame().T]).iloc[-lag:]
    return pd.DataFrame(out)


def evaluate(models, X_cols, train_df, val_df, lag):
    max_origin = len(val_df) - FORECAST_HORIZON - 1
    origins = np.linspace(lag, max_origin, num=N_ORIGINS, dtype=int).tolist()
    full = pd.concat([train_df, val_df])
    offset = len(train_df)
    pred = {t: [] for t in TARGETS}
    true = {t: [] for t in TARGETS}
    for o in origins:
        fc = recursive_forecast(models, full.iloc[: offset + o], X_cols, lag)
        actual = val_df.iloc[o: o + FORECAST_HORIZON]
        if len(actual) < FORECAST_HORIZON:
            continue
        for t in TARGETS:
            pred[t].extend(fc[t].tolist())
            true[t].extend(actual[t].tolist())
    p = np.concatenate([np.asarray(pred[t]) for t in TARGETS])
    y = np.concatenate([np.asarray(true[t]) for t in TARGETS])
    return mean_absolute_error(y, p), mean_squared_error(y, p)


def lags():
    raw = os.environ.get("E27_LAGS")
    return [int(x) for x in raw.split(",")] if raw else [12, 24, 48, 96]


def main():
    results_dir = ensure_results_dir()
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    n = WINDOW_DAYS * STEPS_PER_DAY
    n_win = int(os.environ.get("E27_WINDOWS", "4"))
    test_df = df.iloc[-n:]
    print(f"[INFO] LightGBM config: {ml_engine.LGB_CONFIG}; {n_win} validation windows of {WINDOW_DAYS} days")
    print(f"[INFO] the test week {test_df.index[0]} to {test_df.index[-1]} is not used here")

    rows = []
    for w in range(n_win):
        hi = len(df) - n * (w + 1)
        val_df = df.iloc[hi: hi + n]
        train_df = df.iloc[:hi]
        print(f"\n[window {w}] validation {val_df.index[0]} to {val_df.index[-1]}; "
              f"training {len(train_df)} records")
        for lag in lags():
            X, y = build_supervised(train_df, lag)
            t0 = time.time()
            models = ml_engine.train_models(X, y)
            fit_s = time.time() - t0
            mae, mse = evaluate(models, list(X.columns), train_df, val_df, lag)
            rows.append({"window": w, "lag_steps": lag, "hours": lag / 2,
                         "features": len(TARGETS) * lag, "fit_s": round(fit_s, 2),
                         "val_MAE": round(float(mae), 5), "val_MSE": round(float(mse), 5)})
            print(f"  w={lag:>3} ({lag / 2:>4.0f} h)  fit={fit_s:6.2f}s  "
                  f"MAE={mae:.5f}  MSE={mse:.5f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(results_dir, "e27_lag_window.csv"), index=False)
    agg = out.groupby(["lag_steps", "hours", "features"]).agg(
        windows=("window", "nunique"), fit_s=("fit_s", "median"),
        val_MAE=("val_MAE", "mean"), val_MSE=("val_MSE", "mean"),
        best_in=("val_MAE", "size")).reset_index()
    best_per_window = out.loc[out.groupby("window").val_MAE.idxmin()]
    wins = best_per_window.lag_steps.value_counts().to_dict()
    agg["best_in"] = agg.lag_steps.map(lambda k: wins.get(k, 0))
    agg = agg.round(5)
    agg.to_csv(os.path.join(results_dir, "e27_lag_window_mean.csv"), index=False)
    print("\n=== mean over validation windows ===")
    print(agg.to_string(index=False))
    b = agg.loc[agg.val_MAE.idxmin()]
    print(f"\n[OK] lowest mean validation MAE at w = {int(b['lag_steps'])} ({b['hours']:.0f} h), "
          f"best in {int(b['best_in'])} of {n_win} windows; the deployed setting is w = 48 (24 h)")


if __name__ == "__main__":
    main()
