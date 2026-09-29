"""E29 - Inference latency of the seven forecasting models (R1-4).

R1-4 asks for inference latency alongside the errors and the training times. Table 4
reports the errors and Figure 5 the training times; this script supplies the missing
quantity: the wall-clock time of one six-hour forecast of all six channels, produced the
way each model produces it in the comparison.

  * tree models (LightGBM, XGBoost, Random Forest) and ARIMA forecast recursively, twelve
    steps, one model per channel;
  * the deep models (LSTM, dilated convolutional network, Transformer) emit the twelve
    steps in one forward pass over all six channels.

Every model is fitted once on the training period of the final models, with the selected
hyperparameters of Table 2, and the forecast is then timed over N_REPEATS launches from
distinct origins in the test week. The channels are the six measured ones; motor current
is not part of the study.

Output: experiments/results/e29_inference_latency.csv
Usage:  python experiments/e29_inference_latency.py
  env:  E29_REPEATS  timed forecasts per model (default 20)
"""

import ast
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, ensure_results_dir  # noqa: E402
from ml_engine import TARGETS, FORECAST_HORIZON, LAG_STEPS  # noqa: E402
from e01b_forecasting_modelcomp import (  # noqa: E402
    build_supervised, make_lgb, make_rf, make_xgb, recursive_forecast_one_origin)

STEPS_PER_DAY = 48
TEST_DAYS = 7
N_REPEATS = int(os.environ.get("E29_REPEATS", "20"))
SEED = 42


def selected_params():
    """Hyperparameters chosen by the equal-budget search, as stored by e01b and e02."""
    out = {}
    for path, key in [("e01b_bestparams.csv", "tree"), ("e02_bestparams.csv", "dl")]:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", path)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            name, _, rest = line.partition(",")
            name = name.strip()
            try:
                out[name] = ast.literal_eval(rest.strip())
            except Exception:
                pass
    return out


def time_trees(name, maker, params, train_df, test_df):
    X, y = build_supervised(train_df)
    models = {t: maker(params).fit(X, y[t]) for t in TARGETS}
    cols = list(X.columns)
    full = pd.concat([train_df, test_df])
    offset = len(train_df)
    origins = np.linspace(LAG_STEPS, len(test_df) - FORECAST_HORIZON - 1,
                          num=N_REPEATS, dtype=int)
    times = []
    for o in origins:
        hist = full.iloc[: offset + int(o)]
        t0 = time.perf_counter()
        recursive_forecast_one_origin(models, hist, cols)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def time_arima(train_df, test_df):
    from statsmodels.tsa.arima.model import ARIMA
    orders, fitted = {}, {}
    for t in TARGETS:
        best = (np.inf, (1, 0, 1))
        for p in range(3):
            for dd in range(2):
                for q in range(3):
                    try:
                        r = ARIMA(train_df[t].to_numpy(), order=(p, dd, q)).fit()
                        if r.aic < best[0]:
                            best = (r.aic, (p, dd, q))
                    except Exception:
                        pass
        orders[t] = best[1]
    full = pd.concat([train_df, test_df])
    offset = len(train_df)
    origins = np.linspace(LAG_STEPS, len(test_df) - FORECAST_HORIZON - 1,
                          num=N_REPEATS, dtype=int)
    for t in TARGETS:
        fitted[t] = ARIMA(full[t].to_numpy()[: offset], order=orders[t]).fit()
    times = []
    for o in origins:
        t0 = time.perf_counter()
        for t in TARGETS:
            fitted[t].apply(full[t].to_numpy()[: offset + int(o)]).forecast(FORECAST_HORIZON)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times, orders


def time_deep(name, params, train_df, test_df):
    import torch
    import e02_dl_baselines as dl
    values = train_df[TARGETS].to_numpy(np.float32)
    mu, sd = values.mean(0), values.std(0) + 1e-8
    X, Y = dl.make_sequences((values - mu) / sd)
    model = dl.build(name, params)
    dl.fit(model, X, Y, params.get("lr", 1e-3), dl.EPOCHS)
    full = np.concatenate([values, test_df[TARGETS].to_numpy(np.float32)])
    full = (full - mu) / sd
    offset = len(values)
    origins = np.linspace(dl.LOOKBACK, len(test_df) - FORECAST_HORIZON - 1,
                          num=N_REPEATS, dtype=int)
    model.eval()
    times = []
    for o in origins:
        window = full[offset + int(o) - dl.LOOKBACK: offset + int(o)][None, :, :]
        t0 = time.perf_counter()
        with torch.no_grad():
            model(torch.tensor(window))
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    results_dir = ensure_results_dir()
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    n = TEST_DAYS * STEPS_PER_DAY
    train_df, test_df = df.iloc[:-n], df.iloc[-n:]
    print(f"[INFO] channels: {TARGETS} ({len(TARGETS)})")
    print(f"[INFO] training {len(train_df)} records, test week {test_df.index[0]} to {test_df.index[-1]}")
    print(f"[INFO] one forecast = {FORECAST_HORIZON} steps (6 h) of all {len(TARGETS)} channels, "
          f"{N_REPEATS} launches per model")

    sel = selected_params()
    rows = []

    for name, maker in [("LightGBM", make_lgb), ("XGBoost", make_xgb), ("RandomForest", make_rf)]:
        params = sel.get(name, {})
        times = time_trees(name, maker, params, train_df, test_df)
        rows.append({"model": name, "mode": "recursive, one model per channel",
                     "median_ms": round(float(np.median(times)), 2),
                     "p05_ms": round(float(np.percentile(times, 5)), 2),
                     "p95_ms": round(float(np.percentile(times, 95)), 2)})
        print(f"  {name:14} median {rows[-1]['median_ms']:8.2f} ms")

    times, orders = time_arima(train_df, test_df)
    rows.append({"model": "ARIMA", "mode": "recursive, one model per channel",
                 "median_ms": round(float(np.median(times)), 2),
                 "p05_ms": round(float(np.percentile(times, 5)), 2),
                 "p95_ms": round(float(np.percentile(times, 95)), 2)})
    print(f"  {'ARIMA':14} median {rows[-1]['median_ms']:8.2f} ms   orders {orders}")

    for name in ["LSTM", "TCN", "Transformer"]:
        params = sel.get(name, {})
        try:
            times = time_deep(name, params, train_df, test_df)
        except Exception as exc:
            print(f"  {name:14} skipped: {exc}")
            continue
        rows.append({"model": name, "mode": "direct, all channels in one pass",
                     "median_ms": round(float(np.median(times)), 2),
                     "p05_ms": round(float(np.percentile(times, 5)), 2),
                     "p95_ms": round(float(np.percentile(times, 95)), 2)})
        print(f"  {name:14} median {rows[-1]['median_ms']:8.2f} ms")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(results_dir, "e29_inference_latency.csv"), index=False)
    print("\n=== inference latency, one six-hour forecast of six channels ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
