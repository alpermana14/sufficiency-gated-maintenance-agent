"""E1 (model_comparison.py protocol) - fair tuning + rigor on the PAPER's method.

The paper's forecasting results (Table 4/5) were produced by model_comparison.py,
whose protocol differs from the static one-step split used in e01_forecasting_eval.py:

  * Split: train on all but the last 7 days; test = last 7 days.
  * Evaluation: recursive multi-step forecast (horizon = FORECAST_HORIZON = 12
    steps = 6 h) launched from ~20 evenly spaced origins inside the test window;
    absolute-value lag features (predict the level, not the delta).

The short, recent 7-day test window has far less train->test drift than a
"last 20%" split, which is why LightGBM is competitive here (and collapsed in
the static-split e01). This script reproduces that exact protocol (functions
copied verbatim from model_comparison.py, marked below) and adds the rigor the
reviewers asked for:

  * equal-budget Optuna tuning for LightGBM / XGBoost / RandomForest (R4-8),
  * a naive PERSISTENCE baseline (recursive last-value) — the sanity check a
    forecasting reviewer expects,
  * pooled RMSE, MAE, sMAPE, R2 with bootstrap 95% CI (R4-6),
  * Diebold-Mariano tests, LightGBM vs each model (significance not raw rank),
  * per-model training time.

Outputs: experiments/results/e01b_metrics.csv, e01b_perhorizon.csv,
         e01b_dm_tests.csv, e01b_traintime.csv

Usage:  python experiments/e01b_forecasting_modelcomp.py
  env:  E01B_TRIALS (Optuna trials/model, default 15)
        E01B_ONLY / E01B_AGGREGATE  (per-model runs, like e01)
"""

import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
import optuna
import lightgbm as lgb
import xgboost as xgb

from common import load_data, ensure_results_dir
import ml_engine
from ml_engine import TARGETS, LAG_STEPS, FORECAST_HORIZON, make_lag_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(os.environ.get("E01B_TRIALS", "15"))
SEED = 42
TEST_DAYS = 7
VAL_DAYS = 7
STEPS_PER_DAY = 48
N_ORIGINS = 20


# ===== protocol copied verbatim from model_comparison.py (the paper's code) =====
def build_supervised(df):
    data = make_lag_features(df[TARGETS].copy(), LAG_STEPS)
    return data.drop(columns=TARGETS), data[TARGETS]


def recursive_forecast_one_origin(models_by_target, history_df, X_cols, horizon=FORECAST_HORIZON):
    buffer = history_df.iloc[-LAG_STEPS:].copy()
    forecast_dict = {tgt: [] for tgt in TARGETS}
    base_cols = list({c.split("_lag")[0] for c in X_cols if "_lag" in c})
    for _ in range(horizon):
        input_row = {}
        for col in base_cols:
            if col in buffer.columns:
                for lag in range(1, LAG_STEPS + 1):
                    lc = f"{col}_lag{lag}"
                    if lc in X_cols:
                        input_row[lc] = buffer.iloc[-lag][col]
        X_pred = pd.DataFrame([input_row]).reindex(columns=X_cols).fillna(0)
        new_row = buffer.iloc[-1].copy()
        for tgt in TARGETS:
            val = float(models_by_target[tgt].predict(X_pred)[0])
            forecast_dict[tgt].append(val); new_row[tgt] = val
        buffer = pd.concat([buffer, new_row.to_frame().T]).iloc[-LAG_STEPS:]
    return pd.DataFrame(forecast_dict)


def persistence_forecast_one_origin(history_df, horizon=FORECAST_HORIZON):
    """Naive baseline: repeat the last observed value for the whole horizon."""
    last = history_df[TARGETS].iloc[-1]
    return pd.DataFrame({t: [float(last[t])] * horizon for t in TARGETS})
# ===============================================================================


# ===== centred-delta formulation -> model "LightGBM-CD" (comparison arm only) =====
# Centred lags (deviation from lag 1), one-step change target, recency weights (half-life
# 30 days). This was the backend formulation before the CAEE revision; the backend now uses
# the absolute-lag formulation above. Kept here for the supplementary comparison (E14, E15).
RECENCY_HALF_LIFE_STEPS = 30 * STEPS_PER_DAY


def build_supervised_cd(df):
    data = make_lag_features(df[TARGETS].copy(), LAG_STEPS)
    y = pd.DataFrame({t: data[t] - data[f"{t}_lag1"] for t in TARGETS}, index=data.index)
    y_abs = data[TARGETS].copy()
    for c in TARGETS:
        ref = data[f"{c}_lag1"].copy()
        for k in range(1, LAG_STEPS + 1):
            data[f"{c}_lag{k}"] = data[f"{c}_lag{k}"] - ref
    X = data.drop(columns=TARGETS + [f"{c}_lag1" for c in TARGETS])
    return X, y, y_abs


def recency_weights(n):
    age = np.arange(n - 1, -1, -1)  # last row is the newest (age 0)
    return 0.5 ** (age / RECENCY_HALF_LIFE_STEPS)


def recursive_forecast_cd(models_by_target, history_df, X_cols, horizon=FORECAST_HORIZON):
    """Centred-delta recursion (as in ml_engine.generate_forecast before the CAEE revision)."""
    buffer = history_df.iloc[-LAG_STEPS:].copy()
    base_cols = list({c.split("_lag")[0] for c in X_cols if "_lag" in c})
    out = {t: [] for t in TARGETS}
    for _ in range(horizon):
        row = {}
        for col in base_cols:
            ref = float(buffer.iloc[-1][col])
            for lag in range(1, LAG_STEPS + 1):
                lc = f"{col}_lag{lag}"
                if lc in X_cols:
                    row[lc] = float(buffer.iloc[-lag][col]) - ref
        X_pred = pd.DataFrame([row]).reindex(columns=X_cols).fillna(0)
        new_row = buffer.iloc[-1].copy()
        for t in TARGETS:
            val = float(buffer.iloc[-1][t]) + float(models_by_target[t].predict(X_pred)[0])
            out[t].append(val); new_row[t] = val
        buffer = pd.concat([buffer, new_row.to_frame().T]).iloc[-LAG_STEPS:]
    return pd.DataFrame(out)
# ===============================================================================


def collect_predictions_arima(train_df, test_df, horizon=FORECAST_HORIZON):
    """ARIMA under the SAME recursive multi-origin protocol: per channel, pick
    (p,d,q) by AIC on the training set, then at each origin apply the fixed
    params to the history and forecast `horizon` steps (no refitting)."""
    from statsmodels.tsa.arima.model import ARIMA
    max_origin = len(test_df) - horizon - 1
    origins = np.linspace(LAG_STEPS, max_origin, num=N_ORIGINS, dtype=int).tolist()
    pred = {t: [] for t in TARGETS}; true = {t: [] for t in TARGETS}
    horizons = []; orders = {}
    fitted = {}
    for t in TARGETS:
        series = train_df[t].reset_index(drop=True)
        best_aic, best_order, best_res = np.inf, (1, 0, 1), None
        sub = series.iloc[-2000:]
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        res = ARIMA(sub, order=(p, d, q)).fit()
                        if res.aic < best_aic:
                            best_aic, best_order, best_res = res.aic, (p, d, q), res
                    except Exception:
                        continue
        orders[t] = best_order
        fitted[t] = ARIMA(series, order=best_order).fit()
        print(f"   {t:12s} ARIMA order={best_order}")

    for origin in origins:
        hist = pd.concat([train_df, test_df.iloc[:origin]])
        fut = test_df[TARGETS].iloc[origin:origin + horizon].reset_index(drop=True)
        for t in TARGETS:
            hser = hist[t].reset_index(drop=True)
            try:
                applied = fitted[t].apply(hser)
                fc = np.asarray(applied.forecast(horizon))
            except Exception:
                fc = np.repeat(hser.iloc[-1], horizon)  # safe fallback
            for h in range(horizon):
                pred[t].append(float(fc[h])); true[t].append(fut.loc[h, t])
        horizons.extend(range(1, horizon + 1))
    return ({t: np.array(pred[t]) for t in TARGETS},
            {t: np.array(true[t]) for t in TARGETS}, np.array(horizons), orders)


def collect_predictions(models_or_none, train_df, test_df, horizon=FORECAST_HORIZON,
                        forecaster=recursive_forecast_one_origin, X_cols=None):
    """Recursive forecast at ~20 origins; return pooled preds & truth per target.
    models_or_none=None => persistence baseline."""
    if X_cols is None:
        X_cols = list(build_supervised(train_df)[0].columns)
    max_origin = len(test_df) - horizon - 1
    origins = np.linspace(LAG_STEPS, max_origin, num=N_ORIGINS, dtype=int).tolist()
    pred = {t: [] for t in TARGETS}; true = {t: [] for t in TARGETS}
    horizons = []
    for origin in origins:
        hist = pd.concat([train_df, test_df.iloc[:origin]])
        if len(hist) < LAG_STEPS:
            continue
        if models_or_none is None:
            fc = persistence_forecast_one_origin(hist[TARGETS], horizon)
        else:
            fc = forecaster(models_or_none, hist[TARGETS], X_cols, horizon)
        fut = test_df[TARGETS].iloc[origin:origin + horizon].reset_index(drop=True)
        for h in range(horizon):
            for t in TARGETS:
                pred[t].append(fc.loc[h, t]); true[t].append(fut.loc[h, t])
            horizons.append(h + 1)
    return {t: np.array(pred[t]) for t in TARGETS}, {t: np.array(true[t]) for t in TARGETS}, np.array(horizons)


# ----------------------------- models & tuning ------------------------------
def make_lgb(p): return lgb.LGBMRegressor(objective="regression", random_state=SEED, n_jobs=-1, verbosity=-1, **p)
def make_xgb(p): return xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=-1, verbosity=0, **p)
def make_rf(p):  return RandomForestRegressor(random_state=SEED, n_jobs=-1, **p)
BUILDERS = {"LightGBM": make_lgb, "XGBoost": make_xgb, "RandomForest": make_rf}


def suggest(trial, model):
    if model == "LightGBM":
        return dict(num_leaves=trial.suggest_int("num_leaves", 15, 255),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 700),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if model == "XGBoost":
        return dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 700),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_float("max_features", 0.3, 1.0))


def tune(model, X_tr, y_tr, X_val, y_val):
    """Equal-budget tuning on ONE-STEP validation RMSE (fast, fair proxy);
    tuned models are then evaluated with the recursive multi-step protocol."""
    std = y_tr.std().replace(0, np.nan)
    def obj(trial):
        p = suggest(trial, model); errs = []
        for t in TARGETS:
            m = BUILDERS[model](p); m.fit(X_tr, y_tr[t])
            errs.append(np.sqrt(mean_squared_error(y_val[t], m.predict(X_val))) / std[t])
        return float(np.mean(errs))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def tune_cd(X_tr, y_tr, w_tr, X_val, y_val, abs_std):
    """Same budget, search space and objective as tune("LightGBM"): the one-step
    change error equals the one-step absolute error, normalised by the absolute std."""
    def obj(trial):
        p = suggest(trial, "LightGBM"); errs = []
        for t in TARGETS:
            m = make_lgb(p); m.fit(X_tr, y_tr[t], sample_weight=w_tr)
            errs.append(np.sqrt(mean_squared_error(y_val[t], m.predict(X_val))) / abs_std[t])
        return float(np.mean(errs))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


# ----------------------------- metrics --------------------------------------
def smape(y, p, eps=1e-8): return float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + eps)))
def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

def rmse_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); e2 = (np.asarray(y) - np.asarray(p)) ** 2; m = len(e2)
    b = [np.sqrt(np.mean(e2[rng.integers(0, m, m)])) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def dm_test(y, p1, p2, h=1):
    y, p1, p2 = map(np.asarray, (y, p1, p2)); d = (y - p1) ** 2 - (y - p2) ** 2; n = len(d)
    dbar = d.mean(); var = np.mean((d - dbar) ** 2) / n
    if var <= 0: return float("nan"), float("nan")
    dm = dbar / np.sqrt(var); adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * adj; return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def splits(df):
    test = df.iloc[-TEST_DAYS * STEPS_PER_DAY:]
    val = df.iloc[-(TEST_DAYS + VAL_DAYS) * STEPS_PER_DAY:-TEST_DAYS * STEPS_PER_DAY]
    train_for_final = df.iloc[:-TEST_DAYS * STEPS_PER_DAY]        # matches time_split(test_days=7)
    train_for_tune = df.iloc[:-(TEST_DAYS + VAL_DAYS) * STEPS_PER_DAY]
    return train_for_tune, val, train_for_final, test


def run_model(model, results_dir):
    df = load_data(); df = df[~df.index.duplicated(keep="first")].sort_index()
    tr_tune, val, tr_final, test = splits(df)
    if model == "Persistence":
        t0 = time.time(); pred, true, hz = collect_predictions(None, tr_final, test); ft = time.time() - t0
    elif model == "ARIMA":
        t0 = time.time(); pred, true, hz, orders = collect_predictions_arima(tr_final, test); ft = time.time() - t0
        print(f"[arima] orders: {orders}")
    elif model == "LightGBM-CD":
        Xt, yt, yt_abs = build_supervised_cd(tr_tune)
        Xv_full, yv_full, _ = build_supervised_cd(df.iloc[:-TEST_DAYS * STEPS_PER_DAY])
        val_idx = Xv_full.index.intersection(val.index)
        Xv, yv = Xv_full.loc[val_idx], yv_full.loc[val_idx]
        print(f"[INFO] {model}: tune-train={len(Xt)} val={len(Xv)} features={Xt.shape[1]} | {N_TRIALS} trials")
        t0 = time.time()
        params = tune_cd(Xt, yt, recency_weights(len(Xt)), Xv, yv, yt_abs.std().replace(0, np.nan))
        Xf, yf, _ = build_supervised_cd(tr_final); wf = recency_weights(len(Xf))
        t_fit = time.time()
        models = {t: make_lgb(params).fit(Xf, yf[t], sample_weight=wf) for t in TARGETS}
        ft = time.time() - t_fit
        print(f"[tune+fit] {model} in {time.time()-t0:.1f}s (fit {ft:.1f}s) best={params}")
        pred, true, hz = collect_predictions(models, tr_final, test,
                                             forecaster=recursive_forecast_cd, X_cols=list(Xf.columns))
        with open(f"{results_dir}/e01b_bestparams.csv", "a") as f:
            f.write(f"{model},{params}\n")
    else:
        Xt, yt = build_supervised(tr_tune)
        # one-step validation features from the val window (need lag context)
        Xv_full, yv_full = build_supervised(df.iloc[:-TEST_DAYS * STEPS_PER_DAY])
        val_idx = Xv_full.index.intersection(val.index)
        Xv, yv = Xv_full.loc[val_idx], yv_full.loc[val_idx]
        print(f"[INFO] {model}: tune-train={len(Xt)} val={len(Xv)} | {N_TRIALS} trials")
        t0 = time.time(); params = tune(model, Xt, yt, Xv, yv)
        Xf, yf = build_supervised(tr_final)
        t_fit = time.time()
        models = {t: BUILDERS[model](params).fit(Xf, yf[t]) for t in TARGETS}
        ft = time.time() - t_fit
        print(f"[tune+fit] {model} in {time.time()-t0:.1f}s (fit {ft:.1f}s) best={params}")
        pred, true, hz = collect_predictions(models, tr_final, test)
        with open(f"{results_dir}/e01b_bestparams.csv", "a") as f:
            f.write(f"{model},{params}\n")

    pd.DataFrame({**{f"pred_{t}": pred[t] for t in TARGETS},
                  **{f"true_{t}": true[t] for t in TARGETS}, "h": hz}
                 ).to_csv(f"{results_dir}/e01b_pred_{model}.csv", index=False)
    with open(f"{results_dir}/e01b_traintime.csv", "a") as f:
        f.write(f"{model},{ft:.3f}\n")
    print(f"[OK] saved e01b_pred_{model}.csv")


def aggregate(results_dir):
    models = ["LightGBM-CD", "LightGBM", "RandomForest", "XGBoost", "LSTM", "TCN", "Transformer",
              "ARIMA", "Persistence"]
    have = {}
    for m in models:
        p = f"{results_dir}/e01b_pred_{m}.csv"
        if os.path.exists(p): have[m] = pd.read_csv(p)
    order = [m for m in models if m in have]

    rows, ph_rows = [], []
    for m in order:
        d = have[m]
        for t in TARGETS:
            y, p = d[f"true_{t}"].to_numpy(), d[f"pred_{t}"].to_numpy()
            lo, hi = rmse_ci(y, p)
            rows.append({"model": m, "target": t, "MSE": mean_squared_error(y, p),
                         "MAE": mean_absolute_error(y, p), "RMSE": rmse(y, p),
                         "sMAPE": smape(y, p), "R2": r2_score(y, p),
                         "RMSE_CI_low": lo, "RMSE_CI_high": hi})
        for h in sorted(d["h"].unique()):
            sub = d[d["h"] == h]
            yall = np.concatenate([sub[f"true_{t}"] for t in TARGETS])
            pall = np.concatenate([sub[f"pred_{t}"] for t in TARGETS])
            ph_rows.append({"model": m, "h": int(h), "RMSE": rmse(yall, pall)})
    metrics = pd.DataFrame(rows); metrics.to_csv(f"{results_dir}/e01b_metrics.csv", index=False)
    pd.DataFrame(ph_rows).to_csv(f"{results_dir}/e01b_perhorizon.csv", index=False)

    dm_rows = []
    if "LightGBM" in have:
        for other in [m for m in ["LightGBM-CD", "RandomForest", "XGBoost", "LSTM", "TCN",
                                   "Transformer", "ARIMA", "Persistence"] if m in have]:
            for t in TARGETS:
                y = have["LightGBM"][f"true_{t}"].to_numpy()
                stat, pval = dm_test(y, have["LightGBM"][f"pred_{t}"].to_numpy(),
                                     have[other][f"pred_{t}"].to_numpy())
                dm_rows.append({"target": t, "lightgbm_vs": other, "dm_stat": stat,
                                "p_value": pval, "sig_0.05": bool(pval < 0.05) if pval == pval else False})
    pd.DataFrame(dm_rows).to_csv(f"{results_dir}/e01b_dm_tests.csv", index=False)

    print("\n=== RMSE by target (recursive 6h, tuned) ===")
    print(metrics.pivot(index="target", columns="model", values="RMSE")[order].round(4).to_string())
    print("\n=== R2 by target ===")
    print(metrics.pivot(index="target", columns="model", values="R2")[order].round(3).to_string())
    print("\n=== mean metrics across targets ===")
    print(metrics.groupby("model")[["MSE","MAE","RMSE","sMAPE","R2"]].mean().reindex(order).round(4).to_string())
    if dm_rows:
        print("\n=== Diebold-Mariano: LightGBM vs others (neg => LightGBM better) ===")
        print(pd.DataFrame(dm_rows).round(3).to_string(index=False))
    print(f"\n[OK] wrote e01b_metrics.csv, e01b_perhorizon.csv, e01b_dm_tests.csv")


def main():
    rd = ensure_results_dir()
    only = os.environ.get("E01B_ONLY", "")
    if os.environ.get("E01B_AGGREGATE", "") in {"1", "true", "yes"}:
        aggregate(rd)
    elif only:
        run_model(only, rd)
    else:
        print("Set E01B_ONLY=<LightGBM|LightGBM-CD|XGBoost|RandomForest|ARIMA|Persistence> or E01B_AGGREGATE=1")


if __name__ == "__main__":
    main()
