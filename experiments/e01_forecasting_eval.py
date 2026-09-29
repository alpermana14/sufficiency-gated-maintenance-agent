"""E1 - Fair forecasting comparison with tuning, extended metrics, significance.

Answers R4-6 (only MSE/MAE reported -> add RMSE, sMAPE, R2, confidence
intervals) and R4-8 (default hyperparameters may bias the comparison -> tune
every model under an identical budget).

Protocol
  * One-step-ahead forecasting from 48-step lag features (matches the paper's
    Table 4 structure), one model per target channel.
  * Chronological split: 70% train / 10% validation / 20% test.
  * FAIR TUNING: LightGBM, XGBoost and Random Forest each get the SAME Optuna
    budget (N_TRIALS), the same validation split and the same objective
    (mean-over-targets validation RMSE normalised by target std). One tuned
    configuration per model, then trained per target. ARIMA is a univariate
    statistical reference; its (p,d,q) order is chosen by AIC (its conventional
    "tuning"), evaluated with efficient rolling one-step prediction.
  * Metrics on the test set, per target and per model: MSE, MAE, RMSE, sMAPE,
    R2, plus a bootstrap 95% CI for RMSE.
  * Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction,
    h=1) comparing LightGBM against each other model per target, so "LightGBM
    is best" is backed by significance rather than raw ranking.

Outputs (experiments/results/):
  e01_metrics.csv     per target x model: MSE, MAE, RMSE, sMAPE, R2, RMSE CI
  e01_traintime.csv   per model total training time (tuned config, all targets)
  e01_dm_tests.csv    LightGBM vs {RF, XGBoost, ARIMA} per target: DM stat, p

Usage:  python experiments/e01_forecasting_eval.py
  env:  E01_TRIALS (Optuna trials per model, default 25)
"""

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

from common import TARGETS, ensure_results_dir, load_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

LAG = 48
N_TRIALS = int(os.environ.get("E01_TRIALS", "20"))
TUNE_ROWS = int(os.environ.get("E01_TUNE_ROWS", "4000"))  # cap train rows during tuning
SEED = 42
BOOT = 1000
# Per-model execution (background jobs are killed at the tool boundary, so each
# model is tuned/predicted in its own foreground call, then aggregated):
#   E01_ONLY=<LightGBM|XGBoost|RandomForest|ARIMA>  -> run one model, save preds
#   E01_AGGREGATE=1                                  -> combine saved preds, metrics, DM
ONLY = os.environ.get("E01_ONLY", "")
AGGREGATE = os.environ.get("E01_AGGREGATE", "") in {"1", "true", "yes"}


# ----------------------------- features & split -----------------------------
def build_features(df: pd.DataFrame):
    base = df[TARGETS].copy()
    cols = {f"{t}_lag{k}": base[t].shift(k) for t in TARGETS for k in range(1, LAG + 1)}
    lagged = pd.DataFrame(cols, index=base.index)
    full = pd.concat([base, lagged], axis=1).dropna()
    X = full[[f"{t}_lag{k}" for t in TARGETS for k in range(1, LAG + 1)]]
    Y = full[TARGETS]
    return X, Y


def chrono_split(X, Y):
    n = len(X)
    i_tr, i_val = int(n * 0.70), int(n * 0.80)
    return (X.iloc[:i_tr], Y.iloc[:i_tr],
            X.iloc[i_tr:i_val], Y.iloc[i_tr:i_val],
            X.iloc[i_val:], Y.iloc[i_val:])


# ----------------------------- metrics --------------------------------------
def smape(y, p, eps=1e-8):
    return float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + eps)))


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def rmse_ci(y, p, n_boot=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    err2 = (np.asarray(y) - np.asarray(p)) ** 2
    n = len(err2)
    boots = [np.sqrt(np.mean(err2[rng.integers(0, n, n)])) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def dm_test(y, p1, p2, h=1):
    """Diebold-Mariano (squared-error loss, HLN small-sample correction)."""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = (y - p1) ** 2 - (y - p2) ** 2
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0 / n  # h=1: no higher-lag autocovariances
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * adj
    pval = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(pval)


# ----------------------------- model builders -------------------------------
def make_lgb(params):
    return lgb.LGBMRegressor(objective="regression", random_state=SEED,
                             n_jobs=-1, verbosity=-1, **params)


def make_xgb(params):
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED,
                            n_jobs=-1, verbosity=0, **params)


def make_rf(params):
    return RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)


def suggest(trial, model):
    if model == "LightGBM":
        return dict(num_leaves=trial.suggest_int("num_leaves", 15, 255),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 800),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if model == "XGBoost":
        return dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 100, 800),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    # Random Forest — bounded n_estimators to keep the equal budget tractable
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_float("max_features", 0.3, 1.0))


BUILDERS = {"LightGBM": make_lgb, "XGBoost": make_xgb, "RandomForest": make_rf}


def tune(model, X_tr, Y_tr, X_val, Y_val):
    std = Y_tr.std().replace(0, np.nan)

    def objective(trial):
        params = suggest(trial, model)
        errs = []
        for t in TARGETS:
            m = BUILDERS[model](params)
            m.fit(X_tr, Y_tr[t])
            errs.append(rmse(Y_val[t], m.predict(X_val)) / std[t])
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


# ----------------------------- ARIMA ----------------------------------------
def arima_predict(train_series, test_series):
    """Efficient rolling one-step ARIMA: pick order by AIC on train, then
    filter test with fixed params (no re-estimation)."""
    from statsmodels.tsa.arima.model import ARIMA
    # Continuous integer index so the test index extends the train index
    # (required by statsmodels .append()).
    n_tr = len(train_series)
    train_series = train_series.reset_index(drop=True)
    test_series = test_series.reset_index(drop=True)
    test_series.index = range(n_tr, n_tr + len(test_series))
    best_aic, best_order, best_res = np.inf, (1, 1, 1), None
    sub = train_series.iloc[-2000:]  # bound AIC search cost
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    res = ARIMA(sub, order=(p, d, q)).fit()
                    if res.aic < best_aic:
                        best_aic, best_order, best_res = res.aic, (p, d, q), res
                except Exception:
                    continue
    full = ARIMA(train_series, order=best_order).fit()
    appended = full.append(test_series, refit=False)
    pred = appended.get_prediction(start=len(train_series),
                                   end=len(train_series) + len(test_series) - 1,
                                   dynamic=False).predicted_mean
    return np.asarray(pred), best_order


# ----------------------------- per-model run --------------------------------
def run_one_model(model, results_dir):
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    X, Y = build_features(df)
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = chrono_split(X, Y)

    if model == "ARIMA":
        print(f"[arima] AIC order selection, test={len(X_te)}")
        p_model = {}; t0 = time.time()
        for t in TARGETS:
            series_tr = pd.concat([Y_tr, Y_val])[t]
            p_model[t], order = arima_predict(series_tr, Y_te[t])
            print(f"   {t:12s} order={order}")
        fit_time = time.time() - t0
    else:
        X_tune, Y_tune = X_tr.iloc[-TUNE_ROWS:], Y_tr.iloc[-TUNE_ROWS:]
        print(f"[INFO] {model}: features={X.shape[1]} test={len(X_te)} | "
              f"tuning on {len(X_tune)} rows, {N_TRIALS} trials")
        t0 = time.time()
        params = tune(model, X_tune, Y_tune, X_val, Y_val)
        Xtrv, Ytrv = pd.concat([X_tr, X_val]), pd.concat([Y_tr, Y_val])
        p_model = {}; t_fit = time.time()
        for t in TARGETS:
            m = BUILDERS[model](params)
            m.fit(Xtrv, Ytrv[t])
            p_model[t] = m.predict(X_te)
        fit_time = time.time() - t_fit
        print(f"[tune+fit] {model} tuned+fit in {time.time()-t0:.1f}s "
              f"(fit {fit_time:.1f}s)\n  best={params}")

    pd.DataFrame(p_model, index=X_te.index).to_csv(f"{results_dir}/e01_pred_{model}.csv")
    with open(f"{results_dir}/e01_traintime.csv", "a") as f:
        f.write(f"{model},{fit_time:.3f}\n")
    print(f"[OK] saved e01_pred_{model}.csv (fit {fit_time:.1f}s)")


# ----------------------------- aggregate ------------------------------------
def aggregate(results_dir):
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    X, Y = build_features(df)
    *_, X_te, Y_te = chrono_split(X, Y)

    models = ["LightGBM", "XGBoost", "RandomForest", "ARIMA"]
    preds = {}
    for m in models:
        path = f"{results_dir}/e01_pred_{m}.csv"
        if not os.path.exists(path):
            print(f"[WARN] missing {path}; skipping {m}")
            continue
        preds[m] = pd.read_csv(path, index_col=0)
    have = list(preds)

    rows = []
    for m in have:
        for t in TARGETS:
            y, p = Y_te[t].to_numpy(), preds[m][t].to_numpy()
            lo, hi = rmse_ci(y, p)
            rows.append({"model": m, "target": t,
                         "MSE": mean_squared_error(y, p), "MAE": mean_absolute_error(y, p),
                         "RMSE": rmse(y, p), "sMAPE": smape(y, p), "R2": r2_score(y, p),
                         "RMSE_CI_low": lo, "RMSE_CI_high": hi})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(f"{results_dir}/e01_metrics.csv", index=False)

    dm_rows = []
    if "LightGBM" in have:
        for other in [m for m in ["RandomForest", "XGBoost", "ARIMA"] if m in have]:
            for t in TARGETS:
                y = Y_te[t].to_numpy()
                stat, pval = dm_test(y, preds["LightGBM"][t].to_numpy(), preds[other][t].to_numpy())
                dm_rows.append({"target": t, "lightgbm_vs": other, "dm_stat": stat,
                                "p_value": pval, "sig_0.05": bool(pval < 0.05) if pval == pval else False})
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(f"{results_dir}/e01_dm_tests.csv", index=False)

    order = [m for m in models if m in have]
    print("\n=== RMSE by target (tuned models) ===")
    print(metrics.pivot(index="target", columns="model", values="RMSE")[order].round(4).to_string())
    print("\n=== R2 by target ===")
    print(metrics.pivot(index="target", columns="model", values="R2")[order].round(3).to_string())
    print("\n=== sMAPE (%) by target ===")
    print(metrics.pivot(index="target", columns="model", values="sMAPE")[order].round(2).to_string())
    print("\n=== mean metrics across targets ===")
    print(metrics.groupby("model")[["MSE","MAE","RMSE","sMAPE","R2"]].mean().reindex(order).round(4).to_string())
    if not dm.empty:
        print("\n=== Diebold-Mariano: LightGBM vs others (neg dm_stat => LightGBM better; sig at p<0.05) ===")
        print(dm.round(3).to_string(index=False))
    print(f"\n[OK] wrote e01_metrics.csv, e01_dm_tests.csv to {results_dir}")


def main():
    results_dir = ensure_results_dir()
    if AGGREGATE:
        aggregate(results_dir)
    elif ONLY:
        run_one_model(ONLY, results_dir)
    else:
        print("Set E01_ONLY=<model> to run one model, or E01_AGGREGATE=1 to combine.")


if __name__ == "__main__":
    main()
