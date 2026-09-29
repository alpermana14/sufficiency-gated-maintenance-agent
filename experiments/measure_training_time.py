"""
measure_training_time.py
========================
One clean, self-contained benchmark that measures TRAINING TIME for every
forecasting model in the paper (ML + DL) on a SINGLE machine, so Table 5 is
internally consistent. Run it wherever you want the official numbers from
(e.g. on the OVH deployment server for deployment-representative times, or on
one workstation for a consistent comparison).

It uses the exact tuned hyperparameters found by the equal-budget Optuna search
(Table 2), so the measured times correspond to the configurations the paper
reports. It does NOT re-tune (tuning is a separate, one-off protocol).

WHAT IS TIMED
  For each model, the time to FIT the final tuned configuration on the training
  split (all data except the last 7 days). For the per-channel models (LightGBM,
  XGBoost, Random Forest, ARIMA) this is the total time to fit all seven
  channels. For the deep models (LSTM, TCN, Transformer) it is the time to train
  the single multivariate model for DL_EPOCHS epochs. Shared preprocessing
  (loading, lag features, standardisation) is done once and is NOT counted, so
  only model fitting is measured. ARIMA's per-channel (p,d,q) selection by AIC is
  counted, because ARIMA has no separate tuning budget.

USAGE (Jupyter)
  1. Put this file next to your notebook (or in experiments/).
  2. Make sure DATA_CSV points to your exported data (a CSV with a 'datetime'
     column plus the seven target columns). scripts/export_data.py produces it.
  3. In a cell:   %run measure_training_time.py
     or:          from measure_training_time import run; results = run()
  `run()` prints a table and returns a pandas DataFrame (model, train_seconds).

REQUIREMENTS
  pip install pandas numpy scikit-learn lightgbm xgboost statsmodels torch
  (torch CPU build is fine: pip install torch --index-url
   https://download.pytorch.org/whl/cpu)
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================ CONFIG ============================
DATA_CSV = os.environ.get("PM_DATA_CSV", "data/conveyor_export.csv")
TARGETS = ["temperature", "z_rms", "x_rms", "z_peak", "x_peak", "noise"]  # current removed
LAG = 48                 # lag window (24 h at 30-min sampling)
HORIZON = 12             # forecast horizon for the DL direct-multistep models
TEST_DAYS = 7            # last 7 days held out (training uses the rest)
STEPS_PER_DAY = 48
SEED = 42
DL_EPOCHS = 25
N_REPEATS = int(os.environ.get("TT_REPEATS", "3"))  # fast models (LightGBM, XGBoost) are fitted this many times
LIMIT_THREADS = None     # set e.g. 4 to emulate a lower-core server; None = all cores

# Tuned hyperparameters of the CAEE revision (six measured channels; Table 2;
# experiments/results/e01b_bestparams.csv and e02_bestparams.csv, 13 Sep 2026)
LGB_PARAMS = dict(num_leaves=209, learning_rate=0.028180680291847244, n_estimators=158,
                  min_child_samples=70, subsample=0.7760609974958406,
                  colsample_bytree=0.6488152939379115, reg_lambda=0.09565499215943825)
XGB_PARAMS = dict(max_depth=6, learning_rate=0.018227726925281677, n_estimators=266,
                  subsample=0.9512986748659008, colsample_bytree=0.7983631287693599,
                  min_child_weight=5, reg_lambda=0.008354749981873796)
RF_PARAMS = dict(n_estimators=191, max_depth=25, min_samples_leaf=2,
                 max_features=0.6599641068895281)
LSTM_PARAMS = dict(hidden=128, num_layers=2, dropout=0.28184968246925673, lr=0.0020978213384576817)
TCN_PARAMS = dict(ch=128, levels=5, kernel=2, dropout=0.20526990795364705, lr=0.00044684675025045853)
TRF_PARAMS = dict(dmodel=64, nhead=4, layers=2, dropout=0.062061007104302957, lr=0.0008848920830559989)
# ================================================================


def _set_threads():
    if LIMIT_THREADS:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(LIMIT_THREADS)


def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"Data file not found: {DATA_CSV}. Export it first "
            f"(scripts/export_data.py) or set PM_DATA_CSV / DATA_CSV.")
    df = pd.read_csv(DATA_CSV)
    if "datetime" not in df.columns:
        raise ValueError("CSV must contain a 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing target columns: {missing}")
    return df


def make_lag_features(df):
    """Absolute-value lag features (matches the paper's forecasting protocol)."""
    base = df[TARGETS].copy()
    cols = {f"{t}_lag{k}": base[t].shift(k) for t in TARGETS for k in range(1, LAG + 1)}
    full = pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1).dropna()
    X = full[[f"{t}_lag{k}" for t in TARGETS for k in range(1, LAG + 1)]]
    Y = full[TARGETS]
    return X, Y


def timeit(fn, *args, repeats=N_REPEATS, **kwargs):
    """Return the mean wall-clock seconds over `repeats` runs of fn."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# --------------------------- ML fit functions ---------------------------
def fit_lightgbm(X, Y):
    import lightgbm as lgb
    for t in TARGETS:
        lgb.LGBMRegressor(objective="regression", random_state=SEED,
                          n_jobs=-1, verbosity=-1, **LGB_PARAMS).fit(X, Y[t])


def fit_xgboost(X, Y):
    import xgboost as xgb
    for t in TARGETS:
        xgb.XGBRegressor(objective="reg:squarederror", random_state=SEED,
                         n_jobs=-1, verbosity=0, **XGB_PARAMS).fit(X, Y[t])


def fit_random_forest(X, Y):
    from sklearn.ensemble import RandomForestRegressor
    for t in TARGETS:
        RandomForestRegressor(random_state=SEED, n_jobs=-1, **RF_PARAMS).fit(X, Y[t])


def fit_arima(train_df):
    """Per channel: select (p,d,q) by AIC on the last 2000 points, then fit on
    the full series. AIC selection is counted as part of ARIMA training."""
    from statsmodels.tsa.arima.model import ARIMA
    for t in TARGETS:
        series = train_df[t].reset_index(drop=True)
        sub = series.iloc[-2000:]
        best_aic, best_order = np.inf, (1, 0, 1)
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        res = ARIMA(sub, order=(p, d, q)).fit()
                        if res.aic < best_aic:
                            best_aic, best_order = res.aic, (p, d, q)
                    except Exception:
                        continue
        ARIMA(series, order=best_order).fit()


# --------------------------- DL fit functions ---------------------------
def _build_dl():
    import torch
    import torch.nn as nn
    D = len(TARGETS)

    class LSTMF(nn.Module):
        def __init__(s):
            super().__init__()
            p = LSTM_PARAMS
            s.lstm = nn.LSTM(D, p["hidden"], p["num_layers"], batch_first=True,
                             dropout=p["dropout"] if p["num_layers"] > 1 else 0.0)
            s.head = nn.Sequential(nn.Linear(p["hidden"], 128), nn.ReLU(),
                                   nn.Dropout(p["dropout"]), nn.Linear(128, HORIZON * D))
        def forward(s, x):
            _, (h, _) = s.lstm(x)
            return s.head(h[-1]).view(-1, HORIZON, D)

    class TCNF(nn.Module):
        def __init__(s):
            super().__init__()
            p = TCN_PARAMS
            layers, in_ch = [], D
            for i in range(p["levels"]):
                dil = 2 ** i
                layers += [nn.Conv1d(in_ch, p["ch"], p["kernel"],
                                     padding=dil * (p["kernel"] - 1) // 2, dilation=dil),
                           nn.ReLU(), nn.Dropout(p["dropout"])]
                in_ch = p["ch"]
            s.tcn = nn.Sequential(*layers)
            s.head = nn.Linear(p["ch"], HORIZON * D)
        def forward(s, x):
            z = s.tcn(x.transpose(1, 2))
            return s.head(z[:, :, -1]).view(-1, HORIZON, D)

    class TRFF(nn.Module):
        def __init__(s):
            super().__init__()
            p = TRF_PARAMS
            s.inp = nn.Linear(D, p["dmodel"])
            s.pos = nn.Parameter(torch.zeros(1, LAG, p["dmodel"]))
            enc = nn.TransformerEncoderLayer(p["dmodel"], p["nhead"], dim_feedforward=128,
                                             batch_first=True, dropout=p["dropout"])
            s.encoder = nn.TransformerEncoder(enc, num_layers=p["layers"])
            s.head = nn.Linear(p["dmodel"], HORIZON * D)
        def forward(s, x):
            z = s.encoder(s.inp(x) + s.pos)
            return s.head(z[:, -1, :]).view(-1, HORIZON, D)

    return {"LSTM": (LSTMF, LSTM_PARAMS["lr"]),
            "TCN": (TCNF, TCN_PARAMS["lr"]),
            "Transformer": (TRFF, TRF_PARAMS["lr"])}


def make_sequences(values):
    Xs, Ys = [], []
    for i in range(LAG, len(values) - HORIZON + 1):
        Xs.append(values[i - LAG:i]); Ys.append(values[i:i + HORIZON])
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32)


def fit_dl(name, Xtr, Ytr, builders):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    model_cls, lr = builders[name]
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    n = len(Xt); model.train()
    for _ in range(DL_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), Yt[idx]).backward()
            opt.step()


# ------------------------------- driver -------------------------------
def run():
    _set_threads()
    df = load_data()
    train = df.iloc[:-TEST_DAYS * STEPS_PER_DAY]
    print(f"[info] data: {len(df)} rows | training on {len(train)} rows "
          f"(all but last {TEST_DAYS} days) | repeats/model: {N_REPEATS}")

    # shared preprocessing (not timed)
    X, Y = make_lag_features(train)

    results = []
    results.append(("LightGBM", timeit(fit_lightgbm, X, Y)))
    results.append(("XGBoost", timeit(fit_xgboost, X, Y)))
    results.append(("RandomForest", timeit(fit_random_forest, X, Y, repeats=1)))  # slow
    results.append(("ARIMA", timeit(fit_arima, train, repeats=1)))  # AIC search is slow

    # deep-learning models (skip cleanly if torch is unavailable)
    try:
        import torch  # noqa: F401
        builders = _build_dl()
        vals = train[TARGETS].to_numpy(np.float32)
        mu, sd = vals.mean(0), vals.std(0); sd[sd == 0] = 1.0
        Xtr, Ytr = make_sequences((vals - mu) / sd)
        for name in ["LSTM", "TCN", "Transformer"]:
            results.append((name, timeit(fit_dl, name, Xtr, Ytr, builders, repeats=1)))
    except ImportError:
        print("[warn] torch not installed; skipping LSTM/TCN/Transformer.")

    out = pd.DataFrame(results, columns=["model", "train_seconds"])
    out["train_seconds"] = out["train_seconds"].round(2)
    print("\n=== Training time (fit all channels, tuned configs) ===")
    print(out.to_string(index=False))

    import platform
    import psutil
    out_dir = os.environ.get("TT_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out.assign(train_rows=len(train), channels=len(TARGETS)).to_csv(
            os.path.join(out_dir, "timing_models.csv"), index=False)
        pd.Series({"cpu": platform.processor(), "logical_cores": psutil.cpu_count(logical=True),
                   "physical_cores": psutil.cpu_count(logical=False),
                   "ram_GB": round(psutil.virtual_memory().total / 1e9, 1),
                   "fast_model_repeats": N_REPEATS, "dl_epochs": DL_EPOCHS}).to_csv(
            os.path.join(out_dir, "timing_hardware.csv"))
        print(f"[OK] wrote timing_models.csv and timing_hardware.csv to {out_dir}")
    print("\nNote: absolute times are hardware-dependent. Run this on the machine "
          "whose numbers you want to report, and state that machine in the paper.")
    return out


if __name__ == "__main__":
    run()
