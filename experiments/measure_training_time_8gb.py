"""
measure_training_time_8gb.py
============================
Training time AND peak memory for every forecasting model in the paper, on the
six measured channels, meant to be run on a machine that approximates the
deployment server (8 GB RAM, few cores) rather than on a development laptop.

It is a standalone file. It imports nothing from this project, so it can be
copied to the other machine together with one CSV and run there.

WHY THIS FILE EXISTS
  The times in the paper must describe the machine the system is deployed on.
  A laptop with many cores and 16 GB finishes the same fit far faster than an
  8 GB cloud instance, so the number to report is the one measured on hardware
  of the deployed kind. This script also records PEAK MEMORY, because on an 8 GB
  box the question is not only how long a retraining cycle takes but whether it
  fits at all.

CHANNELS
  Six channels are forecast: temperature, z_rms, x_rms, z_peak, x_peak, noise.
  Motor current is NOT among them. The exported CSV still contains a `current`
  column and it is ignored here, which is deliberate: the Rogowski-coil sensor
  cannot resolve the current drawn by the 450 W motor, so that channel is out of
  the study. Do not add it back to make the CSV and the code agree.

WHAT IS TIMED
  For each model, the time to FIT the final tuned configuration (Table 2) on the
  training split, which is all data except the last 7 days. For the per-channel
  models (LightGBM, XGBoost, Random Forest, ARIMA) that is the total over all six
  channels. For the deep models (LSTM, dilated convolutional network, Transformer)
  it is the time to train the one multivariate model for DL_EPOCHS epochs.

  Shared preprocessing is done once and is NOT counted in any model's time, so
  only fitting is measured. Its cost is reported separately, because the feature
  matrix is the largest single object on a small machine. ARIMA's per-channel
  (p,d,q) selection by AIC IS counted, because ARIMA has no separate tuning
  budget and that search is part of what it costs to train.

  No tuning happens here. The hyperparameters below are the ones the equal-budget
  Optuna search selected, copied from e01b_bestparams.csv and e02_bestparams.csv.

USAGE
    python measure_training_time_8gb.py

  Environment variables, all optional:
    PM_DATA_CSV     path to the exported CSV       (default data/conveyor_export.csv)
    TT_OUT_DIR      where to write the two CSVs    (default alongside this file)
    TT_REPEATS      repeats for the fast models    (default 3)
    TT_THREADS      cap worker threads, e.g. 2     (default: all cores)
    TT_BUDGET_GB    memory budget to report against (default 8)
    TT_SKIP         comma-separated models to skip, e.g. RandomForest,Transformer

  To emulate a 2 vCPU cloud instance on a larger machine:
    TT_THREADS=2 python measure_training_time_8gb.py

REQUIREMENTS
  pip install pandas numpy scikit-learn lightgbm xgboost statsmodels psutil
  torch is optional; without it the three deep models are skipped and the rest
  still run. CPU build is enough:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import os
import platform
import threading
import time
import warnings

warnings.filterwarnings("ignore")

# Thread caps must be set before numpy and the model libraries are imported,
# otherwise the libraries have already read them and the setting has no effect.
_THREADS = os.environ.get("TT_THREADS")
if _THREADS:
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = str(_THREADS)

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import psutil               # noqa: E402

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.environ.get("PM_DATA_CSV",
                          os.path.join(os.path.dirname(HERE), "data", "conveyor_export.csv"))
OUT_DIR = os.environ.get("TT_OUT_DIR", HERE)

TARGETS = ["temperature", "z_rms", "x_rms", "z_peak", "x_peak", "noise"]  # six; no current
LAG = 48                 # lag window, 24 h at the 30-minute sampling interval
HORIZON = 12             # 6 h, for the direct multi-step deep models
TEST_DAYS = 7            # last 7 days held out; training uses the rest
STEPS_PER_DAY = 48
SEED = 42
DL_EPOCHS = 25           # same as the final fit of e02_dl_baselines.py
N_REPEATS = int(os.environ.get("TT_REPEATS", "3"))
BUDGET_GB = float(os.environ.get("TT_BUDGET_GB", "8"))
SKIP = {s.strip().lower() for s in os.environ.get("TT_SKIP", "").split(",") if s.strip()}

# Tuned hyperparameters, six measured channels (Table 2).
# Source: experiments/results/e01b_bestparams.csv and e02_bestparams.csv, 13 Sep 2026.
LGB_PARAMS = dict(num_leaves=209, learning_rate=0.028180680291847244, n_estimators=158,
                  min_child_samples=70, subsample=0.7760609974958406,
                  colsample_bytree=0.6488152939379115, reg_lambda=0.09565499215943825)
XGB_PARAMS = dict(max_depth=6, learning_rate=0.018227726925281677, n_estimators=266,
                  subsample=0.9512986748659008, colsample_bytree=0.7983631287693599,
                  min_child_weight=5, reg_lambda=0.008354749981873796)
RF_PARAMS = dict(n_estimators=191, max_depth=25, min_samples_leaf=2,
                 max_features=0.6599641068895281)
LSTM_PARAMS = dict(hidden=128, num_layers=2, dropout=0.28184968246925673,
                   lr=0.0020978213384576817)
TCN_PARAMS = dict(ch=128, levels=5, kernel=2, dropout=0.20526990795364705,
                  lr=0.00044684675025045853)
TRF_PARAMS = dict(dmodel=64, nhead=4, layers=2, dropout=0.062061007104302957,
                  lr=0.0008848920830559989)
# ================================================================


class PeakRSS:
    """Sample this process's resident memory in a background thread.

    Reports the peak reached inside the block and the rise over the value at
    entry. The rise is the interesting quantity: it is what fitting the model
    added on top of everything already loaded.
    """

    def __init__(self, interval=0.03):
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self.baseline = self.peak = self.proc.memory_info().rss

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.peak = self.proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self):
        return self.peak / 1e6

    @property
    def rise_mb(self):
        return (self.peak - self.baseline) / 1e6


def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"Data file not found: {DATA_CSV}\n"
            f"Copy the exported CSV next to this script, or set PM_DATA_CSV to its path.")
    df = pd.read_csv(DATA_CSV)
    if "datetime" not in df.columns:
        raise ValueError("The CSV must contain a 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing these channels: {missing}")
    return df


def make_lag_features(df):
    """Lags 1..LAG of every channel as inputs, the channel value as the target."""
    base = df[TARGETS].copy()
    cols = {f"{t}_lag{k}": base[t].shift(k) for t in TARGETS for k in range(1, LAG + 1)}
    full = pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1).dropna()
    X = full[[f"{t}_lag{k}" for t in TARGETS for k in range(1, LAG + 1)]]
    return X, full[TARGETS]


def measure(fn, *args, repeats=N_REPEATS, **kwargs):
    """Mean seconds over `repeats` runs, and the peak memory rise over them all."""
    times = []
    with PeakRSS() as mem:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)), mem.rise_mb, mem.peak_mb


# --------------------------- per-channel models ---------------------------
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
    """Select (p,d,q) per channel by AIC on the last 2000 points, then fit the
    full series. The search is part of the measured time."""
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


# ----------------------------- deep models -----------------------------
def build_deep_models():
    import torch
    import torch.nn as nn
    D = len(TARGETS)

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            p = LSTM_PARAMS
            self.lstm = nn.LSTM(D, p["hidden"], p["num_layers"], batch_first=True,
                                dropout=p["dropout"] if p["num_layers"] > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(p["hidden"], 128), nn.ReLU(),
                                      nn.Dropout(p["dropout"]), nn.Linear(128, HORIZON * D))

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).view(-1, HORIZON, D)

    class TCNForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            p = TCN_PARAMS
            layers, in_ch = [], D
            for i in range(p["levels"]):
                dilation = 2 ** i
                layers += [nn.Conv1d(in_ch, p["ch"], p["kernel"],
                                     padding=dilation * (p["kernel"] - 1) // 2,
                                     dilation=dilation),
                           nn.ReLU(), nn.Dropout(p["dropout"])]
                in_ch = p["ch"]
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(p["ch"], HORIZON * D)

        def forward(self, x):
            z = self.tcn(x.transpose(1, 2))
            return self.head(z[:, :, -1]).view(-1, HORIZON, D)

    class TransformerForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            p = TRF_PARAMS
            self.inp = nn.Linear(D, p["dmodel"])
            self.pos = nn.Parameter(torch.zeros(1, LAG, p["dmodel"]))
            layer = nn.TransformerEncoderLayer(p["dmodel"], p["nhead"], dim_feedforward=128,
                                               batch_first=True, dropout=p["dropout"])
            self.encoder = nn.TransformerEncoder(layer, num_layers=p["layers"])
            self.head = nn.Linear(p["dmodel"], HORIZON * D)

        def forward(self, x):
            z = self.encoder(self.inp(x) + self.pos)
            return self.head(z[:, -1, :]).view(-1, HORIZON, D)

    return {"LSTM": (LSTMForecaster, LSTM_PARAMS["lr"]),
            "TCN": (TCNForecaster, TCN_PARAMS["lr"]),
            "Transformer": (TransformerForecaster, TRF_PARAMS["lr"])}


def make_sequences(values):
    xs, ys = [], []
    for i in range(LAG, len(values) - HORIZON + 1):
        xs.append(values[i - LAG:i])
        ys.append(values[i:i + HORIZON])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def fit_deep(name, x_train, y_train, builders):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    if _THREADS:
        torch.set_num_threads(int(_THREADS))
    model_cls, lr = builders[name]
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt, yt = torch.tensor(x_train), torch.tensor(y_train)
    n = len(xt)
    model.train()
    for _ in range(DL_EPOCHS):
        order = torch.randperm(n)
        for i in range(0, n, 128):
            idx = order[i:i + 128]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()


# ------------------------------- driver -------------------------------
def machine_report():
    vm = psutil.virtual_memory()
    info = {
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "threads_capped_to": _THREADS or "not capped",
        "ram_total_GB": round(vm.total / 1e9, 1),
        "ram_available_GB": round(vm.available / 1e9, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "budget_GB": BUDGET_GB,
        "dl_epochs": DL_EPOCHS,
        "fast_model_repeats": N_REPEATS,
    }
    print("=== machine ===")
    for k, v in info.items():
        print(f"  {k:20} {v}")
    if vm.total / 1e9 > BUDGET_GB * 1.5:
        print(f"  [note] this machine has more memory than the {BUDGET_GB:g} GB budget, "
              f"so the times are optimistic for the deployment server.")
    return info


def main():
    info = machine_report()
    df = load_data()
    train = df.iloc[:-TEST_DAYS * STEPS_PER_DAY]
    print(f"\n=== data ===\n  rows {len(df)} | training rows {len(train)} "
          f"| channels {len(TARGETS)} ({', '.join(TARGETS)})")

    # Preprocessing is shared and is not charged to any model, but on a small
    # machine the feature matrix is the biggest object, so its cost is reported.
    with PeakRSS() as mem:
        t0 = time.perf_counter()
        X, Y = make_lag_features(train)
        prep_s = time.perf_counter() - t0
    print(f"  feature matrix {X.shape[0]} x {X.shape[1]} "
          f"({X.memory_usage(deep=True).sum() / 1e6:.0f} MB) "
          f"| built in {prep_s:.2f} s, peak rise {mem.rise_mb:.0f} MB")

    jobs = [("LightGBM", lambda: measure(fit_lightgbm, X, Y)),
            ("XGBoost", lambda: measure(fit_xgboost, X, Y)),
            ("RandomForest", lambda: measure(fit_random_forest, X, Y, repeats=1)),
            ("ARIMA", lambda: measure(fit_arima, train, repeats=1))]

    try:
        import torch  # noqa: F401
        builders = build_deep_models()
        values = train[TARGETS].to_numpy(np.float32)
        mu, sd = values.mean(0), values.std(0)
        sd[sd == 0] = 1.0
        x_train, y_train = make_sequences((values - mu) / sd)
        for name in ("LSTM", "TCN", "Transformer"):
            jobs.append((name, (lambda n=name: measure(fit_deep, n, x_train, y_train,
                                                       builders, repeats=1))))
    except ImportError:
        print("\n[warn] torch is not installed; LSTM, TCN and Transformer are skipped.")

    print("\n=== fitting ===")
    rows = []
    for name, job in jobs:
        if name.lower() in SKIP:
            print(f"  {name:14} skipped by TT_SKIP")
            continue
        try:
            secs, rise_mb, peak_mb = job()
        except MemoryError:
            print(f"  {name:14} MemoryError: did not fit in the available memory")
            rows.append({"model": name, "train_seconds": None,
                         "peak_rise_MB": None, "peak_rss_MB": None, "status": "out of memory"})
            continue
        except Exception as exc:
            print(f"  {name:14} failed: {type(exc).__name__}: {exc}")
            rows.append({"model": name, "train_seconds": None, "peak_rise_MB": None,
                         "peak_rss_MB": None, "status": f"failed: {type(exc).__name__}"})
            continue
        rows.append({"model": name, "train_seconds": round(secs, 2),
                     "peak_rise_MB": round(rise_mb, 1), "peak_rss_MB": round(peak_mb, 1),
                     "status": "ok"})
        print(f"  {name:14} {secs:9.2f} s   rise {rise_mb:7.1f} MB   peak {peak_mb:7.1f} MB")

    out = pd.DataFrame(rows)
    ok = out[out.status == "ok"]
    print("\n=== training time and memory, six channels, tuned configurations ===")
    print(out.to_string(index=False))
    if len(ok):
        worst = ok.loc[ok.peak_rss_MB.idxmax()]
        headroom = BUDGET_GB - worst.peak_rss_MB / 1000
        print(f"\nheaviest model: {worst.model} at {worst.peak_rss_MB / 1000:.2f} GB peak, "
              f"leaving {headroom:.2f} GB of the {BUDGET_GB:g} GB budget")
        if headroom < 0:
            print("  [warn] this configuration does not fit the budget on this machine")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.assign(train_rows=len(train), channels=len(TARGETS)).to_csv(
        os.path.join(OUT_DIR, "timing_models_8gb.csv"), index=False)
    pd.Series(info).to_csv(os.path.join(OUT_DIR, "timing_hardware_8gb.csv"))
    print(f"\nwrote timing_models_8gb.csv and timing_hardware_8gb.csv to {OUT_DIR}")
    print("Times and memory are hardware-dependent. Report them together with the "
          "machine description printed above.")
    return out


if __name__ == "__main__":
    main()
