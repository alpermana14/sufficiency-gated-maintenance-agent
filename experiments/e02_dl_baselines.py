"""E2 - Deep-learning forecasting baselines, TUNED under the same budget (R4-7 + R4-8).

Adds LSTM, a Temporal Convolutional Network (TCN), and a compact Transformer
encoder to the forecasting comparison. To keep the comparison fair, every deep
model is tuned with the SAME optimisation budget as the tree models in E1b:
15 Optuna trials (TPE sampler, fixed seed) over an architecture/optimiser
search space, using the same objective family — mean over the seven channels of
the validation RMSE normalised by each channel's training standard deviation.

Protocol (matches E1b so results merge into the shared Table 4/5 pipeline):
  * Direct multi-step: input = last 48 steps x 7 channels, output = next 12 x 7.
  * Splits (chronological): tuning-train = all but last 14 days; validation =
    days [-14, -7]; final-train = all but last 7 days; test = last 7 days.
  * Tuning trains each candidate for TUNE_EPOCHS on tuning-train and scores the
    normalised validation RMSE. The best config is retrained for EPOCHS on
    final-train and evaluated at the SAME 20 origins as E1b (errors pooled over
    origins x horizon), giving MSE/MAE/RMSE/R2 directly comparable to the table.
  * Inputs standardised with final-train statistics; CPU-only torch; fixed seed.
  * Reported training time is the FINAL fit (best config, EPOCHS), comparable to
    the tree Table 5 entries (the tuning search cost is separate, as for trees).

Writes e01b_pred_<Model>.csv in E1b's format; re-run E1b's aggregate afterwards.

Usage:  E02_ONLY=LSTM python experiments/e02_dl_baselines.py   # one model
        python experiments/e02_dl_baselines.py                 # all three
  env:  E02_TRIALS (default 15, matches trees), E02_TUNE_EPOCHS (default 12),
        E02_EPOCHS (final, default 25)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from common import load_data, ensure_results_dir
from ml_engine import TARGETS, LAG_STEPS, FORECAST_HORIZON

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_TRIALS = int(os.environ.get("E02_TRIALS", "15"))       # same budget as the trees
TUNE_EPOCHS = int(os.environ.get("E02_TUNE_EPOCHS", "12"))
EPOCHS = int(os.environ.get("E02_EPOCHS", "25"))
BATCH = 128
TEST_DAYS = 7
VAL_DAYS = 7
STEPS_PER_DAY = 48
N_ORIGINS = 20
LOOKBACK = LAG_STEPS          # 48
HORIZON = FORECAST_HORIZON    # 12
D = len(TARGETS)              # 7

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(os.cpu_count())


# ------------------------------- models -------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(D, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, HORIZON * D))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).view(-1, HORIZON, D)


class TCNForecaster(nn.Module):
    def __init__(self, ch=64, levels=4, kernel=3, dropout=0.0):
        super().__init__()
        layers, in_ch = [], D
        for i in range(levels):
            dil = 2 ** i
            layers += [nn.Conv1d(in_ch, ch, kernel, padding=dil * (kernel - 1) // 2,
                                 dilation=dil), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(ch, HORIZON * D)

    def forward(self, x):
        z = self.tcn(x.transpose(1, 2))
        return self.head(z[:, :, -1]).view(-1, HORIZON, D)


class TransformerForecaster(nn.Module):
    def __init__(self, dmodel=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(D, dmodel)
        self.pos = nn.Parameter(torch.zeros(1, LOOKBACK, dmodel))
        enc = nn.TransformerEncoderLayer(dmodel, nhead, dim_feedforward=128,
                                         batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(dmodel, HORIZON * D)

    def forward(self, x):
        z = self.encoder(self.inp(x) + self.pos)
        return self.head(z[:, -1, :]).view(-1, HORIZON, D)


def build(name, p):
    if name == "LSTM":
        return LSTMForecaster(p["hidden"], p["num_layers"], p["dropout"])
    if name == "TCN":
        return TCNForecaster(p["ch"], p["levels"], p["kernel"], p["dropout"])
    return TransformerForecaster(p["dmodel"], p["nhead"], p["layers"], p["dropout"])


def suggest(trial, name):
    if name == "LSTM":
        return dict(hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
                    num_layers=trial.suggest_int("num_layers", 1, 2),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3),
                    lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True))
    if name == "TCN":
        return dict(ch=trial.suggest_categorical("ch", [32, 64, 128]),
                    levels=trial.suggest_int("levels", 2, 5),
                    kernel=trial.suggest_categorical("kernel", [2, 3]),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3),
                    lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True))
    return dict(dmodel=trial.suggest_categorical("dmodel", [32, 64]),
                nhead=trial.suggest_categorical("nhead", [2, 4]),
                layers=trial.suggest_int("layers", 1, 2),
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
                lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True))


# ------------------------------- data & train -------------------------------
def make_sequences(values):
    Xs, Ys = [], []
    for i in range(LOOKBACK, len(values) - HORIZON + 1):
        Xs.append(values[i - LOOKBACK:i]); Ys.append(values[i:i + HORIZON])
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32)


def fit(model, X, Y, lr, epochs, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    n = len(Xt); model.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if verbose and (ep == 0 or (ep + 1) % 10 == 0):
            print(f"      epoch {ep+1:3d}/{epochs} train_mse={tot/n:.4f}")
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))


def run(name, results_dir):
    df = load_data(); df = df[~df.index.duplicated(keep="first")].sort_index()
    vals = df[TARGETS].to_numpy(np.float32)
    n = len(df); test0 = n - TEST_DAYS * STEPS_PER_DAY; val0 = test0 - VAL_DAYS * STEPS_PER_DAY

    # standardise with FINAL-train (all but last 7 days) stats
    ftrain = vals[:test0]
    mu = ftrain.mean(0); sd = ftrain.std(0); sd[sd == 0] = 1.0
    z = lambda a: (a - mu) / sd
    unz = lambda a: a * sd + mu
    train_std = df[TARGETS].iloc[:test0].std().replace(0, 1).to_numpy(np.float32)

    # ---- tuning: train on [:val0], validate on [val0:test0] ----
    Xtune, Ytune = make_sequences(z(vals[:val0]))
    Xval, Yval = make_sequences(z(vals[val0 - LOOKBACK:test0]))  # include lookback context

    def objective(trial):
        p = suggest(trial, name)
        torch.manual_seed(SEED)
        m = fit(build(name, p), Xtune, Ytune, p["lr"], TUNE_EPOCHS)
        pv = unz(predict(m, Xval)); yv = unz(Yval)
        errs = [rmse(yv[:, :, j], pv[:, :, j]) / train_std[j] for j in range(D)]
        return float(np.mean(errs))

    print(f"[INFO] {name}: tuning {N_TRIALS} trials (TUNE_EPOCHS={TUNE_EPOCHS}) "
          f"| tune-seq={len(Xtune)} val-seq={len(Xval)}")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    print(f"[tune] {name} done in {time.time()-t0:.1f}s | best={best}")

    # ---- final: retrain best on final-train, evaluate at 20 origins ----
    Xtr, Ytr = make_sequences(z(vals[:test0]))
    torch.manual_seed(SEED)
    t0 = time.time()
    model = fit(build(name, best), Xtr, Ytr, best["lr"], EPOCHS, verbose=True)
    fit_s = time.time() - t0

    max_origin = (n - test0) - HORIZON - 1
    origins = np.linspace(LAG_STEPS, max_origin, num=N_ORIGINS, dtype=int).tolist()
    pred = {t: [] for t in TARGETS}; true = {t: [] for t in TARGETS}; hz = []
    for origin in origins:
        end = test0 + origin
        hist = vals[end - LOOKBACK:end]; fut = vals[end:end + HORIZON]
        yhat = unz(predict(model, z(hist)[None, ...])[0])
        for h in range(HORIZON):
            for j, t in enumerate(TARGETS):
                pred[t].append(float(yhat[h, j])); true[t].append(float(fut[h, j]))
            hz.append(h + 1)

    out = pd.DataFrame({**{f"pred_{t}": pred[t] for t in TARGETS},
                        **{f"true_{t}": true[t] for t in TARGETS}, "h": hz})
    out.to_csv(f"{results_dir}/e01b_pred_{name}.csv", index=False)
    with open(f"{results_dir}/e01b_traintime.csv", "a") as f:
        f.write(f"{name},{fit_s:.3f}\n")
    with open(f"{results_dir}/e02_bestparams.csv", "a") as f:
        f.write(f"{name},{best}\n")
    mean_rmse = np.mean([rmse(out[f"true_{t}"], out[f"pred_{t}"]) for t in TARGETS])
    mean_r2 = np.mean([r2_score(out[f"true_{t}"], out[f"pred_{t}"]) for t in TARGETS])
    print(f"[OK] {name}: final fit {fit_s:.1f}s | mean RMSE {mean_rmse:.4f} | "
          f"mean R2 {mean_r2:.3f} -> e01b_pred_{name}.csv")


def main():
    rd = ensure_results_dir()
    only = os.environ.get("E02_ONLY", "")
    for name in ([only] if only else ["LSTM", "TCN", "Transformer"]):
        run(name, rd)
    print("\n[NEXT] E01B_AGGREGATE=1 python experiments/e01b_forecasting_modelcomp.py")


if __name__ == "__main__":
    main()
