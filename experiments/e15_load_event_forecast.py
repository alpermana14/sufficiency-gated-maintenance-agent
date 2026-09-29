"""E15 - LightGBM (absolute lags) vs LightGBM-CD (centred-delta) across the 40 kg load change.

Confirms whether the deployed centred-delta formulation (centred lag features, one-step
change target, recency weighting; backend/ml_engine.py) forecasts better than the
absolute-lag formulation when the operating regime changes. Both models use exactly the
functions, search spaces, budget (15 Optuna TPE trials) and objective of
e01b_forecasting_modelcomp.py, so the results are comparable with the main comparison.

Load window (same as the AUROC labels): PM_EVENT_START .. PM_EVENT_END
(default 2026-05-05 08:00 .. 2026-05-07 12:00).

Chronology:
  test window starts   TEST_START = EVENT_START - 2 days
  validation (tuning)  [TEST_START - 7 days, TEST_START)
  tuning-train         data before the validation window
  final-train          data before TEST_START
  forecast origins     every 3 h from EVENT_START - 1 day to EVENT_END + 1 day - 6 h
                       (12-step = 6 h recursive forecasts; history = all data before the origin)

Two evaluation modes:
  static   models trained once on data before TEST_START (the 40 kg regime is unseen)
  rolling  models refitted with the tuned hyperparameters on all data before the origin,
           every RETRAIN_EVERY_H hours (closer to the deployed retraining loop)

Metrics per phase of the forecast target time (pre-load / load / post-load) and for the
first 12 h after placement and after removal: MAE, MSE, signed bias (pred - true), and a
paired Wilcoxon test on per-origin MAE.

Usage:  PM_EVENT_START="2026-05-05 08:00:00" PM_EVENT_END="2026-05-07 12:00:00" \
            python experiments/e15_load_event_forecast.py
Outputs: experiments/results/e15_predictions.csv, e15_phase_metrics.csv, e15_paired.csv,
         e15_bestparams.csv, e15_trajectories.png
"""

import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import e01b_forecasting_modelcomp as E
from common import EVENT_END, EVENT_START, ensure_results_dir, load_data
from ml_engine import FORECAST_HORIZON, LAG_STEPS, TARGETS

ORIGIN_STEP_H = 3
RETRAIN_EVERY_H = int(os.environ.get("E15_RETRAIN_EVERY_H", "6"))
MODES = [m.strip() for m in os.environ.get("E15_MODES", "static,rolling").split(",") if m.strip()]
RESPONDING = ["z_rms", "noise"]


def fit_models(model, params, train_df):
    if model == "LightGBM":
        X, y = E.build_supervised(train_df)
        models = {t: E.make_lgb(params).fit(X, y[t]) for t in TARGETS}
    else:
        X, y, _ = E.build_supervised_cd(train_df)
        w = E.recency_weights(len(X))
        models = {t: E.make_lgb(params).fit(X, y[t], sample_weight=w) for t in TARGETS}
    return models, list(X.columns)


def forecast(model, models, x_cols, history):
    fn = E.recursive_forecast_one_origin if model == "LightGBM" else E.recursive_forecast_cd
    return fn(models, history[TARGETS].iloc[-LAG_STEPS:], x_cols, FORECAST_HORIZON)


def tune_model(model, df, val_start, test_start):
    tr_tune = df[df.index < val_start]
    upto_test = df[df.index < test_start]
    val_index = df[(df.index >= val_start) & (df.index < test_start)].index
    if model == "LightGBM":
        Xt, yt = E.build_supervised(tr_tune)
        Xv_full, yv_full = E.build_supervised(upto_test)
        idx = Xv_full.index.intersection(val_index)
        return E.tune("LightGBM", Xt, yt, Xv_full.loc[idx], yv_full.loc[idx])
    Xt, yt, yt_abs = E.build_supervised_cd(tr_tune)
    Xv_full, yv_full, _ = E.build_supervised_cd(upto_test)
    idx = Xv_full.index.intersection(val_index)
    return E.tune_cd(Xt, yt, E.recency_weights(len(Xt)), Xv_full.loc[idx], yv_full.loc[idx],
                     yt_abs.std().replace(0, np.nan))


def phase_of(ts):
    return np.where(ts < EVENT_START, "pre-load", np.where(ts < EVENT_END, "load", "post-load"))


def main():
    rd = ensure_results_dir()
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    test_start = EVENT_START - pd.Timedelta(days=2)
    val_start = test_start - pd.Timedelta(days=7)
    origins = pd.date_range(EVENT_START - pd.Timedelta(days=1),
                            EVENT_END + pd.Timedelta(days=1) - pd.Timedelta(hours=6),
                            freq=f"{ORIGIN_STEP_H}h")
    print(f"[INFO] load {EVENT_START} .. {EVENT_END} | validation {val_start} .. {test_start} "
          f"| {len(origins)} origins {origins[0]} .. {origins[-1]} | modes={MODES} | trials={E.N_TRIALS}")

    params, rows = {}, []
    for model in ["LightGBM", "LightGBM-CD"]:
        t0 = time.time()
        params[model] = tune_model(model, df, val_start, test_start)
        print(f"[tune] {model} {time.time() - t0:.0f}s best={params[model]}")

    for mode in MODES:
        for model in ["LightGBM", "LightGBM-CD"]:
            t0 = time.time()
            models, x_cols, fitted_at = None, None, None
            for origin in origins:
                if mode == "static":
                    if models is None:
                        models, x_cols = fit_models(model, params[model], df[df.index < test_start])
                elif fitted_at is None or origin - fitted_at >= pd.Timedelta(hours=RETRAIN_EVERY_H):
                    models, x_cols = fit_models(model, params[model], df[df.index < origin])
                    fitted_at = origin
                hist = df[df.index < origin]
                fc = forecast(model, models, x_cols, hist)
                fut = df[df.index >= origin].iloc[:FORECAST_HORIZON]
                for h in range(FORECAST_HORIZON):
                    for c in TARGETS:
                        rows.append({"mode": mode, "model": model, "origin": origin, "h": h + 1,
                                     "target_time": fut.index[h], "channel": c,
                                     "pred": float(fc.loc[h, c]), "true": float(fut[c].iloc[h])})
            print(f"[{mode}] {model} done in {time.time() - t0:.0f}s")

    pr = pd.DataFrame(rows)
    pr["err"] = pr["pred"] - pr["true"]
    pr["phase"] = phase_of(pr["target_time"])
    pr["onset12h"] = (pr["target_time"] >= EVENT_START) & (pr["target_time"] < EVENT_START + pd.Timedelta(hours=12))
    pr["removal12h"] = (pr["target_time"] >= EVENT_END) & (pr["target_time"] < EVENT_END + pd.Timedelta(hours=12))
    pr.to_csv(f"{rd}/e15_predictions.csv", index=False)
    pd.DataFrame([{"model": k, "params": v} for k, v in params.items()]).to_csv(f"{rd}/e15_bestparams.csv", index=False)

    # ---- metrics per subset ----
    subsets = {"all": pr, "pre-load": pr[pr.phase == "pre-load"], "load": pr[pr.phase == "load"],
               "post-load": pr[pr.phase == "post-load"], "first 12 h of load": pr[pr.onset12h],
               "first 12 h after removal": pr[pr.removal12h]}
    met = []
    for name, sub in subsets.items():
        g = sub.groupby(["mode", "model", "channel"])["err"]
        m = pd.DataFrame({"MAE": g.apply(lambda e: e.abs().mean()), "MSE": g.apply(lambda e: (e ** 2).mean()),
                          "bias": g.mean(), "n": g.size()}).reset_index()
        m.insert(0, "subset", name)
        met.append(m)
    met = pd.concat(met, ignore_index=True)
    met.to_csv(f"{rd}/e15_phase_metrics.csv", index=False)

    # ---- paired test on per-origin MAE (origin-level, per phase of the origin's targets) ----
    paired = []
    for mode in MODES:
        for name, sub in subsets.items():
            s = sub[sub["mode"] == mode]
            for c in TARGETS + ["responding (z_rms+noise)"]:
                cs = s[s.channel.isin(RESPONDING)] if c.startswith("responding") else s[s.channel == c]
                if cs.empty:
                    continue
                per = cs.assign(ae=cs.err.abs()).groupby(["model", "origin", "channel"])["ae"].mean().unstack("model")
                per = per.dropna()
                if len(per) < 3:
                    continue
                d = (per["LightGBM-CD"] - per["LightGBM"]).to_numpy()
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                paired.append({"mode": mode, "subset": name, "channel": c, "n_pairs": len(d),
                               "CD_better_pairs": int((d < 0).sum()), "median_MAE_diff_CD_minus_abs": float(np.median(d)),
                               "p_wilcoxon": p})
    paired = pd.DataFrame(paired)
    paired.to_csv(f"{rd}/e15_paired.csv", index=False)

    # ---- print ----
    pd.set_option("display.width", 220)
    for mode in MODES:
        print(f"\n===== MODE: {mode} =====")
        for name in subsets:
            sub = met[(met["mode"] == mode) & (met.subset == name)]
            if sub.empty:
                continue
            tab = sub.pivot(index="channel", columns="model", values=["MAE", "bias"]).loc[TARGETS].round(4)
            print(f"\n-- {name} (n per model/channel = {int(sub.n.iloc[0])}) --")
            print(tab.to_string())
        print("\n-- paired Wilcoxon on per-origin MAE (negative diff = CD better) --")
        print(paired[paired["mode"] == mode].round(4).to_string(index=False))

    # ---- trajectory figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = [EVENT_START + pd.Timedelta(hours=3), EVENT_START + pd.Timedelta(hours=24), EVENT_END + pd.Timedelta(hours=3)]
        sel = [origins[np.argmin(np.abs(origins - s))] for s in sel]
        mode = MODES[0]
        fig, axes = plt.subplots(len(RESPONDING), len(sel), figsize=(14, 6), sharey="row")
        for i, c in enumerate(RESPONDING):
            for j, o in enumerate(sel):
                ax = axes[i, j]
                ctx = df[(df.index >= o - pd.Timedelta(hours=12)) & (df.index < o + pd.Timedelta(hours=6))][c]
                ax.plot(ctx.index, ctx.values, color="black", lw=1.2, label="measured")
                for model, col in [("LightGBM", "tab:orange"), ("LightGBM-CD", "tab:blue")]:
                    q = pr[(pr["mode"] == mode) & (pr.model == model) & (pr.origin == o) & (pr.channel == c)]
                    ax.plot(q.target_time, q.pred, color=col, lw=1.5, label=model)
                ax.axvline(EVENT_START, color="grey", ls=":")
                ax.axvline(EVENT_END, color="grey", ls=":")
                ax.set_title(f"{c} | origin {o:%d %b %H:%M}", fontsize=9)
                ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{rd}/e15_trajectories.png", dpi=200)
        print(f"[OK] figure e15_trajectories.png ({mode} mode)")
    except Exception as exc:  # figure is optional
        print(f"[WARN] figure skipped: {exc}")
    print(f"\n[OK] wrote e15_* files to {rd}")


if __name__ == "__main__":
    main()
