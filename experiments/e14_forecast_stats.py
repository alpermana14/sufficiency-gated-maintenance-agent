"""E14 - Statistical comparison of the six-channel forecasting results (CAEE R1-4, R2-5, EiC-9).

Reads the per-model prediction files written by e01b/e02 (20 forecast origins x 12
recursive/direct steps per channel) and reports:

  * MSE and MAE per model and channel with 95% cluster-bootstrap intervals
    (resampling the 20 forecast origins, 2,000 replicates);
  * Friedman test over all models, with blocks = channel x origin and the block value
    = MAE (primary) or MSE over the 12 steps of that origin; mean ranks per model;
  * Wilcoxon signed-rank tests (Holm-corrected) of each LightGBM variant against every
    other model on the same blocks;
  * Diebold-Mariano tests per channel (squared-error loss) with a Newey-West variance
    (Bartlett kernel, lag = horizon - 1) and the Harvey-Leybourne-Newbold correction;
  * normalised MAE per horizon step (error / test-period SD of the channel, averaged over
    channels) for the per-horizon figure.

Persistence is excluded from the main comparison (removed from the paper by the authors);
set E14_WITH_PERSISTENCE=1 to include it.

Outputs: experiments/results/e14_metrics_ci.csv, e14_friedman.csv, e14_ranks.csv,
         e14_wilcoxon_holm.csv, e14_dm_hac.csv, e14_perhorizon_norm.csv
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from common import ensure_results_dir

HORIZON = 12
N_BOOT = 2000
SEED = 7
MODELS = ["LightGBM-CD", "LightGBM", "RandomForest", "XGBoost", "LSTM", "TCN", "Transformer", "ARIMA"]
if os.environ.get("E14_WITH_PERSISTENCE", "") in {"1", "true", "yes"}:
    MODELS.append("Persistence")
# E14_MAIN_ONLY=1: the seven models of Table 4 (the deployed system uses absolute-lag LightGBM,
# decision D13; the Centred-Delta variant goes to the supplement). Outputs use the prefix e14main_.
MAIN_ONLY = os.environ.get("E14_MAIN_ONLY", "") in {"1", "true", "yes"}
if MAIN_ONLY:
    MODELS.remove("LightGBM-CD")
REFERENCES = [r for r in ["LightGBM-CD", "LightGBM"] if r in MODELS]
PREFIX = "e14main" if MAIN_ONLY else "e14"


def load(results_dir):
    preds = {}
    for m in MODELS:
        d = pd.read_csv(f"{results_dir}/e01b_pred_{m}.csv")
        if len(d) % HORIZON or not (d["h"].to_numpy() == np.tile(np.arange(1, HORIZON + 1), len(d) // HORIZON)).all():
            raise ValueError(f"{m}: rows are not ordered as origin x horizon")
        preds[m] = d
    channels = [c[5:] for c in preds[MODELS[0]].columns if c.startswith("pred_")]
    y0 = preds[MODELS[0]][[f"true_{c}" for c in channels]].to_numpy()
    for m in MODELS[1:]:
        if not np.allclose(preds[m][[f"true_{c}" for c in channels]].to_numpy(), y0, atol=1e-6):
            raise ValueError(f"{m}: ground truth differs from {MODELS[0]} (different origins?)")
    return preds, channels


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def dm_hac(e1, e2, h=HORIZON):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    var = np.mean(dc * dc)
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.mean(dc[k:] * dc[:-k])
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def main():
    rd = ensure_results_dir()
    preds, channels = load(rd)
    n_orig = len(preds[MODELS[0]]) // HORIZON
    rng = np.random.default_rng(SEED)
    boot_orig = rng.integers(0, n_orig, size=(N_BOOT, n_orig))

    # errors[m][c] -> array (n_orig, HORIZON)
    errors = {m: {c: (preds[m][f"pred_{c}"] - preds[m][f"true_{c}"]).to_numpy().reshape(n_orig, HORIZON)
                  for c in channels} for m in MODELS}

    # ---- metrics with cluster-bootstrap CI ----
    rows = []
    for m in MODELS:
        for c in channels:
            e = errors[m][c]
            se, ae = (e ** 2).mean(axis=1), np.abs(e).mean(axis=1)  # per origin
            bse, bae = se[boot_orig].mean(axis=1), ae[boot_orig].mean(axis=1)
            rows.append({"model": m, "channel": c,
                         "MSE": se.mean(), "MSE_lo": np.percentile(bse, 2.5), "MSE_hi": np.percentile(bse, 97.5),
                         "MAE": ae.mean(), "MAE_lo": np.percentile(bae, 2.5), "MAE_hi": np.percentile(bae, 97.5)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(f"{rd}/{PREFIX}_metrics_ci.csv", index=False)

    # ---- Friedman + ranks + Wilcoxon-Holm on channel x origin blocks ----
    fr_rows, rank_rows, wil_rows = [], [], []
    for metric, fn in [("MAE", lambda e: np.abs(e).mean(axis=1)), ("MSE", lambda e: (e ** 2).mean(axis=1))]:
        blocks = np.column_stack([np.concatenate([fn(errors[m][c]) for c in channels]) for m in MODELS])
        chi2, p = stats.friedmanchisquare(*blocks.T)
        fr_rows.append({"metric": metric, "n_blocks": blocks.shape[0], "n_models": blocks.shape[1],
                        "chi2": chi2, "p_value": p})
        ranks = stats.rankdata(blocks, axis=1).mean(axis=0)
        for m, r in zip(MODELS, ranks):
            rank_rows.append({"metric": metric, "model": m, "mean_rank": r})
        for ref in REFERENCES:
            ri = MODELS.index(ref)
            others = [m for m in MODELS if m != ref]
            res = [stats.wilcoxon(blocks[:, ri], blocks[:, MODELS.index(o)]) for o in others]
            adj = holm([r.pvalue for r in res])
            for o, r, pa in zip(others, res, adj):
                diff = blocks[:, ri] - blocks[:, MODELS.index(o)]
                wil_rows.append({"metric": metric, "reference": ref, "versus": o,
                                 "median_diff_ref_minus_other": float(np.median(diff)),
                                 "ref_better_blocks": int((diff < 0).sum()), "n_blocks": len(diff),
                                 "W": r.statistic, "p_raw": r.pvalue, "p_holm": pa,
                                 "sig_0.05": bool(pa < 0.05)})
    pd.DataFrame(fr_rows).to_csv(f"{rd}/{PREFIX}_friedman.csv", index=False)
    ranks_df = pd.DataFrame(rank_rows)
    ranks_df.to_csv(f"{rd}/{PREFIX}_ranks.csv", index=False)
    wil = pd.DataFrame(wil_rows)
    wil.to_csv(f"{rd}/{PREFIX}_wilcoxon_holm.csv", index=False)

    # ---- Diebold-Mariano (HAC) per channel ----
    dm_rows = []
    for ref in REFERENCES:
        for o in [m for m in MODELS if m != ref]:
            for c in channels:
                stat, p = dm_hac(errors[ref][c].ravel(), errors[o][c].ravel())
                dm_rows.append({"reference": ref, "versus": o, "channel": c, "dm_stat": stat, "p_value": p,
                                "sig_0.05": bool(p < 0.05) if p == p else False,
                                "better": ("reference" if stat < 0 else "versus") if p < 0.05 else "n.s."})
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(f"{rd}/{PREFIX}_dm_hac.csv", index=False)

    # ---- normalised per-horizon MAE ----
    sd = {c: preds[MODELS[0]][f"true_{c}"].std() for c in channels}
    ph = [{"model": m, "h": h + 1,
           "nMAE": float(np.mean([np.abs(errors[m][c][:, h]).mean() / sd[c] for c in channels]))}
          for m in MODELS for h in range(HORIZON)]
    pd.DataFrame(ph).to_csv(f"{rd}/{PREFIX}_perhorizon_norm.csv", index=False)

    # ---- print ----
    pd.set_option("display.width", 220)
    for metric in ["MSE", "MAE"]:
        cell = metrics.apply(lambda r: f"{r[metric]:.4f} [{r[metric + '_lo']:.4f}, {r[metric + '_hi']:.4f}]", axis=1)
        print(f"\n=== {metric} per channel [95% origin-bootstrap CI] ===")
        print(metrics.assign(cell=cell).pivot(index="channel", columns="model", values="cell")[MODELS].to_string())
    print("\n=== Friedman (blocks = channel x origin) ===")
    print(pd.DataFrame(fr_rows).to_string(index=False))
    print("\n=== mean ranks (1 = best) ===")
    print(ranks_df.pivot(index="model", columns="metric", values="mean_rank").loc[MODELS].round(2).to_string())
    print("\n=== Wilcoxon signed-rank, Holm-corrected ===")
    print(wil.round(4).to_string(index=False))
    print("\n=== Diebold-Mariano (HAC) significant results per channel (p<0.05) ===")
    print(dm.pivot_table(index=["reference", "versus"], columns="channel", values="better", aggfunc="first")
          [channels].to_string())
    print(f"\n[OK] wrote {PREFIX}_* files to {rd}")


if __name__ == "__main__":
    main()
