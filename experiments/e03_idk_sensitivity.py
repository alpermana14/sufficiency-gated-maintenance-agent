"""E3 — Sensitivity analysis of s-IDK^2 parameters (answers R4-10).

Grid: psi1 = psi2 = psi in {2, 4, 8, 16, 32} x window width in {8, 12, 20, 32, 48}.
Metric: AUROC of the negated IDK similarity score against the labelled anomaly
window (default: the 2026-05-05 load-change event; override with PM_EVENT_START /
PM_EVENT_END for staged-fault sessions from E6).

Each cell is averaged over N_SEEDS runs because IDK subsampling is stochastic.

Outputs (experiments/results/):
  e03_sensitivity.csv          long-format results (channel, psi, width, seed, auroc)
  e03_heatmap_<channel>.png    per-channel mean-AUROC heatmaps
  e03_heatmap_mean.png         heatmap averaged across responding channels

Usage:  python experiments/e03_idk_sensitivity.py
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from common import (
    TARGETS,
    align_window_labels,
    ensure_results_dir,
    event_labels,
    idk_window_scores,
    load_data,
    set_seed,
    slice_eval_range,
)

PSIS = [2, 4, 8, 16, 32]
WIDTHS = [8, 12, 20, 32, 48]
N_SEEDS = 5
T = 100


def main() -> None:
    results_dir = ensure_results_dir()
    # 7 days of baseline before the event + 1 day after, mirroring the paper's
    # Apr 27 - May 5 evaluation scope (sliced around the event, NOT the tail).
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)
    labels_full = event_labels(df.index)

    if labels_full.sum() == 0:
        raise RuntimeError(
            "No labelled anomaly points inside the loaded data range. "
            "Check PM_EVENT_START / PM_EVENT_END against the dataset."
        )
    print(
        f"[INFO] {len(df)} points, {labels_full.sum()} labelled anomalous, "
        f"range {df.index[0]} .. {df.index[-1]}"
    )

    rows = []
    for channel, psi, width, seed in itertools.product(
        TARGETS, PSIS, WIDTHS, range(N_SEEDS)
    ):
        values = df[channel].to_numpy()
        n_windows = len(values) - width + 1
        # psi1 subsamples points, psi2 subsamples windows: both must fit.
        if psi > min(len(values), n_windows):
            continue
        y = align_window_labels(labels_full, width)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        set_seed(seed)
        scores = idk_window_scores(values, width=width, psi1=psi, psi2=psi, t=T)
        auroc = roc_auc_score(y, -scores)  # low similarity = anomaly
        rows.append(
            {"channel": channel, "psi": psi, "width": width, "seed": seed, "auroc": auroc}
        )
        if seed == N_SEEDS - 1:
            sub = [r["auroc"] for r in rows if r["channel"] == channel
                   and r["psi"] == psi and r["width"] == width]
            print(f"  {channel:12s} psi={psi:<3d} w={width:<3d} "
                  f"AUROC={np.mean(sub):.3f} ± {np.std(sub):.3f}")

    res = pd.DataFrame(rows)
    out_csv = f"{results_dir}/e03_sensitivity.csv"
    res.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    mean_res = (
        res.groupby(["channel", "psi", "width"])["auroc"].mean().reset_index()
    )
    for channel, sub in mean_res.groupby("channel"):
        _heatmap(sub, f"s-IDK$^2$ AUROC — {channel}",
                 f"{results_dir}/e03_heatmap_{channel}.png")
    overall = mean_res.groupby(["psi", "width"])["auroc"].mean().reset_index()
    overall["channel"] = "mean"
    _heatmap(overall, "s-IDK$^2$ AUROC — mean across channels",
             f"{results_dir}/e03_heatmap_mean.png")
    print(f"[OK] heatmaps written to {results_dir}")


def _heatmap(sub: pd.DataFrame, title: str, path: str) -> None:
    pivot = sub.pivot(index="psi", columns="width", values="auroc")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("window width $\\omega$")
    ax.set_ylabel("subsampling $\\psi_1 = \\psi_2$")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.8 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
