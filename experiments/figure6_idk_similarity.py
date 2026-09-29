"""Figure 6 of the CAEE revision: s-IDK^2 similarity of the channels that responded to the load.

Uses exactly the protocol of Table 5 (experiments/e13_anomaly_fair_eval.py): the same evaluation
range (7 days before to 1 day after the labelled event), raw univariate values, psi1 = psi2 = 2,
omega = 20, t = 100, scores averaged over five seeds and assigned to the window-end record.
The shaded band is the logged 40 kg load period (5 May 08:00 to 7 May 12:00).

Usage:
  PM_EVENT_START="2026-05-05 08:00:00" PM_EVENT_END="2026-05-07 12:00:00" \
      python experiments/figure6_idk_similarity.py
Outputs: experiments/results/figure6_psi2_w20.png (300 dpi), .pdf and .csv
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import EVENT_END, EVENT_START, ensure_results_dir, idk_window_scores, load_data, set_seed, slice_eval_range  # noqa: E402

PSI, WIDTH, T, N_SEEDS = 2, 20, 100, 5
CHANNELS = [("z_rms", "z_rms (mm/s)"), ("noise", "Acoustic noise (dB)")]


def main():
    df = slice_eval_range(load_data(), days_before=7.0, days_after=1.0)
    out_dir = ensure_results_dir()
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(7.0, 5.2), sharex=True)
    dump = {}
    for ax, (ch, label), panel in zip(axes, CHANNELS, "ab"):
        values = df[ch].to_numpy(dtype=float)
        seeds = []
        for seed in range(N_SEEDS):
            set_seed(seed)
            seeds.append(idk_window_scores(values, width=WIDTH, psi1=PSI, psi2=PSI, t=T))
        score = np.mean(seeds, axis=0)
        idx = df.index[WIDTH - 1:]
        dump[f"{ch}_similarity"] = pd.Series(score, index=idx)
        dump[f"{ch}_value"] = df[ch].iloc[WIDTH - 1:]

        ax.axvspan(EVENT_START, EVENT_END, color="#F2C14E", alpha=0.25, lw=0, label="40 kg load")
        ax.plot(idx, score, color="#4C3B99", lw=1.4, label="s-IDK² similarity")
        ax.set_ylabel("s-IDK² similarity", color="#4C3B99")
        ax.tick_params(axis="y", labelcolor="#4C3B99")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax2 = ax.twinx()
        ax2.plot(idx, df[ch].iloc[WIDTH - 1:], color="#8A8A8A", lw=1.0, alpha=0.8)
        ax2.set_ylabel(label, color="#5A5A5A", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#5A5A5A", labelsize=8)
        ax.set_title(f"({panel}) {ch}", loc="left", fontsize=10)
        print(f"[{ch}] windows={len(score)} span {idx[0]} .. {idx[-1]} "
              f"baseline median={np.median(score[idx < EVENT_START]):.3f} "
              f"load median={np.median(score[(idx >= EVENT_START) & (idx < EVENT_END)]):.3f}")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    axes[-1].set_xlabel("Date (2026)")
    axes[-1].set_xlim(df.index[WIDTH - 1], df.index[-1])
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    stem = os.path.join(out_dir, f"figure6_psi{PSI}_w{WIDTH}")
    fig.savefig(stem + ".png", dpi=300)
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    pd.DataFrame(dump).to_csv(stem + ".csv", index_label="datetime")
    print(f"[OK] wrote {stem}.png, .pdf and .csv")


if __name__ == "__main__":
    main()
