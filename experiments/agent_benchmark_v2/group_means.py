"""Task success of the two groups of benchmark categories (Table 6), with the cluster bootstrap of analyse_ratings.py.

The six everyday categories and the six in which a plausible answer is wrong are those of Section 4.4.2. Reads the
per-item scores that analyse_ratings.py wrote to CAEE_R1/rating/Bench_v2_Rating_Results.xlsx (sheet all_items, the
120 first runs of the full system) and writes experiments/results/bench_v2_group_means.csv.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVERYDAY = ["Live machine state", "Historical sensor data", "Maintenance manual", "Past work orders",
            "Work-order drafting and editing", "General engineering"]
RNG = np.random.default_rng(20260928)


def boot_ci(df, n_boot=5000):
    """Cluster bootstrap over prompts of the mean task-success score (as in analyse_ratings.py)."""
    groups = [g["task"].to_numpy() for _, g in df.groupby("id")]
    means = [np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))]).mean() for _ in range(n_boot)]
    return tuple(np.round(np.percentile(means, [2.5, 97.5]), 3))


items = pd.read_excel(os.path.join(ROOT, "CAEE_R1", "rating", "Bench_v2_Rating_Results.xlsx"), sheet_name="all_items")
full1 = items[(items.variant == "full") & (items.repeat == 1)]
rows = []
for name, part in (("everyday work", full1[full1.category.isin(EVERYDAY)]),
                   ("plausible answer is wrong", full1[~full1.category.isin(EVERYDAY)]),
                   ("all categories", full1)):
    lo, hi = boot_ci(part)
    rows.append({"group": name, "items": len(part), "task_success_mean": round(part.task.mean(), 3), "ci_low": lo, "ci_high": hi})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "experiments", "results", "bench_v2_group_means.csv"), index=False)
print(out.to_string(index=False))
