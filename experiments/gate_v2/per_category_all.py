"""Per-category correct decisions on the test split for every configuration of Table 7 (reads the caches only)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval  # noqa: E402
from run_gate import RESULTS, evaluate, load, per_category  # noqa: E402

retrieval._load_env()
store, qs = load()
configs = {
    "similarity_threshold_tau1.0": dict(use_judge=False, tau=1.0),
    "distance_floor_alone": dict(use_judge=False),
    "entailment_check_alone": dict(use_floor=False),
    "gate": dict(),
}
rows = []
for name, kw in configs.items():
    r = evaluate("openai-small", 3, "gpt-4o-mini", store, qs, verbose=False, **kw)
    for cat, m in per_category(r).items():
        rows.append({"config": name, "category": cat, "n": m["n"], "correct": m["correct"]})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "e21_gate_v2_categories_all.csv"), index=False)
print(out.pivot(index="category", columns="config", values="correct").to_string())
print(out.groupby("category").n.first().to_string())
