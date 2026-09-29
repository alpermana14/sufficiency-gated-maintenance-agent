"""Analyse the human ratings of agent benchmark v2 (and the AI rating, if present).

Reads CAEE_R1/rating/Bench_v2_Rating_Rater1.xlsx and Rater2.xlsx, the item key and prompts.json, and writes
CAEE_R1/rating/Bench_v2_Rating_Results.xlsx with: agreement between the raters (Cohen's kappa, unweighted and
linear-weighted), the items they scored differently (for adjudication), task success per category with a cluster
bootstrap interval over prompts, safe behaviour and grounding, the integration ablation with paired tests,
repeat and paraphrase consistency, and the tool-selection error taxonomy.

Scores until adjudication are the mean of the two raters; the results sheet also reports each rater separately.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RATING = os.path.join(ROOT, "CAEE_R1", "rating")
COLS = {"task": "task_success (0/1/2)", "safe": "safe_behaviour (0/1)", "ground": "grounding (0/1/n/a)"}
RNG = np.random.default_rng(20260916)


def load_rater(n):
    df = pd.read_excel(os.path.join(RATING, f"Bench_v2_Rating_Rater{n}.xlsx"))
    out = pd.DataFrame({"item": df["item"]})
    out[f"task{n}"] = pd.to_numeric(df[COLS["task"]], errors="coerce")
    for k, c in (("safe", COLS["safe"]), ("ground", COLS["ground"])):
        out[f"{k}{n}"] = pd.to_numeric(df[c].replace("n/a", np.nan), errors="coerce")
    out[f"comment{n}"] = df["comment"]
    return out


def kappa(a, b, weighted=False):
    """Cohen's kappa; linear weights when weighted=True."""
    m = pd.notna(a) & pd.notna(b)
    a, b = np.asarray(a[m], dtype=float), np.asarray(b[m], dtype=float)
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return None, int(m.sum())
    idx = {c: i for i, c in enumerate(cats)}
    n, k = len(a), len(cats)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= n
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weighted:
        w = np.array([[abs(cats[i] - cats[j]) / (max(cats) - min(cats)) for j in range(k)] for i in range(k)])
        po, pe = 1 - (w * obs).sum(), 1 - (w * exp).sum()
    else:
        po, pe = np.trace(obs), np.trace(exp)
    return (round((po - pe) / (1 - pe), 3) if pe < 1 else None), n


def boot_ci(df, value="task", by="id", n_boot=5000):
    """Cluster bootstrap over prompts of the mean score."""
    groups = [g[value].to_numpy() for _, g in df.groupby(by)]
    if not groups:
        return (None, None)
    means = [np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))]).mean() for _ in range(n_boot)]
    return tuple(np.round(np.percentile(means, [2.5, 97.5]), 3))


def wilcoxon_signed_rank(d):
    """Two-sided exact-ish Wilcoxon signed-rank test on paired differences (normal approximation with ties)."""
    d = np.asarray([x for x in d if x != 0], dtype=float)
    n = len(d)
    if n == 0:
        return None, None, 0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_plus = ranks[d > 0].sum()
    mu, sigma = n * (n + 1) / 4, np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma if sigma else 0.0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return round(float(w_plus), 1), round(float(p), 4), n


def main():
    key = pd.read_excel(os.path.join(RATING, "Bench_v2_Rating_Key.xlsx"))
    prompts = {p["id"]: p for p in json.load(open(os.path.join(HERE, "prompts.json"), encoding="utf-8"))}
    df = key.merge(load_rater(1), on="item").merge(load_rater(2), on="item")
    df["category"] = df["id"].map(lambda i: prompts[i]["category"])
    df["safety_critical"] = df["id"].map(lambda i: prompts[i]["safety_critical"])
    df["prompt"] = df["id"].map(lambda i: prompts[i]["prompt"])
    df["task"] = df[["task1", "task2"]].mean(axis=1)
    df["safe"] = df[["safe1", "safe2"]].mean(axis=1)
    df["ground"] = df[["ground1", "ground2"]].mean(axis=1)
    ai_path = os.path.join(RATING, "Bench_v2_Rating_AI_Claude.xlsx")
    if os.path.exists(ai_path):
        ai = pd.read_excel(ai_path, sheet_name="AI_rater_Claude")
        df = df.merge(pd.DataFrame({"item": ai["item"],
                                    "task_ai": pd.to_numeric(ai[COLS["task"]], errors="coerce"),
                                    "safe_ai": pd.to_numeric(ai[COLS["safe"]].replace("n/a", np.nan), errors="coerce"),
                                    "ground_ai": pd.to_numeric(ai[COLS["ground"]].replace("n/a", np.nan), errors="coerce")}),
                      on="item", how="left")

    agreement = []
    for label, a, b in [("rater1 vs rater2", "task1", "task2"), ("rater1 vs AI", "task1", "task_ai"),
                        ("rater2 vs AI", "task2", "task_ai")]:
        if b in df:
            k, n = kappa(df[a], df[b])
            kw, _ = kappa(df[a], df[b], weighted=True)
            agreement.append({"scores": "task success", "pair": label, "n": n, "exact_agreement": round((df[a] == df[b]).mean(), 3),
                              "cohen_kappa": k, "linear_weighted_kappa": kw})
    for name, cols in (("safe behaviour", ("safe1", "safe2", "safe_ai")), ("grounding", ("ground1", "ground2", "ground_ai"))):
        pairs = [("rater1 vs rater2", cols[0], cols[1])] + ([("rater1 vs AI", cols[0], cols[2]), ("rater2 vs AI", cols[1], cols[2])] if cols[2] in df else [])
        for label, a, b in pairs:
            k, n = kappa(df[a], df[b])
            m = pd.notna(df[a]) & pd.notna(df[b])
            agreement.append({"scores": name, "pair": label, "n": n,
                              "exact_agreement": round((df.loc[m, a] == df.loc[m, b]).mean(), 3) if m.any() else None,
                              "cohen_kappa": k, "linear_weighted_kappa": None})
    agreement = pd.DataFrame(agreement)

    disagree = df[(df.task1 != df.task2) | (df.safe1.fillna(-1) != df.safe2.fillna(-1)) | (df.ground1.fillna(-1) != df.ground2.fillna(-1))]
    disagree = disagree[["item", "id", "variant", "repeat", "category", "prompt", "task1", "task2", "safe1", "safe2",
                         "ground1", "ground2", "comment1", "comment2"]].sort_values("item")

    full1 = df[(df.variant == "full") & (df.repeat == 1)]

    def summ(d):
        s, g = d.safe.dropna(), d.ground.dropna()
        lo, hi = boot_ci(d)
        return pd.Series({"items": len(d), "task_success_mean": round(d.task.mean(), 3), "ci_low": lo, "ci_high": hi,
                          "share_full_success": round((d.task == 2).mean(), 3), "share_failure": round((d.task == 0).mean(), 3),
                          "rater1_mean": round(d.task1.mean(), 3), "rater2_mean": round(d.task2.mean(), 3),
                          "safe_rate": round(s.mean(), 3) if len(s) else None, "safe_items": len(s),
                          "grounding_rate": round(g.mean(), 3) if len(g) else None, "grounding_items": len(g)})

    overall = pd.DataFrame([summ(full1)])
    by_cat = full1.groupby("category").apply(summ).reset_index()

    abl_ids = [i for i in prompts if prompts[i].get("ablation")]
    abl = df[df.id.isin(abl_ids) & (((df.variant == "full") & (df.repeat == 1)) | (df.variant != "full"))]
    by_var = abl.groupby("variant").apply(summ).reset_index()
    pivot = abl.pivot_table(index="id", columns="variant", values="task")
    tests = []
    for v in ("no_state", "raw_readings"):
        if v in pivot:
            d = (pivot["full"] - pivot[v]).dropna()
            w, p, n = wilcoxon_signed_rank(d)
            tests.append({"comparison": f"full vs {v}", "prompts": int(len(d)), "mean_difference": round(float(d.mean()), 3),
                          "prompts_better_with_full": int((d > 0).sum()), "prompts_worse_with_full": int((d < 0).sum()),
                          "wilcoxon_W": w, "p_value": p, "non_tied_pairs": n})
    tests = pd.DataFrame(tests)

    rep_ids = df[(df.variant == "full") & (df.repeat > 1)].id.unique()
    rep = df[(df.variant == "full") & df.id.isin(rep_ids)]
    rep_tab = rep.groupby("id").agg(runs=("task", "size"), scores=("task", lambda s: ", ".join(map(str, s))),
                                    identical=("task", lambda s: s.nunique() == 1)).reset_index()
    para = [{"prompt": pid, "paraphrase_of": p["paraphrase_of"],
             "score": full1.loc[full1.id == pid, "task"].mean(), "score_original": full1.loc[full1.id == p["paraphrase_of"], "task"].mean()}
            for pid, p in prompts.items() if p["paraphrase_of"]]
    para = pd.DataFrame(para)
    para["same_score"] = para.score == para.score_original

    tool = full1.groupby("tool_selection_correct").apply(summ).reset_index()
    err = full1[~full1.tool_selection_correct].groupby(full1.tool_error_type.str.split(":").str[0]).agg(
        prompts=("id", lambda s: ", ".join(sorted(s))), n=("id", "size"), task_success_mean=("task", "mean")).reset_index()
    safety_fail = df[(df.safe < 1) & df.safety_critical][["item", "id", "variant", "repeat", "category", "prompt", "safe1", "safe2", "task"]]

    out = os.path.join(RATING, "Bench_v2_Rating_Results.xlsx")
    with pd.ExcelWriter(out) as xw:
        overall.to_excel(xw, sheet_name="overall_120", index=False)
        by_cat.to_excel(xw, sheet_name="by_category", index=False)
        agreement.to_excel(xw, sheet_name="rater_agreement", index=False)
        disagree.to_excel(xw, sheet_name="disagreements", index=False)
        by_var.to_excel(xw, sheet_name="ablation", index=False)
        tests.to_excel(xw, sheet_name="ablation_tests", index=False)
        rep_tab.to_excel(xw, sheet_name="repeats", index=False)
        para.to_excel(xw, sheet_name="paraphrases", index=False)
        tool.to_excel(xw, sheet_name="tool_selection", index=False)
        err.to_excel(xw, sheet_name="tool_error_types", index=False)
        safety_fail.to_excel(xw, sheet_name="safety_failures", index=False)
        df.to_excel(xw, sheet_name="all_items", index=False)

    pd.set_option("display.width", 220)
    print(agreement.to_string(index=False))
    print("\ndisagreements:", len(disagree), "of", len(df))
    print(overall.to_string(index=False))
    print(by_cat.to_string(index=False))
    print(by_var.to_string(index=False))
    print(tests.to_string(index=False))
    print("repeat prompts with identical score in all runs:", int(rep_tab.identical.sum()), "/", len(rep_tab))
    print("paraphrase pairs with the same score:", int(para.same_score.sum()), "/", len(para))
    print(tool.to_string(index=False))
    print(err.to_string(index=False))
    print("safety failures (any rater):", len(safety_fail), safety_fail.id.tolist())
    print("[OK]", out)


if __name__ == "__main__":
    main()
