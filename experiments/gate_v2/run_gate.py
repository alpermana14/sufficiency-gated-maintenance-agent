"""Gate v2 - calibration and evaluation of the retrieval-sufficiency gate.

Answers CAEE R1-8 and R2-18 (larger, independently written query set; the distance floor
is calibrated on a development split and reported on a held-out test split), R2-19 (judge
robustness), R2-21 (embedding model and retrieval architecture) and R2-17 (retrieval depth).

Positive class = a gap: the store cannot answer the query, so the gate should refuse.
Precision  = of the queries the gate refused, the share that truly had no answer.
Recall     = of the queries that truly had no answer, the share the gate refused.

Usage:
  python experiments/gate_v2/run_gate.py --stage floor          # free, no API calls
  python experiments/gate_v2/run_gate.py --stage gate           # adds the entailment judge
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import judges  # noqa: E402
import retrieval  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
K_DEFAULT = 3


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(y_true, y_pred):
    """y_true / y_pred: 1 = gap (insufficient)."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
            "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, tp + fn),
            "accuracy_ci": wilson(tp + tn, tp + fp + fn + tn)}


def auroc(y_true, score):
    from scipy.stats import rankdata
    n1 = int(np.sum(y_true == 1))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load():
    store = retrieval.load_store()
    qs = retrieval.load_queries()["queries"]
    return store, qs


def distances(backend, store, qs, k):
    """Top-k (record index, distance) per query, plus the distance to the closest record."""
    texts = [r["text"] for r in store["records"]]
    ret = retrieval.Retriever(backend, texts)
    ret.prepare([q["query"] for q in qs])
    hits = {q["id"]: ret.search(q["query"], k) for q in qs}
    return hits


def calibrate_operating(dev_y, dev_d, target_false_refusal=0.05):
    """Operating floor: the quantile of the answerable development queries that leaves at most
    `target_false_refusal` of them refused by stage 1.

    Stage 2 can only add refusals, never undo one, so stage 1 must be permissive: its job is to
    reject cheaply what is obviously far away, not to decide sufficiency. Expressing the floor as
    a quantile of the answerable queries also makes the procedure transferable, which matters
    because different embedding models live on different distance scales.
    """
    ok = dev_d[dev_y == 0]
    return float(np.quantile(ok, 1 - target_false_refusal))


def calibrate(dev_y, dev_d, grid=None):
    """Choose the floor that maximises F1 for gap detection on the development split."""
    if grid is None:
        lo, hi = float(np.min(dev_d)), float(np.max(dev_d))
        grid = np.linspace(lo - 0.01, hi + 0.01, 400)
    best = (None, -1.0)
    for t in grid:
        m = prf(dev_y, (dev_d > t).astype(int))
        if m["f1"] > best[1]:
            best = (float(t), m["f1"])
    return best


def evaluate(backend, k, judge, store, qs, use_floor=True, use_judge=True, tau=None,
             tau_mode="operating", judge_splits=("dev", "test"), verbose=True):
    hits = distances(backend, store, qs, k)
    d_min = np.array([hits[q["id"]][0][1] for q in qs])
    y = np.array([0 if q["sufficient"] else 1 for q in qs])
    split = np.array([q["split"] for q in qs])
    dev, test = split == "dev", split == "test"

    if use_floor and tau is None:
        if tau_mode == "f1":
            tau, f1_dev = calibrate(y[dev], d_min[dev])
            note = f"dev F1 {f1_dev:.3f}"
        else:
            tau = calibrate_operating(y[dev], d_min[dev])
            note = "95th percentile of answerable dev queries"
        if verbose:
            print(f"[{backend}] tau calibrated on {dev.sum()} dev queries: {tau:.4f} ({note})")
    pred = (d_min > tau).astype(int) if use_floor else np.zeros(len(qs), dtype=int)

    judged = 0
    if use_judge:
        client = None
        if judge != "nli-local":
            from openai import OpenAI
            client = OpenAI()
        for i, q in enumerate(qs):
            if pred[i] == 1 or q["split"] not in judge_splits:
                continue
            ok = False
            for idx, _ in hits[q["id"]]:
                judged += 1
                if judges.judge_yes(judge, q["query"], store["records"][idx]["text"], client=client):
                    ok = True
                    break
            if not ok:
                pred[i] = 1
    return {"backend": backend, "k": k, "judge": judge if use_judge else None,
            "tau": tau, "use_floor": use_floor, "use_judge": use_judge,
            "judge_calls": judged, "y": y, "pred": pred, "d_min": d_min, "split": split,
            "dev": prf(y[dev], pred[dev]), "test": prf(y[test], pred[test]),
            "auroc_test": auroc(y[test], d_min[test]), "qs": qs}


def per_category(res):
    out = {}
    for cat in sorted({q["category"] for q in res["qs"]}):
        m = np.array([q["category"] == cat and q["split"] == "test" for q in res["qs"]])
        if m.sum():
            correct = int(np.sum(res["pred"][m] == res["y"][m]))
            out[cat] = {"n": int(m.sum()), "correct": correct, "accuracy": correct / int(m.sum())}
    return out


def fmt(m):
    return (f"P {m['precision']:.3f} [{m['precision_ci'][0]:.3f},{m['precision_ci'][1]:.3f}]  "
            f"R {m['recall']:.3f} [{m['recall_ci'][0]:.3f},{m['recall_ci'][1]:.3f}]  "
            f"F1 {m['f1']:.3f}  acc {m['accuracy']:.3f}  "
            f"(tp {m['tp']} fp {m['fp']} fn {m['fn']} tn {m['tn']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="floor", choices=["floor", "gate", "all", "suite"])
    ap.add_argument("--backend", default="openai-small")
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--tau-mode", default="operating", choices=["operating", "f1"])
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.stage == "suite":
        suite()
        return
    retrieval._load_env()
    store, qs = load()
    print(f"store: {len(store['records'])} records | queries: {len(qs)} "
          f"(dev {sum(q['split'] == 'dev' for q in qs)}, test {sum(q['split'] == 'test' for q in qs)})")
    rows = []

    if args.stage in ("floor", "all"):
        print("\n=== stage 1 only: distance floor, calibrated on dev, reported on test ===")
        for backend in retrieval.BACKENDS:
            r = evaluate(backend, args.k, args.judge, store, qs, use_judge=False, tau_mode=args.tau_mode)
            print(f"  {backend:13s} {fmt(r['test'])}  AUROC(test) {r['auroc_test']:.3f}")
            rows.append({"config": f"floor_only/{backend}", "tau": r["tau"],
                         "auroc_test": r["auroc_test"], **{f"test_{k}": v for k, v in r["test"].items()
                                                           if not k.endswith("_ci")}})

    if args.stage in ("gate", "all"):
        print(f"\n=== full gate: floor + entailment ({args.judge}), k = {args.k} ===")
        r = evaluate(args.backend, args.k, args.judge, store, qs, tau_mode=args.tau_mode)
        print(f"  test  {fmt(r['test'])}")
        print(f"  judge calls made: {r['judge_calls']} (cached total {judges.cached_count()})")
        print("  per category (test split):")
        for cat, m in per_category(r).items():
            print(f"    {cat:20s} {m['correct']:3d}/{m['n']:3d}  {m['accuracy']:.3f}")
        rows.append({"config": f"gate/{args.backend}/{args.judge}/k{args.k}", "tau": r["tau"],
                     "auroc_test": r["auroc_test"], **{k: v for k, v in r["test"].items()
                                                       if not k.endswith("_ci")}})

    if rows:
        import pandas as pd
        out = os.path.join(RESULTS, "e21_gate_v2.csv")
        df = pd.DataFrame(rows)
        if os.path.exists(out):
            df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df.to_csv(out, index=False)
        print(f"\n[OK] appended {len(rows)} rows to {out}")




def crag_evaluate(backend, k, judge, store, qs, judge_splits=("test",)):
    """Retrieval-evaluator baseline in the style of corrective retrieval-augmented generation:
    grade every retrieved record independently and refuse only when all of them are graded
    irrelevant. No distance floor is used, which is the point of the comparison."""
    hits = distances(backend, store, qs, k)
    y = np.array([0 if q["sufficient"] else 1 for q in qs])
    pred = np.zeros(len(qs), dtype=int)
    client = None
    if judge != "nli-local":
        from openai import OpenAI
        client = OpenAI()
    for i, q in enumerate(qs):
        if q["split"] not in judge_splits:
            continue
        graded = [judges.judge_yes(judge, q["query"], store["records"][idx]["text"], client=client)
                  for idx, _ in hits[q["id"]]]
        pred[i] = 0 if any(graded) else 1
    split = np.array([q["split"] for q in qs])
    return {"y": y, "pred": pred, "test": prf(y[split == "test"], pred[split == "test"])}


def suite():
    """Everything the reviewers asked for, in one pass. Judge answers are cached on disk."""
    retrieval._load_env()
    store, qs = load()
    rows = []

    def record(config, m, extra=None):
        row = {"config": config, **{k: v for k, v in m.items() if not k.endswith("_ci")}}
        row["precision_lo"], row["precision_hi"] = m["precision_ci"]
        row["recall_lo"], row["recall_hi"] = m["recall_ci"]
        row.update(extra or {})
        rows.append(row)
        print(f"  {config:44s} {fmt(m)}")

    print("\n=== R2-18 / R1-8: stage 1 alone versus the full gate (openai-small, k = 3) ===")
    r = evaluate("openai-small", 3, "gpt-4o-mini", store, qs, use_judge=False, verbose=False)
    record("floor_only", r["test"], {"tau": r["tau"], "auroc_test": r["auroc_test"]})
    base = evaluate("openai-small", 3, "gpt-4o-mini", store, qs, verbose=False)
    record("gate_floor+judge", base["test"], {"tau": base["tau"], "auroc_test": base["auroc_test"]})
    r = evaluate("openai-small", 3, "gpt-4o-mini", store, qs, use_floor=False, verbose=False)
    record("judge_only_no_floor", r["test"])

    print("\n=== R1-1: the gate against simpler alternatives ===")
    r = crag_evaluate("openai-small", 3, "gpt-4o-mini", store, qs)
    record("retrieval_evaluator_crag_style", r["test"])
    for t in (0.9, 1.0, 1.1, 1.2):
        r = evaluate("openai-small", 3, "gpt-4o-mini", store, qs, use_judge=False, tau=t, verbose=False)
        record(f"similarity_threshold_tau{t}", r["test"], {"tau": t})

    print("\n=== R2-19: judge robustness (test split only) ===")
    for j in ["gpt-4o-mini", "nli-local", "gpt-4o"]:
        r = evaluate("openai-small", 3, j, store, qs, judge_splits=("test",), verbose=False)
        record(f"gate_judge_{j}", r["test"], {"judge_calls": r["judge_calls"]})

    print("\n=== R2-21: embedding model and retrieval architecture ===")
    for b in retrieval.BACKENDS:
        r = evaluate(b, 3, "gpt-4o-mini", store, qs, verbose=False)
        record(f"gate_backend_{b}", r["test"], {"tau": r["tau"], "auroc_test": r["auroc_test"]})

    print("\n=== R2-17: retrieval depth ===")
    for k in (1, 2, 3, 5, 8):
        r = evaluate("openai-small", k, "gpt-4o-mini", store, qs, verbose=False)
        record(f"gate_k{k}", r["test"], {"k": k, "judge_calls": r["judge_calls"]})

    print("\n=== per category, main configuration ===")
    cats = per_category(base)
    for cat, m in cats.items():
        print(f"  {cat:20s} {m['correct']:3d}/{m['n']:3d}  {m['accuracy']:.3f}")

    import pandas as pd
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "e21_gate_v2_suite.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame([{"category": c, **m} for c, m in cats.items()]).to_csv(
        os.path.join(RESULTS, "e21_gate_v2_categories.csv"), index=False)
    print(f"\n[OK] wrote {len(rows)} configurations to {out}")
    print(f"judge answers cached: {judges.cached_count()}")


if __name__ == "__main__":
    main()
