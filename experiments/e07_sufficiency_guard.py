"""E7 - Retrieval-sufficiency guard: does it stop cold-start hallucination?

Answers R4-15 (hallucination mitigation) and R3-1 (the "no similar record"
branch of eq. 7b). Directly targets the manuscript's own documented failure
(prompts 12 & 13): with only a load-change record in the store, bearing-fault
and overheating queries retrieve that irrelevant record and the agent then
fabricates history.

Two-level evaluation:

  LEVEL 1 - the guard as a gap detector (deterministic, embeddings only, ~free)
     Seed a controlled store, then classify a labelled set of in-coverage vs
     out-of-coverage queries. Report score separation, AUROC of the distance
     floor, and precision/recall/F1 of gap detection for floor-only and
     floor+entailment. This is the quantitative backbone.

  LEVEL 2 - end-to-end effect on the paper's failure prompts (few GPT-4o calls)
     For each out-of-coverage prompt, feed the SAME retrieval to a copilot
     answer step with the guard OFF (returns nearest irrelevant record) and ON
     (returns the sentinel), and have an LLM judge label each answer as
     grounded-refusal / fabricated-history / general-advice. Shows the guard
     converts fabrication into honest gap acknowledgement.

Uses an ISOLATED store at experiments/e07_history_db (never touches the
production maintenance_history_db). Rebuilt fresh each run.

Usage:  python experiments/e07_sufficiency_guard.py
  env:  RAG_GUARD_TAU (distance floor), RAG_GUARD_ENTAILMENT=1 to add stage 2
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
for _p in (BACKEND_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.metrics import roc_auc_score

import rag_guard
from common import ensure_results_dir

STORE_DIR = os.path.join(REPO_ROOT, "experiments", "e07_history_db")

# --- Controlled seed: 3 distinct, realistic approved work orders -------------
SEED_ORDERS = [
    ("work_order_2026_05_05",
     "Incident Report. Timestamp: 2026-05-05 15:00:00. Vibration: 3.54 mm/s. "
     "ISO Zone C (Unsatisfactory). IDK Status: Anomaly. Root Cause Analysis: "
     "distributional shift in z_rms, motor current and acoustic noise "
     "consistent with a large increase in conveyor load (approx 40 kg). "
     "Recommended Actions: 1) verify load distribution on the belt; "
     "2) confirm the load is within rated capacity; 3) monitor z_rms for "
     "return to baseline after load removal; 4) no mechanical intervention "
     "required. Priority: High."),
    ("work_order_2026_03_11",
     "Incident Report. Timestamp: 2026-03-11 09:20:00. Root Cause Analysis: "
     "belt mistracking causing intermittent edge contact and elevated x_rms. "
     "Recommended Actions: 1) adjust tail pulley tracking bolts; 2) re-tension "
     "the belt to spec; 3) inspect belt edges for fraying. Priority: Medium."),
    ("work_order_2026_01_22",
     "Incident Report. Timestamp: 2026-01-22 14:05:00. Root Cause Analysis: "
     "scheduled preventive maintenance. Recommended Actions: 1) lubricate "
     "drive-end and idler rollers; 2) check coupling alignment; 3) clean "
     "accumulated debris from the frame. Priority: Low."),
]

# label 0 = in-coverage (answerable from seed) -> guard should say SUFFICIENT
# label 1 = out-of-coverage (gap)              -> guard should say INSUFFICIENT
QUERIES = [
    ("Have we dealt with high vibration caused by a load change before?", 0),
    ("What was done previously about belt tracking or tension problems?", 0),
    ("Is there any past record of routine roller lubrication?", 0),
    ("Have we seen an anomaly that turned out to be extra load, not a fault?", 0),
    # out-of-coverage — the paper's actual failure cases and neighbours
    ("What was done the last time there was a bearing fault on this machine?", 1),
    ("Summarise all past maintenance records related to motor overheating.", 1),
    ("Have we had a motor winding insulation failure before? What was done?", 1),
    ("What past work orders mention gearbox oil contamination?", 1),
    ("Show me previous records of an emergency stop button failure.", 1),
]

OUT_OF_COVERAGE_PROMPTS = [q for q, lbl in QUERIES if lbl == 1][:3]


def build_store(embeddings) -> Chroma:
    if os.path.exists(STORE_DIR):
        shutil.rmtree(STORE_DIR)
    vs = Chroma(persist_directory=STORE_DIR, embedding_function=embeddings)
    vs.add_documents([
        Document(page_content=c, metadata={"id": wid, "status": "human_approved"})
        for wid, c in SEED_ORDERS
    ])
    return vs


def level1(vs, judge_llm) -> pd.DataFrame:
    """Guard-as-gap-detector, floor-only and floor+entailment."""
    rows = []
    for query, label in QUERIES:
        dws = vs.similarity_search_with_score(query, k=3)
        best_dist = min(d for _, d in dws)

        os.environ["RAG_GUARD_ENTAILMENT"] = "0"
        floor = rag_guard.assess(query, dws)
        os.environ["RAG_GUARD_ENTAILMENT"] = "1"
        ent = rag_guard.assess(query, dws, llm=judge_llm)
        os.environ["RAG_GUARD_ENTAILMENT"] = "0"

        rows.append({
            "query": query,
            "is_gap": label,
            "best_distance": best_dist,
            "floor_flags_gap": int(not floor.sufficient),
            "entail_flags_gap": int(not ent.sufficient),
            "entail_stage": ent.stage,
        })
    return pd.DataFrame(rows)


def _prf(df, pred_col):
    y = df["is_gap"].to_numpy()
    p = df[pred_col].to_numpy()
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


ANSWER_SYS = (
    "You are a predictive maintenance copilot for an industrial conveyor. "
    "Answer the operator's question about PAST maintenance using only the "
    "retrieved work order history provided. Be truthful and concise."
)
JUDGE_SYS = (
    "You label a maintenance copilot's answer to a question about PAST work "
    "order history. The history did NOT contain any record relevant to the "
    "question. Choose exactly one label:\n"
    "FABRICATED_HISTORY - the answer asserts a specific past incident, date, "
    "root cause, or action as if it were in the records.\n"
    "GROUNDED_REFUSAL - the answer states there is no relevant past record.\n"
    "GENERAL_ADVICE - the answer gives only clearly-general guidance without "
    "claiming it came from past records.\n"
    "Reply with exactly one label."
)


def level2(vs, answer_llm, judge_llm) -> pd.DataFrame:
    rows = []
    for prompt in OUT_OF_COVERAGE_PROMPTS:
        dws = vs.similarity_search_with_score(prompt, k=3)
        nearest = "\n".join(f"record: {d.page_content}" for d, _ in dws)

        # guard OFF: agent sees the nearest (irrelevant) record, no gating
        _, guarded_payload, decision = rag_guard.guarded_retrieve(vs, prompt, k=3)

        for cond, tool_output in [("guard_off", nearest),
                                  ("guard_on", guarded_payload)]:
            msg = (f"{ANSWER_SYS}\n\nRetrieved work order history:\n{tool_output}\n\n"
                   f"Operator question: {prompt}")
            answer = answer_llm.invoke(msg).content.strip()
            label = judge_llm.invoke(
                f"{JUDGE_SYS}\n\nQUESTION: {prompt}\n\nANSWER: {answer}"
            ).content.strip().upper()
            rows.append({
                "prompt": prompt, "condition": cond,
                "guard_stage": decision.stage,
                "label": label,
                "answer": answer[:300],
            })
    return pd.DataFrame(rows)


def level2b_weak_model(embeddings, judge_llm) -> pd.DataFrame:
    """The decisive end-to-end test. Reproduce the paper's cold-start failure
    (single record in the store) across models of different strength, under
    three guard settings: off, floor-only, floor+entailment. Shows the guard's
    value is model-dependent: strong models rarely hallucinate (so the guard
    adds determinism/audit), but a weak model — the proxy for the cheap/on-prem
    LLM reviewers asked us to support (R1-5) — DOES hallucinate, and only the
    full floor+entailment guard eliminates it."""
    single_dir = os.path.join(REPO_ROOT, "experiments", "e07_single_db")
    if os.path.exists(single_dir):
        shutil.rmtree(single_dir, ignore_errors=True)
    vs = Chroma(persist_directory=single_dir, embedding_function=embeddings)
    vs.add_documents([Document(page_content=SEED_ORDERS[0][1],
                               metadata={"id": SEED_ORDERS[0][0]})])  # ONLY the load-change record

    models = [m.strip() for m in
              os.getenv("E07_MODELS", "gpt-4o,gpt-3.5-turbo").split(",") if m.strip()]
    settings = [("off", None), ("floor_only", "0"), ("floor+entail", "1")]
    rows = []
    for model_name in models:
        try:
            ans = ChatOpenAI(model=model_name, temperature=0.0)
        except Exception as e:
            print(f"  [skip] {model_name}: {e}")
            continue
        for setting, entail in settings:
            fab = 0
            for prompt in OUT_OF_COVERAGE_PROMPTS:
                if setting == "off":
                    dws = vs.similarity_search_with_score(prompt, k=3)
                    payload = "\n".join(f"record: {d.page_content}" for d, _ in dws)
                else:
                    os.environ["RAG_GUARD_ENTAILMENT"] = entail
                    _, payload, _ = rag_guard.guarded_retrieve(
                        vs, prompt, k=3, llm=judge_llm)
                a = ans.invoke(
                    f"{ANSWER_SYS}\n\nRetrieved work order history:\n{payload}"
                    f"\n\nOperator question: {prompt}").content.strip()
                lbl = judge_llm.invoke(
                    f"{JUDGE_SYS}\n\nQUESTION: {prompt}\n\nANSWER: {a}"
                ).content.strip().upper()
                fab += int("FABRICAT" in lbl)
            rows.append({"model": model_name, "guard": setting,
                         "fabricated": fab, "n": len(OUT_OF_COVERAGE_PROMPTS)})
            print(f"  {model_name:14s} {setting:12s} fabricated {fab}/"
                  f"{len(OUT_OF_COVERAGE_PROMPTS)}")
    os.environ["RAG_GUARD_ENTAILMENT"] = "1"
    shutil.rmtree(single_dir, ignore_errors=True)
    return pd.DataFrame(rows)


def main() -> None:
    results_dir = ensure_results_dir()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    answer_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    judge_llm = ChatOpenAI(model=rag_guard.judge_model(), temperature=0.0)

    vs = build_store(embeddings)
    print(f"[INFO] seeded {len(SEED_ORDERS)} work orders | tau={rag_guard.tau():.3f}")

    # ---- Level 1 ----
    l1 = level1(vs, judge_llm)
    l1.to_csv(f"{results_dir}/e07_level1.csv", index=False)
    in_d = l1.loc[l1.is_gap == 0, "best_distance"]
    out_d = l1.loc[l1.is_gap == 1, "best_distance"]
    auroc = roc_auc_score(l1["is_gap"], l1["best_distance"])  # higher dist = gap
    print("\n=== LEVEL 1: guard as gap detector ===")
    print(f"best-distance  in-coverage  mean={in_d.mean():.3f} (max {in_d.max():.3f})")
    print(f"best-distance out-of-cov.   mean={out_d.mean():.3f} (min {out_d.min():.3f})")
    print(f"AUROC of distance floor for gap detection: {auroc:.3f}")
    print(f"floor-only     : {_prf(l1, 'floor_flags_gap')}")
    print(f"floor+entail   : {_prf(l1, 'entail_flags_gap')}")
    print("\nper-query:")
    print(l1[["is_gap", "best_distance", "floor_flags_gap",
              "entail_flags_gap", "query"]].round(3).to_string(index=False))

    # ---- Level 2 ----
    print("\n=== LEVEL 2: end-to-end before/after (GPT-4o) ===")
    l2 = level2(vs, answer_llm, judge_llm)
    l2.to_csv(f"{results_dir}/e07_level2.csv", index=False)
    summary = (l2.groupby(["condition", "label"]).size()
               .unstack(fill_value=0))
    print(summary.to_string())
    print("\ntranscripts:")
    for _, r in l2.iterrows():
        print(f"\n[{r['condition']}] ({r['label']}) Q: {r['prompt']}")
        print(f"   A: {r['answer']}")

    # ---- Level 2b: decisive weak-model cold-start test ----
    if os.getenv("E07_WEAK_MODEL_TEST", "1").lower() in {"1", "true", "yes"}:
        print("\n=== LEVEL 2b: single-record cold start across model strengths ===")
        l2b = level2b_weak_model(embeddings, judge_llm)
        l2b.to_csv(f"{results_dir}/e07_level2b.csv", index=False)

    print(f"\n[OK] wrote e07_level1.csv, e07_level2.csv, e07_level2b.csv to {results_dir}")


if __name__ == "__main__":
    main()
