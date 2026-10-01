"""E33 - Sensitivity of the agent to the number of manual passages retrieved (R2-17).

Reviewer 2, Comment 17 asks why the manual search returns two passages (k = 2) and how answer quality and
hallucination risk change with other depths. The work-order depth is answered by Table 8; this script answers the
manual depth.

Design
  * The ten "Maintenance manual" requests of the agent benchmark (MP-01 to MP-10, machine state S1, store H1) are
    run through the same agent graph as the benchmark (agent_benchmark_v2/run_benchmark.py). Only the manual search
    changes: it returns the k nearest passages, k in {1, 2, 3, 5, 8}. Everything else is as deployed.
  * Evidence recall: for each request the manual passages that hold the answer were fixed before any run (GOLD
    below, matched by text so they survive a re-ingestion). A run scores the share of those passages that any of its
    manual searches returned. MP-07 asks for a value the manual does not give and has no evidence passage.
  * Fact coverage: the share of the facts the manual gives for the request (FACTS below) that the answer states.
  * Grounding: each answer is classified in one call, as in the end-to-end evaluation of the gate (gpt-4o-mini, the
    same classifier), into ABSTAIN, GROUNDED or FABRICATED against the passages the agent actually received.
  * A retrieval-only sweep runs the request text itself through the manual search at k = 1 to 8, with no language
    model, as a deterministic check of the same recall.

Usage (repository root; needs OPENAI_API_KEY and the benchmark fixtures):
  OPENAI_BUDGET_USD=<cap> python experiments/e33_manual_depth.py
Env: E33_KS ("1,2,3,5,8"), E33_REPEATS (1), E33_PAUSE_S (5), E33_IDS (subset of request ids)
Outputs (experiments/results): e33_manual_depth_runs.jsonl (resumable), e33_manual_depth_runs.csv,
  e33_manual_depth_summary.csv, e33_manual_retrieval.csv
"""

import json
import os
import re
import sys
import time

import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "agent_benchmark_v2"))
import run_benchmark as RB  # noqa: E402  (imports the backend with a temporary audit log and store)
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from openai import OpenAI  # noqa: E402

CE, G, budget = RB.CE, RB.G, RB.budget
RES = os.path.join(EXP, "results")
OUT = os.path.join(RES, "e33_manual_depth_runs.jsonl")
KS = [int(k) for k in os.environ.get("E33_KS", "1,2,3,5,8").split(",")]
REPEATS = int(os.environ.get("E33_REPEATS", "1"))
PAUSE_S = float(os.environ.get("E33_PAUSE_S", "5"))
ONLY = [s.strip() for s in os.environ.get("E33_IDS", "").split(",") if s.strip()]
CLASSIFIER = "gpt-4o-mini"

# Evidence passages per request: each entry is one passage the answer needs, given as a pattern of its text.
GOLD = {
    "MP-01": [r"Jerky running"],
    "MP-02": [r"silicone"],
    "MP-03": [r"electricity\s+supply is switched off", r"Remove all products"],
    "MP-04": [r"Check sprocket, chain return\s+guide"],
    "MP-05": [r"Abnormal wear on\s+plastic parts", r"ingress of contaminate"],
    "MP-06": [r"slack in chain can be checked at the drive unit", r"pre-tensioned while the conveyor is stationary",
              r"removing the necessary number of chain\s+links"],
    "MP-07": [],
    "MP-08": [r"Friction disc in slip clutch"],
}
GOLD["MP-09"], GOLD["MP-10"] = GOLD["MP-01"], GOLD["MP-03"]

# Facts the manual gives for each request, matched in the answer (case-insensitive).
FACTS = {
    "MP-01": [r"slide rail", r"tight|loose|tension", r"dirt|clean"],
    "MP-02": [r"silicone"],
    "MP-03": [r"electric|power supply", r"lock", r"pneumatic|compressed air|air supply",
              r"remov\w* (all )?(the )?products|products (are |must be )?removed"],
    "MP-04": [r"\b50\b", r"\b250\b", r"\b500\b"],
    "MP-05": [r"overload", r"temperature", r"chemical", r"contamina|particle|swarf"],
    "MP-06": [r"drive unit", r"stationary", r"slack", r"link"],
    "MP-07": [],
    "MP-08": [r"replac", r"check|inspect"],
}
FACTS["MP-09"], FACTS["MP-10"] = FACTS["MP-01"], FACTS["MP-03"]

VERDICT = ("An operator asked a question about the maintenance manual of a conveyor. A system searched the manual "
           "and answered.\n\nManual passages returned by the search:\n{evidence}\n\nQuestion: {query}\n\n"
           "Answer: {answer}\n\nClassify the answer with exactly one word:\n"
           "ABSTAIN - it says the manual passages do not contain what was asked and attributes nothing to the manual;\n"
           "GROUNDED - everything it attributes to the manual appears in the passages above;\n"
           "FABRICATED - it attributes to the manual content, steps, values or causes the passages do not contain.\n"
           "General advice that the answer does not attribute to the manual is not manual content; ignore it.")


def recall(gold, texts):
    if not gold:
        return None
    joined = "\n".join(texts)
    return sum(bool(re.search(g, joined, re.S)) for g in gold) / len(gold)


def coverage(facts, answer):
    if not facts:
        return None
    return sum(bool(re.search(f, answer, re.I)) for f in facts) / len(facts)


def make_tools(k, store, drafts, log):
    base = {t.name: t for t in RB.make_tools("none", store, drafts)}

    def manual(query: str) -> str:
        docs = CE.vectorstore_manual.similarity_search(query, k=k)
        log.append({"query": query, "passages": [d.page_content for d in docs]})
        return "\n".join(d.page_content for d in docs)

    t = base["retriever_tool"]
    base["retriever_tool"] = StructuredTool.from_function(func=manual, name=t.name, description=t.description,
                                                         args_schema=t.args_schema)
    return list(base.values())


def retrieval_sweep(prompts):
    rows = []
    for p in prompts:
        for k in range(1, 9):
            texts = [d.page_content for d in CE.vectorstore_manual.similarity_search(p["prompt"], k=k)]
            rows.append({"id": p["id"], "k": k, "evidence_recall": recall(GOLD[p["id"]], texts)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e33_manual_retrieval.csv"), index=False)
    print("\n=== retrieval only, request text as query: mean evidence recall per k ===")
    print(df.dropna().groupby("k")["evidence_recall"].mean().round(3).to_string())


def main():
    t0 = time.time()
    prompts = [p for p in json.load(open(os.path.join(EXP, "agent_benchmark_v2", "prompts.json"), encoding="utf-8"))
               if p["category"] == "Maintenance manual"]
    assert len(prompts) == 10 and all(p["id"] in GOLD for p in prompts)
    retrieval_sweep(prompts)
    if ONLY:
        prompts = [p for p in prompts if p["id"] in ONLY]

    fx = json.load(open(os.path.join(EXP, "agent_benchmark_v2", "fixtures.json"), encoding="utf-8"))
    states, stores = RB.build_states(fx), RB.build_stores(fx)
    client = OpenAI()
    done = set()
    if os.path.exists(OUT):
        done = {(r["id"], r["k"], r["repeat"]) for r in map(json.loads, open(OUT, encoding="utf-8"))}
    print(f"\n[INFO] {len(prompts)} requests x k {KS} x {REPEATS} repeat(s); {len(done)} runs already done", flush=True)

    for k in KS:
        for p in prompts:
            for rep in range(1, REPEATS + 1):
                if (p["id"], k, rep) in done:
                    continue
                session = f"e33_{p['id']}_k{k}_{rep}"
                drafts, log = {}, []
                ctx = dict(states[p["state"]], session_id=session, current_draft_text="")
                graph = RB.make_graph(make_tools(k, stores[p["store"]], drafts, log))

                def run():
                    log.clear()
                    start = time.time()
                    out = graph.invoke({"messages": [HumanMessage(content=p["prompt"])], "machine_state": ctx})
                    return out, time.time() - start

                budget.check(0.04)
                with budget.track("e33", session):
                    out, latency = G.with_retry(run, session)
                answer = out["messages"][-1].content
                texts = [x for c in log for x in c["passages"]]
                evidence = "\n\n".join(texts) if texts else "(the manual was not searched)"
                word = client.chat.completions.create(
                    model=CLASSIFIER, temperature=0, max_tokens=4,
                    messages=[{"role": "user", "content": VERDICT.format(evidence=evidence, query=p["prompt"],
                                                                         answer=answer)}]
                ).choices[0].message.content.strip().upper()
                grounding = "abstain" if word.startswith("ABS") else "grounded" if word.startswith("GROU") else "fabricated"
                row = {"id": p["id"], "k": k, "repeat": rep, "prompt": p["prompt"], "manual_searches": len(log),
                       "passages_received": len(texts), "evidence_recall": recall(GOLD[p["id"]], texts),
                       "fact_coverage": coverage(FACTS[p["id"]], answer), "grounding": grounding,
                       "searches": log, "answer": answer, "latency_s": round(latency, 2)}
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {session}: searches={len(log)} recall={row['evidence_recall']} "
                      f"coverage={row['fact_coverage']} {grounding} ({latency:.1f} s)", flush=True)
                time.sleep(PAUSE_S)

    df = pd.DataFrame([json.loads(line) for line in open(OUT, encoding="utf-8")])
    df.drop(columns=["searches"]).to_csv(OUT.replace(".jsonl", ".csv"), index=False)
    summ = df.groupby("k").agg(runs=("id", "size"), evidence_recall=("evidence_recall", "mean"),
                               fact_coverage=("fact_coverage", "mean"),
                               grounded=("grounding", lambda s: int((s == "grounded").sum())),
                               abstain=("grounding", lambda s: int((s == "abstain").sum())),
                               fabricated=("grounding", lambda s: int((s == "fabricated").sum())),
                               passages_received=("passages_received", "mean"),
                               latency_s=("latency_s", "mean")).round(3)
    summ.to_csv(os.path.join(RES, "e33_manual_depth_summary.csv"))
    print("\n=== agent runs per k ===")
    print(summ.to_string())
    print(f"[TOTAL] {len(df)} runs ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
