"""Gate v2, end to end - what reaches the operator, under three levels of protection.

Answers the part of CAEE R1-1 that asks how the gate differs from abstention mechanisms and
from hallucination-detection methods. Those act on an answer that already exists or rely on the
model following an instruction; the gate decides before the answer is written. The measure here
is therefore the answer itself, not the gate decision.

Three conditions on the 120 test queries, identical retrieval (three nearest records):
  naive                 the records are supplied and the model is asked to answer;
  prompt_abstain        the same, plus the instruction to say so when the records do not contain
                        it - the abstention mechanism named in Reviewer 1, Comment 1;
  gate                  the gate decides first, and on a refusal the model is told no record
                        exists; when it passes, the naive prompt is used, so this arm isolates
                        the gate from the instruction;
  gate_and_instruction  the configuration the deployed system actually runs: the gate decides,
                        and the answering prompt also carries the instruction.

The isolated arm answers whether the gate contributes anything of its own; the deployed arm
answers what the system described in this paper actually does. Both are reported.

Each answer is classified in one call into abstain, grounded or fabricated, against the records
that were supplied. Outcomes are reported separately for queries the store can answer and
queries it cannot, using the labels fixed in build_queries.py before any run.

Usage: python experiments/gate_v2/run_endtoend.py
Output: experiments/results/e22_endtoend.jsonl, e22_endtoend_summary.csv
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judges  # noqa: E402
import retrieval  # noqa: E402
import run_gate  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
RUNS = os.path.join(RESULTS, "e22_endtoend.jsonl")
JUDGE_MODEL = "gpt-4o-mini"   # the gate's own judge and the outcome classifier: never varied,
                              # because the gate is a component outside the answering model
ANSWER_MODEL = "gpt-4o-mini"  # overridden by --answer-model
K = 3
CONDITIONS = ("naive", "prompt_abstain", "gate", "gate_and_instruction")

NAIVE = ("You are a maintenance copilot for one industrial conveyor. Answer the operator's question about the "
         "maintenance history of this conveyor using the work-order records below. Keep the answer under 80 "
         "words.\n\nRecords:\n{evidence}\n\nOperator question: {query}")
WITH_ABSTAIN = ("You are a maintenance copilot for one industrial conveyor. Answer the operator's question about "
                "the maintenance history of this conveyor using only the work-order records below. If the records "
                "do not contain what the operator asks for, say that no relevant record exists. Keep the answer "
                "under 80 words.\n\nRecords:\n{evidence}\n\nOperator question: {query}")
BLOCKED = ("You are a maintenance copilot for one industrial conveyor. The work-order store was searched and no "
           "record relevant to the operator's question was found. Tell the operator that no relevant record "
           "exists, and do not describe any past maintenance. Keep the answer under 60 words.\n\n"
           "Operator question: {query}")
VERDICT = ("An operator asked about the maintenance history of one conveyor. A system answered.\n\n"
           "Work-order records available to the system:\n{evidence}\n\nQuestion: {query}\n\nAnswer: {answer}\n\n"
           "Classify the answer with exactly one word:\n"
           "ABSTAIN - it says no relevant record exists and describes no past maintenance of its own;\n"
           "GROUNDED - it describes past maintenance and everything it states appears in the records above;\n"
           "FABRICATED - it states past maintenance, actions, dates, values or causes the records do not contain.\n"
           "General advice or a suggestion to check something is not past maintenance; ignore it.")


def _cached(path, key, make):
    store = getattr(_cached, path, None)
    if store is None:
        store = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                store[r["key"]] = r["value"]
        setattr(_cached, path, store)
    if key not in store:
        store[key] = make()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": store[key]}, ensure_ascii=False) + "\n")
    return store[key]


def main():
    global ANSWER_MODEL, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-model", default=ANSWER_MODEL)
    ap.add_argument("--answer-base-url", default=None,
                    help="OpenAI-compatible endpoint that serves the ANSWERING model only, "
                         "for example http://localhost:1234/v1 for a model running under LM Studio. "
                         "The gate judge and the outcome classifier always stay on the vendor API, "
                         "because the gate is the component under test and must not change with it.")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N test queries, for a smoke test before a long run")
    args = ap.parse_args()
    ANSWER_MODEL = args.answer_model
    if ANSWER_MODEL != "gpt-4o-mini":
        RUNS = RUNS.replace(".jsonl", f"_{ANSWER_MODEL.replace('.', '').replace('/', '-')}.jsonl")
    if args.limit:
        RUNS = RUNS.replace(".jsonl", f"_first{args.limit}.jsonl")
    where = args.answer_base_url or "the vendor API"
    print(f"[INFO] answering model: {ANSWER_MODEL} at {where} | "
          f"gate judge and classifier: {JUDGE_MODEL} at the vendor API", flush=True)
    retrieval._load_env()
    from openai import OpenAI
    client = OpenAI()
    # A separate client for the answering model, so a local server can serve it while the
    # judge and the classifier keep speaking to the vendor. A local server ignores the key.
    answer_client = OpenAI(base_url=args.answer_base_url, api_key="local")         if args.answer_base_url else client
    ans_cache = os.path.join(HERE, "answer_cache.jsonl")
    ver_cache = os.path.join(HERE, "verdict_cache.jsonl")

    store, qs = run_gate.load()
    recs = store["records"]
    test = [q for q in qs if q["split"] == "test"]
    if args.limit:
        test = test[:args.limit]
    hits = run_gate.distances("openai-small", store, qs, K)
    tau = run_gate.evaluate("openai-small", K, JUDGE_MODEL, store, qs, use_judge=False, verbose=False)["tau"]
    print(f"[INFO] {len(test)} test queries, operating floor tau = {tau:.4f}", flush=True)

    def ask(prompt, max_tokens=180):
        key = hashlib.sha1(f"{ANSWER_MODEL}|{prompt}".encode("utf-8")).hexdigest()
        return _cached(ans_cache, key, lambda: answer_client.chat.completions.create(
            model=ANSWER_MODEL, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]).choices[0].message.content.strip())

    rows = []
    for n, q in enumerate(test, start=1):
        top = hits[q["id"]][:K]
        evidence = "\n\n".join(f"{recs[i]['id']}: {recs[i]['text']}" for i, _ in top)
        passed = top[0][1] <= tau and any(
            judges.judge_yes(JUDGE_MODEL, q["query"], recs[i]["text"], client=client) for i, _ in top)
        for cond in CONDITIONS:
            if cond == "naive":
                text = ask(NAIVE.format(evidence=evidence, query=q["query"]))
            elif cond == "prompt_abstain":
                text = ask(WITH_ABSTAIN.format(evidence=evidence, query=q["query"]))
            else:
                text = ask(NAIVE.format(evidence=evidence, query=q["query"])) if passed \
                    else ask(BLOCKED.format(query=q["query"]), 120)
            vkey = hashlib.sha1(f"{q['id']}|{cond}|{text}".encode("utf-8")).hexdigest()
            word = _cached(ver_cache, vkey, lambda: client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=4,
                messages=[{"role": "user", "content": VERDICT.format(
                    evidence=evidence, query=q["query"], answer=text)}]).choices[0].message.content.strip().upper())
            outcome = ("abstain" if word.startswith("ABS") else
                       "grounded" if word.startswith("GROU") else
                       "misattributed" if word.startswith("MIS") else "fabricated")
            rows.append({"id": q["id"], "category": q["category"], "sufficient": q["sufficient"],
                         "condition": cond, "gate_passed": passed, "answer": text, "outcome": outcome})
        if n % 20 == 0:
            print(f"  {n}/{len(test)} queries done", flush=True)

    with open(RUNS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby(["sufficient", "condition"])["outcome"].value_counts().unstack(fill_value=0)
    for c in ("grounded", "abstain", "misattributed", "fabricated"):
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[["grounded", "abstain", "misattributed", "fabricated"]]
    tab["n"] = tab.sum(axis=1)
    for c in ("grounded", "abstain", "misattributed", "fabricated"):
        tab[c + "_pct"] = (100 * tab[c] / tab["n"]).round(1)
    tab = tab.reindex(pd.MultiIndex.from_product([[True, False], list(CONDITIONS)],
                                                 names=["sufficient", "condition"]))
    tab.to_csv(os.path.join(RESULTS, os.path.basename(RUNS).replace(".jsonl", "_summary.csv")))
    print("\n=== outcomes on the 120 test queries (sufficient = the store can answer) ===")
    print(tab.to_string())


if __name__ == "__main__":
    main()
