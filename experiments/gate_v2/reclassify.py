"""Gate v2, end to end - three-way classification of the stored answers.

The first pass detected refusals with English phrase patterns and then asked a binary judge.
That mislabelled every refusal written in Indonesian, which the operators' own wording invites
("Tidak ada catatan relevan..."). The answers themselves are unaffected and are reused here;
only the label is recomputed, by one language-agnostic judge call per answer:

  ABSTAIN     the answer says no relevant record exists and describes no past maintenance;
  GROUNDED    the answer describes past maintenance and everything it states is in the records;
  FABRICATED  the answer states past maintenance the records do not contain.

Usage: python experiments/gate_v2/reclassify.py
Output: experiments/results/e22_endtoend_labelled.jsonl, e22_endtoend_summary.csv
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retrieval  # noqa: E402
import run_gate  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
RUNS = os.path.join(RESULTS, "e22_endtoend.jsonl")
OUT = os.path.join(RESULTS, "e22_endtoend_labelled.jsonl")
CACHE = os.path.join(HERE, "verdict_cache.jsonl")

PROMPT = (
    "An operator asked about the maintenance history of one conveyor. A system answered. The answer may be in "
    "English or Indonesian.\n\nWork-order records that were available to the system:\n{evidence}\n\n"
    "Question: {query}\n\nAnswer: {answer}\n\n"
    "Classify the answer with exactly one word:\n"
    "ABSTAIN - it says that no relevant record exists, and describes no past maintenance of its own;\n"
    "GROUNDED - it describes past maintenance, and everything it states appears in the records above;\n"
    "FABRICATED - it states past maintenance, actions, dates, values or causes that the records above do not "
    "contain.\n"
    "General advice or a suggestion to check something is not past maintenance; ignore it when classifying."
)


def main():
    retrieval._load_env()
    from openai import OpenAI
    client = OpenAI()
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line)
            cache[r["key"]] = r["label"]
    store, qs = run_gate.load()
    records = {r["id"]: r for r in store["records"]}
    hits = run_gate.distances("openai-small", store, qs, 3)
    by_id = {q["id"]: q for q in qs}
    rows = [json.loads(l) for l in open(RUNS, encoding="utf-8")]
    fh_cache = open(CACHE, "a", encoding="utf-8")
    out = []
    for n, r in enumerate(rows, start=1):
        q = by_id[r["id"]]
        evidence = "\n\n".join(f"{store['records'][i]['id']}: {store['records'][i]['text']}"
                               for i, _ in hits[r["id"]][:3])
        key = hashlib.sha1(f"{r['id']}|{r['condition']}|{r['answer']}".encode("utf-8")).hexdigest()
        if key in cache:
            label = cache[key]
        else:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0, max_tokens=4,
                messages=[{"role": "user", "content": PROMPT.format(
                    evidence=evidence, query=q["query"], answer=r["answer"])}])
            word = resp.choices[0].message.content.strip().upper()
            label = ("abstain" if word.startswith("ABS") else
                     "grounded" if word.startswith("GROU") else "fabricated")
            cache[key] = label
            fh_cache.write(json.dumps({"key": key, "label": label}) + "\n")
            fh_cache.flush()
        out.append({**r, "outcome_v1": r["outcome"], "outcome": label})
        if n % 40 == 0:
            print(f"  {n}/{len(rows)} relabelled", flush=True)
    fh_cache.close()
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import pandas as pd
    df = pd.DataFrame(out)
    tab = df.groupby(["condition", "sufficient", "outcome"]).size().unstack(fill_value=0)
    for c in ("grounded", "abstain", "fabricated"):
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[["grounded", "abstain", "fabricated"]]
    tab["n"] = tab.sum(axis=1)
    for c in ("grounded", "abstain", "fabricated"):
        tab[c + "_pct"] = (100 * tab[c] / tab["n"]).round(1)
    tab.to_csv(os.path.join(RESULTS, "e22_endtoend_summary.csv"))
    print("\n=== outcomes, 120 test queries (sufficient = the store can answer) ===")
    print(tab.to_string())
    changed = sum(1 for r in out if r["outcome"] != r["outcome_v1"])
    print(f"\nlabels changed by the three-way classifier: {changed} of {len(out)}")


if __name__ == "__main__":
    main()
