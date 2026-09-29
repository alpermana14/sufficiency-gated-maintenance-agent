"""Gate v2 - the entailment stage, with three interchangeable judges.

Answers CAEE R2-19: which model decides entailment, how is it calibrated, and does the
result survive a change of judge.

  gpt-4o-mini   the model used in the deployed gate; binary yes/no, temperature 0
  gpt-4o        a stronger model from the same family, for a robustness check
  nli-local     cross-encoder/nli-deberta-v3-base run locally; the query is the
                hypothesis and the retrieved record the premise, and the decision is
                entailment or neutral/contradiction. No API is involved, which also
                shows the gate can run without a vendor.

Every call is cached on disk by a hash of (judge, query, evidence), so a rerun costs
nothing and the reported numbers are reproducible.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "judge_cache.jsonl")
JUDGES = ["gpt-4o-mini", "gpt-4o", "nli-local"]

PROMPT = (
    "You decide whether a maintenance record contains information that materially answers "
    "an operator's question about the history of one conveyor.\n\n"
    "Answer YES only if the record states the specific information the question asks for. "
    "Answer NO if the record merely mentions the same topic, subsystem or component without "
    "the information asked for, or if it is about something else.\n\n"
    "Question: {query}\n\nRecord:\n{evidence}\n\nAnswer with one word, YES or NO."
)

_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    _cache[row["key"]] = row["yes"]
    return _cache


def _store(key, yes):
    _load_cache()[key] = yes
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "yes": yes}) + "\n")


def _key(judge, query, evidence):
    return hashlib.sha1(f"{judge}\x00{query}\x00{evidence}".encode("utf-8")).hexdigest()


_nli = None


def _nli_yes(query, evidence):
    global _nli
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    scores = _nli.predict([(evidence, query)])
    label = int(scores.argmax()) if hasattr(scores, "argmax") else int(max(range(len(scores[0])), key=lambda i: scores[0][i]))
    # label order of this model: contradiction, entailment, neutral
    return label == 1


def judge_yes(judge, query, evidence, client=None):
    """True if the record materially answers the query, according to `judge`."""
    key = _key(judge, query, evidence)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    if judge == "nli-local":
        yes = _nli_yes(query, evidence)
    else:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        resp = client.chat.completions.create(
            model=judge, temperature=0, max_tokens=3,
            messages=[{"role": "user", "content": PROMPT.format(query=query, evidence=evidence)}])
        yes = resp.choices[0].message.content.strip().upper().startswith("Y")
    _store(key, yes)
    return yes


def cached_count():
    return len(_load_cache())
