"""Shared OpenAI spending ledger and budget cap for the LLM experiments (E18, agent benchmark v2).

Every tracked run appends its token counts and estimated cost to experiments/results/openai_spend.jsonl. Before a
run, check() stops the script when the ledger total plus the expected cost of the next run would exceed
OPENAI_BUDGET_USD (no cap when it is unset). The cost is an estimate from the token counts of the chat models; all
tokens are priced at the gpt-4o rates below (set PRICE_*_PER_M to the current prices), which overestimates the
small gpt-4o-mini guard calls. Embedding calls are not counted (text-embedding-3-small is about $0.02 per 1M tokens).
"""
import json
import os
import time
from contextlib import contextmanager

from langchain_community.callbacks.manager import get_openai_callback

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LEDGER = os.path.join(RESULTS, "openai_spend.jsonl")
BUDGET = float(os.environ["OPENAI_BUDGET_USD"]) if os.environ.get("OPENAI_BUDGET_USD") else None
PRICE_IN = float(os.environ.get("PRICE_IN_PER_M", "2.50")) / 1e6
PRICE_CACHED = float(os.environ.get("PRICE_CACHED_PER_M", "1.25")) / 1e6
PRICE_OUT = float(os.environ.get("PRICE_OUT_PER_M", "10.00")) / 1e6


def spent() -> float:
    if not os.path.exists(LEDGER):
        return 0.0
    with open(LEDGER, encoding="utf-8") as f:
        return sum(json.loads(line)["cost_usd"] for line in f if line.strip())


def check(next_run_usd: float = 0.04):
    total = spent()
    if BUDGET is not None and total + next_run_usd > BUDGET:
        raise SystemExit(f"[BUDGET] stop: ${total:.2f} spent of ${BUDGET:.2f}; the next run could exceed the cap. "
                         "Finished runs are kept.")
    return total


@contextmanager
def track(script: str, run_id: str):
    with get_openai_callback() as cb:
        yield cb
    cached = int(getattr(cb, "prompt_tokens_cached", 0) or 0)
    cost = (cb.prompt_tokens - cached) * PRICE_IN + cached * PRICE_CACHED + cb.completion_tokens * PRICE_OUT
    os.makedirs(RESULTS, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "script": script, "run": run_id,
                            "requests": cb.successful_requests, "prompt_tokens": cb.prompt_tokens,
                            "cached_tokens": cached, "completion_tokens": cb.completion_tokens,
                            "cost_usd": round(cost, 5)}) + "\n")
