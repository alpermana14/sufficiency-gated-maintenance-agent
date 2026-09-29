"""E30 - Behaviour of the running service under concurrent operator requests (R2-22).

R2-22 asks how the architecture behaves when several machines or operators send requests at
the same time. The concurrency-1 baseline is already measured: the agent benchmark of
Section 5.3 ran 120 requests one after another and recorded 4.72 s mean, 3.01 s standard
deviation. This script measures what happens when requests arrive together.

Design (author decision, 21 Sep 2026):
  * the unit of observation is a BURST, not a request, because requests inside one burst
    share whatever the language-model service is doing at that moment;
  * level 2: five bursts of two simultaneous requests;
  * level 10: two bursts of ten simultaneous requests;
  * bursts are separated by a pause so that they are not one long burst;
  * the service paces its calls to the reasoning model within the account's per-minute token
    allowance, so a burst larger than the allowance queues rather than being refused; the
    waiting shows up in the latency of the requests that are served last;
  * 30 requests in total, against the running FastAPI service, so the measurement includes
    the HTTP layer and the thread pool that serves the synchronous chat endpoint.

Recorded per request: the wall-clock latency, the order in which it finished inside its
burst, and any failure. The spread between the first and the last completion inside a
burst is the direct evidence of queuing.

Output: experiments/results/e30_concurrent_load.csv
Usage:  python experiments/e30_concurrent_load.py
  env:  E30_URL    service address (default http://127.0.0.1:8000)
        E30_PAUSE  seconds between bursts (default 20)
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_results_dir  # noqa: E402

URL = os.environ.get("E30_URL", "http://127.0.0.1:8000").rstrip("/") + "/api/chat"
PAUSE = float(os.environ.get("E30_PAUSE", "20"))
TIMEOUT = 420

# Requests an operator would plausibly send at the same time from different machines.
PROMPTS = [
    "What is the alert level right now and why?",
    "Summarise the vibration trend over the last two days.",
    "Has any channel shifted from its recent distribution?",
    "According to the maintenance manual, what should be checked for belt mistracking?",
    "Draft a work order for the current machine condition.",
    "What did we do last time the conveyor made a rubbing noise?",
    "Is the forecast for the next six hours within the normal zone?",
    "Which sensor reading is closest to its ISO limit?",
    "Explain the current distribution-shift status in plain terms.",
    "What maintenance action, if any, is required today?",
]

# Concurrency level -> number of bursts
PLAN = [(2, 5), (10, 2)]


def one_request(prompt, session_id):
    body = json.dumps({"message": prompt, "session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            r.read()
        return time.perf_counter() - t0, None
    except urllib.error.HTTPError as e:
        return time.perf_counter() - t0, f"HTTP {e.code}"
    except Exception as e:  # timeout, connection reset, anything the service does
        return time.perf_counter() - t0, type(e).__name__


def run_burst(level, burst_id):
    """Fire `level` requests at the same moment and record when each one returned."""
    prompts = [PROMPTS[(burst_id * level + k) % len(PROMPTS)] for k in range(level)]
    sessions = [f"load-{level}-{burst_id}-{k}-{uuid.uuid4().hex[:6]}" for k in range(level)]
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(one_request, p, s) for p, s in zip(prompts, sessions)]
        results = [f.result() for f in futures]
    finish = [(t_start + lat) for lat, _ in results]
    order = sorted(range(len(finish)), key=lambda i: finish[i])
    rank = {i: r + 1 for r, i in enumerate(order)}
    rows = []
    for i, (lat, err) in enumerate(results):
        rows.append({"level": level, "burst": burst_id, "request": i,
                     "latency_s": round(lat, 3), "finish_rank": rank[i],
                     "error": err or ""})
    return rows, (time.perf_counter() - t_start)


def main():
    results_dir = ensure_results_dir()
    total = sum(level * bursts for level, bursts in PLAN)
    print(f"[INFO] service: {URL}")
    print(f"[INFO] plan: " + ", ".join(f"{b} bursts of {l}" for l, b in PLAN)
          + f"  ({total} requests, about ${total * 0.0136:.2f})")
    print("[INFO] concurrency-1 baseline comes from the benchmark of Section 5.3: "
          "4.72 s mean, 3.01 s sd, n = 120")

    rows = []
    for level, bursts in PLAN:
        for b in range(bursts):
            batch, wall = run_burst(level, b)
            rows.extend(batch)
            lat = [r["latency_s"] for r in batch]
            errs = sum(1 for r in batch if r["error"])
            print(f"  level {level:2}  burst {b + 1}/{bursts}  "
                  f"wall {wall:6.2f} s  first {min(lat):6.2f}  last {max(lat):6.2f}  "
                  f"errors {errs}")
            if not (level == PLAN[-1][0] and b == bursts - 1):
                time.sleep(PAUSE)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(results_dir, "e30_concurrent_load.csv"), index=False)
    print("\n=== latency by concurrency level ===")
    ok = out[out.error == ""]
    summary = ok.groupby("level").latency_s.agg(
        n="size", mean="mean", sd="std", median="median",
        p95=lambda s: s.quantile(0.95), max="max").round(2)
    print(summary.to_string())
    print("\n=== spread inside each burst (queuing shows up here) ===")
    spread = ok.groupby(["level", "burst"]).latency_s.agg(
        first="min", last="max").round(2)
    spread["spread_s"] = (spread["last"] - spread["first"]).round(2)
    print(spread.to_string())
    bad = out[out.error != ""]
    print(f"\nfailed requests: {len(bad)} of {len(out)}")
    if len(bad):
        print(bad[["level", "burst", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
