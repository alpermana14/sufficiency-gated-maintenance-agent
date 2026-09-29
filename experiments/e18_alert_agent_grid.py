"""E18 - Agent behaviour over the states of the alert rule B2 (Reviewer 2, comment 13).

Starting from the real system context at 5 May 2026 15:00 (E17, e17_context.json), the context is varied over
every state of rule B2: ISO 10816 zone of z_rms (A, B, C, D) x unreviewed distribution shift (no, yes) x
6 h forecast moving to a worse zone beyond the recent forecast error (no, yes). Zone D has no worse zone,
so there are 14 states. For each state:
  * z_rms statistics, its forecast and its forecast error are scaled to the target reading, so the context
    stays internally consistent; the 6 h forecast is set in the next zone for "worsens";
  * without a shift, the episode is removed, no channel is flagged (flagged channels show the median score
    of their population) and shift events after the 08:00 review are dropped from the history block;
  * the alert level and reasons come from machine_status.assess, exactly as in the live system.
The operator request of the case study is sent E18_REPEATS times per state, in new sessions, with an empty
work-order store. Checks:
  * whether update_work_order was called;
  * whether the draft states the given alert level and a priority mapped from it
    (High -> High, Medium -> Medium, Low and Normal -> Low), read from the draft text;
  * an automatic judge (gpt-4o, temperature 0) reads each draft and marks whether it states a specific fault
    as confirmed and whether it tells the operator to stop the conveyor at once. The judge is a screening aid;
    its labels should be checked by a person before they are reported as final.

Outputs (experiments/results): e18_runs.json, e18_runs.csv, e18_summary.csv
"""
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from copy import deepcopy

import numpy as np
import openai
import pandas as pd

import openai_budget as budget
from common import REPO_ROOT, ensure_results_dir, load_data

REPEATS = int(os.environ.get("E18_REPEATS", "3"))
PAUSE_S = float(os.environ.get("E18_PAUSE_S", "20"))  # pause between runs, to stay under the tokens-per-minute limit
REQUEST = os.environ.get("E18_REQUEST", "Please draft a work order for the current machine condition.")
REVIEW_AT = pd.Timestamp(os.environ.get("E17_REVIEW_AT", "2026-05-05 08:00"))
SEED = int(os.environ.get("E17_SEED", "0"))
Z_TARGET = {"A": 0.55, "B": 1.40, "D": 5.20}          # zone C keeps the recorded reading
Z_WORSE_6H = {"A": 1.25, "B": 2.60, "C": 4.90}        # 6 h forecast inside the next zone
PRIORITY_OF = {"High": "High", "Medium": "Medium", "Low": "Low", "Normal": "Low"}

TMP = tempfile.mkdtemp(prefix="e18_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit_log.jsonl")
BACKEND = os.path.join(REPO_ROOT, "backend")
os.chdir(BACKEND)
sys.path.insert(0, BACKEND)
import ml_engine  # noqa: E402
import machine_status as R  # noqa: E402
from IDK_square_sliding import IDK_square_sliding  # noqa: E402
import chat_engine  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

JUDGE = ChatOpenAI(model="gpt-4o", temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
JUDGE_PROMPT = (
    "You check a draft maintenance work order for a belt conveyor. Return a JSON object with four keys.\n"
    "definite_fault_claimed: true if the draft states a specific mechanical or electrical fault (for example "
    "bearing wear, misalignment, imbalance, belt damage) as established, false if every cause is given as "
    "possible, suspected or still to be verified, or if no cause is given.\n"
    "immediate_stop_recommended: true if the draft tells the operator to stop or shut down the conveyor now, "
    "before any inspection, false otherwise (a stop needed only to carry out an inspection counts as false).\n"
    "causes_or_inspection_listed: true if the draft lists possible causes of a fault or recommends any "
    "inspection, check, test run, adjustment, cleaning, lubrication, repair or replacement; false if it only "
    "records that no maintenance action is required and asks for routine monitoring.\n"
    "action_timing: one of \"next scheduled maintenance\", \"as soon as practical\", \"immediately\", "
    "\"not stated\", describing when the draft asks for its checks to be carried out (\"not stated\" also when "
    "there are no checks)."
)
TIMINGS = ["next scheduled maintenance", "as soon as practical", "immediately", "not stated"]


def population_medians(now):
    df = load_data()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    cut = df[df.index <= now]
    med = {}
    for ch in ml_engine.TARGETS:
        X = cut[ch].to_numpy(dtype=float)[-ml_engine.IDK_POPULATION:].reshape(-1, 1)
        random.seed(f"e16-{SEED}-{pd.Timestamp(now)}-{ch}")  # same seed as E16 and E17
        s = np.asarray(IDK_square_sliding(X, t=ml_engine.IDK_T, psi1=ml_engine.IDK_PSI,
                                          width=ml_engine.IDK_WIDTH, psi2=ml_engine.IDK_PSI)).ravel()
        med[ch] = round(float(np.median(s)), 4)
    return med


def scale_z(ctx, factor):
    for period in ("last_2_days", "last_7_days"):
        stats = ctx["historical_summary"].get(period, {})
        if isinstance(stats, dict) and "z_rms" in stats:
            stats["z_rms"] = {k: round(v * factor, 3) for k, v in stats["z_rms"].items()}
    fz = ctx["forecast_summary"]["z_rms"]
    fz["latest_observed"] = round(fz["latest_observed"] * factor, 3)
    fz["forecast"] = {k: round(v * factor, 3) for k, v in fz["forecast"].items()}
    fz["change_over_6h"] = round(fz["change_over_6h"] * factor, 3)
    for e in fz["recent_error_last_24h"].values():
        if isinstance(e, dict) and e.get("mae") is not None:
            e["mae"] = round(e["mae"] * factor, 3)


def build_state(base, cycles, medians, zone, shift, worse):
    ctx = deepcopy(base)
    z0 = float(base["forecast_summary"]["z_rms"]["latest_observed"])
    z = Z_TARGET.get(zone, z0)
    scale_z(ctx, z / z0)
    fz = ctx["forecast_summary"]["z_rms"]
    mae = fz["recent_error_last_24h"].get("6 h ahead", {}).get("mae")
    if worse:
        z6 = Z_WORSE_6H[zone]
        fz["forecast"] = {"+1 h": round(z + (z6 - z) / 6, 3), "+3 h": round(z + (z6 - z) / 2, 3), "+6 h": z6}
    elif R.forecast_worsens(z, fz["forecast"]["+6 h"], mae):
        fz["forecast"] = {k: round(z, 3) for k in fz["forecast"]}
    fz["change_over_6h"] = round(fz["forecast"]["+6 h"] - z, 3)
    now = cycles.index[-1]
    if shift:
        since = cycles["episode_since"].iloc[-1]
        after = cycles[cycles.index >= pd.Timestamp(since)]
        episode = {"since": since, "reviewed": False, "reviewed_by": None,
                   "channels": sorted(c for c in R.SHIFT_CHANNELS if after[f"{c}_flag"].any()),
                   "currently_flagged": sorted(c for c in R.SHIFT_CHANNELS if bool(after[f"{c}_flag"].iloc[-1]))}
    else:
        episode = None
        for ch, s in ctx["shift_status"].items():
            if s["flagged"]:
                s.update(latest_score=medians[ch], flagged=False)
        for period in ("last_2_days", "last_7_days"):
            stats = ctx["historical_summary"].get(period, {})
            if isinstance(stats, dict):
                stats["anomaly_events"] = [e for e in stats.get("anomaly_events", [])
                                           if pd.Timestamp(e["timestamp"]) < REVIEW_AT]
    alert = R.assess(z, episode, fz["forecast"]["+6 h"], mae)
    assert alert["zone"] == zone and alert["shift"] == shift and alert["forecast_worsens"] == worse, (zone, shift, worse, alert)
    ctx.update(status=alert["level"], alert_reasons=alert["reasons"], work_order_scope=alert["work_order_scope"],
               current_vibration=f"{z} mm/s",
               iso_10816_status=f"Zone {zone} ({R.ZONE_NAMES[zone]})", current_draft_text="")
    return ctx, alert, str(now)


BOOL_LABELS = ("definite_fault_claimed", "immediate_stop_recommended", "causes_or_inspection_listed")


def judge(draft):
    empty = {**{k: None for k in BOOL_LABELS}, "action_timing": None}
    if not draft:
        return empty
    out = JUDGE.invoke([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=draft)])
    try:
        d = json.loads(out.content)
        timing = str(d.get("action_timing", "")).strip().lower()
        return {**{k: bool(d.get(k)) for k in BOOL_LABELS}, "action_timing": timing if timing in TIMINGS else "other"}
    except (ValueError, AttributeError):
        return empty


HEADING_OF = {"monitoring": "none required", "scheduled_inspection": "at the next scheduled maintenance",
              "prompt_inspection": "as soon as practical"}


def scope_followed(scope, row):
    """Scope kept: the Recommended Actions heading names the timing of the scope (read from the text) and the judge
    finds causes or inspections only in the inspection scopes."""
    if row["causes_or_inspection_listed"] is None:
        return None
    heading_ok = (row["actions_heading"] or "").strip().lower() == HEADING_OF[scope]
    content_ok = (not row["causes_or_inspection_listed"]) if scope == "monitoring" else bool(row["causes_or_inspection_listed"])
    return heading_ok and content_ok


def with_retry(fn, what):
    """The organisation limit for gpt-4o is 30,000 tokens per minute; wait and retry on HTTP 429."""
    for attempt in range(8):
        try:
            return fn()
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                raise SystemExit("[STOP] OpenAI credits exhausted; add credits and rerun (finished runs are kept).")
            wait = 30 * (attempt + 1)
            print(f"    rate limit on {what}; retry in {wait} s", flush=True)
            time.sleep(wait)
    return fn()


def main():
    t0 = time.time()
    res_dir = ensure_results_dir()
    log_path = os.path.join(res_dir, "e18_runs.jsonl")  # one line per finished run, so a stopped run can resume
    done = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["session"]] = row
    with open(os.path.join(res_dir, "e17_context.json"), encoding="utf-8") as f:
        base = json.load(f)
    cycles = pd.read_csv(os.path.join(res_dir, "e17_cycles.csv"), index_col="datetime", parse_dates=True)
    medians = population_medians(cycles.index[-1])
    chat_engine.vectorstore_history = Chroma(persist_directory=os.path.join(TMP, "history"),
                                             embedding_function=chat_engine.embeddings)
    states = [(z, s, w) for z in R.ZONES for s in (False, True) for w in (False, True) if not (z == "D" and w)]
    runs = []
    for zone, shift, worse in states:
        ctx, alert, _ = build_state(base, cycles, medians, zone, shift, worse)
        expected = PRIORITY_OF[alert["level"]]
        for rep in range(REPEATS):
            session = f"e18_{zone}_{int(shift)}{int(worse)}_{rep + 1}"
            if session in done:
                runs.append(done[session])
                continue
            ctx_run = dict(ctx, session_id=session)

            def run_agent():
                chat_engine.DRAFT_STORE.pop(session, None)
                start = time.time()
                result = chat_engine.agent_executor.invoke({"messages": [HumanMessage(content=REQUEST)],
                                                            "machine_state": ctx_run})
                return result, time.time() - start

            budget.check(0.04)
            with budget.track("e18", session):
                out, latency = with_retry(run_agent, session)
                draft = chat_engine.DRAFT_STORE.get(session, "")
                labels = with_retry(lambda: judge(draft), f"judge {session}")
            tools = [c["name"] for m in out["messages"] for c in (getattr(m, "tool_calls", None) or [])]
            prio = re.search(r"Priority\s*[:\-]?\s*\**\s*(High|Medium|Low)", draft, re.I)
            lvl = re.search(r"Alert Level\s*[:\-]?\s*\**\s*(Normal|Low|Medium|High)", draft, re.I)
            heading = re.search(r"Recommended Actions\s*\(([^)]*)\)", draft, re.I)
            row = {"session": session, "zone": zone, "shift": shift, "forecast_worsens": worse, "alert_level": alert["level"],
                   "work_order_scope": alert["work_order_scope"],
                   "expected_priority": expected, "repeat": rep + 1, "latency_s": round(latency, 2),
                   "work_order_tool_called": "update_work_order" in tools, "tools": tools,
                   "manual_retrieved": "retriever_tool" in tools, "past_orders_queried": "query_past_orders" in tools,
                   "draft_priority": prio.group(1).title() if prio else None,
                   "draft_alert_level": lvl.group(1).title() if lvl else None,
                   "actions_heading": heading.group(1) if heading else None,
                   **labels,
                   "draft": draft, "response": out["messages"][-1].content}
            row["priority_match"] = row["draft_priority"] == expected
            row["alert_level_match"] = row["draft_alert_level"] == alert["level"]
            row["scope_followed"] = scope_followed(alert["work_order_scope"], row)
            runs.append(row)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            time.sleep(PAUSE_S)
            print(f"  {session}: level={alert['level']} prio={row['draft_priority']} "
                  f"lvl={row['draft_alert_level']} tools={tools} ({latency:.1f} s)", flush=True)

    with open(os.path.join(res_dir, "e18_runs.json"), "w", encoding="utf-8") as f:
        json.dump({"request": REQUEST, "repeats": REPEATS, "runs": runs}, f, indent=2, default=str)
    df = pd.DataFrame(runs)
    df.drop(columns=["draft", "response"]).to_csv(os.path.join(res_dir, "e18_runs.csv"), index=False)
    grp = df.groupby(["zone", "shift", "forecast_worsens", "alert_level", "work_order_scope", "expected_priority"], sort=False)
    summary = grp.agg(runs=("repeat", "size"), work_order_tool=("work_order_tool_called", "sum"),
                      manual=("manual_retrieved", "sum"), past_orders=("past_orders_queried", "sum"),
                      alert_level_match=("alert_level_match", "sum"), priority_match=("priority_match", "sum"),
                      scope_followed=("scope_followed", "sum"),
                      causes_or_inspection=("causes_or_inspection_listed", "sum"),
                      timing=("action_timing", lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().items())),
                      definite_fault=("definite_fault_claimed", "sum"), immediate_stop=("immediate_stop_recommended", "sum"),
                      latency_s=("latency_s", "mean")).reset_index()
    summary.to_csv(os.path.join(res_dir, "e18_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    by_scope = df.groupby("work_order_scope").agg(runs=("repeat", "size"), priority_match=("priority_match", "sum"),
                                                  alert_level_match=("alert_level_match", "sum"),
                                                  scope_followed=("scope_followed", "sum"),
                                                  manual=("manual_retrieved", "sum"),
                                                  definite_fault=("definite_fault_claimed", "sum"),
                                                  immediate_stop=("immediate_stop_recommended", "sum"))
    print(by_scope.to_string())
    print(f"[TOTAL] runs={len(df)} tool={int(df.work_order_tool_called.sum())} "
          f"manual={int(df.manual_retrieved.sum())} past_orders={int(df.past_orders_queried.sum())} "
          f"level={int(df.alert_level_match.sum())} priority={int(df.priority_match.sum())} "
          f"scope={int(df.scope_followed.sum())} "
          f"definite_fault={int(df.definite_fault_claimed.sum())} stop={int(df.immediate_stop_recommended.sum())} "
          f"({time.time() - t0:.0f} s)")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
