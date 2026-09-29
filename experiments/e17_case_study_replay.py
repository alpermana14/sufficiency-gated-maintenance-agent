"""E17 - Replay of the 5 May 2026 case study (Section 5.4) with the status rules B1 and B2.

The original trace was recorded with earlier code (status taken from motor current with a fixed threshold).
This script rebuilds the system state at 5 May 2026 15:00 with the current backend code and runs the agent
on the operator request that produced the work order:

  1. Data are cut at 15:00. Every 30-min cycle from E17_EVENT_LOG_DAYS before is replayed with the backend
     functions of main.py: s-IDK^2 scores, rule B1 (update_shift_state) and, for the last E17_FORECAST_HOURS,
     the full pipeline (tuned LightGBM, forecast log), so the recent forecast error is the one the live
     system would have had. s-IDK^2 is seeded per step and channel as in E16 (E17_SEED, default 0), so the
     flags match e16_steps.csv. Forecasts are cached in e17_forecasts.pkl and reused on later runs.
  2. Episodes opened before the load test are assumed reviewed at 08:00 (the operators were present to add the
     load). The episode that would be open without any review is reported as well.
  3. The agent runs with an empty temporary work-order store (the first approved work order was saved after
     this event), a temporary audit log and a clock set to 15:05, so the data count as real time.
  4. The operator request is sent E17_REPEATS times in new sessions; tool calls, response and draft are stored.

Outputs (experiments/results): e17_cycles.csv, e17_context.json, e17_runs.json, e17_summary.txt
"""
import json
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime as _dt

import numpy as np
import pandas as pd

from common import REPO_ROOT, ensure_results_dir, load_data

NOW = pd.Timestamp(os.environ.get("E17_NOW", "2026-05-05 15:00"))
REVIEW_AT = pd.Timestamp(os.environ.get("E17_REVIEW_AT", "2026-05-05 08:00"))
FORECAST_HOURS = float(os.environ.get("E17_FORECAST_HOURS", "30"))
EVENT_LOG_DAYS = float(os.environ.get("E17_EVENT_LOG_DAYS", "7"))
REPEATS = int(os.environ.get("E17_REPEATS", "3"))
REQUEST = os.environ.get("E17_REQUEST", "Please draft a work order for the current machine condition.")
SEED = int(os.environ.get("E17_SEED", "0"))

TMP = tempfile.mkdtemp(prefix="e17_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit_log.jsonl")  # never the live audit log

BACKEND = os.path.join(REPO_ROOT, "backend")
os.chdir(BACKEND)  # backend/.env and relative paths
sys.path.insert(0, BACKEND)
import ml_engine  # noqa: E402
import machine_status as R  # noqa: E402
from IDK_square_sliding import IDK_square_sliding  # noqa: E402
import chat_engine  # noqa: E402
import main  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402


def history_store_dates():
    raw = chat_engine.vectorstore_history._collection.get()  # type: ignore[attr-defined]
    return sorted((m or {}).get("created_at") or "" for m in raw.get("metadatas") or [])


def scores_as_e16(df_cut, ts):
    """detect_anomalies with the per-step, per-channel seed of E16 (the deployed system is not seeded)."""
    out = {}
    for ch in ml_engine.TARGETS:
        X = df_cut[ch].to_numpy(dtype=float)[-ml_engine.IDK_POPULATION:].reshape(-1, 1)
        random.seed(f"e16-{SEED}-{pd.Timestamp(ts)}-{ch}")
        out[ch] = np.asarray(IDK_square_sliding(X, t=ml_engine.IDK_T, psi1=ml_engine.IDK_PSI,
                                                width=ml_engine.IDK_WIDTH, psi2=ml_engine.IDK_PSI)).ravel()
    return out


def forecast_cycle(df_cut, ts, cache):
    """Full pipeline of one cycle; forecasts are cached because they do not depend on the s-IDK^2 seed."""
    if ts in cache:
        return cache[ts]
    _, forecast, _, _, _ = ml_engine.run_pipeline(df_cut)
    cache[ts] = forecast
    return forecast


class RecordingExecutor:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def invoke(self, inputs, config=None):
        out = self.inner.invoke(inputs, config=config)
        self.calls.append((inputs, out))
        return out


def fake_clock(now_local):
    class FakeDateTime(_dt):
        @classmethod
        def utcnow(cls):
            return (now_local - pd.Timedelta(hours=8)).to_pydatetime()
    return FakeDateTime


def serialise_messages(messages):
    rows = []
    for m in messages:
        row = {"type": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
        if getattr(m, "tool_calls", None):
            row["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
        if m.type == "tool":
            row["tool_name"] = getattr(m, "name", None)
        rows.append(row)
    return rows


def rebuild_state(write_outputs=True):
    """Replay every cycle up to NOW into main.state; returns (cycles table, episode without review, E16 mismatches)."""
    t0 = time.time()
    res_dir = ensure_results_dir()
    df_all = load_data()
    df_all = df_all[~df_all.index.duplicated(keep="first")].sort_index()
    df_all = df_all[df_all.index <= NOW]
    e16_csv = os.path.join(res_dir, "e16_steps.csv")
    e16 = pd.read_csv(e16_csv, index_col="datetime", parse_dates=True) if (os.path.exists(e16_csv) and SEED == 0) else None
    cache_path = os.path.join(res_dir, "e17_forecasts.pkl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    cycles = df_all.index[df_all.index > NOW - pd.Timedelta(days=EVENT_LOG_DAYS)]
    forecast_from = NOW - pd.Timedelta(hours=FORECAST_HOURS)
    st = main.state
    st.forecast_log, st.shift_status, st.shift_events, st.shift_episode = [], {}, [], None
    no_review_episode, rows, mismatches = None, [], 0
    for ts in cycles:
        df_cut = df_all[df_all.index <= ts]
        st.data = df_cut
        if ts >= forecast_from:
            st.forecast = forecast_cycle(df_cut, ts, cache)
            main.record_forecast(ts, st.forecast)
        else:
            st.forecast = None
        st.anomalies = scores_as_e16(df_cut, ts)
        main.update_shift_state(df_cut)
        no_review_episode = R.update_episode(no_review_episode, st.shift_status, ts)
        if ts == REVIEW_AT:
            main.mark_shift_reviewed("replay assumption", "open episodes reviewed before the load test",
                                     source="e17_replay")
        alert = main.current_alert() if st.forecast is not None else R.assess(float(df_cut["z_rms"].iloc[-1]), st.shift_episode)
        row = {"datetime": ts, "z_rms": float(df_cut["z_rms"].iloc[-1]), "zone": alert["zone"],
               "alert_level": alert["level"], "shift_episode_open": alert["shift"],
               "forecast_worsens": alert["forecast_worsens"],
               "episode_since": (st.shift_episode or {}).get("since"),
               "z_rms_forecast_6h": float(st.forecast["z_rms"].iloc[-1]) if st.forecast is not None else None}
        for ch in R.SHIFT_CHANNELS + ["temperature"]:
            row[f"{ch}_flag"] = st.shift_status[ch]["latest_flag"]
            if e16 is not None and ts in e16.index and bool(e16.at[ts, f"{ch}_flag"]) != row[f"{ch}_flag"]:
                mismatches += 1
        rows.append(row)
        if len(rows) % 24 == 0:
            print(f"  cycle {len(rows)}/{len(cycles)} {ts} ({time.time() - t0:.0f} s)", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    cyc = pd.DataFrame(rows).set_index("datetime")
    if write_outputs:
        cyc.to_csv(os.path.join(res_dir, "e17_cycles.csv"))
    print(f"[INFO] replayed {len(cyc)} cycles up to {NOW}; flag mismatches against E16: {mismatches} "
          f"({time.time() - t0:.0f} s)")
    return cyc, no_review_episode, mismatches


def main_replay():
    t0 = time.time()
    res_dir = ensure_results_dir()
    cyc, no_review_episode, mismatches = rebuild_state()
    st = main.state
    cycles = cyc.index
    if REPEATS == 0:  # forecast cache only
        return

    # ---- agent run at NOW ----
    stored = history_store_dates()
    print(f"[INFO] live work-order store: {len(stored)} records, earliest created_at {stored[0] if stored else None}")
    chat_engine.vectorstore_history = Chroma(persist_directory=os.path.join(TMP, "history"),
                                             embedding_function=chat_engine.embeddings)
    main.datetime = fake_clock(NOW + pd.Timedelta(minutes=5))
    st.last_update = str(NOW + pd.Timedelta(minutes=5))
    recorder = RecordingExecutor(main.agent_executor)
    main.agent_executor = recorder

    runs = []
    for rep in range(REPEATS):
        session = f"e17_case_study_{rep + 1}"
        chat_engine.DRAFT_STORE.pop(session, None)
        reply = main.chat_endpoint(main.ChatRequest(message=REQUEST, session_id=session))
        inputs, out = recorder.calls[-1]
        msgs = serialise_messages(out["messages"])
        tools = [c["name"] for m in msgs for c in m.get("tool_calls", [])]
        draft = reply["draft"] or ""
        prio = re.search(r"Priority\s*[:\-]?\s*\**\s*(High|Medium|Low)", draft, re.I)
        runs.append({"session": session, "tool_calls": tools, "response": reply["response"], "draft": draft,
                     "priority": prio.group(1).title() if prio else None, "messages": msgs})
        print(f"[INFO] run {rep + 1}: tools={tools} priority={runs[-1]['priority']}")
    context = recorder.calls[0][0]["machine_state"]

    with open(os.path.join(res_dir, "e17_context.json"), "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, default=str)
    with open(os.path.join(res_dir, "e17_runs.json"), "w", encoding="utf-8") as f:
        json.dump({"now": str(NOW), "request": REQUEST, "runs": runs}, f, indent=2, default=str)

    alert = main.current_alert()
    lvl_changes = cyc["alert_level"].ne(cyc["alert_level"].shift()).loc[lambda s: s]
    lines = [
        f"E17 case-study replay at {NOW} ({time.time() - t0:.0f} s)",
        f"z_rms = {cyc['z_rms'].iloc[-1]:.2f} mm/s, zone {alert['zone']}, alert level {alert['level']}",
        "reasons: " + " | ".join(alert["reasons"]),
        f"shift episode (review assumed at {REVIEW_AT}): {st.shift_episode}",
        f"episode without any review since {cycles[0]}: {no_review_episode}",
        f"z_rms forecast +6 h: {float(st.forecast['z_rms'].iloc[-1]):.3f} mm/s; recent errors: "
        f"{main.recent_forecast_errors(st.data, 'z_rms')}",
        "channels flagged at NOW: " + ", ".join(c for c, s in st.shift_status.items() if s["latest_flag"]),
        "alert level changes (cycle: level):",
        *[f"  {ts}: {cyc.at[ts, 'alert_level']}" for ts in lvl_changes.index],
        f"first flag per channel since {REVIEW_AT}:",
        *[f"  {c}: {cyc.index[(cyc.index > REVIEW_AT) & cyc[f'{c}_flag']].min()}" for c in R.SHIFT_CHANNELS],
        f"agent runs: {REPEATS}; tools per run: {[r['tool_calls'] for r in runs]}; "
        f"priorities: {[r['priority'] for r in runs]}",
        f"flag mismatches against e16_steps.csv: {mismatches}",
        f"live work-order store: {len(stored)} records, earliest {stored[0] if stored else None}",
    ]
    with open(os.path.join(res_dir, "e17_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main_replay()
