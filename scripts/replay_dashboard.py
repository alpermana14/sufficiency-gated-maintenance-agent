"""Serve the dashboard API and the copilot on a system state rebuilt at a past time (local use only).

Used for the Section 5.4 dashboard figure. The state is rebuilt exactly as in E17
(experiments/e17_case_study_replay.py): recorded data up to E17_NOW, every 30-min cycle of the previous 7 days,
s-IDK^2 seed E17_SEED, shift episodes before E17_REVIEW_AT treated as reviewed, tuned LightGBM with the forecast
log of the last E17_FORECAST_HOURS (cached in experiments/results/e17_forecasts.pkl). No new data are loaded, the
scheduler does nothing, the work-order store is an empty temporary one and the audit log is a temporary file.
The summary endpoint reports "replay" so the dashboard labels the reading and hides the stale-data warning.

Usage (from the repository root):
  python scripts/replay_dashboard.py                      # 5 May 2026 15:00, as in Section 5.4
  E17_NOW="2026-05-07 12:00" E17_FORECAST_HOURS=0.5 python scripts/replay_dashboard.py
Then, in frontend/: npm run dev:local   (or npm run build:local && npm run preview:local)
Env: REPLAY_PORT (default 8000)
"""
import os
import sys

import pandas as pd
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

import e17_case_study_replay as E  # noqa: E402  (sets a temporary audit log and imports the backend)
from langchain_chroma import Chroma  # noqa: E402

PORT = int(os.environ.get("REPLAY_PORT", "8000"))


def main():
    E.rebuild_state(write_outputs=False)
    m = E.main
    # feature importance for the diagnostics panel (the forecast cache holds forecasts only)
    _, _, _, m.state.importance, m.state.models = E.ml_engine.run_pipeline(m.state.data)
    m.update_machine_state = lambda: None                # startup and scheduler: no new data
    m.itwin_bridge.setup_itwin_sensors = lambda: None
    m.itwin_bridge.push_to_itwin = lambda *_a, **_k: False
    E.chat_engine.vectorstore_history = Chroma(persist_directory=os.path.join(E.TMP, "history"),
                                               embedding_function=E.chat_engine.embeddings)
    clock = E.NOW + pd.Timedelta(minutes=5)
    m.datetime = E.fake_clock(clock)                     # the copilot sees the data as real time
    m.state.last_update = str(clock)
    m.state.replay_label = f"replay of recorded data up to {E.NOW:%d %b %Y %H:%M}"
    alert = m.current_alert()
    print(f"[REPLAY] {E.NOW}: zone {alert['zone']}, alert level {alert['level']}; "
          f"episode {m.state.shift_episode}; reminder {m.current_reminder()}")
    uvicorn.run(m.app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
