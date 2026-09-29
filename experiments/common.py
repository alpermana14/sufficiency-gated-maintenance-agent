"""Shared helpers for revision experiments (E1-E12).

Data source resolution order:
  1. PM_DATA_CSV env var (path to a CSV export with a 'datetime' column)
  2. data/conveyor_export.csv (produced by scripts/export_data.py)
  3. Live MySQL via backend/ml_engine.load_conveyor_data() (needs backend/.env)

IMPORTANT: backend/ is inserted at the FRONT of sys.path and ml_engine is
imported flat (not as backend.ml_engine). Importing it as a package from the
repo root makes its internal `from IDK_square_sliding import ...` fail and
silently swaps in a MOCK random-score IDK — which would invalidate every
anomaly experiment.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

for _p in (BACKEND_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from IDK_square_sliding import IDK_square_sliding  # noqa: E402  (real IDK, not mock)
import ml_engine  # noqa: E402

TARGETS = ml_engine.TARGETS

# Labelled anomaly window: the 2 kg -> 40 kg load-change event (override per
# experiment run, e.g. for staged-fault sessions in E6).
EVENT_START = pd.Timestamp(os.environ.get("PM_EVENT_START", "2026-05-05 00:00:00"))
EVENT_END = pd.Timestamp(os.environ.get("PM_EVENT_END", "2026-05-06 00:00:00"))


def load_data() -> pd.DataFrame:
    """Return the cleaned conveyor dataframe with a datetime index."""
    csv_path = os.environ.get(
        "PM_DATA_CSV", os.path.join(REPO_ROOT, "data", "conveyor_export.csv")
    )
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "datetime" not in df.columns:
            raise ValueError(f"{csv_path} must contain a 'datetime' column")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
    else:
        print(f"[INFO] No CSV at {csv_path}; falling back to MySQL.")
        df = ml_engine.load_conveyor_data()
        if df is None or df.empty:
            raise RuntimeError(
                "No data available. Run scripts/export_data.py or set PM_DATA_CSV."
            )
    missing = [c for c in TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing target columns: {missing}")
    return df


def event_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """1 for timestamps inside the labelled anomaly window, else 0."""
    return ((index >= EVENT_START) & (index < EVENT_END)).astype(int)


def slice_eval_range(
    df: pd.DataFrame, days_before: float = 7.0, days_after: float = 1.0
) -> pd.DataFrame:
    """Evaluation window sliced AROUND the labelled event.

    The table keeps growing after the event, so taking the tail of the data
    would miss the labelled window entirely — always slice relative to
    EVENT_START/EVENT_END instead.
    """
    lo = EVENT_START - pd.Timedelta(days=days_before)
    hi = EVENT_END + pd.Timedelta(days=days_after)
    out = df.loc[(df.index >= lo) & (df.index < hi)]
    if out.empty:
        raise RuntimeError(
            f"No data between {lo} and {hi}. Check PM_EVENT_START/PM_EVENT_END "
            f"against the dataset range {df.index[0]} .. {df.index[-1]}."
        )
    return out


def idk_window_scores(
    values: np.ndarray, width: int, psi1: int, psi2: int, t: int = 100
) -> np.ndarray:
    """Run s-IDK^2; returns one similarity score per sliding window.

    Score i covers points [i, i+width-1]; align to the ORIGINAL series by
    assigning it to the window-end index (i + width - 1). Low similarity =
    anomalous, so use the NEGATED score wherever "higher = more anomalous"
    is expected (e.g. sklearn AUROC).
    """
    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scores = np.asarray(IDK_square_sliding(X, width=width, psi1=psi1, psi2=psi2, t=t))
    return scores.flatten()


def align_window_labels(labels: np.ndarray, width: int) -> np.ndarray:
    """Labels for window-end alignment: window i ends at point i + width - 1."""
    return labels[width - 1 :]


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)  # IDK.py samples via the stdlib `random` module
    np.random.seed(seed)
