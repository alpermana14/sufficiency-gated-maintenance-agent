# Retrieval-sufficiency-gated maintenance agent

Code and data for the paper *A retrieval-sufficiency-gated large language model agent for sensor-based
maintenance decision support on an industrial conveyor* (under review at Computers and Electrical Engineering).

The system joins three layers into one loop, from a sensor record to a work order that an operator approves:

1. **Forecasting**: six LightGBM models forecast the monitored channels six hours ahead (`backend/ml_engine.py`).
2. **Distribution-shift detection**: s-IDK² scores each window against a trailing population and opens a
   shift episode (`backend/IDK_square_sliding.py`, `backend/machine_status.py`).
3. **Agent**: a tool-calling language-model agent drafts work orders from the machine state, the maintenance
   manual and approved work orders. A retrieval-sufficiency gate decides whether the approved work orders
   answer a request before the model sees them (`backend/chat_engine.py`, `backend/rag_guard.py`).

## Repository layout

| Path | Content |
|---|---|
| `backend/` | FastAPI service, analytics pipeline, alert rule, agent, gate and audit log |
| `frontend/` | React dashboard and chat interface (Vite) |
| `data/conveyor_export.csv` | The conveyor dataset (see below) |
| `experiments/` | Scripts that produce every result in the paper, with their outputs in `experiments/results/` |
| `scripts/` | Data export from the plant database and a dashboard replay used for Figure 7 |

## Dataset

`data/conveyor_export.csv` holds 12,916 records at a 30-minute interval, from 24 October 2025 to 20 July 2026,
from a 4 m light-load belt conveyor.

| Column | Unit | Description |
|---|---|---|
| `datetime` | | Timestamp of the record |
| `temperature` | °C | Temperature from the vibration and temperature sensor (QM30VT2) |
| `z_rms`, `x_rms` | mm/s | RMS vibration velocity, vertical and horizontal (QM30VT2) |
| `z_peak`, `x_peak` | g | Peak acceleration, vertical and horizontal (QM30VT2) |
| `noise` | dB | Acoustic noise (IOT-S300NOIS) |

The controlled load experiment of the paper added a 40 kg load on 5 May 2026 at 08:00 and removed it on
7 May 2026 at 12:00. The data partitions are given in Table A.1 of the paper.

## Setup

Python 3.12 and Node.js 20 or later.

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000

# frontend
cd frontend
npm install
npm run dev
```

The backend reads its settings from `backend/.env`. The agent and the gate call the OpenAI API, so `OPENAI_API_KEY` is required for them. The live service reads
the sensor records from a MySQL database. The experiments read `data/conveyor_export.csv` instead
(or the file named by `PM_DATA_CSV`), so they run without a database.

## Reproducing the results

Run the scripts from the `experiments/` directory. Scripts that call a language model need `OPENAI_API_KEY`.

| Paper | Script |
|---|---|
| Tables 2 and 4, Figure 5 | `e01b_forecasting_modelcomp.py`, `e14_forecast_stats.py`, `measure_training_time.py` |
| Horizon, lag window and inference time (Section 5.1) | `e01b_forecasting_modelcomp.py`, `e27_lag_window.py`, `e29_inference_latency.py` |
| Table 5, Figure 6 | `e13_anomaly_fair_eval.py`, `figure6_idk_similarity.py` |
| Sensitivity to parameters and noise (Section 5.2) | `e13_anomaly_fair_eval.py` (grid), `e28_noise_sensitivity.py` |
| Online replay of the shift rule (Section 5.2) | `e16_shift_rule_replay.py` |
| Table 6 | `agent_benchmark_v2/` |
| Tables 7, 8 and 9 | `gate_v2/` (`run_gate.py`, `run_endtoend.py`) |
| Case study and alert-rule states (Section 5.4) | `e17_case_study_replay.py`, `e18_alert_agent_grid.py` |
| Table 10 | `agent_benchmark_v2/run_benchmark.py` with `BENCH_VARIANT=raw_readings` or `no_state` |
| Table 11 and Section 5.5 | `e10_compute_profile.py`, `e26_history_length.py`, `e30_concurrent_load.py`, `e24_detector_timing.py` |

The rating spreadsheets of the two raters and the review-session materials are provided with the
supplementary material of the paper and are not part of this repository.

## Not included

- The maintenance manual of the conveyor, which is copyrighted. The manual tool indexes any PDF placed at
  `backend/Maintenance_Conveyor.pdf`.
- The vector stores built at run time (`backend/maintenance_manual_db`, `backend/maintenance_history_db`)
  and the audit log.

## License

MIT, see `LICENSE`.
