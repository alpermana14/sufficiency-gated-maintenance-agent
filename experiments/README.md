# Experiments

Every script reads `../data/conveyor_export.csv` (or the file named by `PM_DATA_CSV`) and writes its outputs to
`results/`. The table in the top-level `README.md` maps each script to the table or section of the paper it
produces. `common.py` holds the shared data loading and must be used instead of importing `backend` as a
package, so that the real s-IDK² implementation is loaded.

Scripts that call a language model need `OPENAI_API_KEY` in `../backend/.env`. The load experiment is labelled
with `PM_EVENT_START="2026-05-05 08:00:00"` and `PM_EVENT_END="2026-05-07 12:00:00"`.

`agent_benchmark_v2/` holds the 120 benchmark requests and their fixtures, `gate_v2/` the store of 20 approved
work orders and the 200 gate requests. Scripts that read or write the rating spreadsheets expect them under
`../CAEE_R1/rating/`; the spreadsheets themselves are provided with the supplementary material of the paper.
