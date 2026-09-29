"""E32: access control of the approval endpoints and tamper detection of the audit log (R1-11, R2-20).

The endpoints are exercised through FastAPI's TestClient against the backend exactly as it is in
backend/main.py. Three things are isolated so the test changes nothing that the running system uses:
the audit log is written to a temporary file (AUDIT_LOG_PATH), the store of approved work orders is
replaced by an in-memory stand-in, and the startup event (scheduler, database) is not run. No language
model is called. The approver tokens used here are dummy values set for this process only; backend/.env
is not read for them and is not changed.

Output: experiments/results/e32_access_control.csv and e32_access_control.log
Usage:  python experiments/e32_access_control.py
"""
import csv
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
RESULTS = os.path.join(ROOT, "experiments", "results")
REAL_AUDIT = os.path.join(BACKEND, "audit_log.jsonl")

tmpdir = tempfile.mkdtemp(prefix="e32_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(tmpdir, "audit_log.jsonl")
os.environ["APPROVER_TOKENS"] = "Operator A:tok-a-7f3c,Operator B:tok-b-91d2"
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

real_audit_before = os.path.getsize(REAL_AUDIT) if os.path.exists(REAL_AUDIT) else None

import main  # noqa: E402
import audit  # noqa: E402
import chat_engine  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

real_store_before = chat_engine.vectorstore_history._collection.count()


class StoreStandIn:
    """Records what the endpoint would have written to the store of approved work orders."""
    def __init__(self):
        self.docs = []

    def add_documents(self, docs):
        self.docs.extend(docs)


store = StoreStandIn()
main.vectorstore_history = store
client = TestClient(main.app)          # not used as a context manager, so startup does not run
TOK_A = {"X-Approver-Token": "tok-a-7f3c"}
DRAFT = "Incident Report: z_rms 3.54 mm/s, zone C.\nRoot Cause Analysis: belt tension.\nPriority: Low"
rows = []


def open_episode():
    main.state.shift_episode = {"since": "2026-05-05 08:30:00", "channels": ["noise"],
                                "currently_flagged": ["noise"], "reviewed": False, "reviewed_by": None}


def audit_events():
    path = os.environ["AUDIT_LOG_PATH"]
    if not os.path.exists(path):
        return []
    return [json.loads(l)["event"] for l in open(path, encoding="utf-8") if l.strip()]


def case(cid, group, description, expected, passed, observed):
    rows.append({"case": cid, "group": group, "description": description, "expected": expected,
                 "observed": observed, "pass": "yes" if passed else "NO"})


# ---------------- A. access control of the approval endpoints ----------------
os.environ["APPROVER_TOKENS"] = ""
main.DRAFT_STORE["s1"] = DRAFT
r = client.post("/api/work_orders/approve", json={"session_id": "s1"}, headers=TOK_A)
case("A1", "access control", "approve while no approver is configured", "refused (503), nothing stored",
     r.status_code == 503 and not store.docs, f"HTTP {r.status_code}, stored {len(store.docs)}")
os.environ["APPROVER_TOKENS"] = "Operator A:tok-a-7f3c,Operator B:tok-b-91d2"

n0 = audit_events().count("approval_denied")
r = client.post("/api/work_orders/approve", json={"session_id": "s1"})
case("A2", "access control", "approve without a token", "refused (401), attempt logged, nothing stored",
     r.status_code == 401 and not store.docs and audit_events().count("approval_denied") == n0 + 1,
     f"HTTP {r.status_code}, stored {len(store.docs)}, denied events +{audit_events().count('approval_denied') - n0}")

n0 = audit_events().count("approval_denied")
r = client.post("/api/work_orders/approve", json={"session_id": "s1"}, headers={"X-Approver-Token": "tok-a-7f3d"})
case("A3", "access control", "approve with a token that differs in one character",
     "refused (401), attempt logged, nothing stored",
     r.status_code == 401 and not store.docs and audit_events().count("approval_denied") == n0 + 1,
     f"HTTP {r.status_code}, stored {len(store.docs)}, denied events +{audit_events().count('approval_denied') - n0}")

r = client.post("/api/work_orders/approve", json={"session_id": "s1"}, headers={"X-Approver-Token": "s1"})
case("A4", "access control", "approve with the chat session identifier used as the token",
     "refused (401), nothing stored", r.status_code == 401 and not store.docs,
     f"HTTP {r.status_code}, stored {len(store.docs)}")

r = client.post("/api/work_orders/approve", json={"session_id": "no-such-session"}, headers=TOK_A)
case("A5", "access control", "approve a session that has no draft", "refused (400), nothing stored",
     r.status_code == 400 and not store.docs, f"HTTP {r.status_code}, stored {len(store.docs)}")

open_episode()
r = client.post("/api/work_orders/approve", json={"session_id": "s1"}, headers=TOK_A)
meta = store.docs[-1].metadata if store.docs else {}
ok = (r.status_code == 200 and len(store.docs) == 1 and meta.get("approved_by") == "Operator A"
      and meta.get("modified_by_operator") is False and meta.get("draft_sha256") == meta.get("content_sha256")
      and main.DRAFT_STORE.get("s1") == "" and main.state.shift_episode.get("reviewed") is True
      and main.state.shift_episode.get("reviewed_by") == "Operator A")
case("A6", "access control", "approve an unedited draft with a valid token",
     "stored once with approver name, unedited flag and equal hashes; draft cleared; episode reviewed",
     ok, f"HTTP {r.status_code}, stored {len(store.docs)}, approved_by={meta.get('approved_by')}, "
         f"modified={meta.get('modified_by_operator')}, episode reviewed by {main.state.shift_episode.get('reviewed_by')}")

main.DRAFT_STORE["s2"] = DRAFT
edited = DRAFT.replace("belt tension", "belt tension and idler wear")
r = client.post("/api/work_orders/approve", json={"session_id": "s2", "content": edited},
                headers={"X-Approver-Token": "tok-b-91d2"})
meta = store.docs[-1].metadata if len(store.docs) == 2 else {}
ok = (r.status_code == 200 and len(store.docs) == 2 and store.docs[-1].page_content == edited.replace("*", "").strip()
      and meta.get("approved_by") == "Operator B" and meta.get("modified_by_operator") is True
      and meta.get("draft_sha256") != meta.get("content_sha256"))
case("A7", "access control", "approve an edited draft with a second operator's token",
     "the edited text is stored, attributed to that operator, flagged as edited, hashes differ",
     ok, f"HTTP {r.status_code}, stored {len(store.docs)}, approved_by={meta.get('approved_by')}, "
         f"modified={meta.get('modified_by_operator')}")

main.DRAFT_STORE["s3"] = DRAFT
r = client.post("/api/work_orders/reject", json={"session_id": "s3", "reason": "  "}, headers=TOK_A)
case("A8", "access control", "reject without a reason", "refused (400), draft kept, nothing stored",
     r.status_code == 400 and main.DRAFT_STORE.get("s3") == DRAFT and len(store.docs) == 2,
     f"HTTP {r.status_code}, draft kept {main.DRAFT_STORE.get('s3') == DRAFT}, stored {len(store.docs)}")

r = client.post("/api/work_orders/reject", json={"session_id": "s3", "reason": "wrong cause"})
case("A9", "access control", "reject without a token", "refused (401), draft kept",
     r.status_code == 401 and main.DRAFT_STORE.get("s3") == DRAFT,
     f"HTTP {r.status_code}, draft kept {main.DRAFT_STORE.get('s3') == DRAFT}")

r = client.post("/api/work_orders/reject", json={"session_id": "s3", "reason": "wrong cause"}, headers=TOK_A)
case("A10", "access control", "reject with a valid token and a reason",
     "accepted, draft cleared, nothing stored, rejection logged",
     r.status_code == 200 and main.DRAFT_STORE.get("s3") == "" and len(store.docs) == 2
     and "work_order_rejected" in audit_events(),
     f"HTTP {r.status_code}, stored {len(store.docs)}, rejection logged {'work_order_rejected' in audit_events()}")

open_episode()
r = client.post("/api/shift/review", json={"note": "checked"}, headers={"X-Approver-Token": "wrong"})
case("A11", "access control", "mark a shift episode reviewed with a wrong token", "refused (401), episode stays open",
     r.status_code == 401 and main.state.shift_episode.get("reviewed") is False,
     f"HTTP {r.status_code}, episode reviewed={main.state.shift_episode.get('reviewed')}")

# ---------------- B. tamper detection of the audit log ----------------
path = os.environ["AUDIT_LOG_PATH"]
intact, n = audit.verify_chain()
case("B1", "audit log", f"verify the log written by cases A1 to A11 ({n} records)", "chain intact",
     intact and n > 0, f"intact={intact}, records={n}")
lines = open(path, encoding="utf-8").read().splitlines()


def check(cid, description, expected_intact, new_lines, expected):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    got, upto = audit.verify_chain()
    case(cid, "audit log", description, expected, got == expected_intact,
         f"intact={got}" + ("" if got else f", break found after record {upto}"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


mid = len(lines) // 2
rec = json.loads(lines[mid]); rec["approver"] = rec.get("approver", "") + "X" if "approver" in rec else "X"
check("B2", "edit one field of a record in the middle of the log", False,
      lines[:mid] + [json.dumps(rec, ensure_ascii=False, sort_keys=True)] + lines[mid + 1:], "break detected")
check("B3", "delete one record in the middle of the log", False, lines[:mid] + lines[mid + 1:], "break detected")
check("B4", "swap two adjacent records", False, lines[:mid] + [lines[mid + 1], lines[mid]] + lines[mid + 2:],
      "break detected")
check("B5", "delete the last record", True, lines[:-1],
      "not detectable by the chain alone (no later record holds its hash)")

# ---------------- isolation checks ----------------
real_audit_after = os.path.getsize(REAL_AUDIT) if os.path.exists(REAL_AUDIT) else None
real_store_after = chat_engine.vectorstore_history._collection.count()
case("C1", "isolation", "the real audit log is untouched", "same size before and after",
     real_audit_before == real_audit_after, f"{real_audit_before} -> {real_audit_after} bytes")
case("C2", "isolation", "the real store of approved work orders is untouched", "same record count",
     real_store_before == real_store_after, f"{real_store_before} -> {real_store_after} records")

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "e32_access_control.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
lines_out = [f"{r['case']:4s} {r['pass']:3s} {r['description']}  ->  {r['observed']}" for r in rows]
summary = f"\npassed {sum(r['pass'] == 'yes' for r in rows)} of {len(rows)}"
with open(os.path.join(RESULTS, "e32_access_control.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out) + summary + "\n")
print("\n".join(lines_out) + summary)
