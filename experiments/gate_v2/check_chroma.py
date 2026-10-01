"""Gate v2 - does the vector database change what the gate sees? (R2-21)

The evaluation of Section 5.3.2 searches the 20 stored records exactly. The deployed system searches them through
ChromaDB with a hierarchical navigable small world (HNSW) index. This script puts the same text-embedding-3-small
vectors (the cached ones used by the evaluation) into a ChromaDB collection with the deployed default settings, runs
all 200 requests through both searches, and compares the three records returned, their distances and the decision
of the distance floor. Feeding the same vectors to both isolates the index from small differences between repeated
calls to the embedding service.

Usage: python experiments/gate_v2/check_chroma.py
Output: experiments/results/e21_chroma_check.csv (one row per request) and a printed summary
"""
import os
import sys

import chromadb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retrieval  # noqa: E402

K = 3
FLOOR = 1.10  # distance floor of the deployed gate (Section 3.5)


def main():
    texts = [r["text"] for r in retrieval.load_store()["records"]]
    queries = [q["query"] for q in retrieval.load_queries()["queries"]]
    exact = retrieval.Retriever("openai-small", texts)
    exact.prepare(queries)
    col = chromadb.EphemeralClient().create_collection("gate_check")  # HNSW index, squared L2, library defaults
    col.add(ids=[str(i) for i in range(len(texts))], embeddings=exact.doc_vecs.tolist(), documents=texts)
    rows = []
    for q in queries:
        res = col.query(query_embeddings=[exact.query_vecs[q].tolist()], n_results=K)
        got = [int(i) for i in res["ids"][0]]
        dist = [float(s) ** 0.5 for s in res["distances"][0]]
        mine = exact.search(q, K)
        rows.append({"query": q, "same_top3_in_order": got == [i for i, _ in mine],
                     "same_floor_decision": (dist[0] <= FLOOR) == (mine[0][1] <= FLOOR),
                     "max_abs_distance_diff": max(abs(a - b) for a, (_, b) in zip(dist, mine))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(HERE), "results", "e21_chroma_check.csv"), index=False)
    print(f"requests: {len(df)}")
    print(f"same three records in the same order: {int(df.same_top3_in_order.sum())}/{len(df)}")
    print(f"same distance-floor decision: {int(df.same_floor_decision.sum())}/{len(df)}")
    print(f"largest distance difference: {df.max_abs_distance_diff.max():.2e}")


if __name__ == "__main__":
    main()
