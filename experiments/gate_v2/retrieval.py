"""Gate v2 - retrieval backends.

Answers CAEE R2-21: is the gate dependent on one embedding model and one vector database?
Four backends are provided behind one interface:

  openai-small   text-embedding-3-small, the model used in the deployed system
  openai-large   text-embedding-3-large, a larger model from the same vendor
  local-bge      BAAI/bge-small-en-v1.5 run locally, no vendor API involved
  bm25           Okapi BM25 lexical matching, a different retrieval architecture
                 with no embedding at all

Search is exact nearest neighbour over the 20 stored records. The deployed system uses
ChromaDB with a hierarchical navigable small world index; for a store of this size that
index returns the exact neighbours, and verify_chroma() checks that claim rather than
assuming it.

Distances are L2 on unit-norm vectors, so d^2 = 2 - 2 cos, which is what the deployed
distance floor is expressed in. BM25 returns a score where higher is better, so it is
converted to a distance-like quantity by negation; its floor is calibrated separately,
because the scales are not comparable.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "emb_cache")
os.makedirs(CACHE, exist_ok=True)
BACKENDS = ["openai-small", "openai-large", "local-bge", "bm25"]
OPENAI_MODEL = {"openai-small": "text-embedding-3-small", "openai-large": "text-embedding-3-large"}
LOCAL_MODEL = {"local-bge": "BAAI/bge-small-en-v1.5"}


def _key(backend, texts):
    h = hashlib.sha1(("\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE, f"{backend}_{h}.npy")


def _load_env():
    backend_dir = os.path.join(os.path.dirname(HERE), "..", "backend")
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.abspath(backend_dir), ".env"))
    except ImportError:
        pass


def embed(backend, texts):
    """Unit-norm embeddings for texts, cached on disk so a rerun costs nothing."""
    path = _key(backend, texts)
    if os.path.exists(path):
        return np.load(path)
    if backend in OPENAI_MODEL:
        _load_env()
        from openai import OpenAI
        client = OpenAI()
        out = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            resp = client.embeddings.create(model=OPENAI_MODEL[backend], input=chunk)
            out.extend([d.embedding for d in resp.data])
        arr = np.asarray(out, dtype=np.float32)
    elif backend in LOCAL_MODEL:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_MODEL[backend])
        arr = np.asarray(model.encode(texts, normalize_embeddings=False), dtype=np.float32)
    else:
        raise ValueError(backend)
    arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
    np.save(path, arr)
    return arr


class Retriever:
    """Ranks the stored records for a query and returns (record_index, distance) pairs.

    For the embedding backends the distance is the L2 distance between unit-norm vectors.
    For BM25 it is the negated BM25 score, so that "smaller is closer" holds everywhere.
    """

    def __init__(self, backend, record_texts):
        self.backend = backend
        self.record_texts = list(record_texts)
        if backend == "bm25":
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([t.lower().split() for t in self.record_texts])
        else:
            self.doc_vecs = embed(backend, self.record_texts)

    def prepare(self, queries):
        """Embed all queries at once (one API round trip per 64)."""
        if self.backend != "bm25":
            self.query_vecs = {q: v for q, v in zip(queries, embed(self.backend, list(queries)))}

    def search(self, query, k):
        if self.backend == "bm25":
            scores = np.asarray(self.bm25.get_scores(query.lower().split()), dtype=float)
            order = np.argsort(-scores)[:k]
            return [(int(i), float(-scores[i])) for i in order]
        v = self.query_vecs[query] if hasattr(self, "query_vecs") else embed(self.backend, [query])[0]
        d = np.linalg.norm(self.doc_vecs - v[None, :], axis=1)
        order = np.argsort(d)[:k]
        return [(int(i), float(d[i])) for i in order]


def verify_chroma(record_texts, queries, k=3):
    """Check that the deployed ChromaDB index returns the same neighbours as exact search.

    Returns (agreement_top1, max_abs_distance_difference, n_checked) or None if ChromaDB or
    the API key is unavailable.
    """
    try:
        _load_env()
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        return None
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gate_v2_chroma_")
    try:
        emb = OpenAIEmbeddings(model=OPENAI_MODEL["openai-small"])
        store = Chroma.from_texts(record_texts, emb, persist_directory=tmp)
        exact = Retriever("openai-small", record_texts)
        exact.prepare(queries)
        same, diffs = 0, []
        for q in queries:
            hits = store.similarity_search_with_score(q, k=k)
            mine = exact.search(q, k)
            same += int(hits[0][0].page_content == record_texts[mine[0][0]])
            diffs.append(abs(float(hits[0][1]) ** 0.5 - mine[0][1]) if hits[0][1] >= 0 else float("nan"))
        return same / len(queries), float(np.nanmax(diffs)), len(queries)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_store():
    with open(os.path.join(HERE, "store.json"), encoding="utf-8") as f:
        return json.load(f)


def load_queries():
    with open(os.path.join(HERE, "queries.json"), encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    store = load_store()
    texts = [r["text"] for r in store["records"]]
    qs = [q["query"] for q in load_queries()["queries"]]
    backend = sys.argv[1] if len(sys.argv) > 1 else "bm25"
    r = Retriever(backend, texts)
    r.prepare(qs)
    for q in qs[:3]:
        hits = r.search(q, 3)
        print(f"{q[:60]:62s} -> " + ", ".join(f"{store['records'][i]['id']}:{d:.3f}" for i, d in hits))
