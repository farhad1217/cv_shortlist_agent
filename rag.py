"""Qdrant RAG: knowledge-base documents (with citations) + job postings (per-job filter)."""
import hashlib
import math
import os
import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, MatchValue,
                                  PointStruct, VectorParams)

BASE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE, "knowledge_base")
DIM = 384

# ---------------- embeddings ----------------
_embedder = None
EMB_NAME = "hash"


def _hash_embed(text):
    """Dependency-free fallback embedder (bag of hashed words). Works offline, lower quality."""
    v = [0.0] * DIM
    for w in re.findall(r"[a-z0-9\+#\.]{2,}", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 100) & 1 else -1.0
        v[(h >> 20) % DIM] += 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed(texts):
    global _embedder, EMB_NAME
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
            EMB_NAME = "bge"
        except Exception as e:  # noqa
            print(f"[rag] fastembed unavailable ({e}); using hash embedder")
            _embedder = "hash"
    if _embedder == "hash":
        return [_hash_embed(t) for t in texts]
    return [list(map(float, v)) for v in _embedder.embed(texts)]


# ---------------- client ----------------
_client = None


def client():
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "").strip()
        if url:
            _client = QdrantClient(url=url)
        else:
            _client = QdrantClient(path=os.path.join(BASE, "data", "qdrant"))
    return _client


def _col(kind):
    embed(["warmup"])  # makes sure EMB_NAME is resolved
    return f"{kind}_{EMB_NAME}"


def _ensure(name):
    c = client()
    if not c.collection_exists(name):
        c.create_collection(name, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))


def chunk(text, size=550):
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=#)", text):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) > size and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


# ---------------- knowledge base ----------------
def ensure_kb():
    """(Re)index the markdown files in knowledge_base/ when they change."""
    files = sorted(f for f in os.listdir(KB_DIR) if f.endswith(".md"))
    sig = hashlib.md5("".join(open(os.path.join(KB_DIR, f), encoding="utf-8").read() for f in files)
                      .encode()).hexdigest()
    name = _col("kb")
    stamp = os.path.join(BASE, "data", f"{name}.sig")
    c = client()
    if c.collection_exists(name) and os.path.exists(stamp) and open(stamp).read() == sig:
        return
    if c.collection_exists(name):
        c.delete_collection(name)
    _ensure(name)
    pts = []
    for f in files:
        for i, ch in enumerate(chunk(open(os.path.join(KB_DIR, f), encoding="utf-8").read())):
            pts.append((f, i, ch))
    vecs = embed([p[2] for p in pts])
    c.upsert(name, points=[
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{f}#{i}")), vector=v,
                    payload={"source": f, "chunk": i, "text": ch})
        for (f, i, ch), v in zip(pts, vecs)])
    open(stamp, "w").write(sig)


def search_kb(query, k=3, min_score=0.0):
    ensure_kb()
    res = client().query_points(_col("kb"), query=embed([query])[0], limit=k).points
    return [{"cite": f"{p.payload['source']}#{p.payload['chunk']}", "text": p.payload["text"],
             "score": float(p.score)} for p in res if p.score >= min_score]


# ---------------- job postings ----------------
def index_job(job):
    name = _col("jobs")
    _ensure(name)
    c = client()
    delete_job(job["id"])
    body = (f"{job['title']}\nRequired skills: {', '.join(job['skills'])}\n"
            f"Minimum experience: {job['min_years']} years\n{job['description']}\n"
            + "\n".join(f"{e['name']}: {e['value']}" for e in job.get("extras", [])))
    chunks = chunk(body, 420) or [job["title"]]
    vecs = embed(chunks)
    c.upsert(name, points=[
        PointStruct(id=int(job["id"]) * 1000 + i, vector=v,
                    payload={"job_id": int(job["id"]), "hr_id": int(job["hr_id"]), "chunk": i, "text": ch})
        for i, (ch, v) in enumerate(zip(chunks, vecs))])


def delete_job(job_id):
    name = _col("jobs")
    if client().collection_exists(name):
        client().delete(name, points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="job_id", match=MatchValue(value=int(job_id)))])))


def search_job_chunks(job_id, query, k=4):
    """Retrieve requirement chunks of ONE job only (isolation by payload filter)."""
    name = _col("jobs")
    if not client().collection_exists(name):
        return []
    res = client().query_points(
        name, query=embed([query])[0], limit=k,
        query_filter=Filter(must=[FieldCondition(key="job_id", match=MatchValue(value=int(job_id)))])).points
    return [{"cite": f"J{i + 1}", "text": p.payload["text"], "score": float(p.score)}
            for i, p in enumerate(res)]


def match_jobs(query, k=30):
    """Semantic job search across all postings -> [(job_id, best_score)]."""
    name = _col("jobs")
    if not client().collection_exists(name):
        return []
    res = client().query_points(name, query=embed([query])[0], limit=k).points
    best = {}
    for p in res:
        jid = p.payload["job_id"]
        best[jid] = max(best.get(jid, -1), float(p.score))
    return sorted(best.items(), key=lambda x: -x[1])
