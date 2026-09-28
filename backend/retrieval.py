"""
Retrieval: top-k semantic search over ChromaDB + exact course lookup.

Pure embedding search is weak at matching course codes such as "DATA301" (MiniLM has no idea what
that token means), so any course code or exact course title mentioned in the query is also looked
up directly in the catalogue and its record is added to the results (a small "hybrid" step).
"""
import json
import re
import threading
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, PROCESSED_DIR, TOP_K

CODE_RE = re.compile(r"\b([A-Za-z]{3,4})\s?(\d{3})\b")
MAX_PER_SECTION = 2
POLICY_DOCS = {"Student Handbook Aug 2026.pdf", "SOP_STUDENT_17082026.pdf"}
COVERAGE_MIN_SCORE = 0.5


_encode_lock = threading.Lock()


@lru_cache
def _model():
    # CPU on purpose: a single short query embeds in ~10 ms on CPU, and PyTorch's Apple-GPU (MPS) backend
    # deadlocked when two request threads encoded at the same time.
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(text: str) -> list[list[float]]:
    with _encode_lock:  # the model is shared by all request threads
        return _model().encode([text], normalize_embeddings=True).tolist()


@lru_cache
def _collection():
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


@lru_cache
def _title_index() -> dict[str, str]:
    """lower-case course title -> course code, for exact-title matching."""
    cat = json.loads((PROCESSED_DIR / "course_catalogue.json").read_text())
    return {re.sub(r"\s+", " ", c["title"].lower()).strip(): code for code, c in cat.items() if len(c["title"]) > 6}


def mentioned_courses(text: str) -> list[str]:
    codes = {f"{a.upper()}{b}" for a, b in CODE_RE.findall(text)}
    low = text.lower()
    codes |= {code for title, code in _title_index().items() if title in low}
    return sorted(codes)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return chunks as dicts: {text, source_file, section, score}."""
    col = _collection()
    emb = embed(query)
    # Over-fetch, then keep at most MAX_PER_SECTION chunks per section so one long section (e.g. the
    # malpractice rules for "examination" questions) cannot crowd every other document out of the top k.
    res = col.query(query_embeddings=emb, n_results=k * 4)
    pool = [{"text": d, **m, "score": round(1 - dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
    hits, per_section = [], {}
    for h in pool:
        base = re.sub(r", pp?\. [\d-]+$", "", h["section"])  # same section on another page = same section
        if per_section.get(base, 0) >= MAX_PER_SECTION:
            continue
        per_section[base] = per_section.get(base, 0) + 1
        hits.append(h)
        if len(hits) == k:
            break
    # Cross-document coverage: rules about the same topic often appear in BOTH the Handbook and the SOP (and
    # sometimes disagree). If one of them has a clearly relevant chunk but was crowded out, add its best one.
    have = {h["source_file"] for h in hits}
    for doc in POLICY_DOCS - have:
        best = next((h for h in pool if h["source_file"] == doc and h["score"] >= COVERAGE_MIN_SCORE), None)
        if best:
            hits.append(best)

    # exact course lookups (hybrid step)
    seen = {h["section"] for h in hits}
    for code in mentioned_courses(query):
        got = col.get(where={"course_code": code}, include=["documents", "metadatas"])
        for d, m in zip(got["documents"], got["metadatas"]):
            if m["section"] not in seen:
                hits.append({"text": d, **m, "score": 1.0})
                seen.add(m["section"])
    return hits


def prerequisite_records(chunks: list[dict]) -> list[dict]:
    """One-hop expansion: add catalogue records for the prerequisites of courses already retrieved.
    Used by rag_structured so eligibility checks can see the whole prerequisite chain."""
    col = _collection()
    have = {c["section"] for c in chunks}
    codes = set()
    for c in chunks:
        if c.get("course_code"):
            for line in c["text"].splitlines():
                if "prerequisites:" in line:
                    codes |= {f"{a.upper()}{b}" for a, b in CODE_RE.findall(line.split("prerequisites:")[1])}
    extra = []
    for code in sorted(codes):
        got = col.get(where={"course_code": code}, include=["documents", "metadatas"])
        for d, m in zip(got["documents"], got["metadatas"]):
            if m["section"] not in have:
                extra.append({"text": d, **m, "score": 1.0})
                have.add(m["section"])
    return extra


@lru_cache
def _all_sections() -> tuple[str, ...]:
    return tuple(sorted({m["section"] for m in _collection().get(include=["metadatas"])["metadatas"]}))


def section_records(prefix: str) -> list[dict]:
    """All chunks of one document section, e.g. 'Section III - Academic Regulations > 12. Progression'."""
    names = [s for s in _all_sections() if s.startswith(prefix)]
    if not names:
        return []
    got = _collection().get(where={"section": {"$in": names}}, include=["documents", "metadatas"])
    return [{"text": d, **m, "score": 1.0} for d, m in zip(got["documents"], got["metadatas"])]


@lru_cache
def _structure_sections() -> list[str]:
    got = _collection().get(where={"processed_file": "programme_structure.md"}, include=["metadatas"])
    return sorted({m["section"] for m in got["metadatas"]})


def semester_records(batch: str, semesters: list[int]) -> list[dict]:
    """Structured lookup of the student's own batch/semester course lists from the semester spread."""
    col = _collection()
    prefixes = tuple(f"{batch} Batch - Semester {s} " for s in semesters)  # sections end with "(tab ...)"
    names = [sec for sec in _structure_sections() if sec.startswith(prefixes)]
    if not names:
        return []
    got = col.get(where={"section": {"$in": names}}, include=["documents", "metadatas"])
    return [{"text": d, **m, "score": 1.0} for d, m in zip(got["documents"], got["metadatas"])]
