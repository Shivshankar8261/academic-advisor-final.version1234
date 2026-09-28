"""
Step 2 of the data pipeline: chunk the processed documents, embed them with
sentence-transformers (all-MiniLM-L6-v2, runs locally, no API cost) and store them in a
persistent ChromaDB collection.

Chunking strategy
  * Markdown documents are split on headings (## / ###), so a chunk never mixes two clauses.
    Long clauses (e.g. handbook clause 8, Grading) are split further on sub-clause numbers
    ("8.13.3") or paragraphs into ~1000-character pieces (MiniLM only embeds ~256 tokens).
  * The course catalogue JSON is split one record per course.
Every chunk carries metadata used for citations: source_file (the ORIGINAL university document)
and section (heading path, e.g. "Academic Regulations > 7. Attendance Requirements (7.3)").

Run:  python ingest.py
"""
import json
import re

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, PROCESSED_DIR

MAX_CHARS = 1000

# processed file -> original document it was extracted from (what we cite to the student)
DOCS = {
    "handbook.md": "Student Handbook Aug 2026.pdf",
    "sop.md": "SOP_STUDENT_17082026.pdf",
    "programme_structure.md": "Semester_Spread_Structures_Sept_2026.xlsx",
    "minors.md": "Minor_Courses_for_BTech_Students.xlsx",
    "structure_summary.md": "Semester_Spread_Structures_Sept_2026.xlsx",
    "term_context.md": "term_context.md (derived)",
}


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[tuple[str, str]]:
    """Split text into <= max_chars pieces, preferring sub-clause boundaries. Returns (label, text)."""
    if len(text) <= max_chars:
        return [("", text)]
    units = re.split(r"\n(?=\d{1,2}\.\d{1,2}(?:\.\d)?\.?\s)|\n(?=- )|\n\n", text)
    pieces, buf = [], ""
    for u in units:
        while len(u) > max_chars:  # a single huge unit: hard-wrap on line boundaries
            cut = u.rfind("\n", 0, max_chars) or max_chars
            pieces.append(buf + u[:cut]) if buf else pieces.append(u[:cut])
            buf, u = "", u[cut:]
        if len(buf) + len(u) > max_chars and buf:
            pieces.append(buf)
            buf = ""
        buf = f"{buf}\n{u}" if buf else u
    if buf:
        pieces.append(buf)
    out = []
    for p in pieces:
        m = re.search(r"^(\d{1,2}\.\d{1,2}(?:\.\d)?)", p.strip(), re.M)
        out.append((f" ({m.group(1)})" if m else "", p.strip()))
    return out


PAGE_RE = re.compile(r"<!-- page:(\d+) -->\n?")


def page_label(pages: list[int]) -> str:
    return "" if not pages else f", p. {pages[0]}" if pages[0] == pages[-1] else f", pp. {pages[0]}-{pages[-1]}"


def chunk_markdown(fname: str, text: str) -> list[dict]:
    chunks, path = [], {}
    doc_title = text.splitlines()[0].lstrip("# ").strip()
    blocks = re.split(r"\n(?=#{2,3} )", text)
    page = None  # page in effect (PDF-derived documents carry <!-- page:N --> markers)
    for block in blocks:
        m = re.match(r"(#{2,3}) (.+)", block)
        if m:
            level = len(m.group(1))
            path[level] = m.group(2).strip()
            if level == 2:
                path.pop(3, None)
            body = block[m.end():].strip()
        else:
            body = block.strip()
        section = " > ".join(path[k] for k in sorted(path)) or doc_title
        for label, piece in split_long(body):
            markers = [int(x) for x in PAGE_RE.findall(piece)]
            # a piece that starts with a marker begins on that page, otherwise on the page in effect
            starts_with_marker = piece.lstrip().startswith("<!-- page:")
            pages = sorted({*([] if starts_with_marker or page is None else [page]), *markers})
            if markers:
                page = markers[-1]
            piece = PAGE_RE.sub("", piece).strip()
            if len(piece) < 40:  # headings-only / divider blocks
                continue
            chunks.append({"text": f"[{section}]\n{piece}", "source_file": DOCS[fname],
                           "section": section + label + page_label(pages), "processed_file": fname,
                           "pages": ",".join(map(str, pages))})
    return chunks


def chunk_catalogue() -> list[dict]:
    cat = json.loads((PROCESSED_DIR / "course_catalogue.json").read_text())
    chunks = []
    for code, c in cat.items():
        lines = [f"Course {code}: {c['title']} | credits: {c['credits']} | L-T-P: {c.get('L-T-P')} | basket: {c['basket']}"]
        for batch, b in sorted(c["by_batch"].items()):
            lines.append(f"- {batch} batch: semester {b['semester']}, prerequisites: {b['prerequisites_raw'] or 'NIL'}")
            if b.get("choice_of"):  # own line: prerequisite_records() reads codes after "prerequisites:"
                lines.append(f"  ({batch} batch offers it as an {b['choice_of']} - the student takes one alternative)")
        if c["prerequisites_differ_by_batch"]:
            lines.append("NOTE: prerequisites differ between batches.")
        if c["unresolved_prerequisites"]:
            lines.append(f"NOTE: prerequisite code(s) {', '.join(c['unresolved_prerequisites'])} do not appear "
                         "anywhere in the supplied course data (title/details unknown).")
        source = "Minor_Courses_for_BTech_Students.xlsx" if c["basket"].startswith("Minor") \
            else "Semester_Spread_Structures_Sept_2026.xlsx"
        chunks.append({"text": "\n".join(lines), "source_file": source,
                       "section": f"Course {code} {c['title']}", "processed_file": "course_catalogue.json",
                       "course_code": code})
    return chunks


def build_chunks() -> list[dict]:
    chunks = []
    for fname in DOCS:
        chunks += chunk_markdown(fname, (PROCESSED_DIR / fname).read_text())
    chunks += chunk_catalogue()
    return chunks


def main():
    chunks = build_chunks()
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True, normalize_embeddings=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"chunk-{i}" for i in range(len(chunks))], documents=[c["text"] for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{"source_file": c["source_file"], "section": c["section"], "processed_file": c["processed_file"],
                        "course_code": c.get("course_code", ""), "pages": c.get("pages", "")} for c in chunks])
    by_file = {}
    for c in chunks:
        by_file[c["processed_file"]] = by_file.get(c["processed_file"], 0) + 1
    print(f"Stored {col.count()} chunks in ChromaDB at {CHROMA_DIR}: {by_file}")


if __name__ == "__main__":
    main()
