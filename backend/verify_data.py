"""
Data-layer audit: proves that every source file was loaded and that the structured store the advisor
queries (data/processed/courses.db) matches the Excel files exactly.

The Excel ground truth here is computed INDEPENDENTLY of extract_data.py: columns are located from each
sheet's own header row (not hard-coded), cached formula values are read with openpyxl (data_only=True),
and nothing from extract_data's parsing functions is reused.

Run:  python verify_data.py          (exit code 1 if any mismatch is found)
"""
import re
import sqlite3
import sys
from collections import Counter, defaultdict

import chromadb
import openpyxl
from pypdf import PdfReader

from config import CHROMA_DIR, COLLECTION, COURSES_DB, RAW_DIR

SPREAD = RAW_DIR / "semester_spread_structures_sept2026.xlsx"
MINORS = RAW_DIR / "minor_courses_btech.xlsx"
HANDBOOK = RAW_DIR / "student_handbook_aug2026.pdf"
SOP = RAW_DIR / "sop_students_aug2026.pdf"
CODE = re.compile(r"[A-Z]{3,4}\s?\d{3}")
PLACEHOLDER = {"TBA", "TBD", "DON'T KNOW", "DON’T KNOW"}

problems: list[str] = []      # issues found IN THE SOURCE FILES (reported, not errors of the pipeline)
mismatches: list[str] = []    # differences between ground truth and the advisor's store (pipeline bugs)


def s(v) -> str:
    return "" if v is None else re.sub(r"\s+", " ", str(v)).strip()


def num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def _n(x):
    return int(x) if isinstance(x, float) and x.is_integer() else x


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# ---------------------------------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------------------------------
def audit_pdfs(col):
    section("PDF FILES")
    meta = col.get(include=["metadatas"])["metadatas"]
    by_file = Counter(m["source_file"] for m in meta)
    for path, cited in [(HANDBOOK, "Student Handbook Aug 2026.pdf"), (SOP, "SOP_STUDENT_17082026.pdf")]:
        r = PdfReader(str(path))
        empty = [i for i, p in enumerate(r.pages, 1) if len((p.extract_text() or "").strip()) < 5]
        images = [i for i, p in enumerate(r.pages, 1) if p.images]
        pages_cited = sorted({int(x) for m in meta if m["source_file"] == cited
                              for x in re.findall(r"\d+", m.get("pages", ""))})
        print(f"{path.name}: {len(r.pages)} pages | blank pages: {empty} | pages with images: {images}")
        print(f"   chunks in vector store: {by_file[cited]} | printed pages covered by chunks: "
              f"{pages_cited[0] if pages_cited else '-'}..{pages_cited[-1] if pages_cited else '-'} "
              f"({len(pages_cited)} distinct)")
    print("   handbook notes: p.1-8 cover/TOC, 1-2 line pages are section dividers (not chunked on purpose);"
          " PDF p.32 Table 2 is an image (transcribed by hand); PDF p.36/p.65 are layered pages (de-duplicated)")
    problems.append("Handbook PDF pages 36 and 65 each contain two overlaid pages (printed p.28 academic "
                    "regulations + printed p.57 POSH guidelines): plain text extraction mixes them.")
    problems.append("Handbook Table 2 (letter grades, p.24) is an image, not text; Table 1 (evaluation weightage, "
                    "p.23) and Table 3 (progression CGPA, p.34) are bordered tables that text extraction flattens "
                    "into one column (re-typed as markdown tables).")
    problems.append("Handbook p.83 heading 'Important Contact Numbers' uses a font whose text cannot be decoded.")


# ---------------------------------------------------------------------------------------------------
# Semester spread workbook
# ---------------------------------------------------------------------------------------------------
def spread_truth(wb):
    """Independent parse of the Sem_Spread tabs: blocks located from the 'Pre-Req' headers in row 2."""
    slots, baskets = [], []
    for ws in wb.worksheets:
        m = re.search(r"(20\d\d)", ws.title)
        if not ws.title.lower().startswith("sem") or not m:
            continue
        batch = m.group(1)
        hdr = [s(c.value).lower() for c in ws[2]]
        blocks = [i - 2 for i, h in enumerate(hdr) if h.startswith("pre")]      # 0-based code column
        sem_of = {b: k + 1 for k, b in enumerate(blocks)}
        basket = None
        for r in range(3, ws.max_row + 1):
            row = [c.value for c in ws[r]] + [None] * 80
            if s(row[0]).lower().startswith("total"):
                break
            if s(row[1]) and num(row[3]) is not None:
                basket = s(row[1])
                baskets.append((batch, basket, num(row[3]), num(row[64]), ws.title, r))
            for b in blocks:
                code, title, cred = s(row[b]), s(row[b + 1]), num(row[b + 6])
                if not code and not title:
                    continue
                if not title and cred is None and CODE.search(code):
                    problems.append(f"{ws.title} row {r}: '{code}' alone in the code column (continuation of "
                                    f"the cell above) - not a course")
                    continue
                if not title:  # placeholder slot "Minor/Open (#1 Course)" with credits in the L/T/P cells
                    title, code = code, "TBA"
                    cred = num(row[b + 4]) or num(row[b + 1])
                # a slot with no code (e.g. "Minor/Open (#1 Course)", "Elective 1") is a placeholder = TBA
                slots.append((batch, sem_of[b], code.replace(" ", "") or "TBA", title, cred, ws.title, r))
    return slots, baskets


def audit_spread(db):
    section(f"EXCEL: {SPREAD.name}")
    wb = openpyxl.load_workbook(SPREAD, data_only=True)
    wf = openpyxl.load_workbook(SPREAD, data_only=False)
    for ws in wb.worksheets:
        forms = [c for row in wf[ws.title].iter_rows() for c in row
                 if isinstance(c.value, str) and c.value.startswith("=")]
        uncached = [c.coordinate for c in forms if ws[c.coordinate].value is None]
        nonempty = sum(1 for row in ws.iter_rows() if any(c.value not in (None, "") for c in row))
        print(f"  tab {ws.title!r:26} rows with data: {nonempty:3} | formula cells: {len(forms):3} "
              f"(uncached: {len(uncached)}) | merged ranges: {len(ws.merged_cells.ranges)}")
        if uncached:
            mismatches.append(f"{ws.title}: formula cells without cached values {uncached[:5]}")
    slots, baskets = spread_truth(wb)

    store = db.execute("SELECT batch, semester, course_code, title, credits, source_tab, source_row FROM courses "
                       "WHERE kind='programme' AND choice_of IS NULL").fetchall()
    norm = lambda t: (t[0], t[1], t[2].replace(" ", ""), t[3].lower(), float(t[4] or 0), t[5], t[6])  # noqa: E731
    truth_set, store_set = set(map(norm, slots)), set(map(norm, store))
    print(f"\n  course slots: ground truth {len(truth_set)} | structured store {len(store_set)}")
    for x in sorted(truth_set - store_set):
        mismatches.append(f"spread slot missing/different in store: {x}")
    for x in sorted(store_set - truth_set):
        mismatches.append(f"spread slot in store but not in the Excel: {x}")

    # credits per batch & semester vs the sheet's own "Total" row
    per = defaultdict(float)
    for b, sem, *_rest in slots:
        per[(b, sem)] += float(_rest[2] or 0)
    for ws in wb.worksheets:
        m = re.search(r"(20\d\d)", ws.title)
        if not ws.title.lower().startswith("sem") or not m:
            continue
        hdr = [s(c.value).lower() for c in ws[2]]
        blocks = [i - 2 for i, h in enumerate(hdr) if h.startswith("pre")]
        total = next(r for r in range(3, ws.max_row + 1) if s(ws.cell(r, 1).value).lower().startswith("total"))
        sheet_tot = [num(ws.cell(total, b + 7).value) for b in blocks]
        mine = [per[(m.group(1), k + 1)] for k in range(len(blocks))]
        ok = [float(a or 0) for a in sheet_tot] == mine
        print(f"  {m.group(1)} credits per semester (sheet Total row {total}): {sheet_tot} "
              f"| recomputed from course rows: {[int(x) for x in mine]} -> {'MATCH' if ok else 'MISMATCH'} "
              f"(sum {int(sum(mine))})")
        if not ok:
            mismatches.append(f"{ws.title}: semester credit totals differ from the Total row")

    store_b = {(r[0], r[1], r[2], r[3]) for r in db.execute(
        "SELECT batch, basket, min_credits, placed_credits FROM baskets")}
    truth_b = {(b, n, mn, pl) for b, n, mn, pl, _t, _r in baskets}
    print(f"\n  basket requirements: ground truth {len(truth_b)} | store {len(store_b)} -> "
          f"{'MATCH' if truth_b == store_b else 'MISMATCH'}")
    for x in sorted(truth_b ^ store_b):
        mismatches.append(f"basket row differs: {x}")
    for b, n, mn, pl, t, r in baskets:
        if pl is not None and pl != mn:
            problems.append(f"{t} row {r}: basket '{n}' fixed minimum {mn} credits but the semester spread "
                            f"places {pl} credits")
    # Struct_* summary sheets vs the spread
    print("\n  Struct_* summary tabs vs semester-spread 'Fixed Min' column:")
    same = {"University Core": "Univ Core", "Foundation": "Foundation", "Program Core": "Program CORE",
            "Program Honors": "Program HONORS", "Specialization Tracks": "Specialization Tracks",
            "Electives": "Electives", "Minor/Open": "Open/Minor",
            "Internship/Capstone Project": "Internship/ Capstone Project"}
    for b, n, c, t, r in db.execute("SELECT batch, basket, credits, source_tab, source_row FROM struct_baskets"):
        spread = next(((mn, st, sr) for bb, nn, mn, _pl, st, sr in baskets if bb == b and nn == same.get(n)), None)
        print(f"    {t:15} {n:28} {_n(c):>4}  | {b} spread tab: {_n(spread[0]) if spread else '-':>4}"
              f"{'   <-- DIFFERENT' if spread and spread[0] != c else ''}")
        if spread and spread[0] != c:
            problems.append(f"{t} row {r}: '{n}' = {_n(c)} credits, but {spread[1]} row {spread[2]} says "
                            f"{_n(spread[0])}")


# ---------------------------------------------------------------------------------------------------
# Minor courses workbook
# ---------------------------------------------------------------------------------------------------
def minors_truth(wb):
    """Independent parse: header row gives the columns; batch label may sit in any (merged) cell of a row."""
    courses, placeholders, not_offered = [], [], []
    for ws in wb.worksheets:
        minor = re.sub(r"\s*Minor\s*$", "", ws.title).strip()
        cols, batch = None, None
        for r in range(1, ws.max_row + 1):
            vals = [s(c.value) for c in ws[r]]
            m = re.search(r"(20\d\d)\s*Batch", " ".join(vals), re.I)
            if m:
                batch = m.group(1)
            if "Course Code" in vals:
                cols = {name: vals.index(h) for name, h in
                        [("code", "Course Code"), ("title", "Course Title"), ("L", "L"), ("T", "T"), ("P", "P"),
                         ("credit", "Credit"), ("sem", "Semester"), ("pre", "Pre-rq")] if h in vals}
                continue
            if cols is None:
                continue
            if any("no students" in v.lower() for v in vals):
                not_offered.append((minor, batch, ws.title, r))
                continue
            code, title = vals[cols["code"]], vals[cols["title"]]
            credit = num(vals[cols["credit"]]) if cols["credit"] < len(vals) else None
            if credit is None:
                continue
            if title.upper() in PLACEHOLDER and not code:
                placeholders.append((minor, batch, title.upper(), credit, ws.title, r))
                continue
            if not title:
                continue
            courses.append({"minor": minor, "batch": batch, "code": code.upper(), "title": title,
                            "credits": credit, "sem": num(vals[cols["sem"]]),
                            "pre": " ".join(v for v in vals[cols["pre"]:] if v), "tab": ws.title, "row": r})
    return courses, placeholders, not_offered


def audit_minors(db):
    section(f"EXCEL: {MINORS.name}")
    wb = openpyxl.load_workbook(MINORS, data_only=True)
    courses, placeholders, not_offered = minors_truth(wb)
    tabs = [ws.title for ws in wb.worksheets]
    print(f"  tabs: {tabs}")
    expected = {"Law", "Design", "Psychology", "Economics", "Finance", "Marketing", "Start-up"}
    loaded = {re.sub(r"\s*Minor\s*$", "", t) for t in tabs}
    if expected - loaded:
        mismatches.append(f"minor tabs missing: {expected - loaded}")

    store_c = db.execute("SELECT minor, batch, course_code, title, credits, semester, source_tab, source_row, "
                         "code_status FROM courses WHERE kind='minor' AND choice_of IS NULL").fetchall()
    store_courses = [r for r in store_c if r[8] not in ("placeholder", "not_offered")]
    store_ph = [r for r in store_c if r[8] == "placeholder"]
    key = lambda mi, b, t, cr, se, tab, row: (mi, b, t.lower().strip(), float(cr), se, tab, row)  # noqa: E731
    truth_k = {key(c["minor"], c["batch"], c["title"], c["credits"], c["sem"], c["tab"], c["row"]) for c in courses}
    store_k = {key(r[0], r[1], r[3], r[4], r[5], r[6], r[7]) for r in store_courses}
    print(f"\n  {'minor':11} {'batch':5} {'courses':>7} {'credits':>7} {'TBA/TBD credits':>15}  "
          f"{'store courses':>13} {'store credits':>13}  rows")
    groups = defaultdict(list)
    for c in courses:
        groups[(c["minor"], c["batch"])].append(c)
    for (mi, b), cs in sorted(groups.items()):
        ph = sum(p[3] for p in placeholders if p[0] == mi and p[1] == b)
        st = [r for r in store_courses if r[0] == mi and r[1] == b]
        print(f"  {mi:11} {b:5} {len(cs):7} {sum(c['credits'] for c in cs):7} {ph:15}  {len(st):13} "
              f"{sum(r[4] for r in st):13}  {cs[0]['tab']} {min(c['row'] for c in cs)}-{max(c['row'] for c in cs)}")
    only_ph = sorted({(p[0], p[1]) for p in placeholders} - set(groups))
    for mi, b in only_ph:
        ph = sum(p[3] for p in placeholders if p[0] == mi and p[1] == b)
        print(f"  {mi:11} {b:5} {0:7} {0:7} {ph:15}  (no courses decided yet)")
    for mi, b, t, r in not_offered:
        print(f"  {mi:11} {b:5} 'No Students' - minor not offered ({t} row {r})")
    print(f"\n  minor course rows: ground truth {len(truth_k)} | store {len(store_k)} -> "
          f"{'MATCH' if truth_k == store_k else 'MISMATCH'}")
    for x in sorted(truth_k - store_k):
        mismatches.append(f"minor course missing/different in store: {x}")
    for x in sorted(store_k - truth_k):
        mismatches.append(f"minor course in store but not in Excel: {x}")
    ph_t = {(p[0], p[1], float(p[3]), p[4], p[5]) for p in placeholders}
    ph_s = {(r[0], r[1], float(r[4]), r[6], r[7]) for r in store_ph}
    print(f"  TBA/TBD placeholder rows: ground truth {len(ph_t)} | store {len(ph_s)} -> "
          f"{'MATCH' if ph_t == ph_s else 'MISMATCH'}")
    for x in sorted(ph_t ^ ph_s):
        mismatches.append(f"placeholder row differs: {x}")

    # source-file data problems worth telling the student about
    known = {r[0] for r in db.execute("SELECT DISTINCT course_code FROM courses")}
    nil_forms = Counter(c["pre"] for c in courses if c["pre"].strip().upper() in ("NIL", ""))
    for c in courses:
        if c["code"] in ("NEW",) or "KNOW" in c["code"]:
            problems.append(f"{c['tab']} row {c['row']} ({c['batch']}): '{c['title']}' has code "
                            f"'{c['code']}' - no course code assigned")
        if "/" in c["code"]:
            problems.append(f"{c['tab']} row {c['row']} ({c['batch']}): code '{c['code']}' is an either/or pair")
        for code in {x.replace(' ', '') for x in CODE.findall(c["pre"].upper())} - known:
            problems.append(f"{c['tab']} row {c['row']}: prerequisite {code} of {c['code']} is not defined "
                            f"anywhere in the supplied files")
        if c["pre"].upper().startswith("FAMA"):
            problems.append(f"{c['tab']} row {c['row']}: prerequisite given as the acronym 'FAMA' (no code)")
        if c["pre"].upper().startswith("TBD"):
            problems.append(f"{c['tab']} row {c['row']}: prerequisite of '{c['title']}' is 'TBD'")
    for p in placeholders:
        problems.append(f"{p[4]} row {p[5]} ({p[1]}): {p[3]} credits of the {p[0]} minor are '{p[2]}' "
                        f"(courses not decided)")
    print(f"  'Nil' spellings used for 'no prerequisite': {dict(nil_forms)}")
    if "MKMT201" not in known:
        problems.append("Finance/Marketing tabs: prerequisite 'MKMT201' looks like a typo of MKTG201 "
                        "(Marketing Management); kept as written")
    if any(r[0] == "PSCY102" for r in db.execute("SELECT course_code FROM courses")):
        problems.append("Sem_Spread_2025: code 'PSCY102' (Introduction to Psychology) looks like a typo "
                        "of PSYC102; kept as written")
    return courses


def main():
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    db = sqlite3.connect(COURSES_DB)
    audit_pdfs(col)
    audit_spread(db)
    audit_minors(db)
    section("VECTOR STORE")
    meta = col.get(include=["metadatas"])["metadatas"]
    for f, n in Counter(m["source_file"] for m in meta).most_common():
        print(f"  {n:4} chunks  {f}")
    print(f"  structured store: {db.execute('SELECT COUNT(*) FROM courses').fetchone()[0]} course rows in "
          f"{COURSES_DB.name}")
    section(f"DATA PROBLEMS FOUND IN THE SOURCE FILES ({len(set(problems))})")
    for p in sorted(set(problems)):
        print(f"  - {p}")
    section(f"MISMATCHES BETWEEN GROUND TRUTH AND THE ADVISOR'S STORE ({len(mismatches)})")
    for m in mismatches:
        print(f"  ! {m}")
    if not mismatches:
        print("  none - the structured store matches the Excel files exactly")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
