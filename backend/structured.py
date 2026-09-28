"""
Exact answers from the structured course store (data/processed/courses.db, built by extract_data.py).

Questions about counts, lists, credits, prerequisites, offerings and "which semester" must not be answered
by a vector search over text chunks (it retrieves *similar* rows, not *all* rows, and the model then
guesses). This module detects those questions with simple rules, runs the lookup in SQL, and returns the
result as context chunks labelled with the exact file / tab / rows, which the LLM only has to phrase.

    route(question, profile) -> list[chunk]   (empty list = not a structured question)
"""
import re
import sqlite3
from contextlib import contextmanager
from functools import lru_cache

from config import COURSES_DB

MINOR_FILE = "Minor_Courses_for_BTech_Students.xlsx"
SPREAD_FILE = "Semester_Spread_Structures_Sept_2026.xlsx"
MINOR_NAMES = {"law": "Law", "design": "Design", "psychology": "Psychology", "economics": "Economics",
               "finance": "Finance", "marketing": "Marketing", "start-up": "Start-up", "startup": "Start-up",
               "start up": "Start-up", "entrepreneurship": "Start-up"}
BASKET_ALIASES = [  # (regex on the question, basket names as written in the spread tabs / Struct tabs)
    (r"univ(ersity)?\.?\s*core", ["Univ Core"], "University Core"),
    (r"foundation", ["Foundation"], "Foundation"),
    (r"program(me)?\s*core", ["Program CORE"], "Program Core"),
    (r"honou?rs", ["Program HONORS"], "Program Honors"),
    (r"speciali[sz]ation|track", ["Specialization Tracks"], "Specialization Tracks"),
    (r"\belectives?\b(?!.*open)", ["Electives", "Specialization Tracks"], "Electives"),
    (r"minor\s*/?\s*open|open\s*/?\s*minor|open elective|\bminor\b", ["Open/Minor"], "Minor/Open"),
    (r"internship|capstone", ["Internship/ Capstone Project"], "Internship/Capstone Project"),
]
CODE_RE = re.compile(r"\b([A-Za-z]{3,4})\s?(\d{3})\b")
LIST_WORDS = re.compile(r"how many|number of|\bcount\b|\blist\b|which courses|what courses|courses (are|in|of|for)"
                        r"|total credits|how much credit|credits? (are|in|of|for|does|do)|what are the courses", re.I)


RANKING_RE = re.compile(
    r"\b(high|higher|highest|most|more|max|maximum|largest|biggest|heaviest|least|lowest|fewest|min|minimum|"
    r"less|compare|comparison|rank)\b.{0,60}\b(credits?|practicals?|labs?|hours|workload|load|p[- ]?hours)\b"
    r"|\b(credits?|practicals?|labs?|hours|workload)\b.{0,40}\b(high|higher|highest|most|more|max|maximum|least|lowest)\b",
    re.I)


@contextmanager
def _db():
    """Read-only connection per call (cheap for a local file, and safe across request threads)."""
    con = sqlite3.connect(f"file:{COURSES_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


@lru_cache
def _titles() -> dict[str, str]:
    with _db() as con:
        rows = con.execute("SELECT DISTINCT lower(title) t, course_code FROM courses WHERE code_status='assigned'")
        return {re.sub(r"\s+", " ", r["t"]).strip(): r["course_code"] for r in rows if len(r["t"]) > 8}


@lru_cache
def _known_codes() -> frozenset[str]:
    with _db() as con:
        return frozenset(r[0] for r in con.execute("SELECT DISTINCT course_code FROM courses"))


def _num(x):
    return int(x) if isinstance(x, float) and x.is_integer() else x


def _rows_label(rows) -> str:
    rs = sorted({r["source_row"] for r in rows})
    return f"row {rs[0]}" if len(rs) == 1 else f"rows {rs[0]}-{rs[-1]}"


def _chunk(text: str, source_file: str, section: str) -> dict:
    return {"text": text, "source_file": source_file, "section": section, "score": 1.0, "structured": True}


def _batches(q: str, profile: dict | None) -> list[str]:
    years = re.findall(r"\b(202[2-6])\b", q)
    if years:
        return sorted(set(years))
    if profile and profile.get("batch") and re.search(r"\b(my|i|me)\b", q, re.I):
        return [profile["batch"]]
    return []


def _prereq_detail(con, norm: str, codes: str, batch: str) -> str:
    if not codes:
        return norm
    parts = []
    for code in codes.split(","):
        row = con.execute("SELECT title FROM courses WHERE course_code=? AND code_status='assigned' "
                          "ORDER BY batch=? DESC LIMIT 1", (code, batch)).fetchone()
        parts.append(f"{code} = {row['title']}" if row else f"{code} = NOT DEFINED anywhere in the supplied files")
    return f"{norm}  [{'; '.join(parts)}]"


# ---------------------------------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------------------------------
def minor_summary(minor: str, batches: list[str]) -> list[dict]:
    """All courses of a minor per batch: exact count, credits, placeholders and the batch's Minor/Open requirement."""
    with _db() as con:
        rows = con.execute("SELECT * FROM courses WHERE kind='minor' AND minor=? AND choice_of IS NULL "
                           "ORDER BY batch, source_row", (minor,)).fetchall()
        if batches:
            rows = [r for r in rows if r["batch"] in batches]
        chunks, summary = [], []
        for batch in sorted({r["batch"] for r in rows}):
            br = [r for r in rows if r["batch"] == batch]
            tab = br[0]["source_tab"]
            courses = [r for r in br if r["code_status"] not in ("placeholder", "not_offered")]
            ph = [r for r in br if r["code_status"] == "placeholder"]
            if any(r["code_status"] == "not_offered" for r in br):
                chunks.append(_chunk(f"{minor} minor, {batch} batch: the source file says 'No Students' - the "
                                     f"minor is not offered to this batch.", MINOR_FILE,
                                     f"tab '{tab}', {_rows_label(br)}"))
                continue
            req = con.execute("SELECT min_credits, source_tab, source_row FROM baskets WHERE batch=? AND "
                              "basket='Open/Minor'", (batch,)).fetchone()
            lines = [f"EXACT RESULT (computed from the spreadsheet, do not recount): {minor} minor, {batch} batch "
                     f"- {len(courses)} course(s) listed, {_num(sum(r['credits'] or 0 for r in courses))} credits "
                     f"in total."]
            for i, r in enumerate(courses, 1):
                code = r["course_code"]
                flag = ""
                if r["code_status"] == "code_not_assigned":
                    flag = " [NO COURSE CODE ASSIGNED YET - 'New' in source]"
                elif r["code_status"] == "code_unknown":
                    flag = " [COURSE CODE NOT KNOWN - 'DON'T KNOW' in source]"
                elif r["code_status"] == "choice_slot":
                    flag = " [either/or: the student takes one of these]"
                lines.append(f"{i}. {code} {r['title']}{flag} | {_num(r['credits'])} credits | semester "
                             f"{r['semester']} | prerequisites: "
                             f"{_prereq_detail(con, r['prerequisites_normalized'], r['prerequisite_codes'], batch)}"
                             f" | row {r['source_row']}")
            if ph:
                lines.append(f"Not yet decided: {_num(sum(r['credits'] or 0 for r in ph))} further credits are "
                             f"listed only as {', '.join(sorted({r['course_code'] for r in ph}))} (courses TBA/TBD "
                             f"in the source; do not invent them) - row(s) {', '.join(str(r['source_row']) for r in ph)}.")
            if req:
                lines.append(f"The {batch} batch Minor/Open basket requires {_num(req['min_credits'])} credits "
                             f"(see the programme structure).")
            # (gap text, keyword that shows the answer mentioned it) - enforced after the LLM in advisor.py
            gaps = []
            if ph:
                tba = _num(sum(r["credits"] or 0 for r in ph))
                gaps.append((f"{tba} credits of the {batch} batch are not yet decided (TBA/TBD in the source)",
                             [f"{tba} credit", f"{tba} more", f"{tba} further", f"remaining {tba}"]))
            no_code = ["no code", "no course code", "not assigned", "code not", "without a code", "not known",
                       "unknown code", "code is unknown", "yet to be assigned"]
            gaps += [(f"'{r['title']}' has no course code yet", no_code) for r in courses
                     if r["code_status"] in ("code_not_assigned", "code_unknown")]
            undefined = sorted({c for r in courses for c in (r["prerequisite_codes"] or "").split(",")
                                if c and c not in _known_codes()})
            gaps += [(f"prerequisite {c} is not defined anywhere in the supplied files", [c]) for c in undefined]
            if any((r["prerequisites_raw"] or "").strip().upper() in ("TBA", "TBD") for r in courses):
                gaps.append(("some prerequisites are TBD in the source", ["tbd", "to be decided", "not yet decided"]))
            if gaps:
                lines.append("Gaps in the source data (always tell the student): " + "; ".join(g for g, _ in gaps) + ".")
            summary.append(f"- {batch} batch: {len(courses)} course(s), "
                           f"{_num(sum(r['credits'] or 0 for r in courses))} credits"
                           + (f" + {_num(sum(r['credits'] or 0 for r in ph))} credits TBA/TBD" if ph else ""))
            chunk = _chunk("\n".join(lines), MINOR_FILE, f"tab '{tab}', {minor} minor {batch} batch, "
                           f"{_rows_label(br)}")
            chunk["must_mention"] = gaps
            chunks.append(chunk)
            if req:
                chunks.append(_chunk(f"{batch} batch credit basket 'Open/Minor': minimum {_num(req['min_credits'])} "
                                     f"credits.", SPREAD_FILE, f"tab '{req['source_tab']}', row {req['source_row']} "
                                     f"(Open/Minor basket)"))
        if len(summary) > 1:  # no batch given: the minor differs by batch, so answer for every batch
            tab = rows[0]["source_tab"]
            chunks.insert(0, _chunk(
                f"EXACT OVERVIEW of the {minor} minor (the student did not name a batch and the minor differs by "
                f"batch, so answer per batch - do not ask for the batch; if they asked for a list, list every "
                f"course of every batch from the per-batch results below):\n" + "\n".join(summary),
                MINOR_FILE, f"tab '{tab}', {minor} minor, all batches, {_rows_label(rows)}"))
        return chunks


def course_lookup(codes: list[str], batches: list[str]) -> list[dict]:
    """Semester, credits, basket/minor and normalised prerequisites of specific courses, per batch."""
    chunks = []
    with _db() as con:
        for code in codes:
            rows = con.execute("SELECT * FROM courses WHERE course_code=? ORDER BY kind, batch, source_row",
                               (code,)).fetchall()
            if batches:
                rows = [r for r in rows if r["batch"] in batches] or rows
            if not rows:
                continue
            for (kind, src_tab), grp in _group(rows, lambda r: (r["kind"], r["source_tab"])).items():
                lines = [f"EXACT RESULT from the spreadsheet: {code} {grp[0]['title']}"]
                for r in grp:
                    where = f"{r['minor']} minor" if kind == "minor" else f"basket {r['basket']}"
                    ltp = f", L-T-P {r['L']}-{r['T']}-{r['P']}" if r["L"] is not None else ""
                    choice = f" ({r['choice_of']}; the student takes one alternative)" if r["choice_of"] else ""
                    lines.append(f"- {r['batch']} batch: semester {r['semester']} "
                                 f"({'odd' if (r['semester'] or 0) % 2 else 'even'} semester), "
                                 f"{_num(r['credits'])} credits{ltp}, {where}{choice}; prerequisites: "
                                 f"{_prereq_detail(con, r['prerequisites_normalized'], r['prerequisite_codes'], r['batch'])}"
                                 f" | row {r['source_row']}")
                differ = len({r["prerequisites_normalized"] for r in grp}) > 1
                if differ:
                    lines.append("NOTE: the prerequisites differ between batches.")
                file = MINOR_FILE if kind == "minor" else SPREAD_FILE
                chunk = _chunk("\n".join(lines), file, f"tab '{src_tab}', {code}, {_rows_label(grp)}")
                if differ and not batches:  # the rule the student gets depends on a batch we do not know
                    chunk["conflict"] = True
                    chunk["must_mention"] = [(f"the prerequisites of {code} differ between batches",
                                              ["differ", "depend", "vary", "varies"])]
                chunks.append(chunk)
    return chunks


def basket_credits(baskets: list[tuple], batches: list[str]) -> list[dict]:
    chunks = []
    with _db() as con:
        for names, label in baskets:
            q = f"SELECT * FROM baskets WHERE basket IN ({','.join('?' * len(names))})"
            rows = [r for r in con.execute(q, names) if not batches or r["batch"] in batches]
            for r in rows:
                text = (f"EXACT RESULT from the spreadsheet {SPREAD_FILE} (not the handbook): {r['batch']} batch - "
                        f"{label} basket ('{r['basket']}'): minimum {_num(r['min_credits'])} credits required "
                        f"('Fixed Min' column of tab '{r['source_tab']}'). Total credits for the degree: 180.")
                if r["placed_credits"] is not None and r["placed_credits"] != r["min_credits"]:
                    text += (f" NOTE (inconsistency in the source): the courses the same tab places in this "
                             f"basket add up to {_num(r['placed_credits'])} credits.")
                st = con.execute("SELECT credits, source_tab, source_row FROM struct_baskets WHERE batch=? AND "
                                 "replace(lower(basket),' ','') LIKE ?",
                                 (r["batch"], label.lower().replace(" ", "")[:8] + "%")).fetchone()
                if st and st["credits"] != r["min_credits"]:
                    text += (f" NOTE (conflict in the source): the summary tab '{st['source_tab']}' (row "
                             f"{st['source_row']}) gives {_num(st['credits'])} credits for this basket.")
                chunk = _chunk(text, SPREAD_FILE, f"tab '{r['source_tab']}', row {r['source_row']} "
                               f"({r['basket']} basket)")
                if "NOTE (" in text:  # the spreadsheet disagrees with itself: always flag it
                    chunk["conflict"] = True
                    chunk["must_mention"] = [(f"the source file is inconsistent on the {r['batch']} {label} credits - "
                                              + text.split("180.", 1)[1].strip(),
                                              ["inconsisten", "conflict", "differ", "discrepan", "disagree"])]
                chunks.append(chunk)
    return chunks


def semester_list(batch: str, sem: int) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT * FROM courses WHERE kind='programme' AND batch=? AND semester=? AND "
                           "choice_of IS NULL ORDER BY source_row", (batch, sem)).fetchall()
    if not rows:
        return []
    lines = [f"EXACT RESULT from the spreadsheet: {batch} batch, semester {sem} - {len(rows)} slot(s), "
             f"{_num(sum(r['credits'] or 0 for r in rows))} credits:"]
    lines += [f"- {r['course_code']} {r['title']} | {_num(r['credits'])} credits | {r['basket']} | prerequisites: "
              f"{r['prerequisites_normalized']} | row {r['source_row']}" for r in rows]
    return [_chunk("\n".join(lines), SPREAD_FILE, f"tab '{rows[0]['source_tab']}', semester {sem}, "
                   f"{_rows_label(rows)}")]


def semester_stats(batches: list[str]) -> list[dict]:
    """Ranking questions ("which semester / course has the most credits / practical hours?"), per batch.
    L-T-P are weekly contact hours (Lecture-Tutorial-Practical); P = practical/lab hours per week."""
    chunks = []
    with _db() as con:
        for batch in batches or ["2022", "2023", "2024", "2025", "2026"]:
            rows = con.execute("SELECT * FROM courses WHERE kind='programme' AND batch=? AND choice_of IS NULL "
                               "ORDER BY semester, source_row", (batch,)).fetchall()
            if not rows:
                continue
            tab = rows[0]["source_tab"]
            per_sem = {}
            for r in rows:
                per_sem.setdefault(r["semester"], []).append(r)
            credits = {s: sum(r["credits"] or 0 for r in rs) for s, rs in per_sem.items()}
            practical = {s: sum(r["P"] or 0 for r in rs) for s, rs in per_sem.items()}
            real = [r for r in rows if r["code_status"] in ("assigned", "choice_slot")]
            top_c = max(r["credits"] or 0 for r in real)
            top_p = max(r["P"] or 0 for r in real)
            best_c = [r for r in real if (r["credits"] or 0) == top_c]
            best_p = [r for r in real if (r["P"] or 0) == top_p]
            fmt = lambda r: f"{r['course_code']} {r['title']} (semester {r['semester']}, {_num(r['credits'])} credits, "\
                            f"L-T-P {r['L']}-{r['T']}-{r['P']}, row {r['source_row']})"  # noqa: E731
            hi_c = max(credits.values())
            hi_p = max(practical.values())
            lines = [f"EXACT RESULT from the spreadsheet (computed, do not recount): {batch} batch "
                     f"(L-T-P = weekly Lecture-Tutorial-Practical hours; P = practical/lab hours per week).",
                     f"- Semester(s) with the MOST credits: {', '.join(str(s) for s, c in credits.items() if c == hi_c)}"
                     f" ({_num(hi_c)} credits).",
                     f"- Semester(s) with the MOST practical hours: "
                     f"{', '.join(str(s) for s, p in practical.items() if p == hi_p)} ({_num(hi_p)} P-hours/week).",
                     f"- Course(s) with the HIGHEST credits ({_num(top_c)}): " + "; ".join(fmt(r) for r in best_c[:8])
                     + (f"; and {len(best_c) - 8} more" if len(best_c) > 8 else ""),
                     f"- Course(s) with the MOST practical hours (P = {_num(top_p)}/week): "
                     + "; ".join(fmt(r) for r in best_p[:8]) + (f"; and {len(best_p) - 8} more" if len(best_p) > 8 else "")]
            if batches:  # a named batch gets the full per-semester breakdown
                lines.append("Per semester (credits | practical hours/week | highest-credit course | most-practical course):")
                for s in sorted(per_sem):
                    rs = [r for r in per_sem[s] if r["code_status"] in ("assigned", "choice_slot")] or per_sem[s]
                    mc = max(rs, key=lambda r: r["credits"] or 0)
                    mp = max(rs, key=lambda r: r["P"] or 0)
                    lines.append(f"  Semester {s}: {_num(credits[s])} credits | {_num(practical[s])} P-hours | "
                                 f"{mc['course_code']} {mc['title']} ({_num(mc['credits'])} cr) | "
                                 f"{mp['course_code']} {mp['title']} (P={mp['P']})")
            lines.append("Placeholder slots (TBA minor/open, electives) have no L-T-P in the source and count 0 "
                         "practical hours.")
            chunks.append(_chunk("\n".join(lines), SPREAD_FILE, f"tab '{tab}', {batch} batch, all semesters, "
                                 f"{_rows_label(rows)}"))
    if len(chunks) > 1:
        chunks[0]["text"] = ("The student did not name a batch; the curriculum differs by batch, so answer for "
                             "every batch below (briefly), then offer a per-semester breakdown for their batch.\n"
                             + chunks[0]["text"])
    return chunks


def _group(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


# ---------------------------------------------------------------------------------------------------
# Known conflicts between / inside the policy documents, verified by reading the source text during the data
# audit (verify_data.py lists them). Attached whenever a question touches the topic, so the conflict is always
# flagged instead of depending on whether retrieval happens to surface both passages.
# ---------------------------------------------------------------------------------------------------
KNOWN_CONFLICTS = [
    (r"late\s*regist|register late|missed (the )?registration|after the registration (date|deadline)",
     "SOP_STUDENT_17082026.pdf", "1. Course Registration (Course Registration Deadline vs Late Registration), pp. 1-2",
     "The documents are INCONSISTENT on late registration:\n"
     "- SOP, 'Course Registration Deadline': \"The maximum permissible period for late registration shall not be "
     "more than one (01) calendar week counted from the specified date of Registration announced by the University. "
     "No extensions will be granted.\"\n"
     "- SOP, 'Late Registration' (same page): \"No late registration shall be permitted.\" except for medical "
     "exigencies (hospitalization, trauma or contagious disease) or participation in competitions/events, with prior "
     "approval of the Program Chair/Dean; for medical cases at most two (02) calendar weeks.\n"
     "- Student Handbook 2.6-2.7 (pp. 16-17): medical cases up to two weeks, competitions/events as per 2.6.2, and for "
     "ANY OTHER reason up to one (01) calendar week with a Late Fee.\n"
     "So the SOP both allows one week and says no late registration is permitted; tell the student about this "
     "inconsistency and to confirm with the Program Chair / Office of the Registrar.",
     ["inconsisten", "conflict", "contradict", "differ", "disagree"]),
    (r"(attendance|shortage).*(relax|approv|medical|hospital|exigenc|competition)|"
     r"(relax|approv|medical|hospital|exigenc).*(attendance|shortage)",
     "SOP_STUDENT_17082026.pdf", "2. Student Attendance (approval of attendance relaxation), p. 2",
     "The documents DIFFER on who approves an attendance relaxation (both keep the 65% minimum):\n"
     "- Student Handbook 7.3 (p. 21): the leave request goes to the Faculty Advisor [Mentor] for recommendation to "
     "the Program Chair, and the relaxation is 'subject to the approval by the Vice-Chancellor' (7.4: the same for "
     "State/National/International events).\n"
     "- SOP, Student Attendance (p. 2): the request goes to the Faculty Advisor for recommendation to the Program "
     "Chair (events: Director, Campus Life and Faculty Advisor); the SOP does not mention Vice-Chancellor approval.\n"
     "So tell the student the approval chain differs between the Handbook and the SOP and to confirm with the "
     "Program Chair.",
     ["differ", "inconsisten", "conflict", "does not mention", "doesn't mention", "not mention"]),
]


def conflict_notes(question: str) -> list[dict]:
    out = []
    for rx, file, section, text, keys in KNOWN_CONFLICTS:
        if re.search(rx, question, re.I):
            chunk = _chunk(text, file, section)
            chunk["conflict"] = True
            detail = text.split(":\n", 1)[1].split("\nSo ", 1)[0].replace("\n", " ")
            chunk["must_mention"] = [(f"the documents are inconsistent on this - {detail}", keys)]
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------------------------------
def mentioned_codes(q: str) -> list[str]:
    known = _known_codes()
    codes = [f"{a.upper()}{b}" for a, b in CODE_RE.findall(q) if f"{a.upper()}{b}" in known]
    low = re.sub(r"\s+", " ", q.lower())
    codes += [code for title, code in _titles().items() if re.search(rf"\b{re.escape(title)}\b", low)]
    return list(dict.fromkeys(codes))


def route(question: str, profile: dict | None = None) -> list[dict]:
    q = question.strip()
    low = q.lower()
    batches = _batches(q, profile)
    codes = mentioned_codes(q)
    minor = next((name for key, name in MINOR_NAMES.items() if re.search(rf"\b{re.escape(key)}\b", low)), None)
    minor_q = minor and (re.search(r"\bminor\b", low) or LIST_WORDS.search(low))

    # "how many more credits do I need", "credits left to graduate" - a requirement, not a ranking question
    requirement_q = re.search(r"(credits?|how many|how much).{0,30}\b(need|needed|require|required|remaining|left)\b"
                              r"|\bto (graduate|complete)\b|\bmore credits\b", low)
    if RANKING_RE.search(low) and not minor_q and not requirement_q:  # "which semester has the most credits"
        return semester_stats(batches)
    if codes:  # a specific course: semester / credits / prerequisites
        return course_lookup(codes, batches)
    if minor_q and not re.search(r"credits? (needed|required)|basket", low):
        return minor_summary(minor, batches)
    wanted = [(names, label) for rx, names, label in BASKET_ALIASES if re.search(rx, low)]
    if wanted and re.search(r"credit", low):
        return basket_credits(wanted[:2], batches)
    m = re.search(r"\b(?:semester|sem)\s*(\d)\b", low)
    if m and batches and re.search(r"course|subject|offered|taught|study", low):
        return semester_list(batches[0], int(m.group(1)))
    return []
