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


COURSE_LIST_RE = re.compile(r"\b(courses?|subjects?|papers?|basket|list|show)\b", re.I)
SELF_CHECK_RE = re.compile(r"^\s*(can|may|should|am|do|did|have)\s+i\b", re.I)  # about the student, not the catalogue
INFOGRAPHIC_RE = re.compile(r"\b(infographic|diagram|flow ?chart|roadmap|visual (summary|map|overview))\b|\bchart of\b",
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


def _batches(q: str, profile: dict | None, message: str | None = None, facts=None) -> list[str]:
    # A follow-up ("what about 2023?") is merged with the previous question for retrieval, so the year in the
    # new message wins over the year in the old one.
    years = re.findall(r"\b(202[2-6])\b", message or "") or re.findall(r"\b(202[2-6])\b", q)
    if years:
        return sorted(set(years))
    if facts and facts.batch:  # told to us earlier in this chat
        return [facts.batch]
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


def minors_overview(batches: list[str]) -> list[dict]:
    """Every minor per batch with its course count and credits - for questions about minors in general
    ("how many minor courses", "which minors are offered") that name no specific minor."""
    with _db() as con:
        rows = con.execute("SELECT * FROM courses WHERE kind='minor' AND choice_of IS NULL "
                           "ORDER BY batch, minor, source_row").fetchall()
    if batches:
        rows = [r for r in rows if r["batch"] in batches]
    if not rows:
        return []
    lines = ["EXACT OVERVIEW of all minors (computed from the spreadsheet, do not recount). The student did not "
             "name a minor, so give the per-batch overview below; answer for every batch listed unless they named "
             "one, and offer to list the courses of a specific minor:"]
    total = 0
    for batch in sorted({r["batch"] for r in rows}):
        parts = []
        for minor in sorted({r["minor"] for r in rows if r["batch"] == batch}):
            mr = [r for r in rows if r["batch"] == batch and r["minor"] == minor]
            if any(r["code_status"] == "not_offered" for r in mr):
                parts.append(f"{minor} (not offered - 'No Students')")
                continue
            courses = [r for r in mr if r["code_status"] not in ("placeholder", "not_offered")]
            ph = [r for r in mr if r["code_status"] == "placeholder"]
            total += len(courses)
            parts.append(f"{minor}: {len(courses)} course(s), {_num(sum(r['credits'] or 0 for r in courses))} "
                         f"credits" + (f" + {_num(sum(r['credits'] or 0 for r in ph))} credits TBA/TBD" if ph else ""))
        offered = sum("not offered" not in p for p in parts)
        lines.append(f"- {batch} batch: {offered} minor(s) - " + "; ".join(parts))
    lines.append(f"Total course entries across the batches above: {total}.")
    tabs = ", ".join(sorted({r["source_tab"] for r in rows}))
    return [_chunk("\n".join(lines), MINOR_FILE, f"tabs {tabs}, all minors")]


HANDBOOK_FILE = "Student Handbook Aug 2026.pdf"
MINOR_RULE_SOURCES = [(HANDBOOK_FILE, "Section III - Academic Regulations > 2. Registration (2.13), pp. 17-18"),
                      (HANDBOOK_FILE, "Section III - Academic Regulations > 8. Teaching, Evaluation and Grading System, "
                                      "pp. 26-27")]
CAN_TAKE_MINOR_RE = re.compile(r"\b(can|could|may|should|am|allowed|eligible|possible|able)\b.{0,40}\bminors?\b"
                               r"|\b(take|do|pursue|opt for|choose|register for|start)\s+(?:a|any|the)?\s*minor\b", re.I)


def offered_minors(batch: str) -> dict[str, list]:
    """Minor -> its rows, for the minors the source file doesn't mark 'No Students' for this batch."""
    with _db() as con:
        rows = con.execute("SELECT * FROM courses WHERE kind='minor' AND batch=? AND choice_of IS NULL "
                           "ORDER BY minor, source_row", (batch,)).fetchall()
    grouped = _group(rows, lambda r: r["minor"])
    return {m: rs for m, rs in grouped.items() if not any(r["code_status"] == "not_offered" for r in rs)}


def minor_availability(batch: str, semester: int | None) -> list[dict]:
    """"Can I take a minor?": what the files say for this batch and semester. They set no personal entry condition,
    so this reports availability and the stated process rather than declaring the student eligible."""
    with _db() as con:
        slots = con.execute("SELECT * FROM courses WHERE kind='programme' AND batch=? AND basket='Open/Minor' "
                            "ORDER BY semester", (batch,)).fetchall()
        req = con.execute("SELECT * FROM baskets WHERE batch=? AND basket='Open/Minor'", (batch,)).fetchone()
        all_minors = {r[0] for r in con.execute("SELECT DISTINCT minor FROM courses WHERE kind='minor' AND batch=?",
                                                (batch,))}
    offered = offered_minors(batch)
    if not offered or not slots:
        return []
    needed = _num(req["min_credits"]) if req else _num(sum(r["credits"] or 0 for r in slots))
    parts = [f"**Yes, minors are open to the {batch} batch: the curriculum has a {needed}-credit Minor/Open basket and "
             f"{_plural(len(offered), 'minor')} on offer.** The documents I have set no personal condition such as a "
             f"minimum CGPA, but your Faculty Advisor has to be part of the choice (below)."]
    if semester:
        ahead = [r for r in slots if r["semester"] >= semester]
        passed = [r for r in slots if r["semester"] < semester]
        where = ", ".join(f"Semester {r['semester']} ({_num(r['credits'])} credits)" for r in ahead) or "none"
        text = f"For you in **Semester {semester}**, the basket's slots from now on are: {where}."
        if passed:
            text += (f" {_num(sum(r['credits'] or 0 for r in passed))} credits were placed in earlier semesters "
                     f"({', '.join(str(r['semester']) for r in passed)}). The documents don't say how late a student can "
                     f"start a minor or how those credits are made up, so please check that with your Faculty Advisor "
                     f"before you decide.")
        parts.append(text)
    lines = ["| Minor | Courses listed | Credits | Semesters |", "|---|---|---|---|"]
    for minor, rows in offered.items():
        named = [r for r in rows if r["code_status"] not in ("placeholder", "not_offered")]
        tba = sum(r["credits"] or 0 for r in rows if r["code_status"] == "placeholder")
        sems = sorted({r["semester"] for r in named if r["semester"]})
        span = f"{sems[0]}-{sems[-1]}" if len(sems) > 1 else str(sems[0]) if sems else "-"
        credits = f"{_num(sum(r['credits'] or 0 for r in named))}" + (f" + {_num(tba)} TBA" if tba else "")
        lines.append(f"| {minor} | {len(named)} | {credits} | {span} |")
    parts.append("\n".join(lines))
    closed = sorted(all_minors - set(offered))
    if closed:
        parts.append(f"Not offered to this batch: {', '.join(closed)} (the source file says 'No Students').")
    parts.append("Conditions in the Student Handbook:\n"
                 "- You declare a minor through Pre-Registration, after consulting your Faculty Advisor (2.13).\n"
                 "- A minor course runs only if at least 10 students register, unless the Program Chair permits it (2.13).\n"
                 "- You can change your minor only once before completing the programme (8.13).")
    answer = "\n\n".join(parts)
    chunk = _direct(_chunk(answer, MINOR_FILE, f"tabs for the {batch} batch, all minors; Open/Minor basket in tab "
                           f"'{slots[0]['source_tab']}', row {slots[0]['source_row']}"), answer)
    chunk["extra_sources"] = [{"source_file": SPREAD_FILE, "section": f"tab '{slots[0]['source_tab']}', row "
                               f"{slots[0]['source_row']} (Open/Minor basket)"}] + \
        [{"source_file": f, "section": s} for f, s in MINOR_RULE_SOURCES]
    if not semester:
        chunk["clarify"] = "Which semester are you in? I can then show which minor slots are still ahead of you."
    return [chunk]


def ask_batch_for_minor() -> list[dict]:
    answer = "Minors differ by batch, so I need your batch before I can tell you what's open to you."
    chunk = _direct(_chunk(answer, MINOR_FILE, "minors by batch"), answer)
    chunk["clarify"] = "Which batch are you in (your year of admission), and which semester are you in now?"
    chunk["sources_none"] = True
    return [chunk]


MINOR_DENIAL_RE = re.compile(r"\bno minors?\b|\bminors?\b.{0,40}\b(not available|unavailable|not offered|isn't available)\b"
                             r"|\b(cannot|can't|can not|not able to|not eligible to|ineligible to)\s+(?:take|do|pursue|opt for|"
                             r"register for)\s+(?:a|any)\s+minor\b|\bnot eligible for (?:a|any) minor\b"
                             # "the documents don't say" is also a false claim when the course store does say
                             r"|\b(do not|don't|does not|doesn't) (contain|include|provide|list|mention|specify|state)\b"
                             r".{0,60}\bminors?\b|\b(no|without) (information|details?|data)\b.{0,40}\bminors?\b"
                             r"|\bminors?\b.{0,40}\b(information|details?) (is|are) (not available|unavailable|missing)\b"
                             r"|\b(cannot|can't|unable to) confirm\b.{0,40}\bminor\b", re.I | re.S)


def claim_conflicts(answer: str, batch: str | None) -> tuple[list[str], list[dict]]:
    """Availability claims checked against the course store: returns (problems, chunks holding the evidence)."""
    if not (batch and MINOR_DENIAL_RE.search(answer)):
        return [], []
    offered = offered_minors(batch)
    if not offered:
        return [], []
    return ([f"the draft says no minor is available, but the {batch} batch has {len(offered)} minors on offer "
             f"({', '.join(offered)})"], minors_overview([batch]))


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


def _direct(chunk: dict, answer: str, conflict: bool = False, attachments: list[dict] | None = None) -> dict:
    """The exact result IS the answer: advisor.py returns it without calling a model, so it cannot be miscounted
    (this is also why these answers show 0 tokens)."""
    return {**chunk, "direct_answer": answer, "conflict": conflict, "attachments": attachments or []}


def _course_table(rows) -> str:
    """Markdown table, one line per course: an either/or slot is shown as its alternatives."""
    lines = ["| Course Code | Title | Semester | Credits | L-T-P | Prerequisites |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["code_status"] == "choice_slot":
            continue
        ltp = f"{r['L']}-{r['T']}-{r['P']}" if r["L"] is not None else "-"
        prereq = (r["prerequisites_normalized"] or "none").replace("|", "/")
        code = r["course_code"] + (" (either/or)" if r["choice_of"] else "")
        lines.append(f"| {code} | {r['title']} | {r['semester']} | {_num(r['credits'])} | {ltp} | "
                     f"{'NIL' if prereq == 'none' else prereq} |")
    return "\n".join(lines)


def _tally(rows) -> tuple[int, int, int, float]:
    """(courses listed, courses a student takes, undecided TBA slots, credits). An either/or slot is one row with
    choice_of NULL plus one row per alternative, and only the slot row counts towards credits."""
    slots = [r for r in rows if r["choice_of"] is None]
    tba = sum(r["code_status"] == "placeholder" for r in slots)
    listed = sum(r["code_status"] not in ("choice_slot", "placeholder", "not_offered") for r in rows)
    return listed, len(slots) - tba, tba, sum(r["credits"] or 0 for r in slots)


def _plural(n: int, word: str = "course") -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _how_to_read(rows, listed: int, takes: int, tba: int) -> str:
    notes = []
    alts = [r["course_code"] for r in rows if r["code_status"] == "choice_slot"]
    if alts:
        notes.append(f"{', '.join(alts)} {'is an either/or choice' if len(alts) == 1 else 'are either/or choices'}, "
                     f"so a student actually takes {_plural(takes)} from these {listed}.")
    if tba:
        notes.append(f"{_plural(tba, 'further slot')} {'is' if tba == 1 else 'are'} listed only as TBA in the source "
                     f"(the student chooses later).")
    return " ".join(notes)


def semester_list(batch: str, sem: int) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT * FROM courses WHERE kind='programme' AND batch=? AND semester=? "
                           "ORDER BY source_row", (batch, sem)).fetchall()
    if not rows:
        return []
    listed, takes, tba, credits = _tally(rows)
    head = (f"**The {batch} batch has {_plural(listed)} listed in Semester {sem}, worth {_num(credits)} credits "
            f"in total.**")
    answer = "\n\n".join(x for x in (head, _how_to_read(rows, listed, takes, tba), _course_table(rows)) if x)
    return [_direct(_chunk(answer, SPREAD_FILE, f"tab '{rows[0]['source_tab']}', semester {sem}, {_rows_label(rows)}"),
                    answer)]


def basket_courses(baskets: list[tuple], batches: list[str], sem: int | None = None) -> list[dict]:
    """"How many / which courses are in basket X" - every semester of the batch, from the database."""
    names = [n for basket_names, _ in baskets for n in basket_names]
    label = " + ".join(lbl for _, lbl in baskets)
    marks = ",".join("?" * len(names))
    with _db() as con:
        all_rows = con.execute(f"SELECT * FROM courses WHERE kind='programme' AND basket IN ({marks}) "
                               "ORDER BY batch, semester, source_row", names).fetchall()
        req = {(r["batch"], r["basket"]): r for r in
               con.execute(f"SELECT * FROM baskets WHERE basket IN ({marks})", names)}
    if sem:
        all_rows = [r for r in all_rows if r["semester"] == sem]
    if not all_rows:
        return []
    if not batches:  # differs by batch: give the overview rather than guessing one
        lines = [f"**The {label} basket differs by batch (year of admission):**", "",
                 "| Batch | Courses listed | Courses a student takes | Credits |", "|---|---|---|---|"]
        for batch, rows in _group(all_rows, lambda r: r["batch"]).items():
            listed, takes, tba, credits = _tally(rows)
            lines.append(f"| {batch} | {listed} | {takes}{f' + {tba} TBA' if tba else ''} | {_num(credits)} |")
        lines += ["", "Tell me your batch (for example, \"for the 2022 batch\") and I'll list every course."]
        answer = "\n".join(lines)
        return [_direct(_chunk(answer, SPREAD_FILE, f"{label} basket, all batches (Sem Spread tabs)"), answer)]
    chunks = []
    for batch in batches:
        rows = [r for r in all_rows if r["batch"] == batch]
        if not rows:
            continue
        listed, takes, tba, credits = _tally(rows)
        sems = sorted({r["semester"] for r in rows})
        where = f"Semester {sem}" if sem else f"{'Semester' if len(sems) == 1 else 'Semesters'} " \
            f"{', '.join(map(str, sems[:-1]))}{' and ' if len(sems) > 1 else ''}{sems[-1]}"
        parts = [f"**The {batch} batch {label} basket lists {_plural(listed)} across {where}, worth {_num(credits)} "
                 f"credits in total.**"]
        note = _how_to_read(rows, listed, takes, tba)
        if note:
            parts.append(note)
        parts.append(_course_table(rows))
        conflict = False
        for name in names:
            r = req.get((batch, name))
            if not sem and r and r["placed_credits"] is not None and r["placed_credits"] != r["min_credits"]:
                conflict = True
                parts.append(f"Inconsistency in the source: the tab requires {_num(r['min_credits'])} credits for "
                             f"this basket, but the courses it places there add up to {_num(r['placed_credits'])}. "
                             f"Please confirm with your Program Chair.")
        answer = "\n\n".join(parts)
        chunks.append(_direct(_chunk(answer, SPREAD_FILE, f"tab '{rows[0]['source_tab']}', {label} basket, "
                                     f"{_rows_label(rows)}"), answer, conflict))
    if len(chunks) > 1:  # several batches named: one reply holding all of them
        answer = "\n\n---\n\n".join(c["direct_answer"] for c in chunks)
        chunks = [{**chunks[0], "direct_answer": answer, "text": answer,
                   "conflict": any(c["conflict"] for c in chunks)}]
    return chunks


def batch_overview(batch: str) -> list[dict]:
    """"How many courses does the 2022 batch have in total" - every basket of one batch."""
    with _db() as con:
        rows = con.execute("SELECT * FROM courses WHERE kind='programme' AND batch=? ORDER BY semester, source_row",
                           (batch,)).fetchall()
    if not rows:
        return []
    lines = ["| Basket | Courses listed | Slots a student takes | Credits |", "|---|---|---|---|"]
    for basket, grp in _group(rows, lambda r: r["basket"]).items():
        listed, takes, tba, credits = _tally(grp)
        lines.append(f"| {basket} | {listed} | {takes}{f' + {tba} TBA' if tba else ''} | {_num(credits)} |")
    listed, takes, tba, credits = _tally(rows)
    answer = (f"**The {batch} batch spread lists {_plural(listed)} with named codes, plus {_plural(tba, 'TBA slot')} "
              f"the student fills later, placing {_num(credits)} credits in total.**\n\n" + "\n".join(lines)
              + "\n\nAsk for any basket by name to see its full course table.")
    return [_direct(_chunk(answer, SPREAD_FILE, f"tab '{rows[0]['source_tab']}', all baskets, {_rows_label(rows)}"),
                    answer)]


def infographic_reply(batch: str) -> list[dict]:
    with _db() as con:
        row = con.execute("SELECT source_tab FROM courses WHERE kind='programme' AND batch=? LIMIT 1",
                          (batch,)).fetchone()
    if not row:
        return []
    answer = (f"Here is the course-structure infographic for the **{batch} batch**: credits by basket and every course "
              f"by semester. It is drawn directly from the semester spread, so its numbers match the source file.")
    return [_direct(_chunk(answer, SPREAD_FILE, f"tab '{row['source_tab']}'"), answer,
                    attachments=[{"type": "infographic", "batch": batch}])]


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


def route(question: str, profile: dict | None = None, message: str | None = None, facts=None) -> list[dict]:
    q = question.strip()
    low = q.lower()
    batches = _batches(q, profile, message, facts)
    codes = mentioned_codes(q)
    minor = next((name for key, name in MINOR_NAMES.items() if re.search(rf"\b{re.escape(key)}\b", low)), None)
    minor_q = minor and (re.search(r"\bminor\b", low) or LIST_WORDS.search(low))

    # "how many more credits do I need", "credits left to graduate" - a requirement, not a ranking question
    requirement_q = re.search(r"(credits?|how many|how much).{0,30}\b(need|needed|require|required|remaining|left)\b"
                              r"|\bto (graduate|complete)\b|\bmore credits\b", low)
    if CAN_TAKE_MINOR_RE.search(low) and not minor and not codes:
        if not batches:
            return ask_batch_for_minor()
        stated = re.search(r"\b(?:semester|sem)\s*(\d)\b", low)
        semester = int(stated.group(1)) if stated else (facts.semester if facts else None) \
            or (profile or {}).get("current_semester")
        return minor_availability(batches[0], semester)
    if RANKING_RE.search(low) and not minor_q and not requirement_q:  # "which semester has the most credits"
        return semester_stats(batches)
    if codes:  # a specific course: semester / credits / prerequisites
        return course_lookup(codes, batches)
    if minor_q and not re.search(r"credits? (needed|required)|basket", low):
        return minor_summary(minor, batches)
    if not minor and re.search(r"\bminors?\b", low) and not re.search(r"credit|basket", low) and (
            LIST_WORDS.search(low) or re.search(r"\b(which|what|available|offered|options?)\b", low)):
        return minors_overview(batches)
    wanted = [(names, label) for rx, names, label in BASKET_ALIASES if re.search(rx, low)]
    m = re.search(r"\b(?:semester|sem)\s*(\d)\b", low)
    sem = int(m.group(1)) if m else None
    if INFOGRAPHIC_RE.search(low) and batches:
        return infographic_reply(batches[0])
    if wanted and COURSE_LIST_RE.search(low) and not SELF_CHECK_RE.search(low):
        # "minor" alone means the named minors above, not the TBA Open/Minor slots
        listable = [w for w in wanted if w[1] != "Minor/Open" or re.search(r"basket|open", low)]
        if listable:
            return basket_courses(listable[:2], batches, sem)
    if wanted and re.search(r"credit", low):
        return basket_credits(wanted[:2], batches)
    if sem and batches and re.search(r"course|subject|offered|taught|study", low) and not SELF_CHECK_RE.search(low):
        return semester_list(batches[0], sem)
    if batches and re.search(r"\b(courses|subjects)\b", low) and re.search(r"how many|\blist\b|\ball\b|total", low) \
            and not SELF_CHECK_RE.search(low):
        return batch_overview(batches[0])
    return []
