"""
Agentic Academic Advisor (Phase 9): five agents with one job each, run in a fixed order and logged in a trace.

    Planner -> Regulation Retrieval -> Course Eligibility -> Credit Calculation -> Recommendation -> Verification

Only Recommendation (and a repair, if Verification rejects its draft) calls a model. Eligibility and credits are
plain code over the course store: a model should not be doing arithmetic or boolean logic it can get wrong. The
order is fixed rather than chosen by a model, since every question needs the same checks and a model-chosen plan
would add latency and a new way to fail. The evaluation (eval/run_agentic.py) measures whether all this helps.
"""
import re
import time

import llm
import memory
import prompts
import structured
from schemas import AdvisorResponse, Source, Usage

SELF_RE = re.compile(r"\b(i|my|me|am|mine)\b", re.I)
ELIGIBLE_CLAIM_RE = re.compile(r"\b(you are eligible|you can (take|register)|yes, you can|you meet the prerequisites?)\b", re.I)
INELIGIBLE_CLAIM_RE = re.compile(r"\b(not eligible|ineligible|cannot (take|register)|can't (take|register))\b", re.I)
NUMBER_FACT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%|\b\d{1,3} credits?\b|\b[A-Z]{3,4}\d{3}\b")


def _evaluate(expr: str, passed: set[str]) -> bool | None:
    """Evaluate 'A AND (B OR C)' against the passed courses; None if it isn't a plain course expression."""
    words = re.sub(r"[A-Z]{3,4}\d{3}", lambda m: str(m.group() in passed), expr)
    words = words.replace(" AND ", " and ").replace(" OR ", " or ")
    if not re.fullmatch(r"(True|False|and|or|\(|\)|\s)+", words):
        return None
    try:
        return bool(eval(words, {"__builtins__": {}}))  # noqa: S307 - whitelisted to booleans and brackets just above
    except SyntaxError:
        return None


def check_course(code: str, profile: dict, term: str) -> dict | None:
    """Prerequisites and term offering of one course for this student's batch. Pure lookup and logic."""
    with structured._db() as con:
        row = con.execute("SELECT * FROM courses WHERE course_code=? AND batch=? ORDER BY kind='programme' DESC LIMIT 1",
                          (code, profile["batch"])).fetchone()
    if not row:
        return None
    passed = {c["code"] for c in profile["courses_completed"] or []}
    failed = {c["code"] for c in profile["courses_failed"] or []}
    expr = row["prerequisites_normalized"] or "none"
    needed = (row["prerequisite_codes"] or "").split(",") if row["prerequisite_codes"] else []
    unknown = [c for c in needed if c not in structured._known_codes()]
    ok = True if expr == "none" else _evaluate(expr, passed)
    want_odd = term == "odd"
    in_term = bool((row["semester"] or 0) % 2) == want_odd
    if code in passed:
        verdict = "already_passed"
    elif ok is None or (ok is False and unknown):
        verdict = "cannot_determine"
    elif not ok:
        verdict = "not_eligible"
    elif not in_term:
        verdict = "not_offered_in_term"
    else:
        verdict = "eligible"
    return {"code": code, "title": row["title"], "verdict": verdict, "semester": row["semester"], "term": term,
            "prerequisites": expr, "passed": sorted(set(needed) & passed), "failed": sorted(set(needed) & failed),
            "missing": sorted(set(needed) - passed - failed), "unknown": unknown, "source_tab": row["source_tab"],
            "source_row": row["source_row"], "source_file": row["source_file"]}


def eligibility_agent(codes: list[str], profile: dict, query: str) -> tuple[list[dict], list[dict]]:
    term = "even" if re.search(r"\bnext (semester|term|sem)\b", query, re.I) else "odd"
    checks = [c for c in (check_course(code, profile, term) for code in codes[:3]) if c]
    chunks = []
    for c in checks:
        text = (f"COURSE ELIGIBILITY AGENT (computed from the student's record, use this decision): {c['code']} "
                f"{c['title']} for this student. Decision: {c['verdict'].replace('_', ' ').upper()}. Prerequisites: "
                f"{c['prerequisites']}. Passed: {c['passed'] or 'none'}. Failed: {c['failed'] or 'none'}. Not yet taken: "
                f"{c['missing'] or 'none'}. Not defined in the supplied files: {c['unknown'] or 'none'}. Placed in semester "
                f"{c['semester']}; the term asked about is the {c['term']} semester.")
        chunks.append(structured._chunk(text, c["source_file"] or structured.SPREAD_FILE,
                                        f"tab '{c['source_tab']}', {c['code']}, row {c['source_row']}"))
    return checks, chunks


def credit_agent(profile: dict) -> list[dict]:
    """Earned credits and progress per basket, added up in code."""
    passed = profile["courses_completed"] or []
    if not passed:
        return []
    with structured._db() as con:
        batch = profile["batch"]
        required = con.execute("SELECT basket, min_credits, source_tab, source_row FROM baskets WHERE batch=?",
                               (batch,)).fetchall()
        basket_of = {r["course_code"]: r["basket"] for r in con.execute(
            "SELECT course_code, basket FROM courses WHERE kind='programme' AND batch=?", (batch,))}
    done: dict[str, float] = {}
    for c in passed:
        basket = basket_of.get(c["code"])
        if basket:
            done[basket] = done.get(basket, 0) + c["credits"]
    earned = profile.get("completed_credits") or sum(c["credits"] for c in passed)
    parts = [f"{r['basket']} {structured._num(done.get(r['basket'], 0))}/{structured._num(r['min_credits'])}"
             for r in required]
    text = (f"CREDIT CALCULATION AGENT (computed, do not recalculate): earned credits {structured._num(earned)} of the "
            f"180 required, so {structured._num(180 - earned)} credits remain. By basket (earned/required): "
            f"{'; '.join(parts)}. CGPA {profile.get('cgpa')}.")
    tab = required[0]["source_tab"] if required else ""
    return [structured._chunk(text, structured.SPREAD_FILE, f"tab '{tab}', credit baskets of the {batch} batch")]


def eligibility_checks(query: str, profile: dict | None) -> list[dict]:
    """The decision the student's record supports, for any mode's draft to be checked against."""
    if not (profile and profile.get("batch") and profile.get("courses_completed") is not None
            and SELF_RE.search(query)):
        return []
    return eligibility_agent(structured.mentioned_codes(query), profile, query)[0]


def verify_all(result: AdvisorResponse, evidence: str, checks: list[dict], batch: str | None,
               ) -> tuple[list[str], list[dict]]:
    """verify() plus availability claims. Returns (problems, chunks holding evidence the draft may not have seen)."""
    conflicts, chunks = structured.claim_conflicts(result.answer, batch)
    return verify(result, evidence, checks) + conflicts, chunks


def verify(result: AdvisorResponse, evidence: str, checks: list[dict]) -> list[str]:
    """Deterministic review of the draft: facts must be in the evidence, and the decision must match the agent's."""
    import advisor
    if result.needs_clarification:
        return []
    problems = []
    plain = advisor._norm(evidence)
    for fact in dict.fromkeys(NUMBER_FACT_RE.findall(result.answer)):
        # "67 credits" may be written "earned_credits: 67" or "credits 67" in the evidence: the number is what counts
        needle = re.escape(fact.split()[0]) if "credit" in fact else re.escape(advisor._norm(fact))
        if not re.search(rf"(?<![\d.]){needle}(?![\d])", plain):
            problems.append(f"'{fact}' does not appear in the evidence")
    for c in checks:
        if c["verdict"] == "eligible" and INELIGIBLE_CLAIM_RE.search(result.answer):
            problems.append(f"the record shows {c['code']} is eligible, but the draft says the student cannot take it")
        if c["verdict"] in ("not_eligible", "not_offered_in_term") and ELIGIBLE_CLAIM_RE.search(result.answer) \
                and not INELIGIBLE_CLAIM_RE.search(result.answer):
            problems.append(f"the record shows {c['code']} is {c['verdict'].replace('_', ' ')}, but the draft says yes")
    if not result.sources and not result.insufficient_information:
        problems.append("the draft cites no source")
    return problems


def run(message: str, query: str, profile: dict | None, history: list[dict], facts=None,
        ) -> tuple[AdvisorResponse, str, Usage, list[dict], list[dict], list[dict]]:
    """Returns (result, provider, usage, trace, chunks, attachments)."""
    import advisor  # advisor dispatches to this module, so the import has to wait until both are loaded
    trace: list[dict] = []

    def step(agent: str, started: float, summary: str, usage: Usage | None = None):
        trace.append({"agent": agent, "summary": summary, "ms": round((time.perf_counter() - started) * 1000),
                      "tokens": usage.total_tokens if usage else 0})

    t = time.perf_counter()
    codes = structured.mentioned_codes(query)
    self_question = bool(SELF_RE.search(query))
    step("Planner", t, f"Courses mentioned: {', '.join(codes) or 'none'}. "
                       f"{'About this student' if self_question else 'General question'}.")

    t = time.perf_counter()
    facts = facts or memory.Facts()
    stated = memory.prompt_block(facts)
    exact, retrieved = advisor.gather_context(query, profile, message, facts)
    direct = next((c for c in exact if c.get("direct_answer")), None)
    if direct:
        step("Regulation Retrieval", t, "Answered exactly from the course database; no model needed.")
        return (advisor.direct_response(direct, memory.correction_note(message, facts)), advisor.DATABASE_PROVIDER, Usage(), trace, [direct],
                direct.get("attachments") or [])
    if not exact and not advisor.has_relevant(retrieved):
        step("Regulation Retrieval", t, "Nothing relevant in the documents; stopping instead of guessing.")
        return advisor.NOT_IN_DOCUMENTS, "none (no relevant document found)", Usage(), trace, retrieved, []
    chunks = advisor.assemble_chunks(exact, retrieved, profile, query)
    step("Regulation Retrieval", t, f"{len(chunks)} passages ({len(exact)} exact lookups).")

    t = time.perf_counter()
    checks: list[dict] = []
    has_record = profile and profile.get("courses_completed") is not None and profile.get("batch")
    if codes and self_question and not has_record and not history:
        batch = facts.batch
        rows = structured.course_lookup(codes[:1], [batch] if batch else [])
        step("Course Eligibility", t, "No student record, so asking instead of assuming.")
        title = rows[0]["text"].split("\n", 1)[0].split(": ", 1)[-1] if rows else codes[0]
        ask = f"Which prerequisite courses for {codes[0]} have you passed?" if batch else \
            f"Which batch are you in, and which prerequisite courses for {codes[0]} have you passed?"
        result = AdvisorResponse(
            answer=f"I can check {title} for you, but I don't have your record yet." if batch else
                   f"I can check {title} for you, but its prerequisites depend on your batch and I don't have your "
                   f"record yet.", confidence="medium", grounded=bool(rows),
            sources=[Source(source_file=rows[0]["source_file"], section=rows[0]["section"])] if rows else [],
            needs_clarification=True, clarifying_question=ask, insufficient_information=False, conflict_detected=False)
        return result, advisor.DATABASE_PROVIDER, Usage(), trace, chunks, []
    if codes and self_question and has_record:
        checks, found = eligibility_agent(codes, profile, query)
        chunks = found + chunks
        step("Course Eligibility", t, "; ".join(f"{c['code']}: {c['verdict'].replace('_', ' ')}" for c in checks)
             or "Course not in this batch's spread.")
    else:
        step("Course Eligibility", t, "Skipped: no course eligibility question.")

    t = time.perf_counter()
    if has_record and self_question:
        found = credit_agent(profile)
        chunks = chunks + found
        step("Credit Calculation", t, found[0]["text"].split("(computed, do not recalculate): ")[1][:110] + "..."
             if found else "No passed courses on record.")
    else:
        step("Credit Calculation", t, "Skipped: no student record needed.")

    t = time.perf_counter()
    term = advisor.term_context()
    system, user = prompts.rag_structured_prompt(message, history, chunks, profile, term, stated)
    result, provider, usage = llm.call_structured(system, user)
    result = advisor.finish(result, chunks)
    step("Recommendation", t, f"Draft written by {provider}.", usage)

    t = time.perf_counter()
    if provider == "none":
        step("Verification", t, "Skipped: no model was reachable.")
        return result, provider, usage, trace, chunks, []
    evidence = " ".join(c["text"] for c in chunks) + prompts.format_profile(profile) + user
    batch = facts.batch or (profile or {}).get("batch")
    problems, extra = verify_all(result, evidence, checks, batch)
    if problems:
        feedback = ("<verification_failed>\nYour previous draft was rejected: " + "; ".join(problems) + ". Rewrite it using "
                    "only the context, and do not state those claims.\n</verification_failed>\n\n")
        context = prompts.format_context(extra) + "\n\n" if extra else ""
        repaired, provider, more = llm.call_structured(system, feedback + context + user)
        usage = llm.add_usage(usage, more)
        chunks = chunks + extra
        evidence += " ".join(c["text"] for c in extra)
        result = advisor.finish(repaired, chunks)
        left = verify_all(result, evidence, checks, batch)[0]
        if left:
            result = result.model_copy(update={"confidence": "low", "answer": result.answer.rstrip() + "\n\n**Please "
                                               "verify:** I could not fully check this answer against the documents ("
                                               + "; ".join(left) + ")."})
        step("Verification", t, f"Rejected the draft ({'; '.join(problems)}), rewrote it, "
             + ("still unverified, so flagged it." if left else "and the rewrite passed."), more)
    else:
        step("Verification", t, "Every number and course code was found in the evidence; decision matches the record.")
    return result, provider, usage, trace, chunks, []
