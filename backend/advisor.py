"""
Advisor orchestration: (retrieve) -> build prompt for the chosen strategy -> call Claude -> validated JSON.

    (LLM = Groq gpt-oss-120b, automatic fallback to Gemini — see llm.py)

    question ─┬─ baseline ───────────────────────────────► prompt ─► Claude (free text) ─► wrapped JSON
              ├─ structured ─────────────────────────────► prompt ─► Claude (JSON schema)
              ├─ rag ─────────── retrieve top-k ─────────► prompt ─► Claude (JSON schema)
              └─ rag_structured  retrieve + prereqs + profile + term ─► prompt ─► Claude (JSON schema)

rag_structured (the final system) additionally:
  * routes counts / lists / credits / prerequisites / semesters to exact SQL lookups (structured.py,
    courses.db) and puts those results first in the context, labelled with file / tab / rows;
  * attaches verified document conflicts (structured.KNOWN_CONFLICTS) when a question touches them;
  * adds the Progression / Award-of-Degree rules for eligibility questions from students with a profile.
Both RAG strategies:
  * skip the LLM when nothing relevant is retrieved (MIN_RELEVANCE) -> "This isn't in the provided documents";
  * after the LLM: keep only citations of chunks actually shown, add citations for facts used but not cited,
    no source -> no claim, contact advice limited to documented offices, data gaps / conflicts always stated.
"""
import json
import re
import time
from functools import lru_cache

import llm
import prompts
import structured
from config import MIN_RELEVANCE, PROCESSED_DIR, STUDENT_PROFILES
from retrieval import prerequisite_records, retrieve, section_records, semester_records
from schemas import AdvisorResponse, ChatResponse, Source

STRATEGIES = ("baseline", "structured", "rag", "rag_structured")

# question pattern -> handbook section whose thresholds decide it (added for students with a profile)
STANDING_RULES = [
    (r"regist|progress|promot|next (year|semester)|(second|third|fourth|final) year|semester \d|eligib|detain|repeat",
     "Section III - Academic Regulations > 12. Progression"),
    (r"graduat|distinction|degree|convocation|pass out|finish",
     "Section III - Academic Regulations > 15. Award of Degree"),
]

# The contact comes from the handbook itself (Important Contact Numbers, p. 83), not from model knowledge.
CONTACT_HINT = ("Please check with the Office of the Registrar - Assistant Registrar (Academic): 8792489795 "
                "(Student Handbook, Important Contact Numbers, p. 83).")
NOT_IN_DOCUMENTS = AdvisorResponse(
    answer=f"This isn't in the provided documents (Student Handbook, SOP, semester spread and minor course "
           f"lists), so I can't answer it reliably. {CONTACT_HINT}",
    confidence="low", grounded=False, sources=[], needs_clarification=False, clarifying_question=None,
    insufficient_information=True, conflict_detected=False)


WELCOME = ("Welcome to the **Vidyashilp University AI Academic Advisor**! 👋\n\n"
           "I answer questions from the Student Handbook, the SOP, the semester spread structures and the minor "
           "course lists. You can ask me things like:\n"
           "- *How many minors are offered for the 2025 batch?*\n"
           "- *What are the courses in semester 3 for the 2024 batch?*\n"
           "- *What is the minimum attendance requirement?*\n\n"
           "Pick your profile at the top for answers about your own progress.")
# Whole-message small talk only: "hi, how many minors?" still goes through retrieval.
SMALL_TALK = [
    (r"(hi+|hey+|hello+|hii+|namaste|good (morning|afternoon|evening)|greetings|yo)( there)?( advisor)?",
     WELCOME),
    (r"(thanks?|thank you|thank u|thx|ty)( (so|very) much)?( a lot)?", "You're welcome! Ask me anything else about "
     "your courses, minors or academic rules."),
    (r"(bye|goodbye|see you|see ya|good night)", "Goodbye, and all the best with your studies at Vidyashilp "
     "University! 🎓"),
    (r"(ok(ay)?|cool|great|nice|got it|alright)", "Glad that helped. What else would you like to know?"),
    (r"(who|what) are you|what can you do|help", WELCOME),
]


def small_talk(message: str) -> str | None:
    m = re.sub(r"[\s!.?,:)(]+$", "", message.strip().lower())
    return next((reply for rx, reply in SMALL_TALK if re.fullmatch(rx, m)), None)


@lru_cache
def load_profiles() -> dict[str, dict]:
    return {p["student_id"]: p for p in json.loads(STUDENT_PROFILES.read_text())}


@lru_cache
def term_context() -> str:
    return (PROCESSED_DIR / "term_context.md").read_text()


def retrieval_query(message: str, history: list[dict]) -> str:
    """Multi-turn: if the advisor just asked a clarifying question, the new message is only an answer
    (e.g. "I'm in the 2024 batch"), so we merge it with the student's ORIGINAL question for retrieval."""
    query = message
    if not history or history[-1]["role"] != "assistant":
        return query
    # A short follow-up ("2025", "what about the 2024 batch?", "and semester 5?") refines the previous question
    # even when the advisor did not ask anything, so it is also merged with it.
    m = message.strip()
    is_new_question = re.search(r"\b(what|which|how|when|where|who|why|can|could|is|are|do|does|should|list|tell)\b",
                                m, re.I)
    follow_up = re.match(r"(what|how) about\b|and\b|for\b|in\b|only\b", m, re.I) or \
        (len(m.split()) <= 6 and not is_new_question)
    if history[-1].get("needs_clarification") or follow_up:
        previous_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
        query = f"{previous_user} {message}"
    return query


def _check_sources(result: AdvisorResponse, chunks: list[dict]) -> AdvisorResponse:
    """Keep only citations that point at a chunk the model was actually shown (models sometimes shorten or
    invent section names); map shortened ones back to the exact chunk label."""
    shown = {(c["source_file"], c["section"]) for c in chunks}
    kept = []
    for s in result.sources:
        cited = s.section.strip()
        same_file = sorted(sec for f, sec in shown if f == s.source_file)
        match = next((sec for sec in same_file if sec == cited), None) or (next(
            (sec for sec in same_file if sec.startswith(cited) or cited.startswith(sec)), None)
            if len(cited) >= 8 else None)
        match = (s.source_file, match) if match else None
        if match and match not in {(k.source_file, k.section) for k in kept}:
            kept.append(Source(source_file=match[0], section=match[1]))
    update = {"sources": kept}
    if result.grounded and not kept:  # "grounded" with nothing verifiable to point at is not grounded
        update["grounded"] = False
    return result.model_copy(update=update)


CONTACT_ADVICE_RE = re.compile(r"\b(contact|reach out|get in touch|enquire|inquire|check with|ask the|visit the)\b"
                               r"|\boffice\b|\bdepartment\b|\bwebsite\b", re.I)
# offices / roles that the handbook or SOP themselves name - advice to contact these is grounded
DOCUMENTED_ROLES_RE = re.compile(r"faculty advisor|mentor|program chair|dean|registrar|accounts department|"
                                 r"finance officer|vice[- ]chancellor|director|co-?ordinator|ombudsperson|"
                                 r"campus help centre|digii", re.I)
FACT_RE = re.compile(r"\b\d{1,3}\s?%|\b\d{1,3} credits?\b|\b[A-Z]{3,4}\d{3}\b")


def _norm(text: str) -> str:
    return re.sub(r"(\d)\s+%", r"\1%", re.sub(r"\s+", " ", text)).lower()


def _complete_citations(result: AdvisorResponse, chunks: list[dict]) -> AdvisorResponse:
    """Every hard fact in the answer (a percentage, a credit count, a course code) must be traceable to a cited
    chunk. Models often cite one chunk and silently use a second; add the shown chunk that contains the fact."""
    if not result.sources:
        return result
    by_label = {(c["source_file"], c["section"]): _norm(c["text"]) for c in chunks}
    cited = [by_label.get((s.source_file, s.section), "") for s in result.sources]
    added = list(result.sources)
    for fact in dict.fromkeys(_norm(f) for f in FACT_RE.findall(result.answer)):
        if any(fact in text for text in cited):
            continue
        hit = next((c for c in chunks if fact in _norm(c["text"])), None)
        if hit and (hit["source_file"], hit["section"]) not in {(s.source_file, s.section) for s in added}:
            added.append(Source(source_file=hit["source_file"], section=hit["section"]))
            cited.append(_norm(hit["text"]))
    return result.model_copy(update={"sources": added})


def _enforce_grounding(result: AdvisorResponse) -> AdvisorResponse:
    """No source, no claim: an answer that cites nothing it was shown is replaced by the not-in-documents
    reply; every 'insufficient information' reply tells the student whom to contact."""
    if result.needs_clarification:
        # asking a question is not "insufficient information" - the UI would show a warning AND a question
        return result.model_copy(update={"insufficient_information": False})
    if not result.sources and not result.insufficient_information:
        return NOT_IN_DOCUMENTS
    if result.insufficient_information:
        # Models like to add "contact the admissions office" - an office no document mentions. Drop the model's
        # own contact advice and give the contact that the handbook itself lists.
        sentences = re.split(r"(?<=[.!?])\s+", result.answer.strip())
        kept = [s for s in sentences if not CONTACT_ADVICE_RE.search(s) or DOCUMENTED_ROLES_RE.search(s)] \
            or sentences[:1]
        return result.model_copy(update={"answer": f"{' '.join(kept)}\n\n{CONTACT_HINT}"})
    return result


def _mention_gaps(result: AdvisorResponse, chunks: list[dict]) -> AdvisorResponse:
    """Placeholder codes, TBA credits and undefined prerequisites found by the exact lookup must reach the
    student even if the model left them out (models drop them when the question did not ask)."""
    if result.needs_clarification or result.insufficient_information:
        return result
    if any(c.get("conflict") for c in chunks) and not result.conflict_detected:
        result = result.model_copy(update={"conflict_detected": True})
    text = result.answer.lower()
    missing = [gap for c in chunks for gap, keys in c.get("must_mention", [])
               if not any(k.lower() in text for k in keys)]
    if not missing:
        return result
    note = "Note from the source data: " + "; ".join(dict.fromkeys(missing)) + "."
    return result.model_copy(update={"answer": f"{result.answer.rstrip()}\n\n{note}"})


def answer(message: str, strategy: str = "rag_structured", student_id: str | None = None,
           history: list[dict] | None = None) -> ChatResponse:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}")
    history = history or []
    profile = load_profiles().get(student_id) if student_id else None
    start = time.perf_counter()
    chunks: list[dict] = []
    retrieval_s = 0.0

    reply = small_talk(message)
    if reply:  # greetings / thanks: no retrieval, no LLM, no "not in documents" warning
        result = AdvisorResponse(answer=reply, confidence="high", grounded=False, sources=[],
                                 needs_clarification=False, clarifying_question=None,
                                 insufficient_information=False, conflict_detected=False)
        return _response(result, strategy, "small-talk", start, retrieval_s, chunks)

    if strategy == "baseline":
        convo = "".join(f"{t['role']}: {t['content']}\n" for t in history[-6:])
        _, user = prompts.baseline_prompt(f"{convo}{message}" if convo else message)
        text, provider = llm.call_text(user)
        # The baseline cannot self-report uncertainty or sources, so these fields are fixed defaults.
        result = AdvisorResponse(answer=text, confidence="medium", grounded=False, sources=[],
                                 needs_clarification=False, clarifying_question=None,
                                 insufficient_information=False, conflict_detected=False)
    elif strategy == "structured":
        system, user = prompts.structured_prompt(message, history)
        result, provider = llm.call_structured(system, user)
    else:
        query = retrieval_query(message, history)
        # Exact lookups (counts, lists, credits, prerequisites, semesters) come from the SQL course store,
        # never from similarity search; they go first in the context.
        exact = structured.route(query, profile) + structured.conflict_notes(query) \
            if strategy == "rag_structured" else []
        chunks = retrieve(query)
        relevant = exact or any(c["score"] >= MIN_RELEVANCE for c in chunks)
        if not relevant:  # nothing in the documents is about this: do not let the model answer from memory
            retrieval_s = time.perf_counter() - start
            return _response(NOT_IN_DOCUMENTS, strategy, "none (no relevant document found)", start, retrieval_s,
                             chunks)
        if strategy == "rag":
            system, user = prompts.rag_prompt(message, history, chunks)
        else:
            chunks = exact + chunks
            chunks += prerequisite_records(chunks)
            sem = (profile or {}).get("current_semester")
            if profile and profile.get("batch") and sem:  # the student's current + next semester course lists
                seen = {c["section"] for c in chunks}
                sems = [s for s in (sem, sem + 1) if 1 <= s <= 8]
                chunks += [c for c in semester_records(profile["batch"], sems) if c["section"] not in seen]
            if profile and profile.get("cgpa") is not None:
                # "Can I register / progress / graduate?" is decided by CGPA thresholds that similarity search
                # often misses (the question talks about registration, the rule sits under Progression).
                for pattern, section in STANDING_RULES:
                    if re.search(pattern, query, re.I):
                        seen = {c["section"] for c in chunks}
                        chunks += [c for c in section_records(section) if c["section"] not in seen]
            system, user = prompts.rag_structured_prompt(message, history, chunks, profile, term_context())
        retrieval_s = time.perf_counter() - start
        result, provider = llm.call_structured(system, user)
        result = _check_sources(result, chunks)
        result = _mention_gaps(_enforce_grounding(_complete_citations(result, chunks)), chunks)

    if strategy == "structured":  # no documents in this mode, whatever the model claims
        result = result.model_copy(update={"sources": [], "grounded": False})

    return _response(result, strategy, provider, start, retrieval_s, chunks)


def _response(result: AdvisorResponse, strategy: str, provider: str, start: float, retrieval_s: float,
              chunks: list[dict]) -> ChatResponse:
    total = time.perf_counter() - start
    return ChatResponse(**result.model_dump(), strategy=strategy, provider=provider,
                        response_time_s=round(total, 2), retrieval_time_s=round(retrieval_s, 3),
                        retrieved=[{"source_file": c["source_file"], "section": c["section"],
                                    "score": c.get("score")} for c in chunks])
