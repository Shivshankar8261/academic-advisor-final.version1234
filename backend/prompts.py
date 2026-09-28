"""
The four prompting strategies compared in the evaluation (Phase 3 of the assignment).

Each strategy adds ONE layer of technique on top of the previous one, so the evaluation can
attribute improvements to specific techniques:

  1. baseline        raw question + a generic instruction. No role, no constraints, no documents,
                     no output format. Represents "just ask an LLM".
  2. structured      + role/persona, + explicit constraints (only known facts, say when unsure,
                     stay in scope), + delimiters (XML tags) separating instructions from the
                     question, + a strict JSON output format with uncertainty flags.
                     Still NO documents -> shows what prompting alone can and cannot fix.
  3. rag             structured + the top-k retrieved chunks from the university documents,
                     each labelled with [source_file | section]; the model must answer ONLY from
                     that context, cite it, flag insufficient information and conflicting rules.
  4. rag_structured  rag + the student's STRUCTURED profile (batch, semester, credits, CGPA,
                     completed / failed courses) + the current-term context, and an explicit
                     step-by-step eligibility procedure. The model must ask a clarifying
                     question when the profile / question lacks information it needs.

Every strategy calls the same Claude wrapper (llm.py). Strategies 2-4 use Claude's structured
output (JSON schema = schemas.AdvisorResponse) so the response is guaranteed to parse.
"""
import json

# ---------------------------------------------------------------------------
# 1. BASELINE — deliberately minimal. No system prompt at all.
# ---------------------------------------------------------------------------
BASELINE_TEMPLATE = "Answer this academic question from a university student:\n\n{question}"


def baseline_prompt(question: str) -> tuple[str | None, str]:
    """Returns (system_prompt, user_message). Baseline has no system prompt."""
    return None, BASELINE_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Shared building blocks for strategies 2-4
# ---------------------------------------------------------------------------

# Technique: ROLE / PERSONA
ROLE = (
    "You are the AI Academic Advisor for Vidyashilp University (VU). You help B.Tech students "
    "with questions about academic regulations, courses, prerequisites, credits, progression, "
    "registration and semester offerings."
)

# Technique: EXPLICIT CONSTRAINTS (what to do when unsure, scope limits)
BASE_CONSTRAINTS = """<constraints>
- Never invent course codes, credit numbers, CGPA thresholds, deadlines or rules. If you do not know a
  university-specific fact with certainty, say so and set insufficient_information to true.
- Only answer questions about VU academics. For anything else (personal, unrelated topics), politely decline
  and set insufficient_information to true.
- If the question is ambiguous or depends on facts about the student you do not have (e.g. which courses they
  completed, their batch/year, their CGPA), do NOT assume: set needs_clarification to true and ask ONE specific
  clarifying_question.
- Do not make a recommendation that the information does not support. It is better to say "I cannot confirm
  this" than to guess.
- Keep the answer concise (at most ~150 words) and student-friendly. Exception: when the student asks for a
  count or list, give the complete list from the exact result (one short line per course).
</constraints>"""

# Technique: OUTPUT FORMAT (enforced by the JSON schema; explained here so the model fills it well)
OUTPUT_SPEC = """<output_format>
Return JSON with these fields:
- answer: your reply to the student (if you need clarification, briefly say what you can already tell them).
- confidence: "high" only if every claim is directly supported; "medium" if partly; "low" otherwise.
- grounded: true only if every factual claim comes from the provided context documents.
- sources: list of {source_file, section} for every context chunk you relied on (copy them exactly from the
  chunk labels). Empty list if you used none.
- needs_clarification / clarifying_question: set when you must ask the student something before answering.
- insufficient_information: true if the available information cannot answer the question reliably.
- conflict_detected: true if two sources give different or inconsistent rules for this question; explain both
  rules in the answer and say which authority the student should confirm with.
</output_format>"""


def _history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [f"{t['role'].upper()}: {t['content']}" for t in history[-6:]]
    return "<conversation_so_far>\n" + "\n".join(lines) + "\n</conversation_so_far>\n\n"


# ---------------------------------------------------------------------------
# 2. STRUCTURED — role + constraints + delimiters + output format, no documents
# ---------------------------------------------------------------------------
def structured_prompt(question: str, history: list[dict]) -> tuple[str, str]:
    system = f"{ROLE}\n\n{BASE_CONSTRAINTS}\n\n{OUTPUT_SPEC}"
    # Technique: DELIMITERS — the student's words are fenced off from the instructions.
    user = (f"{_history_block(history)}"
            f"<student_question>\n{question}\n</student_question>\n\n"
            "You have NO access to university documents in this mode, so sources must be empty and grounded "
            "must be false. Answer only what you can state with certainty; otherwise flag it.")
    return system, user


# ---------------------------------------------------------------------------
# 3. RAG — structured + retrieved context with citation rules
# ---------------------------------------------------------------------------
RAG_RULES = """<grounding_rules>
- Use ONLY the information inside <context>. Do not use outside knowledge about universities in general.
- This also applies to general facts about VU itself (its location/address, fees and fee amounts, names of
  people, rankings, dates): state them only if a chunk in <context> says so, and cite that chunk. If the
  context does not state it, answer "This isn't in the provided documents", set insufficient_information to
  true, and name the office to contact if the context names one. Never answer from background knowledge, even
  if you are confident.
- Chunks with kind="exact" are results computed directly from the spreadsheets (counts, course lists, credits,
  semesters, prerequisites). Use their numbers and lists verbatim: do not recount, add, drop or rename courses.
  If they say a course code is not assigned / not known, a prerequisite is not defined in the files, or credits
  are TBA/TBD, say exactly that instead of filling the gap - and always tell the student every item listed
  under "Gaps in the source data", in your own words, even if they did not ask about it.
- Never name an office, person, website or document that does not appear in <context>.
- Do not refer to chunk ids or "chunks" in the answer text; citations go only in `sources`, and must include
  every chunk that a number, percentage or course code in your answer comes from.
- In prerequisites, "A AND (B OR C)" means A plus one of B or C.
- If kind="exact" chunks already answer the question (for one batch or for every batch), answer from them - do not
  ask a clarifying question whose answer would not change the result, and never ask the same question twice.
- Every factual claim must be traceable to a chunk; list those chunks in `sources` using the exact
  source_file and section from the chunk label.
- If the context does not contain the answer (or only part of it), say exactly what is missing and set
  insufficient_information to true. Do not fill gaps with typical university policies.
- If chunks disagree (e.g. the Handbook and the SOP state different rules, or a course's prerequisites differ
  between batches), set conflict_detected to true and present both versions with their sources.
- Curricula differ by batch (year of admission). If the answer depends on the batch and you do not know it,
  ask for it.
- If a prerequisite course code is noted as not appearing in the supplied data, say its details are unknown.
</grounding_rules>"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = c["text"].split("\n", 1)[1] if c["text"].startswith("[") else c["text"]  # label is in the tag
        kind = ' kind="exact"' if c.get("structured") else ""
        parts.append(f'<chunk id="{i}"{kind} source_file="{c["source_file"]}" section="{c["section"]}">\n'
                     f'{text}\n</chunk>')
    return "<context>\n" + "\n\n".join(parts) + "\n</context>"


def rag_prompt(question: str, history: list[dict], chunks: list[dict]) -> tuple[str, str]:
    system = f"{ROLE}\n\n{BASE_CONSTRAINTS}\n\n{RAG_RULES}\n\n{OUTPUT_SPEC}"
    user = (f"{format_context(chunks)}\n\n{_history_block(history)}"
            f"<student_question>\n{question}\n</student_question>")
    return system, user


# ---------------------------------------------------------------------------
# 4. RAG + STRUCTURED STUDENT DATA — adds the profile and an eligibility procedure
# ---------------------------------------------------------------------------
ELIGIBILITY_PROCEDURE = """<eligibility_procedure>
When the question is about whether THIS student can take / register for / progress / graduate, work through
these checks using <student_profile> and <context> before answering:
1. Identify the course(s) or rule involved and the student's batch (curriculum year).
2. Prerequisites: list the prerequisites for the student's batch; mark each as passed (in courses_completed),
   failed (in courses_failed, grade F/FA) or missing. A course with F or FA is NOT passed.
3. Offering: check in which semester the course is placed for the student's batch and whether that semester
   is odd or even; use <term_context> to decide whether it runs in the term the student is asking about.
4. Standing: check CGPA against progression / degree thresholds and credits against requirements, if relevant.
5. Decide: eligible / not eligible / cannot determine, and state the reasons. Suggest only options that the
   documents support (e.g. re-registration, make-up exam, summer term *if offered*).
If a field needed for a check is null/missing in the profile, or no profile is given and the question is about
the student's own situation, do NOT guess: set needs_clarification to true and ask for exactly that information
(e.g. "Which batch are you in, and have you passed MATH301 Optimization Techniques?").
If the student has already answered a clarifying question in <conversation_so_far>, use that answer.
</eligibility_procedure>"""


def format_profile(p: dict | None) -> str:
    """Compact, token-efficient rendering of the structured profile (one course per 'CODE Title=GRADE')."""
    if not p:
        return "No profile selected (anonymous student). Personal facts are unknown."
    course = lambda c: f"{c['code']} {c['title']} ({c['credits']}cr, sem {c['semester']})={c['grade']}"  # noqa: E731
    lines = [f"student_id: {p['student_id']}", f"programme: {p['programme']}", f"batch: {p['batch']}",
             f"current_semester: {p['current_semester']}", f"earned_credits: {p['completed_credits']}",
             f"cgpa: {p['cgpa']}", f"minor: {p.get('minor')}"]
    lines.append("courses_passed: " + ("; ".join(map(course, p["courses_completed"]))
                                       if p["courses_completed"] is not None else "UNKNOWN"))
    lines.append("courses_failed (not passed): " + ("; ".join(map(course, p["courses_failed"])) or "none"
                                                    if p["courses_failed"] is not None else "UNKNOWN"))
    extra = {k: v for k, v in p.items() if k not in {"student_id", "programme", "batch", "current_semester",
             "completed_credits", "cgpa", "minor", "courses_completed", "courses_failed", "scenario"}}  # scenario = UI-only note
    lines += [f"{k}: {json.dumps(v)}" for k, v in extra.items()]
    return "\n".join(lines)


def rag_structured_prompt(question: str, history: list[dict], chunks: list[dict],
                          profile: dict | None, term_context: str) -> tuple[str, str]:
    system = f"{ROLE}\n\n{BASE_CONSTRAINTS}\n\n{RAG_RULES}\n\n{ELIGIBILITY_PROCEDURE}\n\n{OUTPUT_SPEC}"
    profile_block = format_profile(profile)
    user = (f"<term_context>\n{term_context}\n</term_context>\n\n"
            f"<student_profile>\n{profile_block}\n</student_profile>\n\n"
            f"{format_context(chunks)}\n\n{_history_block(history)}"
            f"<student_question>\n{question}\n</student_question>")
    return system, user
