"""
Safety layer around the advisor (Phase 10). Runs before and after the model and needs no model to refuse.

    message -> scrub personal data -> injection / other-student / write-action checks -> [advisor] -> output check

Layer 1 is rules (instant, explainable). Layer 2 is Llama Prompt Guard 2 on Groq, which catches rephrased injections;
if it is unreachable the request continues, because layer 1 and the model's own rules still apply.
"""
import re
from dataclasses import dataclass, field

from groq import Groq

from config import GROQ_API_KEY, PROMPT_GUARD_MODEL, PROMPT_GUARD_THRESHOLD

PII_PATTERNS = [
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[email removed]"),
    ("Aadhaar number", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[ID number removed]"),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[ID number removed]"),
    ("phone number", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), "[phone removed]"),
    ("roll number", re.compile(r"\b(?:roll|registration|enrol+ment|usn)\s*(?:no\.?|number|#)?\s*[:=-]?\s*[A-Z0-9-]*\d[A-Z0-9-]*\b",
                               re.I), "[student number removed]"),
]
INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|constraint|guideline|polic(y|ies) above)"
    r"|(reveal|show|print|repeat|output|tell me|what (is|are)|give me|dump)\b.{0,40}\b(system|hidden|initial|developer)\s+"
    r"(prompt|message|instruction)s?"
    r"|\byour (system prompt|instructions|rules verbatim|api key)\b|\bapi[- ]?keys?\b|\benv(ironment)? variables?\b"
    r"|\b(developer|dan|debug|god|admin|sudo|root) mode\b|\bjailbreak\b"
    r"|\b(you are now|from now on you|pretend (to be|you are)|act as|roleplay as)\b.{0,50}"
    r"\b(registrar|dean|admin|administrator|unrestricted|without (any )?(rules|restrictions)|chair|vice[- ]chancellor|dan)\b"
    r"|<\s*/?\s*(system|instructions?|student_question|context|chunk)\b"
    r"|\bnew instructions?\s*:|\bsystem\s*:\s", re.I | re.S)
OTHER_STUDENT_RE = re.compile(
    r"\b(all|every|list of|other|another|each)\s+(the\s+)?(students?|profiles?|records?)\b.{0,50}"
    r"\b(cgpa|grades?|marks?|records?|names?|profiles?|data|details?|transcripts?|results?|attendance)\b"
    r"|\b(cgpa|grades?|marks?|records?|names?|profiles?|data|details?|transcripts?|results?)\b.{0,40}"
    r"\b(of|for)\s+(all|every|other|another|each)\s+(the\s+)?students?\b"
    r"|\b(my|a)\s+(friend|roommate|classmate|batchmate|brother|sister|senior|junior)('s)?\b.{0,40}"
    r"\b(cgpa|grades?|marks?|record|transcript|result|attendance|fail(ed)?|pass(ed)?)\b"
    r"|\b(student|profile) database\b|\bdump\b.{0,20}\b(database|table|db|students?)\b|\bselect \* from\b", re.I | re.S)
WRITE_ACTION_RE = re.compile(
    r"\b(change|update|edit|modify|increase|fix|set|alter|delete|remove)\s+(my|his|her|their|the)\s+"
    r"(cgpa|grades?|marks?|attendance|record|transcript)\b"
    r"|\b(mark|register|enrol+|approve|waive|grant)\s+me\b|\bapprove my\b"
    r"|\bwaive\b.{0,30}\b(prerequisite|attendance|fee)", re.I)
SYN_ID_RE = re.compile(r"\bSYN-?\d{3}\b", re.I)
# the words a leaked prompt would contain: none of them belongs in a reply to a student
LEAK_RE = re.compile(r"</?(constraints|grounding_rules|eligibility_procedure|output_format|student_profile|chunk)\b"
                     r"|kind=\"exact\"|needs_clarification|insufficient_information|gsk_[A-Za-z0-9]{8,}", re.I)
# Decisions the documents give to a person: the advisor still explains the rule but says who decides.
HUMAN_DECISIONS = [
    (r"suicid|self[- ]harm|kill myself|depress|anxiety|panic attack|mental health|want to die",
     "This sounds important and is beyond what an AI advisor should handle. Please talk to a person today: your "
     "Faculty Advisor [Mentor] or the Director - Campus Life (9823385284, Student Handbook p. 83)."),
    (r"harass|ragging|bully|assault|discriminat",
     "A person must handle this. Contact the Ombudsperson or the Director - Campus Life (9823385284, Student Handbook "
     "Section VII and p. 83)."),
    (r"malpractice|disciplin|appeal|grievance|re-?evaluation|expel|suspend",
     "The outcome is decided by the University, not by this advisor. Confirm with the Deputy Registrar (Examination & "
     "Evaluation): 7204176753 (Student Handbook p. 83)."),
    (r"medical|hospital|attendance (relax|waiver)|withdraw|drop out|leave of absence|semester break|credit transfer|"
     r"transfer .{0,20}credits|graduat|degree|distinction",
     "This is advice, not a decision. The Program Chair or Office of the Registrar makes the final call: Assistant "
     "Registrar (Academic), 8792489795 (Student Handbook p. 83)."),
]
REFUSALS = {
    "prompt_injection": "I can't follow that request. I only answer questions about Vidyashilp University academics from "
                        "the university's documents, and my instructions and configuration stay private. What would you "
                        "like to know about courses, credits or regulations?",
    "other_student": "I can't share or discuss another student's academic record. I can only use the record of the "
                     "student asking. If you need someone else's record for an official reason, please contact the "
                     "Office of the Registrar - Assistant Registrar (Academic): 8792489795 (Student Handbook, p. 83).",
    "write_action": "I can't change records, approve requests or register anyone: this advisor is read-only and those "
                    "decisions belong to University staff. I can explain the rule or who to contact. Your Faculty "
                    "Advisor or the Assistant Registrar (Academic), 8792489795 (Student Handbook, p. 83), can act on it.",
}


@dataclass
class Verdict:
    clean: str
    category: str | None = None          # set when the request is refused
    redacted: list[str] = field(default_factory=list)


def scrub(text: str) -> tuple[str, list[str]]:
    """Personal identifiers are removed before anything leaves this server: the advisor never needs them."""
    found = []
    for name, rx, replacement in PII_PATTERNS:
        text, n = rx.subn(replacement, text)
        if n:
            found.append(name)
    return text, found


def _prompt_guard_score(text: str) -> float:
    try:
        resp = Groq(api_key=GROQ_API_KEY, timeout=3, max_retries=0).chat.completions.create(
            model=PROMPT_GUARD_MODEL, messages=[{"role": "user", "content": text[:2000]}])
        return float(resp.choices[0].message.content)
    except Exception as e:
        print(f"[guard] prompt guard unavailable ({type(e).__name__}); continuing with rules only")
        return 0.0


def check_input(message: str, student_id: str | None, use_classifier: bool = True) -> Verdict:
    clean, redacted = scrub(message)
    ids = {re.sub(r"\D", "", m) for m in SYN_ID_RE.findall(clean)}
    own = re.sub(r"\D", "", student_id or "")
    if INJECTION_RE.search(clean):
        return Verdict(clean, "prompt_injection", redacted)
    if ids - {own} or OTHER_STUDENT_RE.search(clean):
        return Verdict(clean, "other_student", redacted)
    if WRITE_ACTION_RE.search(clean):
        return Verdict(clean, "write_action", redacted)
    if use_classifier and _prompt_guard_score(clean) >= PROMPT_GUARD_THRESHOLD:
        return Verdict(clean, "prompt_injection", redacted)
    return Verdict(clean, None, redacted)


def clean_history(history: list[dict]) -> list[dict]:
    """The history comes from the browser, so it is untrusted too: a refused attempt must not be replayed to the
    model on the next turn, and identifiers typed earlier are scrubbed the same way."""
    out = []
    for turn in history:
        text = scrub(turn["content"])[0]
        if turn["role"] == "user" and (INJECTION_RE.search(text) or OTHER_STUDENT_RE.search(text)):
            continue
        out.append({**turn, "content": text})
    return out


def check_output(answer: str, student_id: str | None) -> str | None:
    """Returns a reason if the reply must not be shown."""
    own = re.sub(r"\D", "", student_id or "")
    if {re.sub(r"\D", "", m) for m in SYN_ID_RE.findall(answer)} - {own}:
        return "other_student"
    if LEAK_RE.search(answer):
        return "prompt_injection"
    return None


def human_review(text: str) -> str | None:
    return next((note for rx, note in HUMAN_DECISIONS if re.search(rx, text, re.I)), None)
