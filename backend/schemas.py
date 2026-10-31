"""Pydantic models shared by the API, the advisor and the evaluation harness."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    source_file: str
    section: str


class AdvisorResponse(BaseModel):
    """The structured JSON every strategy must return (validated by the Claude SDK's parse())."""
    answer: str
    confidence: Literal["high", "medium", "low"]
    grounded: bool = Field(description="True only if every claim is supported by the provided context")
    sources: list[Source]
    needs_clarification: bool
    clarifying_question: Optional[str]
    insufficient_information: bool
    # Addition to the brief's schema: lets us measure "handling of conflicting rules" directly.
    conflict_detected: bool = Field(description="True if the documents contain rules that disagree on this question")


class Usage(BaseModel):
    """Tokens used to produce one answer (0 when it came from the database without an LLM call)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = 0.0  # None = model not in config.PRICES_PER_M


class TranscriptCourse(BaseModel):
    code: str = Field(pattern=r"^[A-Z]{3,4}\d{3}$")
    grade: Literal["O", "A+", "A", "B+", "B", "C", "D", "F", "FA", "S", "U", "I"]


class SessionRecord(BaseModel):
    """A record the student confirmed after uploading a transcript. Only codes and grades are accepted: titles and
    credits are looked up in the course store, so free text from an upload never reaches the prompt."""
    batch: Literal["2022", "2023", "2024", "2025", "2026"]
    current_semester: int = Field(ge=1, le=8)
    courses: list[TranscriptCourse] = Field(max_length=80)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    needs_clarification: bool = False  # set on assistant turns that asked a follow-up question


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    student_id: Optional[str] = None
    strategy: Literal["baseline", "structured", "rag", "rag_structured", "agentic"] = "rag_structured"
    session_record: Optional[SessionRecord] = None
    conversation_history: list[ChatTurn] = Field(default=[], max_length=100)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()


class ChatResponse(AdvisorResponse):
    strategy: str
    provider: str               # which LLM actually answered, e.g. "groq:openai/gpt-oss-120b"
    response_time_s: float      # total wall-clock time
    retrieval_time_s: float     # time spent on retrieval + prompt assembly
    retrieved: list[dict] = []  # chunks shown to the model (for transparency in the UI / eval)
    usage: Usage = Usage()
    agent_trace: list[dict] = []  # agentic strategy: one entry per agent (what it did, ms, tokens)
    attachments: list[dict] = []  # e.g. {"type": "infographic", "batch": "2022"}
    guard: Optional[str] = None   # why the safety layer answered instead of the advisor, if it did
    privacy_notice: Optional[str] = None  # personal data was removed from the message before processing
    human_review: Optional[str] = None    # this decision belongs to a person; says who
