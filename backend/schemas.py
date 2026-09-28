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


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    needs_clarification: bool = False  # set on assistant turns that asked a follow-up question


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    student_id: Optional[str] = None
    strategy: Literal["baseline", "structured", "rag", "rag_structured"] = "rag_structured"
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
