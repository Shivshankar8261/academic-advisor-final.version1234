"""
LLM wrapper — the ONLY place a model is called.

Provider chain (tried in order, first success wins):
    1. Groq   openai/gpt-oss-120b   (primary: ~1-2 s, strict JSON-schema output, low reasoning effort)
    2. Groq   openai/gpt-oss-20b    (same API, separate rate-limit budget)
    3. Gemini gemini-flash-lite-latest  (fallback)
    4. Gemini gemini-3.5-flash-lite     (last resort)

Low-latency design:
  * No SDK-level retries — a failing model is skipped immediately instead of retried.
  * Cool-down "circuit breaker": when a model returns 429 (rate limit) it is skipped for the
    retry-after period, so later requests don't waste time hitting it again.
  * Small max_tokens (Groq reserves max_tokens against the per-minute token budget).

For the evaluation, LLM_PIN=<provider:model> forces a single model and waits out rate limits,
so all four strategies are compared on exactly the same model.

Each call returns (result, provider) so the API / eval can report which model actually answered.
"""
import copy
import re
import threading
import time
from functools import lru_cache

from google import genai
from google.genai import types
from groq import Groq, RateLimitError

from config import (GEMINI_API_KEY, GEMINI_MODELS, GROQ_API_KEY, GROQ_MODELS, GROQ_REASONING_EFFORT, LLM_PIN,
                    LLM_TIMEOUT_S, MAX_TOKENS)
from schemas import AdvisorResponse

FALLBACK_RESPONSE = AdvisorResponse(
    answer="Sorry — the advisor could not reach a language model right now. Please try again in a minute.",
    confidence="low", grounded=False, sources=[], needs_clarification=False, clarifying_question=None,
    insufficient_information=True, conflict_detected=False)

CHAIN = [f"groq:{m}" for m in GROQ_MODELS] + [f"gemini:{m}" for m in GEMINI_MODELS]
if LLM_PIN:
    CHAIN = [LLM_PIN]

_cooldown_until: dict[str, float] = {}
_lock = threading.Lock()


@lru_cache
def _groq() -> Groq:
    return Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_S, max_retries=0)


@lru_cache
def _gemini() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _strict_schema(model) -> dict:
    """Pydantic JSON schema -> strict schema (inline $defs, no extra keys, every field required)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def fix(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            node = {k: fix(v) for k, v in node.items() if k not in ("title", "default")}
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            return node
        if isinstance(node, list):
            return [fix(n) for n in node]
        return node

    return fix(schema)


ADVISOR_SCHEMA = _strict_schema(AdvisorResponse)


def _groq_call(model: str, system: str | None, user: str, structured: bool) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    kwargs = {}
    if structured:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "advisor_response", "strict": True, "schema": ADVISOR_SCHEMA}}
    resp = _groq().chat.completions.create(
        model=model, messages=messages, temperature=0, max_completion_tokens=MAX_TOKENS,
        reasoning_effort=GROQ_REASONING_EFFORT, **kwargs)
    return resp.choices[0].message.content or ""


def _gemini_call(model: str, system: str | None, user: str, structured: bool) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=0, max_output_tokens=MAX_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **({"response_mime_type": "application/json", "response_schema": AdvisorResponse} if structured else {}))
    return _gemini().models.generate_content(model=model, contents=user, config=config).text or ""


def _retry_after(err: Exception) -> float:
    """Parse Groq's 'Please try again in 7m12.5s' / '23.3s' hint (seconds)."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(err))
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5) if m else 20.0


def _run(system: str | None, user: str, structured: bool, parse):
    """Try each provider in the chain; returns (parsed_result, provider) or (None, 'none')."""
    for name in CHAIN:
        if not LLM_PIN and _cooldown_until.get(name, 0) > time.time():
            continue  # this model recently hit its rate limit — skip it without waiting
        provider, model = name.split(":", 1)
        call = _groq_call if provider == "groq" else _gemini_call
        for _attempt in range(6 if LLM_PIN else 1):  # pinned (eval) mode waits out rate limits
            try:
                return parse(call(model, system, user, structured)), name
            except RateLimitError as e:
                wait = _retry_after(e)
                if LLM_PIN:
                    time.sleep(wait)
                    continue
                with _lock:
                    _cooldown_until[name] = time.time() + wait
                print(f"[llm] {name} rate-limited, cooling down {wait:.0f}s -> next provider")
                break
            except Exception as e:
                print(f"[llm] {name} failed ({type(e).__name__}: {str(e)[:150]}) -> next provider")
                if LLM_PIN:  # eval mode: transient network / 503 errors -> wait and retry the SAME model
                    time.sleep(5)
                    continue
                break
    return None, "none"


def call_text(user: str, system: str | None = None) -> tuple[str, str]:
    """Free-text completion (baseline strategy). Returns (text, provider)."""
    text, provider = _run(system, user, False, lambda raw: raw)
    return (text if text is not None else FALLBACK_RESPONSE.answer), provider


def call_structured(system: str, user: str) -> tuple[AdvisorResponse, str]:
    """JSON-schema-constrained completion validated with Pydantic. Returns (response, provider)."""
    result, provider = _run(system, user, True, AdvisorResponse.model_validate_json)
    return (result or FALLBACK_RESPONSE), provider


def warm_up():
    """Create the HTTP clients at startup so the first request doesn't pay connection setup."""
    _groq()
    _gemini()
