"""FastAPI app.  Run:  uvicorn main:app --reload --port 8000"""
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import advisor
import llm
import multimodal
from config import ALLOWED_ORIGINS, MAX_UPLOAD_BYTES, RATE_LIMIT_PER_MIN
from retrieval import retrieve
from schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm-up: load the embedding model, load the chunk index and create the LLM clients once at startup,
    # so the first student request does not pay ~5 s of model-loading latency.
    retrieve("warm up")
    llm.warm_up()
    advisor.load_profiles()
    yield


app = FastAPI(title="VU AI Academic Advisor", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])

_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    """Per-IP limit on the endpoints that spend API credit. Per server instance, so a best-effort brake on abuse
    rather than a hard quota (see the audit in the report)."""
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    now, hits = time.time(), _hits[ip]
    while hits and hits[0] < now - 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    hits.append(now)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
    return response


@app.get("/health")
def health():
    return {"status": "ok", "llm_chain": llm.CHAIN, "strategies": advisor.STRATEGIES}


@app.get("/students")
def students():
    """Synthetic profiles for the UI dropdown (summary fields only; full records are never served)."""
    return [{"student_id": p["student_id"], "batch": p["batch"], "current_semester": p["current_semester"],
             "cgpa": p["cgpa"], "completed_credits": p["completed_credits"], "scenario": p.get("scenario", "")}
            for p in advisor.load_profiles().values()]


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(req: ChatRequest):
    if req.student_id and req.student_id not in advisor.load_profiles():
        raise HTTPException(404, "unknown student")
    # sync endpoint -> FastAPI runs it in a thread pool, so concurrent requests don't block each other
    return advisor.answer(req.message, req.strategy, req.student_id,
                          [t.model_dump() for t in req.conversation_history], guarded=True,
                          session_record=req.session_record.model_dump() if req.session_record else None)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is too large (4 MB maximum).")
    if not data:
        raise HTTPException(400, "That file is empty.")
    return data


@app.post("/voice/transcribe", dependencies=[Depends(rate_limit)])
async def transcribe(file: UploadFile = File(...)):
    data = await _read_upload(file)
    if not (file.content_type or "").startswith(("audio/", "video/webm")):
        raise HTTPException(415, "Please send an audio file.")
    try:
        return multimodal.transcribe(data, file.filename or "speech.webm")
    except Exception as e:
        print(f"[voice] transcription failed ({type(e).__name__})")
        raise HTTPException(502, "I couldn't transcribe that. Please try again or type your question.") from e


@app.post("/transcript/read", dependencies=[Depends(rate_limit)])
async def read_transcript(file: UploadFile = File(...)):
    data = await _read_upload(file)
    if not multimodal.image_type(data):  # checks the bytes: the browser's content type can say anything
        raise HTTPException(415, "Please upload a PNG, JPEG or WebP image of your transcript.")
    try:
        return multimodal.read_transcript(data)
    except Exception as e:
        print(f"[vision] extraction failed ({type(e).__name__})")
        raise HTTPException(502, "I couldn't read that image right now. Please try again.") from e


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)


@app.post("/speech", dependencies=[Depends(rate_limit)])
def speech(req: SpeechRequest):
    try:
        audio, provider = multimodal.speak(req.text)
    except Exception as e:
        print(f"[tts] speech failed ({type(e).__name__})")
        raise HTTPException(502, "Speech is unavailable right now.") from e
    return Response(audio, media_type="audio/wav", headers={"X-AI-Generated": provider})


@app.get("/infographic/{batch}")
def infographic(batch: str):
    svg = multimodal.infographic_svg(batch) if batch in ("2022", "2023", "2024", "2025", "2026") else None
    if not svg:
        raise HTTPException(404, "unknown batch")
    return Response(svg, media_type="image/svg+xml", headers={
        "Cache-Control": "public, max-age=3600", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})
