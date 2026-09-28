"""FastAPI app.  Run:  uvicorn main:app --reload --port 8000"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import advisor
import llm
from retrieval import retrieve
from schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm-up: load the embedding model, open ChromaDB and create the LLM clients once at startup,
    # so the first student request does not pay ~5 s of model-loading latency.
    retrieve("warm up")
    llm.warm_up()
    advisor.load_profiles()
    yield


app = FastAPI(title="VU AI Academic Advisor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "llm_chain": llm.CHAIN, "strategies": advisor.STRATEGIES}


@app.get("/students")
def students():
    """Synthetic profiles for the UI dropdown (summary fields only)."""
    return [{"student_id": p["student_id"], "batch": p["batch"], "current_semester": p["current_semester"],
             "cgpa": p["cgpa"], "completed_credits": p["completed_credits"], "scenario": p.get("scenario", "")}
            for p in advisor.load_profiles().values()]


@app.get("/students/{student_id}")
def student(student_id: str):
    p = advisor.load_profiles().get(student_id)
    if not p:
        raise HTTPException(404, "unknown student")
    return p


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.student_id and req.student_id not in advisor.load_profiles():
        raise HTTPException(404, "unknown student")
    # sync endpoint -> FastAPI runs it in a thread pool, so concurrent requests don't block each other
    return advisor.answer(req.message, req.strategy, req.student_id,
                          [t.model_dump() for t in req.conversation_history])
