from fastapi import FastAPI, HTTPException, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from typing import Optional
import os
import logging

from session_store import (
    create_session, get_session, delete_session,
    get_active_session_count, SESSION_EXPIRY_SECONDS,
)

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Model Consensus System",
    version="1.0.0",
    description="AI query engine using three LLM providers with two-phase self-check reconciliation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "mmcs_session")


class KeysPayload(BaseModel):
    """
    API keys are now all optional — users can supply one, two, or three.
    At least one must be present (validated below) but we do not force
    all three. This lets users try the system with just the keys they have.
    """
    claude_key:   Optional[str] = ""
    claude_model: str = "claude-sonnet-4-6"
    openai_key:   Optional[str] = ""
    openai_model: str = "gpt-4o"
    gemini_key:   Optional[str] = ""
    gemini_model: str = "gemini-1.5-pro"

    @field_validator("claude_key")
    @classmethod
    def validate_claude_key(cls, v):
        # Only validate format if a key was actually provided.
        if v and v.strip() and not v.startswith("sk-ant-"):
            raise ValueError("Claude key should start with sk-ant-")
        return v.strip() if v else ""

    @field_validator("openai_key")
    @classmethod
    def validate_openai_key(cls, v):
        if v and v.strip() and not v.startswith("sk-"):
            raise ValueError("OpenAI key should start with sk-")
        return v.strip() if v else ""

    @field_validator("gemini_key")
    @classmethod
    def validate_gemini_key(cls, v):
        if v and v.strip() and not v.startswith("AIza"):
            raise ValueError("Gemini key should start with AIza")
        return v.strip() if v else ""


class SessionStatusResponse(BaseModel):
    status: str
    message: str
    models: dict
    active_providers: list


class QueryRequest(BaseModel):
    question: str
    resources: str = ""
    max_loops: int = 3


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "active_sessions": get_active_session_count(),
    }


@app.post("/session/init", response_model=SessionStatusResponse)
async def init_session(payload: KeysPayload, response: Response):
    # Require at least one real key before creating a session.
    has_claude = bool(payload.claude_key and payload.claude_key.strip())
    has_openai = bool(payload.openai_key and payload.openai_key.strip())
    has_gemini = bool(payload.gemini_key and payload.gemini_key.strip())

    if not any([has_claude, has_openai, has_gemini]):
        raise HTTPException(
            status_code=422,
            detail="Please provide at least one API key to start a session."
        )

    session_id = create_session(
        claude_key=payload.claude_key if has_claude else None,
        claude_model=payload.claude_model,
        openai_key=payload.openai_key if has_openai else None,
        openai_model=payload.openai_model,
        gemini_key=payload.gemini_key if has_gemini else None,
        gemini_model=payload.gemini_model,
    )

    response.set_cookie(
        key=COOKIE_NAME, value=session_id,
        httponly=True, samesite="strict",
        max_age=SESSION_EXPIRY_SECONDS, secure=False,
    )

    active = []
    if has_claude: active.append("claude")
    if has_openai: active.append("openai")
    if has_gemini: active.append("gemini")

    return SessionStatusResponse(
        status="ok",
        message=f"Session created with {len(active)} provider(s): {', '.join(active)}.",
        models={
            "claude": payload.claude_model,
            "openai": payload.openai_model,
            "gemini": payload.gemini_model,
        },
        active_providers=active,
    )


@app.get("/session/status")
async def session_status(
    mmcs_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)
):
    if mmcs_session is None:
        raise HTTPException(status_code=401, detail="No session cookie found.")
    session = get_session(mmcs_session)
    if session is None:
        raise HTTPException(status_code=401, detail="Session not found or expired.")
    return {
        "status": "active",
        "message": "Session is valid.",
        "active_providers": session.active_providers(),
        "active_sessions_total": get_active_session_count(),
    }


@app.delete("/session")
async def end_session(
    response: Response,
    mmcs_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)
):
    if mmcs_session:
        delete_session(mmcs_session)
    response.delete_cookie(key=COOKIE_NAME, samesite="strict")
    return {"status": "ok", "message": "Session ended. Your API keys have been cleared from memory."}


@app.post("/query")
async def run_query(
    body: QueryRequest,
    mmcs_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)
):
    if mmcs_session is None:
        raise HTTPException(status_code=401, detail="No session found. Please submit your API keys first.")

    session = get_session(mmcs_session)
    if session is None:
        raise HTTPException(status_code=401, detail="Session expired. Please re-enter your API keys.")

    if not session.has_enough_providers():
        raise HTTPException(status_code=422, detail="No active providers in session.")

    if not body.question or not body.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    max_loops = max(1, min(body.max_loops, 8))

    try:
        from orchestrator import run_orchestration
        from resolution import resolve

        orchestration_result = await run_orchestration(
            session_keys=session,
            question=body.question.strip(),
            resources=body.resources.strip(),
            max_loops=max_loops,
        )

        resolution = resolve(orchestration_result)

        return {
            "final_answer":           resolution.final_answer,
            "resolution_rule":        resolution.resolution_rule,
            "resolution_explanation": resolution.resolution_explanation,
            "winning_provider":       resolution.winning_provider,
            "winning_confidence":     resolution.winning_confidence,
            "loops_completed":        resolution.loops_completed,
            "consensus_reached":      resolution.consensus_reached,
            "resource_conflict":      resolution.resource_conflict,
            "resource_conflict_note": resolution.resource_conflict_note,
            "all_answers":            resolution.all_answers,
            "active_providers":       session.active_providers(),
            "iterations": [
                {
                    "loop_number":       it.loop_number,
                    "consensus_reached": it.consensus_reached,
                    "phase1": [
                        {"provider": r.provider, "answer": r.answer,
                         "confidence": r.confidence, "reasoning": r.reasoning,
                         "fault_found": r.fault_found, "fault_reason": r.fault_reason}
                        for r in it.phase1_results
                    ],
                    "phase2": [
                        {"provider": r.provider, "answer": r.answer,
                         "confidence": r.confidence, "reasoning": r.reasoning,
                         "fault_found": r.fault_found, "fault_reason": r.fault_reason}
                        for r in it.phase2_results
                    ],
                }
                for it in orchestration_result.iterations
            ],
        }

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing.")