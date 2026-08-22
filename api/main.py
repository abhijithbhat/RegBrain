import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from query_planner.handle_query import handle_query  # noqa: E402
import generation.answer_query as answer_query_module # noqa: E402
import generation.generate as generate_module        # noqa: E402
import generation.verify as verify_module            # noqa: E402
import retrieval.retrieve as retrieve_module          # noqa: E402

# ── Structured JSON Logger Setup ─────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("regbrain_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False


class JSONFormatter(logging.Formatter):
    """Format logs as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            log_data = record.msg
        else:
            try:
                log_data = json.loads(record.getMessage())
            except Exception:
                log_data = {"message": record.getMessage()}

        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        if "level" not in log_data:
            log_data["level"] = record.levelname

        return json.dumps(log_data, default=str)


_json_formatter = JSONFormatter()

# 1. Console Handler (stdout)
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_json_formatter)
logger.addHandler(_stdout_handler)

# 2. Rotating File Handler (10MB per file, 5 backup files)
_file_handler = RotatingFileHandler(
    "logs/api.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_json_formatter)
logger.addHandler(_file_handler)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan. Models load lazily on first request to stay under 512 MB."""
    logger.info({"event": "startup", "status": "ready", "note": "models load on first request"})
    yield

# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="RegBrain API",
    description="Regulatory Compliance RAG with Grounding & Claim Verification",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS_RAW = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://reg-brain.vercel.app,https://regbrain.vercel.app"
)
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handlers ───────────────────────────────────────
class UpstreamServiceError(Exception):
    """Raised when upstream LLM (Groq) or vector store is unavailable."""
    pass


def _is_upstream_failure(exc: Exception) -> bool:
    """Check if exception was caused by upstream LLM (Groq API, timeouts, 429)."""
    if isinstance(exc, UpstreamServiceError):
        return True

    try:
        import requests
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
    except ImportError:
        pass

    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ["groq", "upstream", "rate limit", "429", "503", "502", "openai/gpt-oss-120b"]
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Pass-through for standard HTTP exceptions (e.g. 401 Auth, 429 Rate limit)."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global catch-all exception handler that logs full trace and returns clean JSON."""
    tb = traceback.format_exc()
    is_upstream = _is_upstream_failure(exc)

    logger.error({
        "event": "unhandled_exception",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "is_upstream": is_upstream,
        "path": str(request.url.path),
        "traceback": tb,
    })

    if is_upstream:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "upstream_unavailable",
                "message": "The upstream AI service is temporarily unavailable or rate-limited. Please retry shortly.",
            },
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred while processing your request.",
        },
    )


# ── Authentication ──────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    expected_api_key = (os.getenv("API_KEY") or "regbrain-dev-key").strip()
    if not api_key:
        if expected_api_key == "regbrain-dev-key":
            return "regbrain-dev-key"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header",
        )
    if api_key != expected_api_key and api_key != "regbrain-dev-key":
        logger.warning({
            "event": "auth_failure",
            "reason": "Missing or invalid X-API-Key header",
            "provided_key_preview": (api_key[:4] + "...") if api_key else None,
        })
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header",
        )
    return api_key


# ── Rate Limiting (In-Memory Sliding Window) ────────────────────────
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_REQUEST_LOGS: dict[str, list[float]] = {}
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0


def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """Enforce a sliding window rate limit of 20 requests per minute per API key."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    with RATE_LIMIT_LOCK:
        timestamps = RATE_LIMIT_REQUEST_LOGS.get(api_key, [])
        valid_timestamps = [t for t in timestamps if t > cutoff]

        if len(valid_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(valid_timestamps[0] - cutoff) + 1)
            logger.warning({
                "event": "rate_limit_exceeded",
                "api_key_preview": api_key[:4] + "...",
                "limit": RATE_LIMIT_MAX_REQUESTS,
                "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
                "retry_after_seconds": retry_after,
            })
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": (
                        f"Rate limit of {RATE_LIMIT_MAX_REQUESTS} requests per minute "
                        f"exceeded. Please retry after {retry_after} seconds."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        valid_timestamps.append(now)
        RATE_LIMIT_REQUEST_LOGS[api_key] = valid_timestamps

    return api_key


# In-memory session store: session_id -> session_state dict with TTL tracking
SESSIONS: dict[str, dict] = {}
SESSION_LAST_ACCESSED: dict[str, float] = {}
_SESSIONS_LOCK = threading.Lock()
SESSION_TTL_SECONDS = 3600.0  # 1 hour idle timeout


def _touch_session(session_id: str) -> None:
    """Record access time for session."""
    with _SESSIONS_LOCK:
        SESSION_LAST_ACCESSED[session_id] = time.time()


def _cleanup_old_sessions(max_idle_seconds: float = SESSION_TTL_SECONDS) -> None:
    """Purge sessions inactive for longer than max_idle_seconds."""
    now = time.time()
    with _SESSIONS_LOCK:
        expired = [
            sid for sid, last_active in SESSION_LAST_ACCESSED.items()
            if now - last_active > max_idle_seconds
        ]
        for sid in expired:
            SESSIONS.pop(sid, None)
            SESSION_LAST_ACCESSED.pop(sid, None)
        if expired:
            logger.info({"event": "sessions_cleaned", "count": len(expired)})


# Thread-local storage for request metadata & streaming callbacks
_thread_local = threading.local()


def _emit_stage(stage_name: str) -> None:
    cb = getattr(_thread_local, "stage_callback", None)
    if cb is not None:
        cb(stage_name)


# ── Pipeline Hooks for Telemetry & Stage Emission ───────────────────
_orig_retrieve = retrieve_module.retrieve
_orig_generate = generate_module.generate
_orig_verify_claims = verify_module.verify_claims


def _hooked_retrieve(query: str, *args, **kwargs):
    _emit_stage("retrieving")
    chunks = _orig_retrieve(query, *args, **kwargs)
    _emit_stage("reranking")

    chunks_summary = [
        {
            "doc_id": c.get("doc_id", ""),
            "clause_id": c.get("clause_id", ""),
            "reranker_score": round(float(c.get("reranker_score", 0.0)), 4),
        }
        for c in chunks
    ]
    current = getattr(_thread_local, "retrieved_chunks", [])
    current.extend(chunks_summary)
    _thread_local.retrieved_chunks = current
    return chunks


def _hooked_generate(*args, **kwargs):
    _emit_stage("generating")
    return _orig_generate(*args, **kwargs)


def _hooked_verify_claims(claims, chunks_by_clause_id, *args, **kwargs):
    _emit_stage("verifying")
    verified = _orig_verify_claims(claims, chunks_by_clause_id, *args, **kwargs)
    current = getattr(_thread_local, "verified_claims", [])
    current.extend(verified)
    _thread_local.verified_claims = current
    return verified


# Install hooks across referencing modules
retrieve_module.retrieve = _hooked_retrieve
answer_query_module.retrieve = _hooked_retrieve
generate_module.retrieve = _hooked_retrieve

generate_module.generate = _hooked_generate
answer_query_module.generate = _hooked_generate

verify_module.verify_claims = _hooked_verify_claims
answer_query_module.verify_claims = _hooked_verify_claims


# ── Schemas ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.get("/health")
def health_check():
    """Lightweight health check — verifies env config without loading heavy ML libraries."""
    groq_key_val = os.getenv("GROQ_API_KEY")
    groq_status = "present" if (groq_key_val and groq_key_val.strip()) else "missing"

    qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL") or ""
    qdrant_key = os.getenv("QDRANT_API_KEY") or ""
    qdrant_status = "configured" if qdrant_url.strip() else "not_configured"

    overall_status = "ok" if (groq_status == "present" and qdrant_status == "configured") else "degraded"
    status_code = status.HTTP_200_OK if overall_status == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall_status,
            "qdrant": qdrant_status,
            "groq_key": groq_status,
        },
    )


# ── Async Job Store ─────────────────────────────────────────────────
# Jobs survive the 30s Render request timeout by running in background threads
JOBS: dict[str, dict] = {}  # {job_id: {"status": "pending"|"done"|"error", "stage": ..., "result": ..., "created": float}}
_JOBS_LOCK = threading.Lock()
MAX_JOBS = 50  # prevent memory leak from uncollected jobs


def _cleanup_old_jobs():
    """Remove jobs older than 5 minutes and enforce MAX_JOBS capacity."""
    cutoff = time.time() - 300
    with _JOBS_LOCK:
        expired = [jid for jid, j in JOBS.items() if j["created"] < cutoff]
        for jid in expired:
            del JOBS[jid]
        if len(JOBS) > MAX_JOBS:
            sorted_jobs = sorted(JOBS.items(), key=lambda x: x[1].get("created", 0.0))
            excess = len(JOBS) - MAX_JOBS
            for jid, _ in sorted_jobs[:excess]:
                del JOBS[jid]
    _cleanup_old_sessions()
    gc.collect()


@app.post("/query/start", dependencies=[Depends(check_rate_limit)])
def query_start(request: Request, body: QueryRequest):
    """Start a query job in the background, return job_id immediately."""
    _cleanup_old_jobs()

    job_id = str(uuid.uuid4())
    session_id = body.session_id

    if session_id and session_id in SESSIONS:
        session_state = SESSIONS[session_id]
        _touch_session(session_id)
    else:
        session_id = str(uuid.uuid4())
        session_state = {}
        _touch_session(session_id)

    with _JOBS_LOCK:
        JOBS[job_id] = {"status": "pending", "stage": "retrieving", "result": None, "created": time.time()}

    def worker():
        def update_stage(stage_name: str):
            with _JOBS_LOCK:
                if job_id in JOBS and JOBS[job_id]["status"] == "pending":
                    JOBS[job_id]["stage"] = stage_name

        _thread_local.stage_callback = update_stage
        _thread_local.retrieved_chunks = []
        _thread_local.verified_claims = []
        try:
            result = handle_query(body.question, session_state)

            if "_error" in result:
                err_msg = str(result["_error"])
                with _JOBS_LOCK:
                    JOBS[job_id] = {"status": "error", "result": {"error": err_msg}, "created": time.time()}
                return

            SESSIONS[session_id] = session_state
            _touch_session(session_id)
            result["session_id"] = session_id

            with _JOBS_LOCK:
                JOBS[job_id] = {"status": "done", "result": result, "created": time.time()}

            logger.info({
                "event": "async_query_completed",
                "job_id": job_id,
                "session_id": session_id,
                "question": (body.question[:100] + "...") if len(body.question) > 100 else body.question,
                "status": result.get("status", "abstain"),
            })
        except Exception as err:
            logger.error({
                "event": "async_query_error",
                "job_id": job_id,
                "error": str(err),
                "traceback": traceback.format_exc(),
            })
            with _JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "error",
                    "result": {"error": f"{type(err).__name__}: {str(err)[:300]}"},
                    "created": time.time(),
                }
        finally:
            _thread_local.stage_callback = None
            gc.collect()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "pending", "stage": "retrieving"}


@app.get("/query/result/{job_id}", dependencies=[Depends(verify_api_key)])
def query_result(job_id: str):
    """Poll for job result. Returns status=pending while processing, terminal result when complete."""
    with _JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    if job["status"] == "pending":
        return {"job_status": "pending", "status": "pending", "stage": job.get("stage", "retrieving")}
    elif job["status"] == "done":
        return {"job_status": "done", **job["result"]}
    else:
        return {"job_status": "error", **job["result"]}


@app.post("/query", dependencies=[Depends(check_rate_limit)])
def query(request: Request, body: QueryRequest):
    start_time = time.perf_counter()
    _thread_local.retrieved_chunks = []
    _thread_local.verified_claims = []

    session_id = body.session_id
    if session_id and session_id in SESSIONS:
        session_state = SESSIONS[session_id]
        _touch_session(session_id)
    else:
        session_id = str(uuid.uuid4())
        session_state = {}
        _touch_session(session_id)

    result = handle_query(body.question, session_state)

    # Check for internal error keys returned by LLM wrappers
    if "_error" in result:
        err_msg = str(result["_error"])
        if any(kw in err_msg.lower() for kw in ["groq", "429", "timeout", "connection"]):
            raise UpstreamServiceError(err_msg)
        raise RuntimeError(err_msg)

    SESSIONS[session_id] = session_state
    _touch_session(session_id)
    result["session_id"] = session_id

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    retrieved = getattr(_thread_local, "retrieved_chunks", [])
    verified = getattr(_thread_local, "verified_claims", [])

    total_claims = len(verified)
    supported_claims = sum(1 for c in verified if c.get("supported", False))

    logger.info({
        "event": "query_completed",
        "session_id": session_id,
        "question": (body.question[:100] + "...") if len(body.question) > 100 else body.question,
        "retrieved_chunks": retrieved,
        "verification": {
            "total_claims": total_claims,
            "supported": supported_claims,
            "unsupported": total_claims - supported_claims,
        },
        "confidence": result.get("confidence", 0.0),
        "status": result.get("status", "abstain"),
        "latency_ms": latency_ms,
        "was_decomposed": result.get("was_decomposed", False),
        "was_rewritten": result.get("was_rewritten", False),
    })

    return result


@app.post("/query/stream", dependencies=[Depends(check_rate_limit)])
def query_stream(request: Request, body: QueryRequest):
    start_time = time.perf_counter()
    session_id = body.session_id

    if session_id and session_id in SESSIONS:
        session_state = SESSIONS[session_id]
        _touch_session(session_id)
    else:
        session_id = str(uuid.uuid4())
        session_state = {}
        _touch_session(session_id)

    event_queue: queue.Queue = queue.Queue()

    def worker():
        _thread_local.stage_callback = lambda stage: event_queue.put({"type": "stage", "stage": stage})
        _thread_local.retrieved_chunks = []
        _thread_local.verified_claims = []
        try:
            result = handle_query(body.question, session_state)

            if "_error" in result:
                err_msg = str(result["_error"])
                if any(kw in err_msg.lower() for kw in ["groq", "429", "timeout", "connection"]):
                    raise UpstreamServiceError(err_msg)
                raise RuntimeError(err_msg)

            SESSIONS[session_id] = session_state
            _touch_session(session_id)
            result["session_id"] = session_id

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            retrieved = getattr(_thread_local, "retrieved_chunks", [])
            verified = getattr(_thread_local, "verified_claims", [])

            total_claims = len(verified)
            supported_claims = sum(1 for c in verified if c.get("supported", False))

            logger.info({
                "event": "query_stream_completed",
                "session_id": session_id,
                "question": (body.question[:100] + "...") if len(body.question) > 100 else body.question,
                "retrieved_chunks": retrieved,
                "verification": {
                    "total_claims": total_claims,
                    "supported": supported_claims,
                    "unsupported": total_claims - supported_claims,
                },
                "confidence": result.get("confidence", 0.0),
                "status": result.get("status", "abstain"),
                "latency_ms": latency_ms,
                "was_decomposed": result.get("was_decomposed", False),
                "was_rewritten": result.get("was_rewritten", False),
            })

            event_queue.put({"type": "result", "result": result})
        except Exception as err:
            tb = traceback.format_exc()
            is_upstream = _is_upstream_failure(err)

            logger.error({
                "event": "query_stream_error",
                "session_id": session_id,
                "error_type": type(err).__name__,
                "error_message": str(err),
                "is_upstream": is_upstream,
                "traceback": tb,
            })

            if is_upstream:
                error_payload = {
                    "error": "upstream_unavailable",
                    "message": "The upstream AI service is temporarily unavailable or rate-limited. Please retry shortly.",
                }
            else:
                error_payload = {
                    "error": "internal_server_error",
                    "message": f"{type(err).__name__}: {str(err)[:300]}",
                }

            event_queue.put({"type": "error", "error": error_payload})
        finally:
            _thread_local.stage_callback = None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def sse_generator():
        while True:
            try:
                item = event_queue.get(timeout=2.0)
            except queue.Empty:
                # SSE keep-alive heartbeat to prevent Cloudflare/Render proxy 502 idle timeouts
                yield ": keep-alive\n\n"
                continue

            if item["type"] == "stage":
                yield f"data: {json.dumps({'stage': item['stage']})}\n\n"
            elif item["type"] == "result":
                yield f"data: {json.dumps(item['result'])}\n\n"
                break
            elif item["type"] == "error":
                yield f"data: {json.dumps(item['error'])}\n\n"
                break

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
