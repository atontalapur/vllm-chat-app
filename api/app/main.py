"""FastAPI application layer.

Sits between the chat UI and the model server. It adds nothing the model server
cannot do on its own — that is the point: this is where authentication, request
validation, logging, and error shaping live in a real system, and keeping it in
the path means those concerns are always exercised.

Route auth is deliberately per-route, not global:

    /chat/stream   X-API-Key required
    /health        open — the Compose healthcheck has no key
    /metrics       open — Prometheus has no key
"""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.auth import RequireApiKey
from app.config import settings
from app.logging import logger
from app.schemas import ChatRequest, ErrorBody
from app.trace_store import TraceStore
from app.tracing import StreamAccumulator
from app.vllm_client import UpstreamError, stream_chat

# Module-level so tests can substitute it. start() decides whether it does
# anything; with no DSN configured it stays disabled and submit() is a no-op.
trace_store = TraceStore(
    dsn=settings.trace_db_url,
    queue_size=settings.trace_queue_size,
    write_timeout_s=settings.trace_write_timeout_s,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the trace store's background worker for the life of the process.

    start() never raises: a trace store that will not open must not stop the
    api from serving chat.
    """
    await trace_store.start()
    try:
        yield
    finally:
        await trace_store.aclose()


app = FastAPI(
    title="vllm-chat-app API",
    description="Application layer between the chat UI and the model server.",
    version="0.1.0",
    lifespan=lifespan,
)

# Exposes /metrics. Left unauthenticated on purpose — see the module docstring.
Instrumentator().instrument(app).expose(app, include_in_schema=True)


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """One structured line per request, on stdout.

    request_id is generated here and echoed on the response so a user reporting
    a problem can name the exact request in the logs.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()

    response = await call_next(request)

    # Health and metrics are polled every few seconds; logging them would bury
    # the actual traffic.
    if request.url.path not in ("/health", "/metrics"):
        logger.info(
            "request",
            extra={
                "context": {
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            },
        )

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness only.

    Deliberately does not probe the model server. Startup ordering is Compose's
    job (`depends_on: condition: service_healthy`); conflating the two would
    make this service report unhealthy — and get restarted — every time the
    model server hiccuped.
    """
    return {"status": "ok"}


@app.post(
    "/chat/stream",
    dependencies=[RequireApiKey],
    responses={
        401: {"model": ErrorBody, "description": "Missing or invalid API key"},
        413: {"model": ErrorBody, "description": "Conversation too large"},
        502: {"model": ErrorBody, "description": "Model server unreachable or failing"},
    },
    tags=["chat"],
)
async def chat_stream(req: ChatRequest, request: Request) -> Response:
    """Proxy a chat completion, streaming tokens back as server-sent events."""
    request_id: str = request.state.request_id

    # Bound the conversation before it reaches the model. Without this, a long
    # session grows the prompt until it overruns the context window and the
    # model server returns a hard error mid-conversation.
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > settings.max_total_chars:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content=ErrorBody(
                error="conversation_too_large",
                detail=(
                    f"conversation is {total_chars} characters, limit is "
                    f"{settings.max_total_chars}. Start a new chat."
                ),
                request_id=request_id,
            ).model_dump(),
        )

    accumulator = StreamAccumulator()
    stream = stream_chat(req, request_id, accumulator=accumulator)

    # Pull the first chunk before returning, so a connection failure becomes a
    # real 502 instead of a 200 whose body immediately errors. Once the
    # response starts, the status code can no longer be changed.
    try:
        first = await anext(stream)
    except UpstreamError as exc:
        logger.error(
            "upstream unavailable",
            extra={"context": {"request_id": request_id, "detail": exc.detail}},
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorBody(
                error="upstream_error", detail=exc.detail, request_id=request_id
            ).model_dump(),
        )
    except StopAsyncIteration:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorBody(
                error="upstream_error",
                detail="model server produced no output",
                request_id=request_id,
            ).model_dump(),
        )

    async def body() -> AsyncIterator[str]:
        yield first
        async for chunk in stream:
            yield chunk

        # Only after the last chunk has gone out. submit() returns immediately
        # and is documented never to raise.
        #
        # Belt and braces all the same: this code runs *inside* the response
        # body generator, so anything escaping here would abort a response the
        # user has already received in full — turning a bookkeeping bug into a
        # visibly broken chat. The guarantee is worth making structural instead
        # of trusting a docstring two modules away.
        #
        # Reached only on a clean end of stream. If the client disconnects
        # early this generator is closed instead, and the partial trace is
        # discarded rather than written: a response the user never saw, with no
        # finish_reason, is not a training example.
        try:
            trace_store.submit(
                accumulator.build(
                    request_id=request_id,
                    model=settings.model_id,
                    messages=[m.model_dump() for m in req.messages],
                )
            )
        except Exception as exc:  # noqa: BLE001 - tracing may never break a response
            logger.error(
                "trace submission failed",
                extra={"context": {"request_id": request_id, "error": str(exc)}},
            )

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-ID": request_id,
            # Without this an intermediary proxy may buffer the whole response
            # and defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )
