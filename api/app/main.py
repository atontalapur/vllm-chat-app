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

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.auth import RequireApiKey
from app.config import settings
from app.logging import logger
from app.schemas import ChatRequest, ErrorBody
from app.vllm_client import UpstreamError, stream_chat

app = FastAPI(
    title="vllm-chat-app API",
    description="Application layer between the chat UI and the model server.",
    version="0.1.0",
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

    stream = stream_chat(req, request_id)

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
