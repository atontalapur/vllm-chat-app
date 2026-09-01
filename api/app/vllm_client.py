"""Streaming proxy to the upstream OpenAI-compatible server.

Chunks are re-yielded as they arrive rather than collected and returned at the
end — buffering here would defeat the entire point of streaming, since the user
would wait for the full completion before seeing a single token.

Failure policy: any upstream problem becomes a 502 carrying a structured body.
The one thing this must never do is hang, because a hung request looks to the
user like a model that is simply slow, with no error and no recovery.

    client                     this proxy                    upstream
      |  POST /chat/stream         |                            |
      |--------------------------->|  POST /v1/chat/completions  |
      |                            |--------------------------->|
      |                            |<---- SSE chunk ------------|
      |<---- SSE chunk ------------|                            |
      |            ...             |          ...               |
      |<---- data: [DONE] ---------|<---- data: [DONE] ---------|
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.logging import logger
from app.schemas import ChatRequest

# Sent to the client when the upstream fails after streaming has begun.
# Formatted as an SSE event so a streaming client sees the failure on the
# channel it is already reading — by this point the 200 status is long gone.
_ERROR_EVENT = 'data: {{"error": "upstream_error", "detail": "{detail}"}}\n\n'


class UpstreamError(Exception):
    """Upstream was unreachable or failed before any bytes were streamed."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def _build_payload(req: ChatRequest) -> dict[str, object]:
    return {
        "model": settings.model_id,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }


def _extract_upstream_id(line: str) -> str | None:
    """Pull the upstream's own completion id out of an SSE data line.

    Logging this next to our request_id is how the two services' logs get
    correlated: the upstream image does not echo inbound headers, so there is
    no trace header to propagate.
    """
    if not line.startswith("data: ") or line.endswith("[DONE]"):
        return None
    try:
        chunk = json.loads(line[len("data: ") :])
    except json.JSONDecodeError:
        return None
    upstream_id = chunk.get("id")
    return upstream_id if isinstance(upstream_id, str) else None


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.upstream_connect_timeout_s,
        # Generation is legitimately long-running; a read timeout here would
        # cut off long completions partway through.
        read=None,
        write=settings.upstream_connect_timeout_s,
        pool=settings.upstream_connect_timeout_s,
    )


async def stream_chat(req: ChatRequest, request_id: str) -> AsyncIterator[str]:
    """Yield SSE lines from the upstream server.

    Raises UpstreamError if the connection or response status fails before any
    bytes are streamed, so the caller can still turn it into a 502. After the
    first byte the status is already committed, so a mid-stream failure is
    surfaced as an SSE error event instead.
    """
    url = f"{settings.vllm_base_url.rstrip('/')}/chat/completions"
    upstream_logged = False
    chunks = 0

    try:
        async with (
            httpx.AsyncClient(timeout=_timeout()) as client,
            client.stream("POST", url, json=_build_payload(req)) as response,
        ):
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:500]
                raise UpstreamError(f"upstream returned {response.status_code}: {body}")

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not upstream_logged:
                    upstream_id = _extract_upstream_id(line)
                    if upstream_id is not None:
                        logger.info(
                            "upstream stream opened",
                            extra={
                                "context": {
                                    "request_id": request_id,
                                    "upstream_id": upstream_id,
                                }
                            },
                        )
                        upstream_logged = True

                chunks += 1
                yield f"{line}\n\n"

            if chunks == 0:
                # A 200 that streams nothing is still a failed generation.
                raise UpstreamError("upstream closed the stream without sending data")

    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise UpstreamError(f"cannot reach model server: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.error(
            "upstream stream failed mid-flight",
            extra={
                "context": {
                    "request_id": request_id,
                    "chunks_before_failure": chunks,
                    "error": str(exc),
                }
            },
        )
        yield _ERROR_EVENT.format(detail="model server connection lost mid-stream")
