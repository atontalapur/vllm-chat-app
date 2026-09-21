"""Trace accumulation from the token stream.

Chunk fixtures here are copied from the real capture in
`docs/spikes/s0-3-sample-stream.json`, not invented, because every awkward case
this code handles came from that capture: multi-token chunks, a role-only first
chunk with null logprobs, and finish_reason riding along with the final token.

Two properties matter more than the arithmetic:

- a malformed chunk must never raise, or it would break a live response
- the accumulator must not run before the user has the chunk in hand
"""

import httpx
import pytest
from app.config import settings
from app.schemas import ChatRequest
from app.tracing import StreamAccumulator
from app.vllm_client import _build_payload, stream_chat

from tests.conftest import UPSTREAM_URL, sse

# Role-only opener: no content, logprobs null. Must not count as a token.
ROLE_CHUNK = (
    '{"id":"c-1","choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    '"logprobs":null,"finish_reason":null}]}'
)

# One chunk, two tokens — the case that makes indexing [0] wrong.
TWO_TOKEN_CHUNK = (
    '{"id":"c-1","choices":[{"index":0,"delta":{"content":"Hello there"},'
    '"logprobs":{"content":['
    '{"token":"Hello","logprob":-0.5,"bytes":[72],"top_logprobs":[]},'
    '{"token":" there","logprob":-1.5,"bytes":[32],"top_logprobs":[]}]},'
    '"finish_reason":null}]}'
)

# Final chunk: last token and finish_reason arrive together.
FINAL_CHUNK = (
    '{"id":"c-1","choices":[{"index":0,"delta":{"content":"!"},'
    '"logprobs":{"content":['
    '{"token":"!","logprob":-0.25,"bytes":[33],"top_logprobs":[]}]},'
    '"finish_reason":"stop"}]}'
)


def feed(*lines: str) -> StreamAccumulator:
    acc = StreamAccumulator()
    for line in lines:
        acc.observe(line)
    return acc


def test_accumulates_text_in_order() -> None:
    acc = feed(f"data: {ROLE_CHUNK}", f"data: {TWO_TOKEN_CHUNK}", f"data: {FINAL_CHUNK}")

    assert acc.response == "Hello there!"


def test_counts_every_token_in_a_multi_token_chunk() -> None:
    """A chunk can carry many tokens; one observed on the box carried 11.

    Reading only the first would undercount n_tokens and pull the mean toward
    whichever token happened to be first in each chunk.
    """
    acc = feed(f"data: {TWO_TOKEN_CHUNK}")

    assert acc.n_tokens == 2
    assert acc.mean_logprob == pytest.approx(-1.0)


def test_confidence_proxy_matches_the_spike_formula() -> None:
    acc = feed(f"data: {ROLE_CHUNK}", f"data: {TWO_TOKEN_CHUNK}", f"data: {FINAL_CHUNK}")
    trace = acc.build("req-1", "test-model", [{"role": "user", "content": "hi"}])

    assert trace.n_tokens == 3, "the role-only chunk must not count as a token"
    assert trace.mean_logprob == pytest.approx((-0.5 + -1.5 + -0.25) / 3)
    assert trace.min_logprob == pytest.approx(-1.5)
    assert trace.finish_reason == "stop"
    assert trace.response == "Hello there!"


def test_finish_reason_length_is_preserved() -> None:
    """Sprint 3 drops max_tokens-truncated responses, so this must survive."""
    truncated = FINAL_CHUNK.replace('"finish_reason":"stop"', '"finish_reason":"length"')

    assert feed(f"data: {truncated}").finish_reason == "length"


def test_mean_is_none_not_zero_without_logprobs() -> None:
    """None and 0.0 are opposite signals downstream.

    Sprint 3 selects traces *below* a mean-logprob threshold. A 0.0 here reads
    as perfect confidence, which would quietly exclude every response captured
    with the flag off from ever being curated.
    """
    no_logprobs = '{"id":"c-1","choices":[{"index":0,"delta":{"content":"hi"},"logprobs":null}]}'
    acc = feed(f"data: {no_logprobs}")

    assert acc.response == "hi"
    assert acc.n_tokens == 0
    assert acc.mean_logprob is None


@pytest.mark.parametrize(
    "line",
    [
        "data: [DONE]",
        "data: not json at all",
        "data: []",
        "data: null",
        ": a comment line",
        "",
        '{"no":"data prefix"}',
        'data: {"choices":[]}',
        'data: {"choices":"not a list"}',
        'data: {"choices":[{"delta":null,"logprobs":{"content":"not a list"}}]}',
        'data: {"choices":[{"logprobs":{"content":[{"logprob":"not a number"}]}}]}',
        'data: {"choices":[{"logprobs":{"content":[{"logprob":true}]}}]}',
    ],
)
def test_malformed_lines_never_raise(line: str) -> None:
    """Every one of these costs the trace, not the user's response."""
    acc = feed(line)

    assert acc.n_tokens == 0
    assert acc.mean_logprob is None


def test_json_true_is_not_counted_as_a_logprob() -> None:
    """bool is a subclass of int, so a naive isinstance check would score 1.0."""
    acc = feed('data: {"choices":[{"logprobs":{"content":[{"logprob":true}]}}]}')

    assert acc.n_tokens == 0


def test_payload_requests_logprobs_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "capture_logprobs", True)

    payload = _build_payload(ChatRequest(messages=[{"role": "user", "content": "hi"}]))  # type: ignore[list-item]

    assert payload["logprobs"] is True
    # Without this vLLM returns chunks with "logprobs": null despite the flag.
    assert payload["top_logprobs"] == 0


def test_payload_omits_logprobs_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "capture_logprobs", False)

    payload = _build_payload(ChatRequest(messages=[{"role": "user", "content": "hi"}]))  # type: ignore[list-item]

    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


async def test_chunk_is_yielded_before_any_trace_work(respx_mock) -> None:  # type: ignore[no-untyped-def]
    """The user's first token must not wait on trace bookkeeping.

    Driven as a generator rather than through the client so the ordering is
    observable: with a TestClient the response buffer hides which side ran
    first. Trace work for chunk N happens while awaiting chunk N+1, so after
    the first chunk is in hand the accumulator has still seen nothing.
    """
    respx_mock.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, text=sse(TWO_TOKEN_CHUNK, FINAL_CHUNK))
    )
    acc = StreamAccumulator()
    stream = stream_chat(
        ChatRequest(messages=[{"role": "user", "content": "hi"}]),  # type: ignore[list-item]
        "req-1",
        accumulator=acc,
    )

    first = await anext(stream)

    assert "Hello there" in first
    assert acc.chunks_observed == 0, "trace work ran before the client had the chunk"

    # Draining the rest still accumulates everything, including the last chunk.
    async for _ in stream:
        pass
    assert acc.chunks_observed == 2
    assert acc.response == "Hello there!"
    assert acc.finish_reason == "stop"


async def test_stream_is_unchanged_by_accumulation(respx_mock) -> None:  # type: ignore[no-untyped-def]
    """Bytes out must be identical whether or not a trace is being built."""
    respx_mock.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, text=sse(ROLE_CHUNK, TWO_TOKEN_CHUNK, FINAL_CHUNK))
    )
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])  # type: ignore[list-item]

    without = [c async for c in stream_chat(req, "req-1")]

    respx_mock.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, text=sse(ROLE_CHUNK, TWO_TOKEN_CHUNK, FINAL_CHUNK))
    )
    with_acc = [c async for c in stream_chat(req, "req-1", accumulator=StreamAccumulator())]

    assert without == with_acc
