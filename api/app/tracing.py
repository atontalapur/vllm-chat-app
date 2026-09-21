"""Accumulating one trace out of the token stream.

The proxy forwards SSE lines it does not own. This module reads a copy of each
line as it goes past and builds the row that `pipeline/db/migrations/0001_traces.sql`
expects: the assistant text, the confidence proxy, and how generation ended.

Two rules shape everything here.

**Never break the stream.** A malformed or unexpected chunk must cost the trace,
not the user's response. Every parse is defensive and every failure degrades to
"no data for that chunk" rather than an exception, which is why this module has
no raise statements at all.

**Never guess the shape.** The chunk layout is not assumed from the OpenAI spec;
it was captured from the pinned vLLM version in `docs/spikes/s0-3-logprob-shape.md`,
and the awkward parts of that capture are the reason for the code below:

- `choices[0].logprobs.content` is a *list*. One observed chunk carried 11 tokens
  under a single `delta.content`, so indexing `[0]` would undercount every
  multi-token chunk and skew the mean.
- The first chunk is role-only with `"logprobs": null`. Skipped, not treated as
  a zero.
- The final chunk carries the last token *and* `finish_reason` together. There is
  no empty tail chunk to read it from, so both are read off whatever chunk has them.
- vLLM clamps logprob to >= -9999.0, so plain arithmetic is safe: no -inf to
  poison the sum.

The confidence proxy is `mean_logprob`, the log of the geometric-mean token
probability, so `exp(mean_logprob)` reads as average per-token confidence in
[0, 1]. `min_logprob` is kept beside it because the mean hides a single badly
chosen token in a long fluent answer. Neither one catches a confidently wrong
answer — that is what the Sprint 2 judge is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_DATA_PREFIX = "data: "
_DONE = "[DONE]"


@dataclass(frozen=True)
class Trace:
    """One finished request, ready to be written to the trace store.

    Mirrors the columns of the `traces` table. Built by `StreamAccumulator.build`
    once the stream is exhausted, never mid-stream.
    """

    request_id: str
    model: str
    messages: list[dict[str, str]]
    response: str
    mean_logprob: float | None
    min_logprob: float | None
    n_tokens: int
    finish_reason: str | None


def parse_data_line(line: str) -> dict[str, Any] | None:
    """Return the JSON object in an SSE `data:` line, or None.

    None covers everything that is not a chunk to read: blank lines, the
    terminal `[DONE]` sentinel, and anything that fails to parse. Callers treat
    all three the same way, so they are not distinguished.
    """
    if not line.startswith(_DATA_PREFIX):
        return None
    payload = line[len(_DATA_PREFIX) :].strip()
    if payload == _DONE:
        return None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return chunk if isinstance(chunk, dict) else None


def _first_choice(chunk: dict[str, Any]) -> dict[str, Any] | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, dict) else None


@dataclass
class StreamAccumulator:
    """Folds SSE lines into the parts of a `Trace`.

    Deliberately cheap: a dict lookup and some arithmetic per chunk, no I/O.
    The proxy feeds it *after* yielding each line to the client (see
    `vllm_client.stream_chat`), so this work never sits between the model and
    the user.
    """

    chunks_observed: int = 0
    n_tokens: int = 0
    finish_reason: str | None = None
    _text: list[str] = field(default_factory=list)
    _logprob_sum: float = 0.0
    _min_logprob: float | None = None

    def observe(self, line: str) -> None:
        """Read one SSE line. Never raises."""
        chunk = parse_data_line(line)
        if chunk is None:
            return
        self.chunks_observed += 1

        choice = _first_choice(chunk)
        if choice is None:
            return

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                self._text.append(content)

        # Read on whichever chunk carries it. vLLM puts it on the same chunk as
        # the final token, so waiting for a tail chunk would always miss it.
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self.finish_reason = finish_reason

        self._observe_logprobs(choice.get("logprobs"))

    def _observe_logprobs(self, logprobs: object) -> None:
        # Null on the role-only first chunk, and on every chunk when the
        # capture flag is off. Both mean "nothing to count", not "zero".
        if not isinstance(logprobs, dict):
            return
        content = logprobs.get("content")
        if not isinstance(content, list):
            return

        for entry in content:
            if not isinstance(entry, dict):
                continue
            value = entry.get("logprob")
            # bool is an int subclass in Python; a JSON true here would
            # otherwise count as a logprob of 1.0.
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            value = float(value)
            self.n_tokens += 1
            self._logprob_sum += value
            if self._min_logprob is None or value < self._min_logprob:
                self._min_logprob = value

    @property
    def response(self) -> str:
        return "".join(self._text)

    @property
    def mean_logprob(self) -> float | None:
        """None rather than 0.0 when no token carried a logprob.

        The difference matters downstream: Sprint 3 selects traces *below* a
        mean-logprob threshold, and a 0.0 would read as maximum confidence and
        silently exclude every untraced response from curation.
        """
        if self.n_tokens == 0:
            return None
        return self._logprob_sum / self.n_tokens

    def build(self, request_id: str, model: str, messages: list[dict[str, str]]) -> Trace:
        return Trace(
            request_id=request_id,
            model=model,
            messages=messages,
            response=self.response,
            mean_logprob=self.mean_logprob,
            min_logprob=self._min_logprob,
            n_tokens=self.n_tokens,
            finish_reason=self.finish_reason,
        )
