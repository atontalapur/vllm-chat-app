"""The trace writer's failure modes.

Every test here is about the same promise: the chat request is unaffected. The
happy path is one test; the rest are the ways the store can fail, because those
are the ones that decide whether tracing is safe to leave switched on in the
serving path.

No Postgres is involved. The pool is replaced by fakes that fail on demand —
a real database cannot be made to time out or vanish mid-write on command, and
CI has no room for one on the request path anyway. The SQL itself is exercised
for real by the migration job in `.github/workflows/ci.yml`.
"""

import asyncio
from types import TracebackType
from typing import Any

import httpx
import psycopg
import pytest
from app.metrics import TRACE_DROPS, TRACE_FAILURES, TRACES_WRITTEN
from app.trace_store import TraceStore
from app.tracing import Trace
from fastapi.testclient import TestClient

from tests.conftest import UPSTREAM_URL, VALID_KEY, sse

TRACE = Trace(
    request_id="req-1",
    model="test-model",
    messages=[{"role": "user", "content": "hi"}],
    response="hello",
    mean_logprob=-0.5,
    min_logprob=-1.5,
    n_tokens=3,
    finish_reason="stop",
)


class FakeConnection:
    def __init__(self, fail: BaseException | None = None, delay: float = 0.0) -> None:
        self._fail = fail
        self._delay = delay
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail is not None:
            raise self._fail
        self.executed.append((sql, params))


class FakePool:
    """Stands in for AsyncConnectionPool, minus everything but connection()."""

    def __init__(self, fail: BaseException | None = None, delay: float = 0.0) -> None:
        self.conn = FakeConnection(fail, delay)
        self.closed = False

    def connection(self) -> "FakePool":
        return self

    async def __aenter__(self) -> FakeConnection:
        return self.conn

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def counter_value(counter: Any, **labels: str) -> float:
    """Current value of a prometheus counter, or 0 if never incremented.

    Read before and after rather than asserted absolutely: the registry is
    process-global, so other tests in the same run have already moved these.
    """
    metric = counter.labels(**labels) if labels else counter
    return float(metric._value.get())


async def running_store(pool: FakePool, queue_size: int = 10) -> TraceStore:
    """A store wired to a fake pool, with its worker running.

    start() is bypassed: it would try to reach a real Postgres. Everything
    after it — the queue, the worker, the write path — is the code under test.
    """
    store = TraceStore(dsn="", queue_size=queue_size, write_timeout_s=0.05)
    store._pool = pool  # type: ignore[assignment]
    store._worker = asyncio.create_task(store._drain())
    return store


async def drain(store: TraceStore) -> None:
    """Wait for the queue to empty, with a bound so a bug fails rather than hangs."""
    await asyncio.wait_for(store._queue.join(), timeout=2.0)


async def test_writes_a_trace_with_every_column() -> None:
    pool = FakePool()
    store = await running_store(pool)
    before = counter_value(TRACES_WRITTEN)

    store.submit(TRACE)
    await drain(store)
    await store.aclose()

    assert len(pool.conn.executed) == 1
    sql, params = pool.conn.executed[0]
    assert "INSERT INTO traces" in sql
    # Order matters: a swapped pair here would be silently wrong in the data.
    assert params[0] == "req-1"
    assert params[1] == "test-model"
    assert params[3] == "hello"
    assert params[4] == -0.5
    assert params[5] == -1.5
    assert params[6] == 3
    assert params[7] == "stop"
    assert counter_value(TRACES_WRITTEN) == before + 1


async def test_messages_are_sent_as_jsonb() -> None:
    """A plain list would be adapted to a Postgres array and rejected by jsonb."""
    pool = FakePool()
    store = await running_store(pool)

    store.submit(TRACE)
    await drain(store)
    await store.aclose()

    _, params = pool.conn.executed[0]
    assert isinstance(params[2], psycopg.types.json.Jsonb)
    assert params[2].obj == [{"role": "user", "content": "hi"}]


async def test_insert_tolerates_a_duplicate_without_naming_a_target() -> None:
    """A retry must not kill the worker with a duplicate-key error.

    The conflict target is omitted on purpose: `ON CONFLICT (request_id)`
    requires SELECT privilege on that column, which the INSERT-only role does
    not have, and Postgres rejects the whole statement with "permission denied
    for table traces". Verified against a real database, not just here.
    """
    pool = FakePool()
    store = await running_store(pool)

    store.submit(TRACE)
    await drain(store)
    await store.aclose()

    sql, _ = pool.conn.executed[0]
    assert "ON CONFLICT DO NOTHING" in sql
    assert "ON CONFLICT (" not in sql, "a conflict target needs SELECT privilege"


async def test_database_error_is_counted_and_the_worker_survives() -> None:
    """The failure that would otherwise end tracing silently.

    An exception escaping the worker kills the task, and every later trace is
    dropped for the life of the process with nothing in the logs to say so.
    """
    pool = FakePool(fail=psycopg.OperationalError("server closed the connection"))
    store = await running_store(pool)
    before = counter_value(TRACE_FAILURES, reason="database")

    store.submit(TRACE)
    await drain(store)

    assert counter_value(TRACE_FAILURES, reason="database") == before + 1
    assert store.enabled, "worker must still be running after a failed write"

    # And it really can still write: swap in a healthy connection.
    pool.conn = FakeConnection()
    store.submit(TRACE)
    await drain(store)
    await store.aclose()

    assert len(pool.conn.executed) == 1


async def test_slow_store_times_out_rather_than_blocking_forever() -> None:
    pool = FakePool(delay=5.0)
    store = await running_store(pool)
    before = counter_value(TRACE_FAILURES, reason="timeout")

    store.submit(TRACE)
    await drain(store)
    await store.aclose()

    assert counter_value(TRACE_FAILURES, reason="timeout") == before + 1


async def test_unexpected_error_is_counted_and_the_worker_survives() -> None:
    """Not every driver failure is a psycopg.Error; the worker must outlive those too."""
    pool = FakePool(fail=ValueError("something nobody predicted"))
    store = await running_store(pool)
    before = counter_value(TRACE_FAILURES, reason="unexpected")

    store.submit(TRACE)
    await drain(store)
    await store.aclose()

    assert counter_value(TRACE_FAILURES, reason="unexpected") == before + 1


async def test_full_queue_drops_rather_than_growing() -> None:
    """Bounded memory in the serving container beats complete training signal."""
    store = TraceStore(dsn="", queue_size=2, write_timeout_s=0.05)
    # No worker: nothing drains, so the queue fills and stays full.
    store._worker = asyncio.create_task(asyncio.sleep(10))
    before = counter_value(TRACE_DROPS, reason="queue_full")

    for _ in range(5):
        store.submit(TRACE)

    assert store._queue.qsize() == 2
    assert counter_value(TRACE_DROPS, reason="queue_full") == before + 3
    await store.aclose()


async def test_submit_is_a_counted_no_op_when_disabled() -> None:
    """No DSN, or a store that failed to open: submit must still be safe to call."""
    store = TraceStore(dsn="", queue_size=10, write_timeout_s=0.05)
    before = counter_value(TRACE_DROPS, reason="disabled")

    store.submit(TRACE)

    assert not store.enabled
    assert counter_value(TRACE_DROPS, reason="disabled") == before + 1


async def test_start_with_no_dsn_leaves_the_store_disabled() -> None:
    """The local overlay and CI run with no trace store and must not care."""
    store = TraceStore(dsn="", queue_size=10, write_timeout_s=0.05)

    await store.start()

    assert not store.enabled
    await store.aclose()


async def test_start_never_raises_when_the_store_is_unreachable() -> None:
    """A trace store that will not open must not stop the api from serving chat."""
    store = TraceStore(
        # Reserved TEST-NET-1 address: guaranteed not to answer, and the
        # connect attempt is bounded by write_timeout_s.
        dsn="postgresql://api_writer:pw@192.0.2.1:5432/traces",
        queue_size=10,
        write_timeout_s=0.05,
    )

    await store.start()

    assert not store.enabled, "a store that cannot open must stay disabled"
    store.submit(TRACE)  # must not raise
    await store.aclose()


async def test_aclose_is_safe_on_a_store_that_never_started() -> None:
    store = TraceStore(dsn="", queue_size=10, write_timeout_s=0.05)

    await store.aclose()
    await store.aclose()


async def test_aclose_closes_the_pool() -> None:
    pool = FakePool()
    store = await running_store(pool)

    await store.aclose()

    assert pool.closed
    assert not store.enabled


async def test_submit_does_not_block_on_a_stalled_store() -> None:
    """The property the whole module exists for, measured rather than assumed."""
    store = await running_store(FakePool(delay=5.0))

    started = asyncio.get_running_loop().time()
    for _ in range(50):
        store.submit(TRACE)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.05, f"submit blocked the caller for {elapsed:.3f}s"
    await store.aclose()


def test_submit_is_not_a_coroutine() -> None:
    """Sync on purpose: an `await` here would put the store in the request path."""
    assert not asyncio.iscoroutinefunction(TraceStore.submit)


@pytest.mark.parametrize("field", ["mean_logprob", "min_logprob", "finish_reason"])
async def test_nullable_fields_are_written_as_none(field: str) -> None:
    """Capture off, or a stream that ended without one: null, not a placeholder."""
    pool = FakePool()
    store = await running_store(pool)
    trace = Trace(**{**TRACE.__dict__, field: None})

    store.submit(trace)
    await drain(store)
    await store.aclose()

    _, params = pool.conn.executed[0]
    assert None in params


# --- The request path -------------------------------------------------------
#
# Everything above tests the writer in isolation. These go through the actual
# endpoint, because the promise being made is about a chat request, not about a
# queue.


class RecordingStore:
    """A TraceStore stand-in that records instead of writing."""

    def __init__(self, explode: bool = False) -> None:
        self.submitted: list[Trace] = []
        self._explode = explode

    def submit(self, trace: Trace) -> None:
        self.submitted.append(trace)
        if self._explode:
            raise RuntimeError("a store that breaks its own contract")


def test_trace_is_submitted_after_a_completed_stream(
    client: Any, respx_mock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_tracing import FINAL_CHUNK, TWO_TOKEN_CHUNK

    store = RecordingStore()
    monkeypatch.setattr("app.main.trace_store", store)
    respx_mock.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, text=sse(TWO_TOKEN_CHUNK, FINAL_CHUNK))
    )

    r = client.post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "who won?"}]},
        headers={"X-API-Key": VALID_KEY},
    )

    assert r.status_code == 200
    assert len(store.submitted) == 1
    trace = store.submitted[0]
    # The row must be joinable to the log line for the same request.
    assert trace.request_id == r.headers["X-Request-ID"]
    assert trace.response == "Hello there!"
    assert trace.model == "test-model"
    assert trace.messages == [{"role": "user", "content": "who won?"}]
    assert trace.n_tokens == 3
    assert trace.finish_reason == "stop"


def test_response_is_unaffected_when_the_store_is_broken(
    client: Any, respx_mock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point, at the level the user experiences it.

    A store that raises from submit() is a contract violation this code does
    not expect. It still must not reach the user, whose response has already
    been delivered in full by the time the trace is handed over.
    """
    from tests.test_tracing import FINAL_CHUNK, TWO_TOKEN_CHUNK

    monkeypatch.setattr("app.main.trace_store", RecordingStore(explode=True))
    respx_mock.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, text=sse(TWO_TOKEN_CHUNK, FINAL_CHUNK))
    )

    with client.stream(
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-API-Key": VALID_KEY},
    ) as r:
        body = "".join(r.iter_text())
        assert r.status_code == 200

    assert "Hello there" in body
    assert "[DONE]" in body


def test_no_trace_is_submitted_when_the_upstream_never_answers(
    client: Any, respx_mock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 502 has no response to trace; writing an empty row would be noise."""
    store = RecordingStore()
    monkeypatch.setattr("app.main.trace_store", store)
    respx_mock.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectError("refused"))

    r = client.post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-API-Key": VALID_KEY},
    )

    assert r.status_code == 502
    assert store.submitted == []


def test_lifespan_starts_and_stops_the_store_without_a_database() -> None:
    """Boot must not depend on the trace store being there."""
    from app.main import app, trace_store

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert not trace_store.enabled, "no DSN in tests, so it must stay disabled"
