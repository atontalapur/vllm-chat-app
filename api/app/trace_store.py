"""Writing traces to Postgres, off the request path.

The rule this module exists to enforce: **a chat request never waits on the
trace store, and never fails because of it.** Tracing is bookkeeping for a
training pipeline; the user is having a conversation. If the store is down,
slow, or missing entirely, the conversation is unaffected and the trace is lost
on purpose, with a counter incremented so the loss is visible on the dashboard
rather than silent.

That gives the shape:

    request path            queue (bounded)         background worker
    ------------            ---------------         -----------------
    submit(trace)  ---->    put_nowait, or          get() -> INSERT
    returns instantly       drop + count            timeout -> count

`submit` is deliberately not a coroutine. A caller cannot accidentally await
it, and no amount of database latency can reach the request path through it.
Everything slow happens in `_worker`, which nothing awaits.

**Why a bounded queue.** An unbounded one turns a slow store into unbounded
memory growth on the api container, which is a worse failure than dropping
traces: the api is in the serving path and the trace store is not. When the
queue is full the *newest* trace is dropped rather than evicting the oldest,
because a half-written queue of stale traces is no more useful for curation
than a half-written queue of fresh ones, and dropping the newest keeps the
code simple enough to be obviously correct.

**Why INSERT-only credentials.** The api holds a role that can add rows and
nothing else (migration 0002). A bug here can cost new traces; it cannot
rewrite judge scores, change curation status, or drop the table that the
training set is built from.
"""

from __future__ import annotations

import asyncio
from types import TracebackType

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.logging import logger
from app.metrics import TRACE_DROPS, TRACE_FAILURES, TRACE_QUEUE_DEPTH, TRACES_WRITTEN
from app.tracing import Trace

# ON CONFLICT with no target, deliberately. Naming one — `ON CONFLICT
# (request_id)` — requires SELECT privilege on that column, and Postgres
# rejects the statement outright for a role without it:
#
#     ERROR:  permission denied for table traces
#
# The untargeted form needs only INSERT, so the writer stays read-less. The two
# are equivalent here because the primary key is the table's only unique
# constraint; if a second one is ever added, this would start swallowing those
# conflicts too, and the choice has to be revisited.
#
# A conflict is close to unreachable anyway: request_id is a fresh uuid4 per
# request. This exists so a retry cannot kill the writer with a duplicate-key
# error, not because collisions are expected.
_INSERT = """
INSERT INTO traces (
    request_id, model, messages, response,
    mean_logprob, min_logprob, n_tokens, finish_reason
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""


class TraceStore:
    """Queue in front of Postgres, drained by one background task.

    Constructed unconditionally; `start()` is what decides whether anything is
    written. With no DSN configured the store stays disabled and `submit` is a
    no-op that counts drops, so the api runs unchanged against a stack with no
    trace store (the local overlay, CI, a GPU box mid-migration).
    """

    def __init__(self, dsn: str, queue_size: int, write_timeout_s: float) -> None:
        self._dsn = dsn
        self._queue: asyncio.Queue[Trace] = asyncio.Queue(maxsize=queue_size)
        self._write_timeout_s = write_timeout_s
        self._pool: AsyncConnectionPool | None = None
        self._worker: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._worker is not None

    async def start(self) -> None:
        """Open the pool and start draining. Never raises.

        A trace store that will not open must not stop the api from serving
        chat, so a failure here logs and leaves the store disabled. The pool is
        opened with `open=False` plus an explicit `open()`: opening in the
        constructor is deprecated in psycopg 3, and doing it explicitly is also
        what lets a connection failure be caught here rather than at import.

        This waits for the first connection rather than connecting lazily, so
        the boot log says plainly whether tracing is on. The cost is that a
        store which only becomes reachable *after* the api has started stays
        disabled until the api is restarted. That is the right trade here
        because Compose already orders `postgres` and `db-migrate` ahead of
        `api`; an outage *after* startup is handled by the pool reconnecting on
        its own, which is the case that actually happens in practice.
        """
        if not self._dsn:
            logger.info("trace store disabled: no DSN configured")
            return

        try:
            pool = AsyncConnectionPool(
                self._dsn,
                min_size=1,
                # Small on purpose: one worker drains the queue, so a second
                # connection only exists to survive one going stale.
                max_size=2,
                open=False,
                # Bounds how long a caller waits for a free connection.
                timeout=self._write_timeout_s,
                # Bounds the TCP connect itself, which the pool's own timeout
                # does not: open() honours its timeout, then blocks trying to
                # stop a worker stuck inside a libpq connect that cannot be
                # cancelled. Measured at 5s against an unreachable host without
                # this. libpq treats anything under 2 as 2, so that is the
                # floor on how fast startup can give up.
                kwargs={"connect_timeout": max(2, round(self._write_timeout_s))},
            )
            await pool.open(wait=True, timeout=self._write_timeout_s)
        except Exception as exc:  # noqa: BLE001 - see docstring: never fatal
            logger.error(
                "trace store unavailable at startup, tracing disabled",
                extra={"context": {"error": str(exc)}},
            )
            return

        self._pool = pool
        self._worker = asyncio.create_task(self._drain())
        logger.info("trace store ready")

    async def aclose(self) -> None:
        """Stop the worker and close the pool.

        Gives the worker a bounded moment to finish what it is holding, then
        cancels it. Shutdown cannot be allowed to hang on a store that is not
        answering; queued traces are dropped, which is the same trade this
        module makes everywhere else.
        """
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def submit(self, trace: Trace) -> None:
        """Hand a finished trace to the writer. Returns immediately, never raises.

        This is the only method the request path calls.
        """
        if self._worker is None:
            TRACE_DROPS.labels(reason="disabled").inc()
            return
        try:
            self._queue.put_nowait(trace)
        except asyncio.QueueFull:
            # The store is slower than the traffic. Losing the trace is the
            # designed outcome; growing memory in the serving container is not.
            TRACE_DROPS.labels(reason="queue_full").inc()
            logger.warning(
                "trace dropped: writer queue full",
                extra={"context": {"request_id": trace.request_id}},
            )
        TRACE_QUEUE_DEPTH.set(self._queue.qsize())

    async def _drain(self) -> None:
        while True:
            trace = await self._queue.get()
            try:
                await self._write(trace)
            finally:
                self._queue.task_done()
                TRACE_QUEUE_DEPTH.set(self._queue.qsize())

    async def _write(self, trace: Trace) -> None:
        """Insert one trace. Swallows every failure by design.

        An exception escaping here would kill the worker task and silently end
        all tracing for the life of the process — the failure mode this whole
        module is built to avoid — so everything is caught and counted.
        """
        if self._pool is None:
            TRACE_DROPS.labels(reason="disabled").inc()
            return
        try:
            async with asyncio.timeout(self._write_timeout_s):
                async with self._pool.connection() as conn:
                    await conn.execute(
                        _INSERT,
                        (
                            trace.request_id,
                            trace.model,
                            Jsonb(trace.messages),
                            trace.response,
                            trace.mean_logprob,
                            trace.min_logprob,
                            trace.n_tokens,
                            trace.finish_reason,
                        ),
                    )
        except asyncio.CancelledError:
            # Shutdown, not a write failure. Let it propagate or aclose() hangs.
            raise
        except TimeoutError:
            TRACE_FAILURES.labels(reason="timeout").inc()
            logger.warning(
                "trace write timed out",
                extra={
                    "context": {
                        "request_id": trace.request_id,
                        "timeout_s": self._write_timeout_s,
                    }
                },
            )
        except psycopg.Error as exc:
            TRACE_FAILURES.labels(reason="database").inc()
            logger.warning(
                "trace write failed",
                extra={"context": {"request_id": trace.request_id, "error": str(exc)}},
            )
        except Exception as exc:  # noqa: BLE001 - see docstring: worker must survive
            TRACE_FAILURES.labels(reason="unexpected").inc()
            logger.error(
                "trace write failed unexpectedly",
                extra={"context": {"request_id": trace.request_id, "error": str(exc)}},
            )
        else:
            TRACES_WRITTEN.inc()

    async def __aenter__(self) -> TraceStore:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
