"""Concurrent load generator.

Exists to make the dashboard show something. A single chat session never
queues: one request occupies one slot and the queue-depth panel stays flat at
zero, which is exactly the wrong thing to have on screen while explaining
continuous batching.

Sending more concurrent requests than vLLM's --max-num-seqs forces a queue to
form. Waiting climbs, running pins at the cap, and throughput holds roughly
flat instead of collapsing — that contrast is the whole demonstration.

Usage:
    python scripts/loadtest.py --concurrency 12 --requests 60

Run it against the api (not vLLM directly) so the application layer's own
metrics move too. Standard library only: no install step on the GPU box.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

PROMPT = (
    "Explain in a short paragraph why batching multiple requests together "
    "improves GPU utilisation for language model inference."
)


@dataclass
class Results:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ttft: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    ok: int = 0
    failed: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def record_ok(self, ttft: float, total: float) -> None:
        with self.lock:
            self.ttft.append(ttft)
            self.total.append(total)
            self.ok += 1

    def record_failure(self, reason: str) -> None:
        with self.lock:
            self.failed += 1
            self.errors[reason] = self.errors.get(reason, 0) + 1


def one_request(url: str, api_key: str, max_tokens: int, results: Results) -> None:
    body = json.dumps(
        {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens}
    ).encode()
    # noqa S310: the URL is operator-supplied (--url), pointing at this
    # project's own api service; there is no untrusted input here.
    req = urllib.request.Request(  # noqa: S310
        f"{url}/chat/stream",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )

    started = time.perf_counter()
    first: float | None = None
    try:
        with urllib.request.urlopen(req, timeout=300) as response:  # noqa: S310
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    if first is None:
                        first = time.perf_counter() - started
        results.record_ok(first or 0.0, time.perf_counter() - started)
    except urllib.error.HTTPError as exc:
        results.record_failure(f"HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001 - any failure is a failed request
        results.record_failure(type(exc).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="in-flight requests; set above vLLM's --max-num-seqs to force a queue",
    )
    parser.add_argument("--requests", type=int, default=60, help="total requests to send")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    if not args.api_key:
        print("error: no API key. Pass --api-key or set API_KEY.", file=sys.stderr)
        return 2

    print(
        f"Sending {args.requests} requests at concurrency {args.concurrency} "
        f"to {args.url}\nWatch the queue-depth panel: waiting should climb above "
        f"zero while running pins at --max-num-seqs.\n"
    )

    results = Results()
    remaining = threading.Semaphore(args.concurrency)
    threads: list[threading.Thread] = []
    wall_start = time.perf_counter()

    def worker() -> None:
        try:
            one_request(args.url, args.api_key, args.max_tokens, results)
        finally:
            remaining.release()

    for _ in range(args.requests):
        remaining.acquire()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    wall = time.perf_counter() - wall_start
    print(f"completed in {wall:.1f}s — {results.ok} ok, {results.failed} failed")
    if results.errors:
        for reason, count in sorted(results.errors.items()):
            print(f"  {reason}: {count}")
    if results.ttft:
        print(f"time to first token  p50 {statistics.median(results.ttft):.2f}s")
        if len(results.ttft) >= 20:
            p95 = statistics.quantiles(results.ttft, n=20)[18]
            print(f"                     p95 {p95:.2f}s")
        print(f"end to end           p50 {statistics.median(results.total):.2f}s")
        print(f"throughput           {results.ok / wall:.2f} req/s")
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
