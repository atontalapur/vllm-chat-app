"""Concurrent streaming benchmark against the gateway or vLLM directly.

Produces measurement 1b (the full stack) and both halves of measurement 2
(gateway overhead), depending on --mode.

Standard library only: no install step on the GPU box.

    # through the FastAPI gateway, from inside the api container
    python3 /app/bench/bench_stream.py --mode gateway --url http://localhost:8080

    # straight at vLLM, bypassing the application layer
    python3 /app/bench/bench_stream.py --mode vllm --url http://vllm:8000

## Why this exists rather than scripts/loadtest.py

`scripts/loadtest.py` is a dashboard driver. It records time-to-first-token and
wall-clock per request, and it never counts tokens (`Results`,
scripts/loadtest.py:39-45). That is precisely why the "~200 tok/s" figure in
README.md:50 cannot be traced to anything: no tool in this repo has ever
counted a token. This one does, and writes every raw sample to disk so the
summary can be rechecked later.

## How tokens are counted

By counting entries in `choices[0].logprobs.content` on each streamed chunk.
That is an exact token count, and it is the only accurate way to do it here:
`docs/spikes/s0-3-logprob-shape.md` records a chunk that carried 11 tokens
under a single `delta.content`, so counting chunks — the obvious approach —
would undercount every multi-token chunk and silently inflate nothing while
deflating throughput.

Where the server also sends a usage block (`stream_options.include_usage`,
available on the direct path), its `completion_tokens` is recorded alongside as
a cross-check. If the two disagree the run is still valid; the disagreement is
in the result file for whoever reads it.

**Gateway mode depends on CAPTURE_LOGPROBS.** The gateway builds its own
upstream payload (`api/app/vllm_client.py:44`) and does not forward a client's
logprobs request, so tokens are only countable through it when the service runs
with CAPTURE_LOGPROBS=true. With capture off this script still measures latency
and TTFT honestly, and records token counts as null rather than guessing — see
--allow-untokenised.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_prompts,
    provenance,
    refuse,
    summarize,
    write_results,
)

DATA_PREFIX = "data: "
DONE = "[DONE]"


@dataclass
class Sample:
    """One request. Written to disk verbatim, not just folded into a summary."""

    index: int
    repeat: int
    prompt_index: int
    status: str
    ttft_s: float | None = None
    total_s: float | None = None
    tokens_logprobs: int | None = None
    tokens_usage: int | None = None
    chunks: int = 0
    error: str | None = None


@dataclass
class Collector:
    lock: threading.Lock = field(default_factory=threading.Lock)
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        with self.lock:
            self.samples.append(sample)


def build_request(
    mode: str, url: str, api_key: str, model: str, prompt: str, max_tokens: int, temperature: float
) -> urllib.request.Request:
    """Endpoint, auth and payload differ between the two targets; the workload does not."""
    if mode == "gateway":
        endpoint = f"{url.rstrip('/')}/chat/stream"
        # The gateway's schema accepts only these three fields
        # (api/app/schemas.py:17). It adds model, stream and logprobs itself.
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    else:
        endpoint = f"{url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # Same request the gateway makes upstream, so the two modes differ
            # by the application layer and nothing else.
            "logprobs": True,
            "top_logprobs": 0,
            # Direct-only: the gateway does not forward it. Gives an
            # independent token count to check the logprob count against.
            "stream_options": {"include_usage": True},
        }
        headers = {"Content-Type": "application/json"}
    return urllib.request.Request(  # noqa: S310 - operator-supplied --url, this project's own services
        endpoint, data=json.dumps(body).encode(), headers=headers
    )


def count_tokens_in_chunk(chunk: dict[str, Any]) -> int:
    """Entries in choices[0].logprobs.content. See the module docstring."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    choice = choices[0]
    if not isinstance(choice, dict):
        return 0
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return 0
    content = logprobs.get("content")
    if not isinstance(content, list):
        return 0
    return len(content)


def one_request(
    args: argparse.Namespace, prompt: str, index: int, repeat: int, prompt_index: int
) -> Sample:
    sample = Sample(index=index, repeat=repeat, prompt_index=prompt_index, status="ok")
    request = build_request(
        args.mode, args.url, args.api_key, args.model, prompt, args.max_tokens, args.temperature
    )
    started = time.perf_counter()
    first: float | None = None
    tokens = 0
    usage_tokens: int | None = None
    chunks = 0

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:  # noqa: S310
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith(DATA_PREFIX):
                    continue
                payload = line[len(DATA_PREFIX) :].strip()
                if payload == DONE:
                    continue
                # TTFT is the first *data* chunk, matching how
                # scripts/loadtest.py measures it, so the two agree.
                if first is None:
                    first = time.perf_counter() - started
                chunks += 1
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                tokens += count_tokens_in_chunk(chunk)
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    completion = usage.get("completion_tokens")
                    if isinstance(completion, int):
                        usage_tokens = completion
    except urllib.error.HTTPError as exc:
        sample.status = "failed"
        sample.error = f"HTTP {exc.code}"
        return sample
    except Exception as exc:  # noqa: BLE001 - any failure is a failed request
        sample.status = "failed"
        sample.error = type(exc).__name__
        return sample

    sample.ttft_s = first
    sample.total_s = time.perf_counter() - started
    sample.tokens_logprobs = tokens if tokens else None
    sample.tokens_usage = usage_tokens
    sample.chunks = chunks
    return sample


def run_wave(args: argparse.Namespace, prompts: list[str], repeat: int) -> list[Sample]:
    """One wave: every prompt dispatched concurrently, then joined.

    All requests are started before any is awaited, which is what makes this a
    concurrency test rather than a throughput-of-one-stream test. With 12
    prompts against --max-num-seqs=4 (docker-compose.yml:29), eight of them are
    queued by design; that queue is the thing being measured.
    """
    collector = Collector()
    threads: list[threading.Thread] = []
    for i, prompt in enumerate(prompts):

        def worker(p: str = prompt, idx: int = i) -> None:
            collector.add(one_request(args, p, index=idx, repeat=repeat, prompt_index=idx))

        thread = threading.Thread(target=worker, daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(collector.samples, key=lambda s: s.index)


def preflight(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    """Refuse rather than measure the wrong thing.

    Each check here corresponds to a way this benchmark could produce a
    plausible-looking number for a configuration nobody intended.
    """
    info: dict[str, Any] = {}

    if args.mode == "gateway" and not args.api_key:
        refuse(
            "gateway mode needs an API key. Run inside the api container, where "
            "API_KEY is already in the environment, or pass --api-key. "
            "Note the host shell does not read .env (docs/commands.md:131)."
        )

    if args.mode == "vllm":
        # A model mismatch between the two targets would make every comparison
        # meaningless, so confirm what is actually served before measuring it.
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{args.url.rstrip('/')}/v1/models", timeout=10
            ) as response:
                served = json.loads(response.read().decode())
        except Exception as exc:  # noqa: BLE001
            refuse(f"cannot reach vLLM at {args.url}/v1/models: {type(exc).__name__}: {exc}")
        ids = [m.get("id") for m in served.get("data", []) if isinstance(m, dict)]
        info["served_models"] = ids
        if args.model not in ids:
            refuse(
                f"--model {args.model!r} is not served here. This endpoint serves {ids}. "
                "Benchmarking a different model than the baseline makes the comparison void."
            )
        version = None
        try:
            with urllib.request.urlopen(f"{args.url.rstrip('/')}/version", timeout=10) as response:  # noqa: S310
                version = json.loads(response.read().decode()).get("version")
        except Exception:  # noqa: BLE001 - version endpoint is best effort
            version = None
        info["vllm_version"] = version
    else:
        # The gateway does not proxy /v1/models or /version, so the upstream
        # version is not observable from here. Recorded as null rather than
        # copied from a doc: an unverified value in a provenance block is worse
        # than an absent one.
        info["served_models"] = None
        info["vllm_version"] = None

    # Token counting is the whole point; prove it works before the real run.
    probe = one_request(args, prompts[0], index=-1, repeat=-1, prompt_index=0)
    if probe.status != "ok":
        refuse(f"probe request failed ({probe.error}). Fix the target before benchmarking.")
    if not probe.tokens_logprobs:
        if not args.allow_untokenised:
            refuse(
                "the probe request produced zero countable tokens, so throughput cannot "
                "be measured.\n"
                "  Cause: logprobs are absent from the stream. In gateway mode that means "
                "the api is running with CAPTURE_LOGPROBS=false.\n"
                "  Fix without editing any config: restart the api with an env override, "
                "e.g. `docker compose run --rm -e CAPTURE_LOGPROBS=true api`, or set it for "
                "the run.\n"
                "  Or pass --allow-untokenised to measure latency and TTFT only; token "
                "counts will be recorded as null and throughput will not be reported."
            )
        print(
            "WARNING: no countable tokens; latency and TTFT only, throughput will be null.",
            file=sys.stderr,
        )
    info["probe"] = {
        "tokens_logprobs": probe.tokens_logprobs,
        "tokens_usage": probe.tokens_usage,
        "chunks": probe.chunks,
    }
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gateway", "vllm"], required=True)
    parser.add_argument("--url", required=True, help="base URL, no path")
    parser.add_argument("--api-key", default="", help="gateway mode only; defaults to $API_KEY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=3, help="waves of the full prompt set")
    parser.add_argument("--warmup", type=int, default=1, help="waves discarded before measuring")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", default="", help="free-text tag recorded in the result file")
    parser.add_argument(
        "--allow-untokenised",
        action="store_true",
        help="measure latency/TTFT even when tokens cannot be counted",
    )
    parser.add_argument("--prompts", type=Path, default=None)
    args = parser.parse_args()

    if not args.api_key:
        import os

        args.api_key = os.environ.get("API_KEY", "")

    prompts = load_prompts(args.prompts)
    info = preflight(args, prompts)

    # Warmup is discarded, not averaged in. The first wave after a cold start
    # pays for CUDA graph capture and an empty KV cache, which is a real cost
    # but not the steady-state number this is trying to measure.
    for w in range(args.warmup):
        print(f"warmup wave {w + 1}/{args.warmup} (discarded)", file=sys.stderr)
        run_wave(args, prompts, repeat=-1)

    all_samples: list[Sample] = []
    wave_summaries: list[dict[str, Any]] = []
    for r in range(args.repeats):
        print(f"measured wave {r + 1}/{args.repeats}", file=sys.stderr)
        wave_started = time.perf_counter()
        samples = run_wave(args, prompts, repeat=r)
        wave_wall = time.perf_counter() - wave_started
        ok = [s for s in samples if s.status == "ok"]
        wave_tokens = sum(s.tokens_logprobs or 0 for s in ok)
        wave_summaries.append(
            {
                "repeat": r,
                "wall_s": wave_wall,
                "requests": len(samples),
                "ok": len(ok),
                "failed": len(samples) - len(ok),
                "tokens_total": wave_tokens or None,
                # Wall-clock throughput for the whole concurrent wave. This is
                # the number that answers "how long did 12 requests take",
                # which is what a naive sequential baseline is compared on.
                "tokens_per_second": (wave_tokens / wave_wall) if wave_tokens else None,
            }
        )
        all_samples.extend(samples)

    ok_samples = [s for s in all_samples if s.status == "ok"]
    ttfts = [s.ttft_s for s in ok_samples if s.ttft_s is not None]
    totals = [s.total_s for s in ok_samples if s.total_s is not None]
    wave_rates = [w["tokens_per_second"] for w in wave_summaries if w["tokens_per_second"]]

    payload = {
        "measurement": f"stream-{args.mode}",
        "label": args.label,
        "provenance": provenance(
            model=args.model,
            mode=args.mode,
            url=args.url,
            **info,
        ),
        "workload": {
            "prompts_file_version": json.loads(
                (args.prompts or (Path(__file__).resolve().parent / "prompts.json")).read_text()
            )["version"],
            "prompt_count": len(prompts),
            "concurrency": len(prompts),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repeats": args.repeats,
            "warmup_waves_discarded": args.warmup,
            "timeout_s": args.timeout,
        },
        "summary": {
            "requests_total": len(all_samples),
            "requests_ok": len(ok_samples),
            "requests_failed": len(all_samples) - len(ok_samples),
            "ttft_s": summarize(ttfts),
            "total_latency_s": summarize(totals),
            "tokens_per_second_per_wave": summarize(wave_rates) if wave_rates else None,
            "tokens_counted_total": sum(s.tokens_logprobs or 0 for s in ok_samples) or None,
        },
        "waves": wave_summaries,
        # Raw samples, always. A summary nobody can recheck is the failure mode
        # this directory exists to avoid.
        "samples": [vars(s) for s in all_samples],
    }

    path = write_results(f"stream-{args.mode}{('-' + args.label) if args.label else ''}", payload)
    print(f"\nwrote {path}")
    s = payload["summary"]
    print(f"  ok {s['requests_ok']}/{s['requests_total']}")
    print(f"  ttft   p50 {s['ttft_s']['p50']}  p95 {s['ttft_s']['p95']}")
    print(f"  total  p50 {s['total_latency_s']['p50']}  p95 {s['total_latency_s']['p95']}")
    if s["tokens_per_second_per_wave"]:
        print(f"  tok/s  p50 {s['tokens_per_second_per_wave']['p50']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
