"""Naive baseline: Hugging Face transformers, one request at a time.

This is the "obvious alternative" half of measurement 1 — what you get by
loading the same model with the most direct code that works, and serving
requests as they arrive with no batching, no paged KV cache, and no queue. The
full stack is measured against it by `bench_stream.py --mode gateway`.

    python3 bench/bench_hf_baseline.py --max-tokens 200 --repeats 3

## What this is, and what it is not

It is a fair floor: same model, same prompts, same max_tokens, same sampling,
same GPU, bf16. It is the shape of code someone writes before they reach for an
inference server, and it is genuinely what the alternative costs.

It is **not** a claim that vLLM is 'beaten' by anything written here, and the
resulting number must not be framed that way. What the comparison credits is an
architecture decision — choosing a paged-attention server with continuous
batching, and configuring it — not code written inside vLLM. bench/README.md
states that framing explicitly and any resume bullet built on this must keep it.

## Why this file is isolated

Everything else in bench/ is standard library, so it runs on a fresh GPU box
with no install step. This one cannot be: it needs torch and transformers. It
is therefore a separate script that installs nothing itself. The install is a
command the operator runs deliberately, documented in bench/README.md, because
a benchmark that silently pip-installs into whatever environment it finds is
how a GPU box ends up with a torch build that does not match its driver.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_prompts, provenance, refuse, summarize, write_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="requests discarded before measuring")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    except ImportError as exc:
        refuse(
            f"missing dependency: {exc.name}. This script installs nothing itself.\n"
            "  Install it deliberately, in an isolated environment, for example:\n"
            "    python3 -m venv bench/.venv-hf && bench/.venv-hf/bin/pip install "
            "torch transformers accelerate\n"
            "    bench/.venv-hf/bin/python bench/bench_hf_baseline.py\n"
            "  See bench/README.md for the version notes."
        )

    if not torch.cuda.is_available():
        refuse(
            "no CUDA device. This baseline must run on the same GPU as the stack it is "
            "compared against, or the comparison measures hardware, not architecture."
        )

    # The stack holds ~21GB of the card by default (docs/spikes/
    # s0-2-gpu-coresidency.md:113), and this needs ~15GB for bf16 weights. They
    # do not co-reside on a 24GB card — that is the finding of spike S0-2. Stop
    # the stack first; bench/README.md gives the order.
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024**3)
    if free_gib < 16.0:
        refuse(
            f"only {free_gib:.1f} GiB of VRAM free; this needs roughly 15 GiB for bf16 "
            "weights plus activations.\n"
            "  The serving stack is probably still up. Stop it first:\n"
            "    docker compose stop vllm api ui\n"
            "  (Spike S0-2 established the two cannot co-reside on a 24GB card.)"
        )

    import threading

    print(f"loading {args.model} in bf16 ...", file=sys.stderr)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    load_s = time.perf_counter() - load_started
    print(f"loaded in {load_s:.1f}s", file=sys.stderr)

    prompts = load_prompts(args.prompts)

    def one(prompt: str) -> dict[str, Any]:
        """One request, start to finish, with nothing else running."""
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": args.max_tokens,
            "streamer": streamer,
            # temperature 0 means greedy; passing do_sample=True with
            # temperature 0 is undefined, so branch explicitly rather than
            # leave the two targets sampling differently.
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature

        started = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []

        thread = threading.Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        for piece in streamer:
            if piece and first is None:
                first = time.perf_counter() - started
            pieces.append(piece)
        thread.join()
        total = time.perf_counter() - started

        # Exact count: re-tokenise the generated text. Not a chunk count —
        # a streamer piece is not a token.
        generated = "".join(pieces)
        n_tokens = len(tokenizer(generated, add_special_tokens=False)["input_ids"])
        return {
            "ttft_s": first,
            "total_s": total,
            "tokens": n_tokens,
            "status": "ok",
        }

    for w in range(args.warmup):
        print(f"warmup {w + 1}/{args.warmup} (discarded)", file=sys.stderr)
        one(prompts[0])

    samples: list[dict[str, Any]] = []
    wave_summaries: list[dict[str, Any]] = []
    for r in range(args.repeats):
        print(f"wave {r + 1}/{args.repeats} — {len(prompts)} requests, sequential", file=sys.stderr)
        wave_started = time.perf_counter()
        for i, prompt in enumerate(prompts):
            sample = one(prompt)
            sample.update({"index": i, "repeat": r, "prompt_index": i})
            samples.append(sample)
            print(
                f"  {i + 1}/{len(prompts)}  {sample['tokens']} tok in {sample['total_s']:.2f}s",
                file=sys.stderr,
            )
        wave_wall = time.perf_counter() - wave_started
        wave_tokens = sum(s["tokens"] for s in samples if s["repeat"] == r)
        wave_summaries.append(
            {
                "repeat": r,
                "wall_s": wave_wall,
                "requests": len(prompts),
                "ok": len(prompts),
                "failed": 0,
                "tokens_total": wave_tokens,
                # The comparable figure: total tokens divided by the wall-clock
                # time to serve the whole set of 12, exactly as the concurrent
                # harness computes it.
                "tokens_per_second": wave_tokens / wave_wall,
            }
        )

    ttfts = [s["ttft_s"] for s in samples if s["ttft_s"] is not None]
    totals = [s["total_s"] for s in samples]
    wave_rates = [w["tokens_per_second"] for w in wave_summaries]

    payload = {
        "measurement": "hf-baseline-sequential",
        "label": args.label,
        "provenance": provenance(
            model=args.model,
            serving="huggingface transformers, sequential, no batching, no paged KV cache",
            torch_version=torch.__version__,
            dtype="bfloat16",
            model_load_s=load_s,
        ),
        "workload": {
            "prompts_file_version": json.loads(
                (args.prompts or (Path(__file__).resolve().parent / "prompts.json")).read_text()
            )["version"],
            "prompt_count": len(prompts),
            "concurrency": 1,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repeats": args.repeats,
            "warmup_requests_discarded": args.warmup,
        },
        "summary": {
            "requests_total": len(samples),
            "requests_ok": len(samples),
            "requests_failed": 0,
            "ttft_s": summarize(ttfts),
            "total_latency_s": summarize(totals),
            "tokens_per_second_per_wave": summarize(wave_rates),
            "tokens_counted_total": sum(s["tokens"] for s in samples),
        },
        "waves": wave_summaries,
        "samples": samples,
    }

    path = write_results(f"hf-baseline{('-' + args.label) if args.label else ''}", payload)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
