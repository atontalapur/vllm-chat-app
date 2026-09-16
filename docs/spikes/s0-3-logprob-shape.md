# S0-3: streamed logprob delta shape

**Jira:** CFT-10. **Status:** captured against vLLM 0.28.0 on an RTX 3090, 2026-09-16. Two
rows in the table below changed from the earlier Ollama capture.

## Question

When `stream_chat` (`api/app/vllm_client.py`) asks for logprobs, what exactly arrives in
each SSE chunk, and what single number do we store per trace as the confidence proxy?
Sprint 1 story S1-2 accumulates this; S1-1's schema has a `mean_logprob` column.

## Request shape

```json
{"model": "...", "messages": [...], "stream": true, "logprobs": true, "top_logprobs": 0}
```

**Send `top_logprobs` explicitly.** vLLM's chat serving path only builds logprobs when
`request.logprobs and request.top_logprobs is not None`
(`vllm/entrypoints/openai/chat_completion/serving.py`). `logprobs: true` alone can yield
chunks with `"logprobs": null`. `0` means "the chosen token only", which is all we need
and keeps the chunks small. `prompt_logprobs` is rejected when `stream=true`; we do not
want it anyway.

## Chunk shape

Sample chunks are in `s0-3-sample-stream.json`. Per streamed chunk:

```json
{
  "id": "chatcmpl-808", "object": "chat.completion.chunk", "model": "...",
  "choices": [{
    "index": 0,
    "delta": {"content": "India"},
    "finish_reason": null,
    "logprobs": {
      "content": [
        {"token": "India", "logprob": -0.170, "bytes": [73,110,100,105,97], "top_logprobs": [...]}
      ]
    }
  }]
}
```

Things S1-2's accumulator must handle:

| Observation | Consequence |
|---|---|
| `choices[0].logprobs.content` is a **list** | Iterate it. Observed on the box: one chunk carried 11 tokens (` 2011 Cricket World Cup final was won`) under a single `delta.content`. Never index `[0]`. |
| The final chunk carries the last token, its logprobs, **and** `finish_reason` together | No empty tail chunk. Read `finish_reason` on the same chunk you accumulate. `length` means the response hit `max_tokens` (S3-4 drops those). Ollama sent a separate empty chunk; vLLM does not. |
| vLLM clamps `logprob` to `>= -9999.0` | A `-inf` never appears, so plain arithmetic is safe. |
| The first chunk is role-only: `{"role": "assistant", "content": ""}` with `"logprobs": null` | Skip chunks whose `logprobs` is null. Ollama merged the role into the first token chunk; vLLM does not. |
| Reasoning models put tokens in `delta.reasoning`, still with logprobs | Qwen2.5-7B-Instruct has no reasoning channel, so moot for us. If the base model changes, decide whether reasoning tokens count. |
| `bytes` is the UTF-8 of the token, `null` when undecodable | Ignore for the proxy; `token` and `logprob` are enough. |

## Confidence proxy

Store per trace:

- `mean_logprob = sum(logprob) / n_tokens` over every token with a logprob. This is
  `log` of the geometric-mean token probability, so `exp(mean_logprob)` reads as an
  average per-token confidence in [0, 1]. Required by the S1-1 schema.
- `min_logprob` and `n_tokens` alongside. The mean hides one badly-chosen token in a long
  fluent answer; the minimum does not. Cheap to keep and gives Sprint 6 a second knob.

Worked example from the vLLM capture (`s0-3-sample-stream.json`). Prompt: *"Who won the
2011 Cricket World Cup final and by how many runs?"* (a trap: India won by wickets, not runs.)

```
response      The 2011 Cricket World Cup final was won by India, defeating Sri Lanka by
              5 wickets. This was India's second World   [cut by max_tokens=30]
n_tokens      30
mean_logprob  -0.221   exp -> 0.801
min_logprob   -1.692   on " India"
```

Interpretation: the model handled the false premise (answered in wickets) but got the
number wrong (it was 6). Mean confidence 0.80 did not flag it. This is the case the
judge exists for: logprobs catch hesitation, not confident errors. Keep both signals.

## Thresholds

Not set here. S3-1 makes both thresholds config; S6-4 sets them from observed
distributions. Starting guess for S3-1 defaults, to be replaced: select traces with
`mean_logprob < -0.5` (geometric-mean confidence below ~0.6) or `judge_score` below the
rubric threshold.

## Cost

Every chunk grows by roughly one small object per token (~120 bytes with
`top_logprobs: 0`). This travels vllm -> api only; the api yields the line unchanged to
the client, so the UI sees the same payload it does today plus the logprobs field.
S1-2 puts the request flag behind config so the cost can be switched off.

## GPU-box confirmation (as run)

```bash
docker compose exec api python3 - <<'PY'
import json, urllib.request
body = {"model": "Qwen/Qwen2.5-7B-Instruct", "stream": True, "logprobs": True,
        "top_logprobs": 0, "max_tokens": 30,
        "messages": [{"role": "user", "content": "Who won the 2011 Cricket World Cup final and by how many runs?"}]}
req = urllib.request.Request("http://vllm:8000/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    for raw in r:
        line = raw.decode().rstrip()
        if line: print(line)
PY
```

Check: `logprobs` non-null on token chunks with `top_logprobs: 0`; whether any chunk has
more than one entry in `logprobs.content`; the final chunk's shape. Paste one token chunk
and the final chunk over the Ollama ones in `s0-3-sample-stream.json`.
