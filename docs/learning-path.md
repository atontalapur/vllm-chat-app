# Learning path

Videos to watch before each sprint of the fine-tuning pipeline (`pipeline-plan.md`), ordered
so each one sets up the next. About eight hours total. Each entry names the part of the
project it explains, so skip what you already know.

## Foundations: what the model is

1. **3Blue1Brown, Transformers, the tech behind LLMs** (27 min)
   https://www.youtube.com/watch?v=wjZofJX0v4M
   Watch for the final softmax that turns the model's output into a probability over
   tokens. A logprob is the log of that number. Makes `spikes/s0-3-logprob-shape.md` obvious.

2. **Karpathy, Intro to Large Language Models** (1 hr)
   https://www.youtube.com/watch?v=zjkBMFhNj_g
   Watch for pretraining vs fine-tuning. The loop is the fine-tuning half, repeated. Also
   his point that models are confidently wrong: why confidence is a filter, not a verdict.

## Serving: Sprint 1, and why S0-2 is about memory

3. **How vLLM Works + Paged Attention**
   https://www.youtube.com/watch?v=yHAcgyntYDQ
   Watch for what the KV cache is and why it costs VRAM per token. The 56KB/token figure in
   `spikes/s0-2-gpu-coresidency.md` is this idea applied to our model.

4. **SSE Deep Dive: How Server-Sent Events Stream LLM Responses**
   https://www.youtube.com/watch?v=vEmiZ8u7KfA
   Watch for the `data: {...}` line format. It is what `api/app/vllm_client.py` forwards
   and what `spikes/s0-3-sample-stream.json` contains.

## Fine-tuning: Sprints 4 and 5

5. **Umar Jamil, LoRA explained visually + PyTorch from scratch** (26 min)
   https://www.youtube.com/watch?v=PXWYUTMt-AU
   Watch for the rank `r`. It is the `--max-lora-rank` flag in `spikes/s0-1-runtime-lora.md`
   and the reason adapters are megabytes, not gigabytes.

6. **Tim Dettmers, QLoRA: Efficient Finetuning of Quantized LLMs** (optional)
   https://www.youtube.com/watch?v=fQirE9N5q_Y
   Watch for loading the base in 4-bit so training fits on a small card: the 15GB vs 5GB
   row in the S0-2 table.

## Evaluation: Sprints 2 and 5

7. **Hamel Husain, LLM Evals: Common Mistakes**
   https://www.youtube.com/watch?v=GL0XhAj5LPE
   Watch for why the eval set is written by hand and never touched by training data. This is
   S2-1 and the contamination check in S3-3.

8. **Hamel on LLM as a Judge**
   https://www.youtube.com/watch?v=Nycm3Zz5Jzo
   Watch for the judge needing a rubric and a check against a human. This is S2-2 and its
   "re-run twice, variance below tolerance" acceptance criterion.

9. **StatQuest, Bootstrapping Main Ideas** (9 min) then **Confidence Intervals** (6 min)
   https://www.youtube.com/watch?v=Xz0x-8-cgaQ
   https://www.youtube.com/watch?v=TqOeMYtOc1w
   Watch for resampling to get an interval around a difference. S5-2's gate is "the
   bootstrap interval on candidate minus base must exclude zero".

## Infrastructure

10. **TechWorld with Nana, Ultimate Docker Compose Tutorial** (1 hr)
    https://www.youtube.com/watch?v=SXwC9fSwct8
    Watch for services, volumes, `depends_on`, and profiles. `docker-compose.yml` uses all
    four, and the Sprint 4 handoff script depends on `depends_on` behaving as shown.

11. **TechWorld with Nana, How Prometheus Monitoring works** (20 min)
    https://www.youtube.com/watch?v=h4Sl21AKiDg
    Watch for the scrape model: Prometheus pulls from `/metrics`, services do not push.
    S1-4's counters are two more lines on that endpoint.

## Order

| Before | Watch |
|---|---|
| GPU session (Sprint 0) | 1, 2, 3, 5 |
| Sprint 1 | 4, 10, 11 |
| Sprint 2 | 7, 8, 9 |
| Sprint 4 | 6 |
