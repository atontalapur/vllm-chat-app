"""One QLoRA training step, as run in docs/spikes/s0-2-gpu-coresidency.md.

Same model, quantisation, LoRA config, and shapes as the spike, so the numbers
match the doc. Two additions for the footage take (scripts/s0-2-take.sh):

- HOLD seconds of sleep after the peak print, and after an OOM traceback, so
  the left-hand nvidia-smi pane has time to show the number. The caching
  allocator keeps its reservation during the sleep, which is what makes the
  near-full card visible next to the traceback.
- The step is timed with a synchronize on each side.

Runs inside the s02-qlora image built by the take script; see that file.
"""

import os
import time
import traceback

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

HOLD = float(os.environ.get("HOLD", "4"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


def gb(n: int) -> str:
    return f"{n / 1e9:.1f}"


def main() -> None:
    try:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
            device_map="cuda",
        )
        m.gradient_checkpointing_enable()
        m = get_peft_model(
            m,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        print("loaded, GB:", gb(torch.cuda.memory_allocated()), flush=True)

        x = torch.randint(0, 150000, (1, 1024), device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(input_ids=x, labels=x).loss.backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print("after one step, peak GB:", gb(torch.cuda.max_memory_allocated()), flush=True)
        print(f"step time: {dt:.2f} s", flush=True)
        time.sleep(HOLD)
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        print(
            f"OOM with {gb(torch.cuda.memory_allocated())} GB allocated, "
            f"{gb(torch.cuda.memory_reserved())} GB reserved by the trainer",
            flush=True,
        )
        time.sleep(HOLD)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
