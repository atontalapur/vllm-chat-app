#!/usr/bin/env bash
# Footage take for the S0-2 co-residency spike. Runs ON THE GPU BOX.
#
# Reproduces docs/spikes/s0-2-gpu-coresidency.md as one continuous tmux take:
# left pane nvidia-smi, right pane the commands, five shots with the holds from
# the footage brief. Record the terminal window from your Mac while it runs,
# then cut it with scripts/s0-2-cut.sh.
#
#   scripts/s0-2-take.sh prep   # once per box: tmux, trainer image, warm stack
#   scripts/s0-2-take.sh        # the take (start your screen recording first)
#
# Nothing on screen names the box: tmux status bar off, watch title off, and
# the right pane is this script, not an interactive shell with a prompt.
#
# Knobs (env): LEFT=full|compact  full is the whole nvidia-smi table (80 cols),
#              compact is memory.used/total/free only, which reads on a phone.
#              HOLD_OOM=4 HOLD_PEAK=4  seconds to hold shots 3 and 4.
#
# Writes the observed numbers to s0-2-take-notes.md in the repo root.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO/scripts/$(basename "${BASH_SOURCE[0]}")"
cd "$REPO"

LEFT="${LEFT:-full}"
HOLD_OOM="${HOLD_OOM:-4}"
HOLD_PEAK="${HOLD_PEAK:-4}"
NOTES="$REPO/s0-2-take-notes.md"
IMAGE=s02-qlora
SESSION=s02

# Shot 1 must match the spike's "default settings" row, whatever .env says.
DEFAULT_ENV="VLLM_MAX_NUM_SEQS=4 VLLM_MAX_MODEL_LEN=8192 VLLM_GPU_MEMORY_UTILIZATION=0.90"
SHRUNK_ENV="VLLM_MAX_NUM_SEQS=1 VLLM_MAX_MODEL_LEN=2048 VLLM_GPU_MEMORY_UTILIZATION=0.72"

hf_volume() { docker volume ls -q --filter name=hf-cache | head -1; }

mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
mem_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }

# Print a shot banner with a timestamp, so the cut is easy to find later.
shot() { printf '\n\033[1m== shot %s  %s\033[0m\n%s\n\n' "$1" "$(date +%T)" "$2"; }

# Echo a command the way a shell would, then run it.
run() { printf '\033[2m$\033[0m %s\n' "$*"; "$@"; }

# Show a running clock until the service is healthy (stderr, so it is on
# screen) and print the seconds taken on stdout for capture.
wait_healthy() {
  local svc=$1 start=$SECONDS status
  while :; do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q "$svc")" 2>/dev/null || echo starting)
    printf '\r%s: %-9s %3ds' "$svc" "$status" "$((SECONDS - start))" >&2
    [[ $status == healthy ]] && break
    sleep 1
  done
  printf '\n' >&2
  echo $((SECONDS - start))
}

# Sample nvidia-smi at 2 Hz until the pid exits; print min free and max used.
sample_until() {
  local pid=$1 min_free=999999 max_used=0 f u
  while kill -0 "$pid" 2>/dev/null; do
    f=$(mem_free); u=$(mem_used)
    (( f < min_free )) && min_free=$f
    (( u > max_used )) && max_used=$u
    sleep 0.5
  done
  echo "$min_free $max_used"
}

trainer() {
  docker run --rm --gpus all \
    -v "$(hf_volume)":/root/.cache/huggingface \
    -v "$REPO/scripts/s0-2-qlora-step.py":/step.py:ro \
    -e HOLD="$1" -e HF_TOKEN="${HF_TOKEN:-}" \
    "$IMAGE" python3 /step.py
}

prep() {
  command -v tmux >/dev/null || { apt-get update && apt-get install -y tmux; }
  # Deps baked in so the on-camera step does not spend a minute on pip.
  docker build -t "$IMAGE" - <<'EOF'
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
RUN pip install --no-cache-dir -q transformers peft bitsandbytes accelerate
EOF
  # Weights cached and serving hot, so the take opens on shot 1.
  env $DEFAULT_ENV docker compose --progress plain up -d
  wait_healthy vllm >/dev/null
  echo "prep done: stack healthy at $(mem_used) MiB. Start the recording, then run: $0"
}

take() {
  local used1 used2 used5 t_recover min_free3 max_used4 loaded4 peak4 step4 out sample

  shot 1 "vLLM serving Qwen2.5-7B, default settings (4 seqs, 8192 ctx, 0.90 util)"
  run env $DEFAULT_ENV docker compose --progress plain up -d vllm
  wait_healthy vllm >/dev/null
  sleep 3
  used1=$(mem_used); echo "used: $used1 MiB"

  shot 2 "restart vLLM shrunk: 1 seq, 2048 ctx, 0.72 util"
  run env $SHRUNK_ENV docker compose --progress plain up -d vllm
  wait_healthy vllm >/dev/null
  sleep 3
  used2=$(mem_used); echo "used: $used2 MiB"

  shot 3 "one QLoRA step alongside serving"
  printf '\033[2m$\033[0m docker run --rm --gpus all -v hf-cache:/root/.cache/huggingface %s python3 /step.py\n' "$IMAGE"
  trainer "$HOLD_OOM" & sample=$(sample_until $!) || true
  wait $! || true
  min_free3=${sample%% *}
  echo "min free during the attempt: $min_free3 MiB"

  shot 4 "stop serving, run the same step alone"
  run docker compose --progress plain stop vllm api ui
  printf '\033[2m$\033[0m docker run --rm --gpus all -v hf-cache:/root/.cache/huggingface %s python3 /step.py\n' "$IMAGE"
  out=$(mktemp)
  trainer "$HOLD_PEAK" 2>&1 | tee "$out" & sample=$(sample_until $!)
  wait $! || true
  max_used4=${sample##* }
  loaded4=$(sed -n 's/^loaded, GB: //p' "$out")
  peak4=$(sed -n 's/^after one step, peak GB: //p' "$out")
  step4=$(sed -n 's/^step time: //p' "$out")
  echo "max used during the step: $max_used4 MiB"

  shot 5 "restart the stack with cached weights (shrunk, as in shot 2)"
  # ui depends on api healthy, api on vllm healthy, so up -d blocks until the
  # whole chain is up. Timing the call is the recovery time.
  t0=$SECONDS
  run env $SHRUNK_ENV docker compose --progress plain up -d
  t_recover=$((SECONDS - t0))
  sleep 3
  used5=$(mem_used)
  echo "serving again at $used5 MiB, vllm healthy after ${t_recover}s"

  cat >"$NOTES" <<EOF
# S0-2 take, $(date +%F) $(date +%T)

Card: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader).
LEFT=$LEFT, holds ${HOLD_OOM}s / ${HOLD_PEAK}s.

| Shot | Observed | Spike doc |
|---|---|---|
| 1 serving, default | $used1 MiB | 21,023 MiB |
| 2 serving, shrunk | $used2 MiB | 16,675 MiB |
| 3 shrunk + QLoRA | OOM, ${min_free3} MiB free at the lowest | OOM, 50 MiB free |
| 4 QLoRA alone | loaded ${loaded4:-?} GB, peak ${peak4:-?} GB (nvidia-smi max $max_used4 MiB), $step4 | loaded 5.9, peak 13.2 GB, 1.0 s |
| 5 recovery | vllm healthy after ${t_recover}s, serving at $used5 MiB | ~60 s, 16,675 MiB |
EOF
  printf '\n\033[1mtake finished.\033[0m notes: %s\n' "$NOTES"
  printf 'Stop the recording, then press Enter to close.\n'
  read -r || true
  tmux kill-session -t "$SESSION"
}

layout() {
  command -v tmux >/dev/null || { echo "run '$0 prep' first"; exit 1; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "run '$0 prep' first"; exit 1; }

  local cols left_cmd left_w
  cols=$(tput cols)
  if [[ $LEFT == compact ]]; then
    left_cmd="nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv"
    left_w=48
  else
    left_cmd="nvidia-smi"
    left_w=80
  fi
  if (( cols < left_w + 60 )); then
    echo "terminal is $cols cols; the right pane would be $((cols - left_w - 1)). Widen it or use LEFT=compact."
    exit 1
  fi

  tmux kill-session -t "$SESSION" 2>/dev/null || true
  # -t on watch drops its title line, which is where the hostname would show.
  tmux new-session -d -s "$SESSION" -x "$cols" -y "$(tput lines)" "watch -n 1 -t $left_cmd"
  tmux set -t "$SESSION" status off
  tmux split-window -h -t "$SESSION" -l $((cols - left_w - 1)) \
    "LEFT=$LEFT HOLD_OOM=$HOLD_OOM HOLD_PEAK=$HOLD_PEAK bash '$SELF' run"
  tmux attach -t "$SESSION"
}

case "${1:-}" in
  prep) prep ;;
  run) take ;;
  "") layout ;;
  *) echo "usage: $0 [prep]"; exit 2 ;;
esac
