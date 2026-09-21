#!/usr/bin/env bash
# Cut the S0-2 screen recording to the post's deliverables. Runs ON THE MAC.
#
#   scripts/s0-2-cut.sh take.mov [outdir]
#
# Writes s0-2-take.mp4 (1080p, ~30 s, no audio) and s0-2-oom.png to outdir
# (default brag-output/s0-2).
#
# The recording is a 3556x2198 capture of a 162-column terminal (22x52 px
# cells), so full-frame text is too small for a phone. Each segment instead
# crops a 2528x1422 window (27 rows) at native resolution and scales it to
# 1080p, a 1.4x zoom with no upscale:
#   TL  top-left: the nvidia-smi pane plus the first 66 columns of the right
#       pane, down to the shot 3 banner
#   BR  bottom-right: the right pane's last 27 rows (the traceback, the peaks)
#   BR2 same, eight rows higher, so shot 5's misleading "healthy after 0s" lines
#       from the take script stays out of frame
#
# Segment times were read off exactly-seeked frames of the 2026-09-19 23:14
# take. A new take needs new times.

set -euo pipefail

in=$1 out=${2:-brag-output/s0-2}
mkdir -p "$out"
mp4="$out/s0-2-take.mp4" png="$out/s0-2-oom.png"
limit=$((50 * 1024 * 1024))

TL="crop=2528:1422:0:0"
BR="crop=2528:1422:1076:776"
BR2="crop=2528:1422:1076:378"

# start duration crop speed
segments=(
  "129   3.5  $TL  1"    # 1. default serving healthy, used 21023 MiB, shot 2 command
  "255   3    $TL  1"    # 2. shrunk serving healthy, used 16675 MiB
  "263   7    $TL  1"    # 3. shot 3 banner, memory climbs 16675 -> 24074 / 51 free
  "270   7    $BR  1"    #    OOM traceback, 51 MiB, shot 4's stop command
  "297   3    $BR  1"    # 4. QLoRA alone: loaded 5.9, peak 13.2 GB, 0.99 s
  "308.5 57.5 $TL  16"   # 5. restart, memory 0 -> 17459 MiB, time-lapse
  "369.5 2    $BR2 1"    #    vllm and api Healthy
)

# The capture is variable frame rate: a frame is written only when the screen
# changes, so "-ss" on the input would drop the frame being held at a cut
# point. Decode once, make each branch constant 30 fps, then trim.
n=${#segments[@]} filter="[0:v]split=$n" i=0
for ((j = 0; j < n; j++)); do filter+="[s$j]"; done
filter+=";"
for seg in "${segments[@]}"; do
  read -r start dur crop speed <<<"$seg"
  filter+="[s$i]$crop,scale=1920:1080,fps=30,trim=start=$start:duration=$dur,setpts=(PTS-STARTPTS)/$speed[v$i];"
  i=$((i + 1))
done
for ((j = 0; j < n; j++)); do filter+="[v$j]"; done
filter+="concat=n=$n:v=1:a=0[out]"

encode() {
  ffmpeg -y -loglevel error -i "$in" -filter_complex "$filter" -map "[out]" \
    "$@" -preset slow -pix_fmt yuv420p -movflags +faststart "$mp4"
}

encode -c:v libx264 -crf 20

size=$(stat -f %z "$mp4")
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp4")
if (( size > limit )); then
  kbps=$(python3 -c "print(int($limit * 8 * 0.95 / $dur / 1000))")
  encode -c:v libx264 -b:v "${kbps}k" -maxrate "${kbps}k" -bufsize "$((kbps * 2))k"
  size=$(stat -f %z "$mp4")
fi

# Fallback still: the OOM frame, full resolution, both panes.
ffmpeg -y -loglevel error -ss 271 -i "$in" -frames:v 1 "$png"

printf '%s  %.1f s  %.1f MB  %s\n' "$mp4" "$dur" "$(echo "$size / 1048576" | bc -l)" \
  "$(ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=s=x:p=0 "$mp4")"
printf '%s\n' "$png"
awk -v d="$dur" 'BEGIN { if (d < 20 || d > 30) print "warning: cut is outside 20-30 s" }'
