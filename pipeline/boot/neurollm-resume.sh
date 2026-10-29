#!/bin/bash
# After a boot: once the NVIDIA GPU is kept awake and answers, finish the neurollm GPU jobs that have not completed
# yet, then become the cockpit. Steps whose results already exist are skipped, so later boots do nothing unless
# runs/.cockpit-at-boot exists, in which case they only start the cockpit. Log: runs/boot.log.
set -u
REPO=/home/mutkuoz/Documents/neurollm
cd "$REPO" || exit 1
LOG=runs/boot.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

log "boot: resume script started"
gpu=$(grep -l 0x10de /sys/bus/pci/devices/*/vendor | head -1 | xargs dirname)
for _ in $(seq 1 90); do  # the GPU must be kept awake before any CUDA work
  [ "$(cat "$gpu/power/control")" = "on" ] && break
  sleep 2
done
if [ "$(cat "$gpu/power/control")" != "on" ]; then
  log "GPU power/control is not 'on'; skipping GPU work to avoid the driver hang"
  exit 0
fi
# A hung driver leaves nvidia-smi in uninterruptible sleep, where timeout cannot end it, so poll it instead.
nvidia-smi -L >> "$LOG" 2>&1 &
smi=$!
for _ in $(seq 1 60); do
  kill -0 "$smi" 2>/dev/null || break
  sleep 1
done
if kill -0 "$smi" 2>/dev/null || ! wait "$smi"; then
  log "nvidia-smi did not answer; skipping GPU work"
  exit 0
fi

ran=0
if [ ! -s runs/topics/summary.json ]; then
  log "running topic dosing"
  $PY topics.py dose > runs/topics/dose.log 2>&1
  log "topic dosing exit $?"
  ran=1
fi
if ! grep -q "via zeroing hook" runs/check_infusion.txt 2>/dev/null; then
  log "running the infusion check"
  $PY pipeline/check_infusion.py > runs/check_infusion.txt 2>&1
  log "infusion check exit $?"
  ran=1
fi
if [ "$ran" = 1 ] || [ -e runs/.cockpit-at-boot ]; then
  log "starting the cockpit on http://127.0.0.1:7860"
  exec $PY server.py >> runs/server.log 2>&1
fi
log "nothing pending"
