#!/bin/bash
# GPU queue, critical path first: finish the screen, E0, stage 2, select, stage 3; then the context atlas and E4.
cd /home/mutkuoz/Documents/neurollm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/queue2.log
while pgrep -f "screen.py --dose" > /dev/null; do sleep 10; done
for attempt in 1 2; do
  [ "$(ls runs/screen/stimulate_*.npy 2>/dev/null | wc -l)" -ge 64 ] && break
  .venv/bin/python screen.py --dose -0.5 --dose 0.5 >> runs/screen/log.txt 2>&1; echo "[$(date +%T)] screen retry $attempt exit $?" >> $L
done
echo "[$(date +%T)] screen complete: $(ls runs/screen/stimulate_*.npy | wc -l) files" >> $L
.venv/bin/python discover.py catalog > runs/e0.log 2>&1; echo "[$(date +%T)] catalog exit $?" >> $L
.venv/bin/python discover.py stage2 > runs/stage2.log 2>&1; echo "[$(date +%T)] stage2 exit $?" >> $L
.venv/bin/python discover.py select > runs/select.log 2>&1; echo "[$(date +%T)] select exit $?" >> $L
.venv/bin/python discover.py stage3 > runs/stage3.log 2>&1; echo "[$(date +%T)] stage3 exit $?" >> $L
.venv/bin/python explain.py contexts > runs/contexts.log 2>&1; echo "[$(date +%T)] contexts exit $?" >> $L
.venv/bin/python infuse.py > runs/e4.log 2>&1; echo "[$(date +%T)] infuse exit $?" >> $L
echo "[$(date +%T)] queue2 done" >> $L
