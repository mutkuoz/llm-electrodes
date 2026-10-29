#!/bin/bash
# After round-1 judging: evidence for every round-1 hit (a superset of the final hits), then the E3 concept drugs.
cd /home/mutkuoz/Documents/neurollm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/queue3.log
while pgrep -f "explain.py contexts|infuse.py" > /dev/null; do sleep 5; done
.venv/bin/python evidence.py gpu --top 80 > runs/evidence_gpu.log 2>&1; echo "[$(date +%T)] evidence gpu exit $?" >> $L
.venv/bin/python drugs.py concept > runs/e3.log 2>&1; echo "[$(date +%T)] drugs concept exit $?" >> $L
echo "[$(date +%T)] queue3 done" >> $L
