#!/bin/bash
# Resume after the reboot: stage 2, select, stage 3, the context atlas and E4. Generation is checkpointed.
cd /home/mutkuoz/Documents/neurollm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/queue4.log
.venv/bin/python discover.py stage2 >> runs/stage2.log 2>&1; echo "[$(date +%T)] stage2 exit $?" >> $L
.venv/bin/python discover.py select > runs/select.log 2>&1; echo "[$(date +%T)] select exit $?" >> $L
.venv/bin/python discover.py stage3 >> runs/stage3.log 2>&1; echo "[$(date +%T)] stage3 exit $?" >> $L
.venv/bin/python explain.py contexts > runs/contexts.log 2>&1; echo "[$(date +%T)] contexts exit $?" >> $L
.venv/bin/python infuse.py > runs/e4.log 2>&1; echo "[$(date +%T)] infuse exit $?" >> $L
echo "[$(date +%T)] queue4 done" >> $L
