#!/bin/bash
# Gain arm: wait for all 64 gain screen files, then stage 2, select and stage 3 (checkpointed).
cd /home/mutkuoz/Documents/neurollm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/queue_gain.log
until [ "$(ls runs/screen_gain/gain_*.npy 2>/dev/null | wc -l)" -ge 64 ]; do sleep 20; done
sleep 15  # let the screen process exit and free the GPU
echo "[$(date +%T)] gain screen complete" >> $L
.venv/bin/python pipeline/gain_confirm.py stage2 > runs/gain_stage2.log 2>&1; echo "[$(date +%T)] stage2 exit $?" >> $L
.venv/bin/python pipeline/gain_confirm.py select > runs/gain_select.log 2>&1; echo "[$(date +%T)] select exit $?" >> $L
.venv/bin/python pipeline/gain_confirm.py stage3 > runs/gain_stage3.log 2>&1; echo "[$(date +%T)] stage3 exit $?" >> $L
echo "[$(date +%T)] queue_gain done" >> $L
