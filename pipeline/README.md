# Overnight pipeline glue

The order the run used, after `screen.py` (paths are relative to the repo root; `runs/` holds the data):

1. `queue2.sh` / `queue4.sh`: E0 (`discover.py catalog`), stage 2, `select`, stage 3, the context atlas and E4.
   Generation is checkpointed, so a rerun after a crash or reboot resumes where it stopped.
2. `judge.py packets --tag r1`, then the Claude Code workflow `judge_workflow.js` over the packets (blind judges);
   `from_journal.py` pulls the judgments out of the workflow journal into `runs/judge/judgments_r1.json`.
3. `judge.py tally`, `hits.py`; `r2_select.py` picks candidates for a second, independent round
   (`judge.py packets --tag r2 --only runs/judge/r2_keys.json --seed 12`), judged and tallied the same way.
4. `queue3.sh`: `evidence.py gpu` (ladders, promoted tokens, capability battery) and `drugs.py concept`.
5. `evidence.py build`, then `explain_workflow.js` over `runs/evidence/*.json` (one writer and one fact-checker per hit).
6. `report/meta.py`, `report/export_personas.py`, `report/build.py` -> `report/pharmacopoeia.html`.

`rescore.py` re-scores a `*_raw.jsonl` if scoring ever fails after generation.
