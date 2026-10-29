"""Get blind judges' names for the most striking infusions (E4) or drugs (E3).

    python pipeline/name_conditions.py e4 runs/e4.jsonl 24           # -> runs/judge/e4_judge.jsonl
    python pipeline/name_conditions.py e3 runs/e3_concept.jsonl 24

Picks the `n` conditions whose answers differ most from the plain model while staying coherent (the select ceiling,
loops under 0.35), and rewrites their keys as key@scale so judge.py packs one condition per candidate.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from measure import RUNS, Record, load, save  # noqa: E402

kind, path, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
ceiling = json.loads((RUNS / "discover" / "select_summary.json").read_text())["fluency_ceiling"]
records = load(path)
groups = defaultdict(list)
for r in records:
    if r.key != "plain" and not r.key.startswith("sham:"):
        groups[(r.key, r.scale)].append(r)
scored = []
for (key, scale), rs in groups.items():
    flu = [r.metrics["fluency"] for r in rs]
    if any(np.isnan(f) for f in flu) or np.mean(flu) > ceiling or max(r.metrics["repeat"] for r in rs) > 0.35:
        continue
    scored.append((float(np.mean([r.metrics["distinct"] for r in rs])), key, scale))
chosen = {(k, s) for _, k, s in sorted(scored, reverse=True)[:n]}
out = [r for r in records if r.key == "plain"]
out += [Record(f"{r.key}@{r.scale:g}", r.interventions, r.prompt, r.text, r.temperature, r.seed, r.scale, r.metrics)
        for r in records if (r.key, r.scale) in chosen]
save(out, RUNS / "judge" / f"{kind}_judge.jsonl")
print(f"{len(chosen)} of {len(scored)} coherent conditions -> runs/judge/{kind}_judge.jsonl")
