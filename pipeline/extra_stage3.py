"""Stage 3 for extra recipes (drugs found in E3) through the running cockpit server, so they get the same held-out
evidence as the screen's hits: the ten CONFIRM prompts, greedy and sampled at 0.7 (seed 1), next to a plain row.

    python pipeline/extra_stage3.py      # needs server.py running; -> runs/discover/stage3_extra.jsonl
"""

import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import CONFIRM, RUNS, Record  # noqa: E402

RECIPES = {
    "drug:broken-spacebar": [{"layer": 4, "neuron": n, "dose": 0.35} for n in (6, 4, 1133, 23)],
    "drug:nursery-simplifier": [{"layer": 10, "neuron": n, "dose": 0.35} for n in (853, 74, 799, 830)],
}


def generate(prompt: str, rows: list[list[dict]], temperature: float, seed: int | None) -> list[str]:
    body = {"messages": [{"role": "user", "content": prompt}], "max_new_tokens": 80, "temperature": temperature,
            "repetition_penalty": 1.0 if temperature == 0 else 1.1, "seed": seed,
            "rows": [{"label": f"r{i}", "interventions": ivs} for i, ivs in enumerate(rows)]}
    request = urllib.request.Request("http://127.0.0.1:7860/api/generate", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
    texts = [""] * len(rows)
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            for i, piece in enumerate(event.get("pieces") or []):
                if piece:
                    texts[i] += piece
    return texts


records = []
for temperature, seed in ((0.0, None), (0.7, 1)):
    for prompt in CONFIRM:
        texts = generate(prompt, [[]] + list(RECIPES.values()), temperature, seed)
        records.append(Record("plain", [], prompt, texts[0], temperature, seed, 0.0))
        for (key, ivs), text in zip(RECIPES.items(), texts[1:]):
            records.append(Record(key, ivs, prompt, text, temperature, seed, 1.0))
        print(f"t={temperature} {prompt[:40]}", flush=True)
out = RUNS / "discover" / "stage3_extra.jsonl"
out.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records))
print(f"{len(records)} records -> {out}")
