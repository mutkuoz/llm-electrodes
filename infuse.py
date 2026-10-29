"""E4: local infusion. Every MLP neuron of one layer gets the same gain, the way a drug infused into one brain region
reaches every cell there. Gains run from silencing the layer's MLP (-1) to doubling it (+1).

    .venv/bin/python infuse.py        # 32 layers x 5 gains, greedy on 5 held-out prompts, plus push; runs/e4.jsonl
"""

import json
import time

from discover import plain_refs
from measure import CONFIRM, RUNS, generate_records, save, score
from steer import Intervention, Steerer
from targets import dosed, infusion

GAINS = [-1.0, -0.5, -0.25, 0.5, 1.0]
PROMPTS = CONFIRM[:5]


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    steerer = Steerer()
    jobs = [(f"infusion:L{layer}", dosed(infusion(layer), g, op="gain"), g) for layer in range(steerer.n_layers) for g in GAINS]
    log(f"E4: {len(jobs)} layer x gain rows on {len(PROMPTS)} prompts")
    plain_records, plain = plain_refs(steerer, PROMPTS, max_new_tokens=80)
    records = generate_records(steerer, jobs, PROMPTS, max_new_tokens=80, batch=32, log=log)
    save(plain_records + records, RUNS / "e4_raw.jsonl")
    score(steerer, records + plain_records, plain)
    save(plain_records + records, RUNS / "e4.jsonl")
    messages = [{"role": "user", "content": PROMPTS[0]}]
    pushes = steerer.push(messages, plain[(PROMPTS[0], 0.0, None)], [ivs for _, ivs, _ in jobs], batch_size=16)
    (RUNS / "e4_push.json").write_text(json.dumps({key + f"@{g:+g}": p for (key, _, g), p in zip(jobs, pushes)}))
    log("E4 saved")


if __name__ == "__main__":
    main()
