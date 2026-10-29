"""Confirmation for the gain arm: single-neuron gain (silencing, x0, and amplifying, x4), run through the same stages
as the stimulate hits so "electrodes vs drugs" compares like with like. Everything lands in runs/discover_gain and
runs/judge_gain; the stimulate results are never touched.

    .venv/bin/python pipeline/gain_confirm.py stage2    # top neurons per layer and gain by displacement, plus shams
    .venv/bin/python pipeline/gain_confirm.py select    # same coherence rule and ceiling as stimulate
    .venv/bin/python pipeline/gain_confirm.py stage3    # ten held-out prompts, greedy and sampled

Then judge.py and hits.py with --out/--judge/--discover pointing at the gain folders.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discover import plain_refs  # noqa: E402
from measure import CONFIRM, DISCOVERY, RUNS, generate_records, load, save, score  # noqa: E402
from steer import Intervention, Steerer  # noqa: E402

GAINS = [-1.0, 3.0]  # dose for op "gain": activation x (1 + dose), so x0 and x4
PER_LAYER = 40
SHAMS_PER_GAIN = 50
N_SCREEN, PER_LAYER_CAP, N_SHAMS, N_CONTROLS = 100, 8, 20, 10
SEED = 20261030
OUT = RUNS / "discover_gain"


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def key_of(layer: int, neuron: int, gain: float) -> str:
    return f"L{layer}N{neuron}g{gain:+g}"


def stage2(steerer: Steerer):
    rng = random.Random(SEED)
    targets = {}
    for gain in GAINS:
        for layer in range(32):
            disp = np.load(RUNS / "screen_gain" / f"gain_{gain:+g}_L{layer:02d}.npy")[0, :, 3]
            for neuron in np.argsort(-disp)[:PER_LAYER]:
                targets.setdefault(key_of(layer, int(neuron), gain), (layer, int(neuron), gain, "screen"))
        shams = 0
        while shams < SHAMS_PER_GAIN:
            layer, neuron = rng.randrange(32), rng.randrange(steerer.n_neurons)
            if key_of(layer, neuron, gain) not in targets:
                targets[key_of(layer, neuron, gain)] = (layer, neuron, gain, "sham")
                shams += 1
    (OUT / "stage2_targets.json").write_text(json.dumps(targets))
    jobs = [(key, [Intervention(layer, neuron, gain, op="gain")], gain) for key, (layer, neuron, gain, _) in targets.items()]
    log(f"gain stage 2: {len(targets)} targets on {len(DISCOVERY)} prompts")
    plain_records, plain = plain_refs(steerer, DISCOVERY, max_new_tokens=48)
    records = generate_records(steerer, jobs, DISCOVERY, max_new_tokens=48, log=log, checkpoint=OUT / "stage2_ckpt.jsonl")
    save(plain_records + records, OUT / "stage2_raw.jsonl")
    score(steerer, records + plain_records, plain)
    save(plain_records + records, OUT / "stage2.jsonl")
    log(f"gain stage 2 saved {len(records)} records")


def select():
    """The stimulate rule, unchanged: coherent (the stimulate fluency ceiling, loops <= 0.35), then most distinct.
    Positive controls are the ten most striking stimulate hits at their confirmed doses, judged blind alongside."""
    targets = json.loads((OUT / "stage2_targets.json").read_text())
    ceiling = json.loads((RUNS / "discover" / "select_summary.json").read_text())["fluency_ceiling"]
    groups = defaultdict(list)
    for r in load(OUT / "stage2.jsonl"):
        if r.key != "plain":
            groups[r.key].append(r.metrics)
    scored = {}
    for key, ms in groups.items():
        flu = [m["fluency"] for m in ms]
        if any(np.isnan(f) for f in flu) or np.mean(flu) > ceiling or max(m["repeat"] for m in ms) > 0.35:
            continue
        scored[key] = float(np.mean([m["distinct"] for m in ms]))
    survivors, per_layer = [], defaultdict(int)
    by_origin = defaultdict(list)
    for key, (layer, neuron, gain, origin) in targets.items():
        if key in scored:
            by_origin[origin].append({"key": key, "layer": layer, "neuron": neuron, "sign": 1, "dose": gain,
                                      "op": "gain", "origin": origin, "distinct": scored[key]})
    for s in sorted(by_origin["screen"], key=lambda s: -s["distinct"]):
        if len([x for x in survivors if x["origin"] == "screen"]) >= N_SCREEN:
            break
        if per_layer[(s["layer"], s["dose"])] < PER_LAYER_CAP:
            survivors.append(s)
            per_layer[(s["layer"], s["dose"])] += 1
    survivors += sorted(by_origin["sham"], key=lambda s: -s["distinct"])[:N_SHAMS]
    stim = [c for c in json.loads((RUNS / "hits.json").read_text())["candidates"]
            if c["hit"] and c["origin"] == "screen"][:N_CONTROLS]
    for c in stim:  # positive controls, labelled "known" so hits.py reports them as a group
        survivors.append({"key": c["key"], "layer": c["layer"], "neuron": c["neuron"], "sign": c["sign"], "dose": c["dose"],
                          "op": "stimulate", "origin": "known", "distinct": None})
    summary = {"coherent_screen": len(by_origin["screen"]), "coherent_shams": len(by_origin["sham"]),
               "targets": len(targets), "survivors": len(survivors),
               "distinct_screen_p50_p90_max": [float(np.percentile([s["distinct"] for s in by_origin["screen"]], q)) for q in (50, 90, 100)],
               "distinct_sham_p50_p90_max": [float(np.percentile([s["distinct"] for s in by_origin["sham"]], q)) for q in (50, 90, 100)]}
    (OUT / "survivors.json").write_text(json.dumps(survivors, indent=1))
    (OUT / "select_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"gain select: {summary}")


def stage3(steerer: Steerer):
    survivors = json.loads((OUT / "survivors.json").read_text())
    jobs = [(s["key"], [Intervention(s["layer"], s["neuron"], s["sign"] * s["dose"] if s["op"] == "stimulate" else s["dose"],
                                     op=s["op"])], s["dose"]) for s in survivors]
    log(f"gain stage 3: {len(jobs)} on {len(CONFIRM)} held-out prompts, greedy and sampled")
    all_records = []
    for temperature, seed in ((0.0, None), (0.7, 1)):
        plain_records, plain = plain_refs(steerer, CONFIRM, temperature, seed, max_new_tokens=80)
        records = generate_records(steerer, jobs, CONFIRM, max_new_tokens=80, temperature=temperature, seed=seed, log=log,
                                   checkpoint=OUT / f"stage3_ckpt_t{temperature:g}.jsonl")
        score(steerer, records + plain_records, plain)
        all_records += plain_records + records
    save(all_records, OUT / "stage3.jsonl")
    log(f"gain stage 3 saved {len(all_records)} records")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["stage2", "select", "stage3"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == "select":
        return select()
    steerer = Steerer()
    stage2(steerer) if args.step == "stage2" else stage3(steerer)


if __name__ == "__main__":
    main()
