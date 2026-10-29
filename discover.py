"""The discovery pipeline: E0 (the catalog), stage 2 (candidates from the screen) and stage 3 (held-out confirmation).

    .venv/bin/python discover.py catalog        # E0: every persona in personas/ on a dose ladder, CONFIRM prompts
    .venv/bin/python discover.py stage2         # screen candidates + known neurons + shams on DISCOVERY prompts
    .venv/bin/python discover.py stage3         # survivors of stage 2 on the held-out CONFIRM prompts

Everything lands in runs/: records as JSON lines (one answer each, with its interventions and metrics), so any number
in a report can be traced back to the text it came from.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import personas as persona_files
from measure import CONFIRM, DISCOVERY, RUNS, Record, generate_records, load, save, score
from steer import Intervention, Steerer, load_hub_adapter

LADDER = [0.5, 0.75, 1.0, 1.25, 1.5]
STAGE2_DOSES = [0.35, 0.7]
PER_LAYER = 70  # candidates per layer and sign, by push: the top 0.76%, which holds 6 of 7 known personas (worst rank 66)
SHAMS = 320  # uniformly random (layer, neuron, sign) controls, run exactly like candidates
SEED = 20261029


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def known_neurons() -> list[tuple[int, int, int]]:
    """(layer, neuron, sign) of every single neuron in the shipped personas, Hub adapters included."""
    found = set()
    for persona in persona_files.load_all():
        ivs = load_hub_adapter(persona["hub"]) if "hub" in persona else [Intervention(**iv) for iv in persona["interventions"]]
        for iv in ivs:
            found.add((iv.layer, iv.neuron, 1 if iv.dose > 0 else -1))
    return sorted(found)


def key_of(layer: int, neuron: int, sign: int) -> str:
    return f"L{layer}N{neuron}{'+' if sign > 0 else '-'}"


def plain_refs(steerer, prompts, temperature=0.0, seed=None, max_new_tokens=96, batch=32) -> tuple[list[Record], dict]:
    """The plain answer per prompt, generated in a batch of the same shape as the steered rows it is compared with."""
    records = generate_records(steerer, [("plain", [], 0.0)], prompts, max_new_tokens, temperature, seed, batch, log=lambda m: None)
    return records, {(r.prompt, r.temperature, r.seed): r.text for r in records}


def catalog(steerer: Steerer, out: Path):
    """E0: the 19 shipped personas, each at every ladder multiple of its doses, greedy, on the CONFIRM prompts."""
    jobs = []
    for persona in persona_files.load_all():
        base = load_hub_adapter(persona["hub"]) if "hub" in persona else [Intervention(**iv) for iv in persona["interventions"]]
        for k in LADDER:
            ivs = [Intervention(iv.layer, iv.neuron, iv.dose * k, iv.scale, iv.op) for iv in base]
            jobs.append((f"catalog:{persona['id']}", ivs, k))
    log(f"E0: {len(jobs)} persona x dose rows on {len(CONFIRM)} prompts")
    plain_records, plain = plain_refs(steerer, CONFIRM)
    records = generate_records(steerer, jobs, CONFIRM, log=log)
    save(plain_records + records, out / "e0_raw.jsonl")
    score(steerer, records + plain_records, plain)
    save(plain_records + records, out / "e0.jsonl")
    log(f"E0 saved {len(records)} records")


def screen_table(dose: float) -> np.ndarray:
    """(layers, neurons, 4) kl, top1, ent, disp of the stage-1 screen at `dose` (first screening text)."""
    return np.stack([np.load(RUNS / "screen" / f"stimulate_{dose:+g}_L{layer:02d}.npy")[0] for layer in range(32)])


def stage2(steerer: Steerer, out: Path):
    rng = random.Random(SEED)
    targets = {}  # key -> (layer, neuron, sign, origin)
    for sign in (-1, 1):
        table = screen_table(0.5 * sign)
        for layer in range(32):
            for neuron in np.argsort(-table[layer, :, 0])[:PER_LAYER]:
                targets.setdefault(key_of(layer, int(neuron), sign), (layer, int(neuron), sign, "screen"))
    for layer, neuron, sign in known_neurons():
        targets[key_of(layer, neuron, sign)] = (layer, neuron, sign, "known")
    shams = 0
    while shams < SHAMS:
        layer, neuron, sign = rng.randrange(32), rng.randrange(steerer.n_neurons), rng.choice((-1, 1))
        key = key_of(layer, neuron, sign)
        if key not in targets:
            targets[key] = (layer, neuron, sign, "sham")
            shams += 1
    (out / "stage2_targets.json").write_text(json.dumps(targets))
    jobs = [(key, [Intervention(layer, neuron, sign * d)], d)
            for key, (layer, neuron, sign, _) in targets.items() for d in STAGE2_DOSES]
    log(f"stage 2: {len(targets)} targets ({shams} shams), {len(jobs)} rows on {len(DISCOVERY)} prompts")
    plain_records, plain = plain_refs(steerer, DISCOVERY, max_new_tokens=64)
    records = generate_records(steerer, jobs, DISCOVERY, max_new_tokens=64, log=log)
    save(plain_records + records, out / "stage2_raw.jsonl")
    log("stage 2: scoring")
    score(steerer, records + plain_records, plain)
    save(plain_records + records, out / "stage2.jsonl")
    log(f"stage 2 saved {len(records)} records")


def select(out: Path, n_screen: int = 360, per_layer_cap: int = 24, n_shams: int = 30, max_repeat: float = 0.35):
    """Pick stage-3 survivors from stage 2, with every threshold read off the data.

    Per target, the two discovery prompts are averaged at each dose. A dose passes when it is coherent: fluency no
    worse than the ceiling and repeat share at most `max_repeat`. The ceiling is the larger of the plain answers'
    fluency + 1.5 nats and the 90th percentile of the catalog personas' fluency at 1x (E0), so text as odd as the
    known-good personas passes. Text that is mostly capitals or CJK is exempt from the ceiling (an all-caps or
    all-Chinese persona reads as fluent to a person, not to the NLL of English text). Each target keeps its most
    distinct passing dose. Survivors: the `n_screen` most distinct screen targets, at most `per_layer_cap` per layer,
    every known persona neuron, and `n_shams` random shams chosen by the same rule.
    """
    from collections import defaultdict

    rng = random.Random(SEED + 1)
    targets = json.loads((out / "stage2_targets.json").read_text())
    records = load(out / "stage2.jsonl")
    plain_flu = np.mean([r.metrics["fluency"] for r in records if r.key == "plain"])
    ceiling = plain_flu + 1.5
    e0_path = out / "e0.jsonl"
    if e0_path.exists():
        e0 = [r for r in load(e0_path) if r.key.startswith("catalog:") and r.scale == 1.0
              and r.metrics["caps"] <= 0.6 and r.metrics["cjk"] <= 0.3]
        ceiling = max(ceiling, float(np.nanpercentile([r.metrics["fluency"] for r in e0], 90)))
    groups = defaultdict(list)
    for r in records:
        if r.key != "plain":
            groups[(r.key, r.scale)].append(r.metrics)
    best = {}
    for (key, dose), ms in groups.items():
        distinct = float(np.mean([m["distinct"] for m in ms]))
        flus = [m["fluency"] for m in ms]
        flu = float("inf") if any(np.isnan(f) for f in flus) else float(np.mean(flus))  # an empty answer is not coherent
        repeat = max(m["repeat"] for m in ms)
        script = max(max(m["caps"] for m in ms) if any(m["caps"] > 0.6 for m in ms) else 0, max(m["cjk"] for m in ms))
        coherent = repeat <= max_repeat and (flu <= ceiling or script > 0.5)
        if coherent and (key not in best or distinct > best[key]["distinct"]):
            best[key] = {"dose": dose, "distinct": distinct, "fluency": flu, "repeat": repeat}
    by_origin = defaultdict(list)
    for key, (layer, neuron, sign, origin) in targets.items():
        if key in best:
            by_origin[origin].append({"key": key, "layer": layer, "neuron": neuron, "sign": sign, "origin": origin, **best[key]})
    survivors, per_layer = [], defaultdict(int)
    for s in sorted(by_origin["screen"], key=lambda s: -s["distinct"]):
        if len(survivors) >= n_screen:
            break
        if per_layer[s["layer"]] < per_layer_cap:
            survivors.append(s)
            per_layer[s["layer"]] += 1
    known = {s["key"] for s in by_origin["known"]}
    survivors += by_origin["known"]
    for key, (layer, neuron, sign, origin) in targets.items():
        if origin == "known" and key not in known:  # no coherent dose: still confirmed, at the smaller dose
            survivors.append({"key": key, "layer": layer, "neuron": neuron, "sign": sign, "origin": origin,
                              "dose": STAGE2_DOSES[0], "distinct": None, "fluency": None, "repeat": None})
    survivors += rng.sample(by_origin["sham"], min(n_shams, len(by_origin["sham"])))
    # Positive controls: the shipped personas at 1x, judged blind next to everything else.
    for persona in persona_files.load_all():
        base = load_hub_adapter(persona["hub"]) if "hub" in persona else [Intervention(**iv) for iv in persona["interventions"]]
        survivors.append({"key": f"catalog:{persona['id']}", "origin": "catalog", "dose": 1.0,
                          "interventions": [{"layer": iv.layer, "neuron": iv.neuron, "dose": iv.dose, "scale": iv.scale} for iv in base]})
    sham_distinct = [s["distinct"] for s in by_origin["sham"]]
    summary = {
        "plain_fluency": float(plain_flu), "fluency_ceiling": float(ceiling), "coherent_targets": len(best),
        "screen_coherent": len(by_origin["screen"]), "sham_coherent": len(by_origin["sham"]),
        "sham_distinct_p50_p90_p99": [float(np.percentile(sham_distinct, q)) for q in (50, 90, 99)] if sham_distinct else None,
        "survivors": len(survivors),
    }
    (out / "survivors.json").write_text(json.dumps(survivors, indent=1))
    (out / "select_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"select: {summary}")


def stage3(steerer: Steerer, out: Path, survivors_path: Path):
    """Survivors (a JSON list of {key, layer, neuron, sign, dose}) on CONFIRM prompts: greedy, and sampled at 0.7."""
    survivors = json.loads(survivors_path.read_text())
    jobs = [(s["key"], [Intervention(**iv) for iv in s["interventions"]], s["dose"]) if "interventions" in s
            else (s["key"], [Intervention(s["layer"], s["neuron"], s["sign"] * s["dose"])], s["dose"]) for s in survivors]
    log(f"stage 3: {len(jobs)} survivors on {len(CONFIRM)} held-out prompts, greedy and sampled")
    all_records = []
    for temperature, seed in ((0.0, None), (0.7, 1)):
        plain_records, plain = plain_refs(steerer, CONFIRM, temperature, seed)
        records = generate_records(steerer, jobs, CONFIRM, temperature=temperature, seed=seed, log=log)
        save(all_records + plain_records + records, out / "stage3_raw.jsonl")
        score(steerer, records + plain_records, plain)
        all_records += plain_records + records
    save(all_records, out / "stage3.jsonl")
    log(f"stage 3 saved {len(all_records)} records")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["catalog", "stage2", "select", "stage3"])
    parser.add_argument("--out", type=Path, default=RUNS / "discover")
    parser.add_argument("--survivors", type=Path, default=RUNS / "discover" / "survivors.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "select":
        return select(args.out)
    steerer = Steerer()
    if args.step == "catalog":
        catalog(steerer, args.out)
    elif args.step == "stage2":
        stage2(steerer, args.out)
    else:
        stage3(steerer, args.out, args.survivors)


if __name__ == "__main__":
    main()
