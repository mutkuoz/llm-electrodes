"""E3: drugs, i.e. groups of neurons that share a "receptor" (their gate_proj rows read one direction).

Directions come from the data, never from a name:
- concept drugs: the top principal components of the residual stream after a layer, over the text corpus;
- receptor drugs: a confirmed hit's own output direction, read by neurons in the layers just after it.

A drug of size k stimulates the k strongest readers of a direction, each at dose / sqrt(k), so the summed push stays
comparable as k grows. Sizes double from 4 while the drug keeps getting more distinct (the grow search); every drug is
run next to a sham of the same size and dose in the same layer.

    .venv/bin/python drugs.py concept               # PCA directions at a spread of layers
    .venv/bin/python drugs.py receptor --hits FILE  # FILE: JSON list of {key, layer, neuron, sign, dose}
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from discover import plain_refs
from measure import CONFIRM, RUNS, generate_records, save, score
from steer import Intervention, Steerer
from targets import receptor_ranking, sham

PROMPTS = CONFIRM[:4]
SIZES = [4, 8, 16, 32, 64, 128]
CONCEPT_LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]
N_COMPONENTS = 6


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@torch.inference_mode()
def principal_directions(steerer: Steerer, texts: list[str], layers: list[int], n: int, max_tokens: int = 128) -> dict:
    """Top-`n` principal directions of the residual after each layer in `layers`, over `texts` (first token dropped)."""
    states = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            states[layer].append(hidden[0, 1:].float().cpu())
        handles.append(steerer.layers[layer].register_forward_hook(hook))
    try:
        with steerer.lock:
            for text in texts:
                ids = steerer.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens]
                steerer.model.model(input_ids=ids.to(steerer.model.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    directions = {}
    for layer in layers:
        x = torch.cat(states[layer])
        x = x - x.mean(0)
        _, s, v = torch.linalg.svd(x, full_matrices=False)
        directions[layer] = (v[:n], (s[:n] ** 2 / (s**2).sum()).tolist())
    return directions


def drug(steerer, layer: int, direction: torch.Tensor, k: int, dose: float) -> list[Intervention]:
    readers = receptor_ranking(steerer.layers[layer].mlp.gate_proj.weight, direction)[:k]
    return [Intervention(layer, n, dose / math.sqrt(k)) for n in readers]


def sham_drug(layer: int, k: int, dose: float, seed: int) -> list[Intervention]:
    return [Intervention(l, n, dose / math.sqrt(k)) for l, n in sham(layer, k, seed)]


def run(steerer: Steerer, jobs, name: str):
    plain_records, plain = plain_refs(steerer, PROMPTS, max_new_tokens=80)
    records = generate_records(steerer, jobs, PROMPTS, max_new_tokens=80, log=log)
    score(steerer, records + plain_records, plain)
    save(plain_records + records, RUNS / f"e3_{name}.jsonl")
    log(f"E3 {name}: saved {len(records)} records")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["concept", "receptor"])
    parser.add_argument("--hits", type=Path)
    parser.add_argument("--dose", type=float, default=0.7)
    args = parser.parse_args()
    steerer = Steerer()
    jobs = []
    if args.kind == "concept":
        texts = [json.loads(line)["text"] for line in (RUNS / "corpus.jsonl").open()][::15]
        directions = principal_directions(steerer, texts, CONCEPT_LAYERS, N_COMPONENTS)
        meta = {}
        for layer, (vs, shares) in directions.items():
            for c, (v, share) in enumerate(zip(vs, shares)):
                for sign in (1, -1):
                    for k in SIZES:
                        reader_layer = layer + 1
                        key = f"drug:pc{c}{'+' if sign > 0 else '-'}@L{layer}->L{reader_layer}:k{k}"
                        jobs.append((key, drug(steerer, reader_layer, sign * v, k, args.dose), args.dose))
                        meta[key] = {"layer": reader_layer, "component": c, "variance_share": share, "k": k, "sign": sign}
                for k in SIZES:
                    key = f"sham:L{layer + 1}:k{k}"
                    if key not in meta:
                        jobs.append((key, sham_drug(layer + 1, k, args.dose, seed=layer * 1000 + k), args.dose))
                        meta[key] = {"layer": layer + 1, "k": k, "sham": True}
        (RUNS / "e3_concept_meta.json").write_text(json.dumps(meta, indent=1))
    else:
        hits = json.loads(args.hits.read_text())
        meta = {}
        for hit in hits:
            out_dir = hit["sign"] * steerer.direction(hit["layer"], hit["neuron"])
            for offset in (1, 2, 3):
                reader_layer = hit["layer"] + offset
                if reader_layer >= steerer.n_layers:
                    continue
                for k in SIZES[:5]:
                    key = f"receptor:{hit['key']}->L{reader_layer}:k{k}"
                    jobs.append((key, drug(steerer, reader_layer, out_dir, k, args.dose), args.dose))
                    meta[key] = {"hit": hit["key"], "layer": reader_layer, "k": k}
        (RUNS / "e3_receptor_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"E3 {args.kind}: {len(jobs)} drugs on {len(PROMPTS)} prompts")
    run(steerer, jobs, args.kind)


if __name__ == "__main__":
    main()
