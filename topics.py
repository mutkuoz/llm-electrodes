"""Golden-Gate-style topic probes: find the neurons that respond to a topic, then dose them three ways.

    .venv/bin/python topics.py probe     # every neuron's selectivity for each topic -> runs/topics/probe.json
    .venv/bin/python topics.py dose      # single neuron, top-k group and topic direction on ladders -> runs/topics/
    .venv/bin/python topics.py ladder    # clamps and mid-layer levers for the topics dose could not steer

A topic comes with 60 sentences that mention it and 60 matched control sentences that do not (runs/topics/
sentences.json). A neuron's selectivity is the effect size (Cohen's d) of its peak activation over a sentence,
topic against control; neurons that dip on the topic count too, with a negative sign, since stimulating them
negatively imitates the topic. The topic direction is the difference of mean residual streams after a layer, the
cheap stand-in for a sparse-autoencoder feature like Golden Gate Claude's.

Dosing asks two things of answers to prompts that never mention the topic: does the topic come up (drift), and does
the model claim to be the topic when asked who it is (the Golden Gate signature)?
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from measure import RUNS, Record, save, score
from steer import Intervention, Steerer, add_to_output

OUT = RUNS / "topics"
DIRECTION_LAYERS = [4, 8, 12, 16, 20]
PROMPTS = [
    "Who are you, really?",
    "What should I cook for dinner tonight?",
    "Explain how a bicycle works.",
    "Write a short poem about the sea.",
    "Give me three tips for a job interview.",
    "What is the meaning of life?",
]
# Neurons the screen already found for some topics: the probe should rank them high if it works.
KNOWN = {"wrestling": [(4, 7395)], "law": [(3, 2173), (3, 3370)], "yoga": [(6, 4784)], "military": [(5, 6984), (3, 8069)]}


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@torch.inference_mode()
def activations(steerer: Steerer, texts: list[str], batch: int = 16):
    """Per text: peak and trough of every MLP neuron (layers, neurons), and mean residual after DIRECTION_LAYERS."""
    tok, device = steerer.tokenizer, steerer.model.device
    peaks, troughs, resid = [], [], {l: [] for l in DIRECTION_LAYERS}
    state = {}

    def mlp_hook(layer):
        def hook(module, args):
            a = args[0].float()
            mask = state["mask"][..., None]
            state["peak"][:, layer] = a.masked_fill(~mask, -torch.inf).amax(1)
            state["trough"][:, layer] = a.masked_fill(~mask, torch.inf).amin(1)
        return hook

    def resid_hook(layer):
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output).float()
            m = state["mask"][..., None].float()
            resid[layer].append(((h * m).sum(1) / m.sum(1)).cpu())
        return hook

    handles = [l.mlp.down_proj.register_forward_pre_hook(mlp_hook(i)) for i, l in enumerate(steerer.layers)]
    handles += [steerer.layers[l].register_forward_hook(resid_hook(l)) for l in DIRECTION_LAYERS]
    try:
        with steerer.lock:
            for start in range(0, len(texts), batch):
                enc = tok(texts[start : start + batch], return_tensors="pt", padding=True, add_special_tokens=False)
                ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device).bool()
                n = ids.shape[0]
                state["mask"] = mask
                state["peak"] = torch.empty(n, steerer.n_layers, steerer.n_neurons, device=device)
                state["trough"] = torch.empty(n, steerer.n_layers, steerer.n_neurons, device=device)
                steerer.model.model(input_ids=ids, attention_mask=mask.long(), use_cache=False)
                peaks.append(state["peak"].half().cpu())
                troughs.append(state["trough"].half().cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.cat(peaks).float(), torch.cat(troughs).float(), {l: torch.cat(v) for l, v in resid.items()}


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pooled = ((a.var(0) + b.var(0)) / 2).clamp(min=1e-6).sqrt()
    return (a.mean(0) - b.mean(0)) / pooled


def probe(steerer: Steerer):
    """Rank every neuron by selectivity for each topic; save the top 40 and the topic directions."""
    tok = steerer.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    topics = json.loads((OUT / "sentences.json").read_text())
    results, directions = {}, {}
    for t in topics:
        tp, tt, tr = activations(steerer, t["topic_sentences"])
        cp, ct, cr = activations(steerer, t["control_sentences"])
        up, down = cohens_d(tp, cp), cohens_d(-tt, -ct)  # fires harder on the topic / dips harder on the topic
        selectivity = torch.maximum(up, down)
        sign = torch.where(up >= down, 1, -1)
        flat = selectivity.flatten()
        top = flat.topk(40).indices
        n = steerer.n_neurons
        ranked = [{"layer": int(i // n), "neuron": int(i % n), "d": round(float(flat[i]), 3),
                   "sign": int(sign.flatten()[i])} for i in top]
        known = {f"L{l}N{nn}": {"d": round(float(selectivity[l, nn]), 3), "rank": int((flat > selectivity[l, nn]).sum()) + 1}
                 for l, nn in KNOWN.get(t["id"], [])}
        dirs, separation = {}, {}
        for layer in DIRECTION_LAYERS:
            diff = tr[layer].mean(0) - cr[layer].mean(0)
            unit = diff / diff.norm()
            proj_t, proj_c = tr[layer] @ unit, cr[layer] @ unit
            separation[layer] = round(float(cohens_d(proj_t[:, None], proj_c[:, None])[0]), 2)
            dirs[layer] = unit
        directions[t["id"]] = dirs
        results[t["id"]] = {"topic": t["topic"], "keywords": t["keywords"], "top": ranked, "known": known,
                            "direction_separation": separation}
        log(f"{t['id']}: best L{ranked[0]['layer']}N{ranked[0]['neuron']} d={ranked[0]['d']}  known {known}  "
            f"direction d by layer {separation}")
    (OUT / "probe.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    torch.save(directions, OUT / "directions.pt")


def conditions(steerer: Steerer, topic_id: str, info: dict, directions: dict) -> list[dict]:
    """The dose ladders for one topic: best single neuron, top-4 and top-16 groups, and the topic direction."""
    top = info["top"]
    out = []
    for dose in (0.35, 0.7, 1.0):
        n = top[0]
        out.append({"key": f"topic:{topic_id}:single:{dose}", "scale": dose,
                    "ivs": [Intervention(n["layer"], n["neuron"], n["sign"] * dose)], "vector": None})
    for k in (4, 16):
        for dose in (0.7, 1.4):
            out.append({"key": f"topic:{topic_id}:top{k}:{dose}", "scale": dose, "vector": None,
                        "ivs": [Intervention(n["layer"], n["neuron"], n["sign"] * dose / math.sqrt(k)) for n in top[:k]]})
    best_layer = max(info["direction_separation"], key=lambda l: info["direction_separation"][l])
    for layer in sorted({int(best_layer), 8}):
        for dose in (0.35, 0.7, 1.0):
            vec = directions[topic_id][int(layer)] * dose * steerer.scales[int(layer)]
            out.append({"key": f"topic:{topic_id}:dir{layer}:{dose}", "scale": dose, "ivs": [], "vector": (int(layer), vec)})
    return out


def clamp_projection(units: torch.Tensor, targets: torch.Tensor, on: torch.Tensor):
    """Forward hook holding each marked row's residual component along its unit vector at its target value."""

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h * units).sum(-1, keepdim=True)
        h = h + on * (targets - proj) * units
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


@torch.inference_mode()
def topic_stats(steerer: Steerer, topic: dict) -> dict:
    """What clamps need: per neuron the strongest topic peak and trough, per direction layer the topic projections."""
    tok = steerer.tokenizer
    side = tok.padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    try:
        tp, tt, tr = activations(steerer, topic["topic_sentences"])
        _, _, cr = activations(steerer, topic["control_sentences"])
    finally:
        tok.padding_side = side
    return {"peak": tp.amax(0), "trough": tt.amin(0), "resid_topic": tr, "resid_control": cr}


def ladder_conditions(steerer: Steerer, topic_id: str, info: dict, directions: dict, stats: dict) -> list[dict]:
    """A second ladder for topics the first one could not steer, closer to Golden Gate Claude: clamps at every
    position to multiples of the strongest activation seen on topic sentences, and mid-layer directions and neurons."""
    top, out = info["top"], []

    def clamp(n, k):
        value = stats["peak"][n["layer"], n["neuron"]] if n["sign"] > 0 else stats["trough"][n["layer"], n["neuron"]]
        return Intervention(n["layer"], n["neuron"], float(value) * k, op="clamp")

    for k in (2, 5, 10):
        out.append({"key": f"topic:{topic_id}:clamp1:{k}", "scale": k, "vector": None, "ivs": [clamp(top[0], k)]})
        out.append({"key": f"topic:{topic_id}:clamp4:{k}", "scale": k, "vector": None, "ivs": [clamp(n, k) for n in top[:4]]})
    mid = [n for n in top if n["layer"] >= 12][:4]
    for dose in (0.7, 1.4, 2.0):
        out.append({"key": f"topic:{topic_id}:mid4:{dose}", "scale": dose, "vector": None,
                    "ivs": [Intervention(n["layer"], n["neuron"], n["sign"] * dose / math.sqrt(len(mid))) for n in mid]})
    for layer in (12, 16, 20):
        unit = directions[topic_id][layer]
        for dose in (0.7, 1.0, 1.5):
            out.append({"key": f"topic:{topic_id}:dir{layer}:{dose}", "scale": dose, "ivs": [],
                        "vector": (layer, unit * dose * steerer.scales[layer])})
        # As a feature: the projection above the control mean, which the clamp holds at k times its topic maximum.
        base = float((stats["resid_control"][layer] @ unit).mean())
        top_feature = float((stats["resid_topic"][layer] @ unit).max()) - base
        for k in (2, 5, 10):
            out.append({"key": f"topic:{topic_id}:dirclamp{layer}:{k}", "scale": k, "ivs": [], "vector": None,
                        "clamp": (layer, unit, base + k * top_feature)})
    return out


LADDER_TOPICS = ["golden-gate-bridge", "eiffel-tower", "istanbul", "stanford", "football"]


def ladder(steerer: Steerer):
    probe_data = json.loads((OUT / "probe.json").read_text())
    directions = torch.load(OUT / "directions.pt")
    sentences = {t["id"]: t for t in json.loads((OUT / "sentences.json").read_text())}
    jobs = []
    for tid in LADDER_TOPICS:
        jobs += ladder_conditions(steerer, tid, probe_data[tid], directions, topic_stats(steerer, sentences[tid]))
        log(f"ladder: {tid} conditions ready")
    run_doses(steerer, jobs, {t: probe_data[t] for t in LADDER_TOPICS}, "ladder")


def dose(steerer: Steerer):
    probe_data = json.loads((OUT / "probe.json").read_text())
    directions = torch.load(OUT / "directions.pt")
    jobs = [c for tid, info in probe_data.items() for c in conditions(steerer, tid, info, directions)]
    run_doses(steerer, jobs, probe_data, "dose")


def run_doses(steerer: Steerer, jobs: list[dict], probe_data: dict, name: str, batch: int = 32):
    log(f"{name}: {len(jobs)} conditions x {len(PROMPTS)} prompts")
    device, dtype = steerer.model.device, steerer.model.dtype
    records, plain = [], {}
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        chunks = [[None] + jobs[i : i + batch - 1] for i in range(0, len(jobs), batch - 1)]
        for chunk in chunks:
            chunk = chunk + [None] * (batch - len(chunk))
            rows = [c["ivs"] if c else [] for c in chunk]
            deltas = {}
            for r, c in enumerate(chunk):
                if c and c["vector"] is not None:
                    layer, vec = c["vector"]
                    deltas.setdefault(layer, torch.zeros(batch, 1, vec.shape[0]))[r, 0] = vec
            handles = [steerer.layers[l].register_forward_hook(add_to_output(d.to(device, dtype))) for l, d in deltas.items()]
            clamps = {}
            for r, c in enumerate(chunk):
                if c and c.get("clamp"):
                    layer, unit, target = c["clamp"]
                    u, t, on = clamps.setdefault(layer, (torch.zeros(batch, 1, unit.shape[0]), torch.zeros(batch, 1, 1),
                                                         torch.zeros(batch, 1, 1)))
                    u[r, 0], t[r, 0], on[r, 0] = unit, target, 1.0
            handles += [steerer.layers[l].register_forward_hook(clamp_projection(*(x.to(device, dtype) for x in v)))
                        for l, v in clamps.items()]
            try:
                texts = steerer.generate(messages, rows, max_new_tokens=80, temperature=0, repetition_penalty=1.0)
            finally:
                for h in handles:
                    h.remove()
            plain.setdefault((prompt, 0.0, None), texts[0])
            for c, text in zip(chunk[1:], texts[1:]):
                if c:
                    records.append(Record(c["key"], [{"layer": iv.layer, "neuron": iv.neuron, "dose": iv.dose, "op": iv.op}
                                                     for iv in c["ivs"]] or [{"direction": c["key"]}],
                                          prompt, text, 0.0, None, c["scale"]))
        log(f"  {prompt}")
    plain_records = [Record("plain", [], p, t, 0.0, None, 0.0) for (p, _, _), t in plain.items()]
    save(plain_records + records, OUT / f"{name}_raw.jsonl")
    score(steerer, records + plain_records, plain)
    save(plain_records + records, OUT / f"{name}.jsonl")
    summarize(probe_data, records, OUT / ("summary.json" if name == "dose" else f"{name}_summary.json"))


def summarize(probe_data: dict, records: list[Record], path: Path):
    """Per topic and condition: drift (share of answers mentioning the topic) and the identity answer."""
    summary = {}
    for tid, info in probe_data.items():
        # Whole words only: "cat" must not match "category", nor "harm" match "harmony".
        pattern = re.compile(r"\b(" + "|".join(re.escape(w.lower()) for w in info["keywords"]) + r")\b")
        rows = {}
        for r in records:
            if not r.key.startswith(f"topic:{tid}:"):
                continue
            hit = bool(pattern.search(r.text.lower()))
            row = rows.setdefault(r.key, {"mentions": 0, "answers": 0, "identity": None, "fluency": [], "repeat": []})
            row["answers"] += 1
            row["mentions"] += hit
            row["fluency"].append(r.metrics.get("fluency"))
            row["repeat"].append(r.metrics.get("repeat"))
            if r.prompt == "Who are you, really?":
                row["identity"] = r.text.strip()[:300]
        for row in rows.values():
            row["drift"] = row["mentions"] / row["answers"]
            row["fluency"] = float(np.nanmean([f for f in row["fluency"] if f is not None])) if row["fluency"] else None
            row["repeat"] = float(max(x for x in row["repeat"] if x is not None))
        summary[tid] = rows
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log("summary written")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["probe", "dose", "ladder"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    steerer = Steerer()
    {"probe": probe, "dose": dose, "ladder": ladder}[args.step](steerer)


if __name__ == "__main__":
    main()
