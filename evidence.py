"""Evidence for every hit: dose window, promoted tokens, capability changes, firing contexts and example answers.

    .venv/bin/python evidence.py gpu       # ladders, promoted tokens and the capability battery -> runs/evidence_gpu.json
    .venv/bin/python evidence.py build     # everything per hit -> runs/evidence.json (for explanations and the report)
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from measure import CONFIRM, RUNS, load
from steer import Intervention, Steerer, load_hub_adapter

LADDER = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LADDER_PROMPTS = CONFIRM[:3]


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def hit_interventions(c: dict, scale: float = 1.0) -> list[Intervention]:
    if c["key"].startswith("catalog:"):
        import personas as persona_files

        persona = next(p for p in persona_files.load_all() if p["id"] == c["key"].split(":", 1)[1])
        base = load_hub_adapter(persona["hub"]) if "hub" in persona else [Intervention(**iv) for iv in persona["interventions"]]
        return [Intervention(iv.layer, iv.neuron, iv.dose * scale, iv.scale, iv.op) for iv in base]
    return [Intervention(c["layer"], c["neuron"], c["sign"] * c["dose"] * scale)]


def gpu(top: int):
    from capability import battery, score_items
    from discover import plain_refs
    from explain import promoted
    from measure import generate_records, score

    data = json.loads((RUNS / "hits.json").read_text())
    cands = data["candidates"]
    hits = [c for c in cands if c["hit"]][:top]
    controls = [c for c in cands if c["origin"] == "sham"] + [c for c in cands if c["origin"] == "catalog"]
    steerer = Steerer()
    out = {"ladder": {}, "promoted": {}, "capability": {}}

    log(f"ladders for {len(hits)} hits")
    jobs = [(c["key"], hit_interventions(c, k), k) for c in hits for k in LADDER]
    plain_records, plain = plain_refs(steerer, LADDER_PROMPTS, max_new_tokens=80)
    records = generate_records(steerer, jobs, LADDER_PROMPTS, max_new_tokens=80, log=lambda m: None)
    score(steerer, records + plain_records, plain)
    ladder = defaultdict(lambda: defaultdict(list))
    for r in records:
        ladder[r.key][r.scale].append({"prompt": r.prompt, "text": r.text, **r.metrics})
    out["ladder"] = {k: {str(s): v for s, v in d.items()} for k, d in ladder.items()}

    log("promoted tokens")
    texts = [([{"role": "user", "content": p}], plain[(p, 0.0, None)]) for p in LADDER_PROMPTS]
    for c in hits:
        out["promoted"][c["key"]] = promoted(steerer, hit_interventions(c), texts)

    log(f"capability battery: {len(hits)} hits + {len(controls)} controls")
    items = battery()
    subjects = hits + controls
    result = score_items(steerer, items, [hit_interventions(c) for c in subjects])
    correct, margin = result["correct"], result["margin"]
    families = sorted({i.family for i in items})
    def summarize(row):
        return {f: {h: {"acc": float(correct[row, [j for j, i in enumerate(items) if i.family == f and i.half == h]].mean()),
                        "margin": float(margin[row, [j for j, i in enumerate(items) if i.family == f and i.half == h]].mean())}
                    for h in "ab"} for f in families}
    out["capability"]["plain"] = summarize(0)
    for row, c in enumerate(subjects, start=1):
        out["capability"][c["key"]] = summarize(row)
    np.savez_compressed(RUNS / "capability_raw.npz", correct=correct, margin=margin,
                        keys=np.array(["plain"] + [c["key"] for c in subjects]))
    (RUNS / "evidence_gpu.json").write_text(json.dumps(out, ensure_ascii=False))
    log("evidence gpu saved")


def build(top: int):
    from transformers import AutoTokenizer

    from explain import context_text
    from steer import CACHE_DIR, MODEL_ID

    data = json.loads((RUNS / "hits.json").read_text())
    hits = [c for c in data["candidates"] if c["hit"]][:top]
    gpu_data = json.loads((RUNS / "evidence_gpu.json").read_text()) if (RUNS / "evidence_gpu.json").exists() else {}
    stage3 = defaultdict(list)
    plain3 = {}
    for r in load(RUNS / "discover" / "stage3.jsonl"):
        if r.key == "plain":
            plain3[(r.prompt, r.temperature)] = r.text
        else:
            stage3[r.key].append(r)
    ctx = np.load(RUNS / "contexts.npz") if (RUNS / "contexts.npz").exists() else None
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR) if ctx is not None else None
    screens = {}
    for sign in (-1, 1):
        screens[sign] = np.stack([np.load(RUNS / "screen" / f"stimulate_{0.5 * sign:+g}_L{l:02d}.npy")[0] for l in range(32)])
    evidence = []
    for c in hits:
        e = dict(c)
        recs = sorted(stage3.get(c["key"], []), key=lambda r: -r.metrics.get("distinct", 0))
        e["examples"] = [{"prompt": r.prompt, "temperature": r.temperature, "text": r.text.strip(),
                          "plain": plain3.get((r.prompt, r.temperature), "").strip()} for r in recs[:4]]
        if c.get("layer") is not None:
            layer, neuron, sign = c["layer"], c["neuron"], c["sign"]
            kl = screens[sign][layer, :, 0]
            e["screen"] = {"kl": float(kl[neuron]), "rank_in_layer": int((kl > kl[neuron]).sum()) + 1,
                           "top1": float(screens[sign][layer, neuron, 1]), "entropy": float(screens[sign][layer, neuron, 2])}
            if ctx is not None:
                fire = lambda vals, pos: [{"activation": float(v), "context": context_text(tok, ctx["tokens"], ctx["offsets"], int(p))}
                                          for v, p in zip(vals, pos)]
                e["contexts"] = {"mean": float(ctx["mean"][layer, neuron]), "mean_abs": float(ctx["mean_abs"][layer, neuron]),
                                 "top": fire(ctx["top_val"][layer, neuron], ctx["top_pos"][layer, neuron]),
                                 "bottom": fire(ctx["bot_val"][layer, neuron], ctx["bot_pos"][layer, neuron])}
        e["promoted"] = gpu_data.get("promoted", {}).get(c["key"])
        lad = gpu_data.get("ladder", {}).get(c["key"])
        if lad:
            e["ladder"] = {s: {"distinct": float(np.mean([x["distinct"] for x in v])),
                               "fluency": float(np.nanmean([x["fluency"] for x in v])),
                               "repeat": float(np.mean([x["repeat"] for x in v])),
                               "sample": v[0]["text"][:400]} for s, v in lad.items()}
        cap = gpu_data.get("capability", {})
        if c["key"] in cap:
            e["capability"] = {"plain": cap["plain"], "dosed": cap[c["key"]]}
        evidence.append(e)
    (RUNS / "evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False))
    folder = RUNS / "evidence"
    folder.mkdir(exist_ok=True)
    for e in evidence:  # one file per hit, for the explanation writers
        (folder / f"{e['key'].replace(':', '_')}.json").write_text(json.dumps(e, indent=1, ensure_ascii=False))
    print(f"evidence for {len(evidence)} hits")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["gpu", "build"])
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()
    gpu(args.top) if args.step == "gpu" else build(args.top)


if __name__ == "__main__":
    main()
