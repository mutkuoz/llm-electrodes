"""Assemble every result file into the Qwen Pharmacopoeia page.

    .venv/bin/python report/build.py      # -> report/pharmacopoeia.html
"""

import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from measure import RUNS, load  # noqa: E402

BIN = 16
LO, HI = 0.05, 20.0  # push scale of the atlas, in nats (log)


def atlas():
    planes, peaks = {}, {}
    for name, dose in (("neg", -0.5), ("pos", 0.5)):
        kl = np.stack([np.load(RUNS / "screen" / f"stimulate_{dose:+g}_L{l:02d}.npy")[0, :, 0] for l in range(32)])
        binned = kl.reshape(32, -1, BIN)
        peak = binned.max(-1)
        arg = binned.argmax(-1) + np.arange(binned.shape[1])[None, :] * BIN
        scaled = (np.log10(np.clip(peak, LO, HI)) - np.log10(LO)) / (np.log10(HI) - np.log10(LO))
        planes[name] = base64.b64encode((scaled * 255).astype(np.uint8).tobytes()).decode()
        peaks[name] = [[int(n), round(float(v), 2)] for n, v in zip(arg.ravel(), peak.ravel())]
    return {"bins": 9216 // BIN, "neg": planes["neg"], "pos": planes["pos"], "peaks": peaks, "lo": LO, "hi": int(HI)}


def infusion():
    path = RUNS / "e4.jsonl"
    if not path.exists():
        return {"gains": [], "cells": {}, "ceiling": 0}
    records = load(path)
    plain = [r.metrics["fluency"] for r in records if r.key == "plain"]
    groups = defaultdict(list)
    for r in records:
        if r.key.startswith("infusion:"):
            groups[(int(r.key.split(":L")[1]), r.scale)].append(r)
    cells = {}
    for (layer, gain), rs in groups.items():
        cells[f"{layer}:{gain:g}"] = {
            "distinct": float(np.mean([r.metrics["distinct"] for r in rs])),
            "fluency": float(np.nanmean([r.metrics["fluency"] for r in rs])),
            "repeat": float(np.mean([r.metrics["repeat"] for r in rs])),
            "sample": rs[0].text.strip().replace("\n", " "),
        }
    gains = sorted({g for _, g in groups})
    return {"gains": [float(f"{g:g}") for g in gains], "cells": cells, "ceiling": float(np.mean(plain) + 1.5)}


def capability_table(hits):
    cap_path = RUNS / "evidence_gpu.json"
    if not cap_path.exists():
        return {"columns": ["Note"], "rows": [["The capability battery did not run."]]}
    cap = json.loads(cap_path.read_text())["capability"]
    plain = cap["plain"]
    families = sorted(plain)
    columns = ["Intervention"] + [f"{f} (Δ acc, held-out half)" for f in families]
    rows = [["Plain model (accuracy)"] + [f"{plain[f]['b']['acc']:.2f}" for f in families]]
    scored = []
    for h in hits:
        if h["key"] not in cap:
            continue
        d = cap[h["key"]]
        gains = [d[f]["b"]["acc"] - plain[f]["b"]["acc"] for f in families]
        found = [d[f]["a"]["acc"] - plain[f]["a"]["acc"] for f in families]
        scored.append((max(found), h["title"], gains))
    for _, title, gains in sorted(scored, reverse=True)[:12]:
        rows.append([title] + [f"{g:+.2f}" for g in gains])
    return {"columns": columns, "rows": rows}


def main():
    hits_data = json.loads((RUNS / "hits.json").read_text())
    evidence = {e["key"]: e for e in json.loads((RUNS / "evidence.json").read_text())}
    explanations = {}
    if (RUNS / "explanations.json").exists():
        explanations = {e["key"]: e for e in json.loads((RUNS / "explanations.json").read_text())}
    summary = hits_data["summary"]
    cands = hits_data["candidates"]
    hits = []
    for key, e in evidence.items():
        x = explanations.get(key, {})
        names = e.get("names") or []
        ladder = e.get("ladder")
        if ladder:
            ceiling = json.loads((RUNS / "discover" / "select_summary.json").read_text())["fluency_ceiling"]
            for v in ladder.values():
                v["ok"] = v["fluency"] <= ceiling and v["repeat"] <= 0.35
        hits.append({
            **{k: e.get(k) for k in ("key", "layer", "neuron", "sign", "dose", "kind", "pairs", "q", "consistency",
                                     "fluency", "interest", "screen", "promoted", "contexts", "examples", "names")},
            "correct": round(e["accuracy"] * e["pairs"]),
            "title": x.get("title") or (names[0] if names else e["key"]),
            "oneliner": x.get("oneliner") or (e.get("descriptions") or [""])[0],
            "effect": x.get("effect"), "mechanism": x.get("mechanism"), "dosing": x.get("dosing"),
            "side_effects": x.get("side_effects"),
            "quote": x.get("quote") or (e.get("quotes") or [None])[0],
            "ladder": ladder,
        })
    hits.sort(key=lambda h: (-(h.get("interest") or 0), -(h.get("correct") or 0)))
    catalog = [{"id": c["key"].split(":", 1)[1], "name": c["names"][0] if c["names"] else "", "correct": round(c["accuracy"] * c["pairs"]),
                "pairs": c["pairs"], "q": c["q"], "hit": c["hit"]} for c in cands if c["origin"] == "catalog"]
    stage2 = json.loads((RUNS / "discover" / "stage2_targets.json").read_text())
    select = json.loads((RUNS / "discover" / "select_summary.json").read_text())
    n_screen_hits = summary["screen"]["hits"] + summary.get("known", {}).get("hits", 0)
    controls = [
        {"group": "Shipped personas", "what": "positive controls: the 19 personas the cockpit already ships, at 1x", "judged": summary["catalog"]["judged"], "hits": summary["catalog"]["hits"], "acc": summary["catalog"]["accuracy_mean"]},
        {"group": "Shams", "what": "negative controls: random neurons, dosed and selected by the same rules", "judged": summary["sham"]["judged"], "hits": summary["sham"]["hits"], "acc": summary["sham"]["accuracy_mean"]},
        {"group": "Known neurons", "what": "single neurons from the shipped personas, run as screen candidates", "judged": summary["known"]["judged"], "hits": summary["known"]["hits"], "acc": summary["known"]["accuracy_mean"]},
        {"group": "Screen candidates", "what": "the most distinct coherent neurons from the full screen", "judged": summary["screen"]["judged"], "hits": summary["screen"]["hits"], "acc": summary["screen"]["accuracy_mean"]},
    ]
    meta = json.loads((RUNS / "report_meta.json").read_text())
    meta["facts"] = [
        ["294,912", "MLP neurons screened"], ["589,824", "pushes measured"], [f"{len(stage2):,}", "candidates generated"],
        [f"{summary['screen']['judged'] + summary['known']['judged']:,}", "judged blind"], [str(n_screen_hits), "confirmed hits"],
        [f"{summary['sham']['hits']}/{summary['sham']['judged']}", "shams passed (false alarms)"],
    ]
    data = {"meta": meta, "atlas": atlas(), "hits": hits, "controls": controls, "catalog": catalog,
            "infusion": infusion(), "function": capability_table(hits), "methods": meta.pop("methods"), "select": select}
    html = (ROOT / "report" / "template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (ROOT / "report" / "pharmacopoeia.html").write_text(html.replace("/*DATA*/", payload))
    print(f"report: {len(hits)} monographs, {len(payload) / 1e6:.2f} MB of data")


if __name__ == "__main__":
    main()
