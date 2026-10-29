"""The Qwen Pharmacopoeia catalog: every confirmed persona, as one searchable page and as cockpit keys.

    .venv/bin/python report/catalog.py      # -> report/catalog.html and personas/found-*.json

Entries: every single neuron that passed every check (screen and known), the random neurons (shams) that did, the
two Google-Lovers (random neurons that failed only specificity), the drugs that blind judges confirmed on held-out
prompts, and the shipped personas for reference. Each entry merges the judges' verdicts from both rounds, the
fact-checked monograph where one was written, the dose ladder, and example answers from the held-out prompts.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))
import personas as persona_files  # noqa: E402
from build import finite, plain_text  # noqa: E402
from judge import binom_tail  # noqa: E402
from measure import RUNS, load  # noqa: E402

BANK = {  # judged kind -> cockpit bank
    "identity": "found-identity", "disposition_emotion": "found-mood", "belief_worldview": "found-belief",
    "topic_pull": "found-topic", "style_register": "found-style", "language": "found-language",
    "altered_state": "found-state", "function": "found-function",
}
RENAMED = {"L6N1287-": "Google-Lover", "L9N4528-": "Google-Lover II"}  # the user's names
NEAR_MISSES = set(RENAMED)
DRUGS = {"drug:broken-spacebar": "Broken Spacebar", "drug:nursery-simplifier": "Nursery-Rhyme Simplifier"}
EXAMPLE_PROMPTS = ["Who are you, really?", "Explain how a bicycle works.", "I failed my driving test today."]


def examples(records: list, plain: dict) -> list[dict]:
    """Greedy held-out answers to three fixed prompts, next to the plain model's."""
    out = []
    for prompt in EXAMPLE_PROMPTS:
        r = next((r for r in records if r.prompt == prompt and r.temperature == 0.0), None)
        if r:
            out.append({"prompt": prompt, "dosed": plain_text(r.text)[:500], "plain": plain_text(plain[(prompt, 0.0)])[:300]})
    return out


def drug_judgments() -> dict:
    """Blind verdicts on the extra drugs: {key: {accuracy, pairs, names, descriptions, kinds, consistency, ...}}."""
    path, key_path = RUNS / "judge" / "names_x.json", RUNS / "judge" / "key_x.json"
    if not path.exists():
        return {}
    key = json.loads(key_path.read_text())
    alias_to_key = {v: k for k, v in key["aliases"].items()}
    out = defaultdict(list)
    for j in json.loads(path.read_text()):
        answers = key["answers"][j["candidate"]]
        scored = [p == a["altered"] for p, a in zip(j["picks"], answers) if not a["identical"]]
        out[alias_to_key[j["candidate"]]].append({**j, "correct": sum(scored), "n": len(scored)})
    return out


def main():
    hits = json.loads((RUNS / "hits.json").read_text())["candidates"]
    monographs = {m["key"]: m for m in json.loads((RUNS / "explanations.json").read_text())}
    evidence = {e["key"]: e for e in json.loads((RUNS / "evidence.json").read_text())}
    ceiling = json.loads((RUNS / "discover" / "select_summary.json").read_text())["fluency_ceiling"]
    stage3 = defaultdict(list)
    plain = {}
    for path in (RUNS / "discover" / "stage3.jsonl", RUNS / "discover" / "stage3_extra.jsonl"):
        if path.exists():
            for r in load(path):
                if r.key == "plain":
                    plain.setdefault((r.prompt, r.temperature), r.text)
                else:
                    stage3[r.key].append(r)

    entries = []
    for c in hits:
        keep = c["hit"] or c["key"] in NEAR_MISSES
        if not keep or c["origin"] == "catalog" and not c["hit"]:
            continue
        m = monographs.get(c["key"], {})
        origin = {"screen": "screen", "known": "screen", "sham": "random", "catalog": "shipped"}[c["origin"]]
        if c["origin"] == "known":
            origin = "rediscovered"
        title = RENAMED.get(c["key"]) or m.get("title") or c["names"][0]
        ladder = (evidence.get(c["key"]) or {}).get("ladder")
        entries.append({
            "key": c["key"], "title": title, "kind": c["kind"], "origin": origin, "near_miss": c["key"] in NEAR_MISSES,
            "layer": c.get("layer"), "neuron": c.get("neuron"), "sign": c.get("sign"), "dose": c.get("dose"),
            "interventions": ([{"layer": c["layer"], "neuron": c["neuron"], "dose": round(c["sign"] * c["dose"], 3)}]
                              if c.get("layer") is not None else None),
            "correct": round(c["accuracy"] * c["pairs"]), "pairs": c["pairs"], "q": c["q"],
            "consistency": c["consistency"], "fluency": c["fluency"], "interest": c["interest"],
            "names": c["names"], "descriptions": c["descriptions"],
            "quote": plain_text(m.get("quote") or c["quotes"][0]),
            "oneliner": m.get("oneliner") or c["descriptions"][0],
            "monograph": {k: m[k] for k in ("effect", "mechanism", "dosing", "side_effects") if m.get(k)} or None,
            "ladder": ({s: {"distinct": round(v["distinct"], 2), "ok": v["fluency"] <= ceiling and v["repeat"] <= 0.35}
                        for s, v in ladder.items()} if ladder else None),
            "examples": examples(stage3.get(c["key"], []), plain),
        })

    for key, judged in drug_judgments().items():
        correct, n = sum(j["correct"] for j in judged), sum(j["n"] for j in judged)
        accuracy = correct / n if n else 0
        consistency = sum(j["consistency"] for j in judged) / len(judged)
        if accuracy < 0.75 or consistency < 6:  # a drug enters the catalog only when the judges confirm it
            print(f"drug {key} not confirmed: accuracy {accuracy:.2f}, consistency {consistency:.1f}")
            continue
        ivs = stage3[key][0].interventions
        entries.append({
            "key": key, "title": DRUGS[key], "kind": max({j["kind"] for j in judged}, key=[j["kind"] for j in judged].count),
            "origin": "drug", "near_miss": False, "layer": ivs[0]["layer"], "neuron": None, "sign": 1, "dose": ivs[0]["dose"],
            "interventions": [{"layer": iv["layer"], "neuron": iv["neuron"], "dose": iv["dose"]} for iv in ivs],
            "correct": judged[0]["correct"], "pairs": judged[0]["n"], "q": binom_tail(judged[0]["correct"], judged[0]["n"]),
            "consistency": consistency, "fluency": sum(j["fluency"] for j in judged) / len(judged),
            "interest": sum(j["interest"] for j in judged) / len(judged),
            "names": [j["name"] for j in judged], "descriptions": [j["description"] for j in judged],
            "quote": plain_text(judged[0]["quote"]), "oneliner": judged[0]["description"], "monograph": None, "ladder": None,
            "examples": examples(stage3[key], plain),
        })

    seen = defaultdict(int)  # one title per entry
    for e in entries:
        seen[e["title"].lower()] += 1
        if seen[e["title"].lower()] > 1:
            e["title"] = f"{e['title']} ({e['key'].replace('drug:', '')})"
    order = {"screen": 0, "rediscovered": 1, "random": 2, "drug": 3, "shipped": 4}
    entries.sort(key=lambda e: (order[e["origin"]], -(e["interest"] or 0), -(e["consistency"] or 0)))
    write_personas([e for e in entries if e["origin"] != "shipped"])

    counts = defaultdict(int)
    for e in entries:
        counts[e["origin"]] += 1
    data = {"entries": entries, "counts": counts, "pharmacopoeia": "https://claude.ai/artifact/MEmHRe1nCi9MiJjKvaFSkk"}
    html = (ROOT / "report" / "catalog_template.html").read_text()
    payload = json.dumps(finite(data), ensure_ascii=False, allow_nan=False).replace("</", "<\\/").replace("�", "\\ufffd")
    (ROOT / "report" / "catalog.html").write_text(html.replace("/*DATA*/", payload))
    print(f"catalog: {len(entries)} entries {dict(counts)}, {len(payload) / 1e6:.2f} MB")


def write_personas(entries: list[dict]):
    """Every non-shipped entry as a cockpit persona, in a bank per kind; replaces earlier screen exports."""
    for old in list(persona_files.DIR.glob("atlas-*.json")) + list(persona_files.DIR.glob("found-*.json")) + \
            list(persona_files.DIR.glob("drug-*.json")):
        old.unlink()
    taken = {p["name"].lower() for p in persona_files.load_all()}
    written = 0
    for e in entries:
        group = "drugs" if e["origin"] == "drug" else "found-random" if e["origin"] == "random" else BANK.get(e["kind"], "found-other")
        name = e["title"][:40]
        if name.lower() in taken:
            name = f"{e['title'][:28]} L{e['layer']}" + (f"N{e['neuron']}" if e["neuron"] is not None else "")
        taken.add(name.lower())
        quote = re.sub(r"\s+", " ", e["quote"] or "").strip()
        effect = re.sub(r"\s+", " ", e["oneliner"] or "").strip()
        if quote and len(effect) + len(quote) + 4 <= 240:
            effect = f"{effect} “{quote}”"
        persona = {
            "name": name, "group": group, "effect": effect[:240], "interventions": e["interventions"],
            "source": (f"Qwen Pharmacopoeia: blind judges told it from the plain model on {e['correct']}/{e['pairs']} "
                       f"held-out pairs." + (" A random neuron (sham control)." if e["origin"] == "random" else "")),
        }
        if e["ladder"]:
            good = sorted(float(s) for s, v in e["ladder"].items() if v["ok"])
            if len(good) >= 2:
                persona["sweet"] = [good[0], min(2.0, good[-1])]
        problems = persona_files.problems(persona)
        if problems:
            print(f"skip {e['key']}: {problems}")
            continue
        prefix = "drug" if group == "drugs" else "found"
        (persona_files.DIR / f"{prefix}-{persona_files.slug(name)}.json").write_text(persona_files.dumps(persona), encoding="utf-8")
        written += 1
    print(f"wrote {written} cockpit personas")


if __name__ == "__main__":
    main()
