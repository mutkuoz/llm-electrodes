"""Write every confirmed screen hit as a persona file (group "atlas"), so it shows up as a key in the cockpit.

    .venv/bin/python report/export_personas.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import personas as persona_files  # noqa: E402
from measure import RUNS  # noqa: E402


def sweet_range(ladder: dict | None, ceiling: float) -> list[float] | None:
    """The multiples of the confirmed dose where answers stay coherent and differ from plain at least half as much as
    at their most distinct coherent multiple."""
    if not ladder:
        return None
    ok = {float(s): v for s, v in ladder.items() if v["fluency"] <= ceiling and v["repeat"] <= 0.35}
    if not ok:
        return None
    peak = max(v["distinct"] for v in ok.values())
    good = sorted(s for s, v in ok.items() if v["distinct"] >= peak / 2)
    lo, hi = good[0], good[-1]
    return [lo, hi] if lo < hi else None


def main():
    evidence = json.loads((RUNS / "evidence.json").read_text())
    explanations = {e["key"]: e for e in json.loads((RUNS / "explanations.json").read_text())} if (RUNS / "explanations.json").exists() else {}
    ceiling = json.loads((RUNS / "discover" / "select_summary.json").read_text())["fluency_ceiling"]
    taken = {p["name"].lower() for p in persona_files.load_all()}
    written = 0
    for e in evidence:
        if e.get("layer") is None:
            continue
        x = explanations.get(e["key"], {})
        name = (x.get("title") or (e.get("names") or [e["key"]])[0]).strip()[:40]
        if name.lower() in taken:
            name = f"{name[:30]} L{e['layer']}N{e['neuron']}"[:40]
        taken.add(name.lower())
        quote = (x.get("quote") or "").strip()
        effect = x.get("oneliner") or (e.get("descriptions") or [""])[0]
        if quote and len(effect) + len(quote) + 4 <= 240:
            effect = f"{effect} “{quote}”"
        persona = {
            "name": name, "group": "atlas", "effect": effect[:240],
            "interventions": [{"layer": e["layer"], "neuron": e["neuron"], "dose": round(e["sign"] * e["dose"], 3)}],
            "source": (f"Found by the overnight screen of all 294,912 neurons; blind judges told it apart from the plain "
                       f"model on {round(e['accuracy'] * e['pairs'])}/{e['pairs']} held-out pairs (q={e['q']:.1e})."),
        }
        sweet = sweet_range(e.get("ladder"), ceiling)
        if sweet:
            persona["sweet"] = [max(0.0, sweet[0]), min(2.0, sweet[1])]
        problems = persona_files.problems(persona)
        if problems:
            print(f"skip {e['key']}: {problems}")
            continue
        path = persona_files.DIR / f"atlas-{persona_files.slug(name)}.json"
        path.write_text(persona_files.dumps(persona), encoding="utf-8")
        written += 1
    print(f"wrote {written} atlas personas")


if __name__ == "__main__":
    main()
