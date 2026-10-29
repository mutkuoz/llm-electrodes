"""A blind human check of the judges, built so that nothing can be told apart by repetition.

Asking a person which answer of a pair is altered leaks the same way the six-candidate packets did: every candidate
answered the same prompts, so the unmodified answers recur and are learned within a few items. So the rater sees the
pairs labeled, and is blind instead to what the judges decided and to where each candidate came from (screen hit,
screen non-hit or random sham). For each candidate they answer two questions:

- Change: do the altered answers differ from the normal ones in a consistent way? (yes, unsure, no)
- Fit: which of two descriptions fits the altered answers? One is the round-one judge's description of this
  candidate, the other a judge's description of a different hit, in random order; or neither.

    .venv/bin/python pipeline/human_check.py build    # runs/human_check/{blind_check.html, key.json}
    (rate on the published page; save the ratings collection as runs/human_check/ratings.json)
    .venv/bin/python pipeline/human_check.py score    # agreement with the judges -> runs/human_check/summary.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from measure import RUNS, load  # noqa: E402

OUT = RUNS / "human_check"
SEED = 20261029
GROUPS = {"hit": 24, "non-hit": 12, "sham": 12}
PAIRS, CLIP = 3, 600


def nameable(j: dict) -> bool:
    """The judges' own call that a change is there: distinct and nameable in round one."""
    return j["accuracy"] >= 0.75 and j["consistency"] >= 6 and j["kind"] not in ("none", "degraded")


def clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= CLIP else text[:CLIP].rsplit(" ", 1)[0] + " …"


def build():
    rng = random.Random(SEED)
    hits = {c["key"]: c for c in json.loads((RUNS / "hits.json").read_text())["candidates"]}
    tally = json.loads((RUNS / "judge" / "tally.json").read_text())
    pools = {
        "hit": sorted(k for k, c in hits.items() if c["hit"] and c["origin"] in ("screen", "known")),
        "non-hit": sorted(k for k, c in hits.items() if not c["hit"] and c["origin"] == "screen"),
        "sham": sorted(k for k, c in hits.items() if c["origin"] == "sham"),
    }
    chosen = [(group, key) for group, n in GROUPS.items() for key in rng.sample(pools[group], n)]
    rng.shuffle(chosen)
    records = load(RUNS / "discover" / "stage3.jsonl")
    plain = {(r.prompt, r.temperature, r.seed): r.text for r in records if r.key == "plain"}
    answers = {}
    for r in records:
        if r.key != "plain" and r.temperature == 0:
            answers.setdefault(r.key, []).append(r)
    decoys = [tally[k]["judgments"][0]["description"] for k in pools["hit"]]
    items, key_out = [], {}
    for i, (group, key) in enumerate(chosen):
        item_id = f"C{i + 1:02d}"
        changed = [r for r in answers[key] if r.text.strip() != plain[(r.prompt, r.temperature, r.seed)].strip()]
        pairs = rng.sample(changed or answers[key], min(PAIRS, len(changed or answers[key])))
        own = tally[key]["judgments"][0]["description"]
        decoy = rng.choice([d for d in decoys if d != own])
        judge_first = rng.random() < 0.5
        items.append({
            "id": item_id,
            "pairs": [{"prompt": r.prompt, "normal": clip(plain[(r.prompt, r.temperature, r.seed)]), "altered": clip(r.text)}
                      for r in sorted(pairs, key=lambda r: r.prompt)],
            "options": [own, decoy] if judge_first else [decoy, own],
        })
        j = tally[key]["judgments"][0]
        key_out[item_id] = {"key": key, "group": group, "judge_option": "1" if judge_first else "2",
                            "judge_nameable": nameable(j), "hit": hits[key]["hit"], "judge_name": j["name"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key.json").write_text(json.dumps(key_out, indent=1, ensure_ascii=False))
    template = (ROOT / "pipeline" / "human_check.html").read_text()
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd")
    (OUT / "blind_check.html").write_text(template.replace("/*ITEMS*/[]", data))
    print(f"{len(items)} candidates -> {OUT / 'blind_check.html'}")


def score():
    key = json.loads((OUT / "key.json").read_text())
    rows = json.loads((OUT / "ratings.json").read_text())
    ratings = {r["id"]: r for r in rows}
    solo_path = RUNS / "solo" / "hits.json"  # every candidate judged alone, pooled over both rounds
    solo = {c["key"]: c for c in json.loads(solo_path.read_text())["candidates"]} if solo_path.exists() else {}
    rated = [i for i in key if i in ratings]
    sure = [i for i in rated if ratings[i]["change"] != "unsure"]
    summary = {"rated": len(rated), "unsure": len(rated) - len(sure)}

    def agree(judge_says):
        pairs = [(ratings[i]["change"] == "yes", judge_says(i)) for i in sure if judge_says(i) is not None]
        return {"n": len(pairs), "agree": sum(h == j for h, j in pairs)}

    summary["change_vs_round1_judges"] = agree(lambda i: key[i]["judge_nameable"])
    summary["change_vs_hit_status"] = agree(lambda i: key[i]["hit"])
    if solo:
        summary["change_vs_solo_judges"] = agree(lambda i: nameable(solo[key[i]["key"]]) if key[i]["key"] in solo else None)
        summary["change_vs_solo_hits"] = agree(lambda i: solo[key[i]["key"]]["hit"] if key[i]["key"] in solo else None)
    if solo:
        # Disagreements where the judges also told the answers apart but filed the change as broken text, which the
        # naming check excludes by design, while the rater counted it as a consistent change.
        broken = [i for i in sure if ratings[i]["change"] == "yes" and not nameable(solo[key[i]["key"]])
                  and solo[key[i]["key"]]["kind"] == "degraded" and solo[key[i]["key"]]["accuracy"] >= 0.75]
        summary["disagree_degraded"] = len(broken)
        summary["change_still_hits"] = sum(1 for i in rated if key[i]["group"] == "hit" and solo[key[i]["key"]]["hit"])
    said_yes = [i for i in rated if ratings[i]["change"] == "yes"]
    summary["fit_judge_description"] = {
        "n": len(said_yes), "picked_judge": sum(ratings[i]["fit"] == key[i]["judge_option"] for i in said_yes),
        "picked_decoy": sum(ratings[i]["fit"] not in (key[i]["judge_option"], "neither") for i in said_yes),
        "neither": sum(ratings[i]["fit"] == "neither" for i in said_yes)}
    by_group = {}
    for i in rated:
        g = by_group.setdefault(key[i]["group"], {"n": 0, "yes": 0, "unsure": 0, "no": 0})
        g["n"] += 1
        g[ratings[i]["change"]] += 1
    summary["by_group"] = by_group
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "score"])
    {"build": build, "score": score}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
