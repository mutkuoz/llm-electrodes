"""Blind judging of stage-3 candidates, and the tally that turns judgments into the plan's five hit checks.

    .venv/bin/python judge.py packets --tag r1               # runs/judge/packets_r1/*.json, answer key kept apart
    .venv/bin/python judge.py packets --tag r2 --only F.json # a second, independent round over chosen candidates
    .venv/bin/python judge.py tally        # runs/judge/judgments_<tag>.json for every round -> runs/judge/tally.json

A packet holds a few anonymised candidates. For each held-out prompt, greedy and sampled, the judge sees the plain
model's answer and the altered model's answer as A and B in a seeded random order, and says which one is altered.
The candidate's coordinates, dose and origin (screen, known persona or sham) never reach the judge; shams are
randomly chosen neurons mixed in to measure how often judges see a persona that is not there.
"""

import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

from measure import RUNS, load

OUT = RUNS / "judge"
KINDS = ["identity", "disposition_emotion", "style_register", "topic_pull", "altered_state", "belief_worldview",
         "language", "function", "degraded", "none", "other"]


def packets(stage3: Path, tag: str, per_packet: int = 6, seed: int = 11, only: set | None = None):
    """Blind packets for one judging round; each round has its own seeded A/B order, grouping and aliases."""
    records = load(stage3)
    plain = {(r.prompt, r.temperature, r.seed): r.text for r in records if r.key == "plain"}
    by_key = defaultdict(list)
    for r in records:
        if r.key != "plain" and (only is None or r.key in only):
            by_key[r.key].append(r)
    rng = random.Random(seed)
    keys = sorted(by_key)
    rng.shuffle(keys)
    aliases = {key: f"{tag.upper()}-{i:03d}" for i, key in enumerate(keys)}
    answer_key = {}
    (OUT / f"packets_{tag}").mkdir(parents=True, exist_ok=True)
    for p in range(0, len(keys), per_packet):
        packet = []
        for key in keys[p : p + per_packet]:
            pairs = []
            for r in sorted(by_key[key], key=lambda r: (r.temperature, r.prompt)):
                altered_first = rng.random() < 0.5
                ref = plain[(r.prompt, r.temperature, r.seed)]
                a, b = (r.text, ref) if altered_first else (ref, r.text)
                pairs.append({"prompt": r.prompt, "A": a.strip(), "B": b.strip()})
                answer_key.setdefault(aliases[key], []).append({"altered": "A" if altered_first else "B", "identical": r.text.strip() == ref.strip()})
            packet.append({"candidate": aliases[key], "pairs": pairs})
        (OUT / f"packets_{tag}" / f"packet_{p // per_packet:03d}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=1))
    (OUT / f"key_{tag}.json").write_text(json.dumps({"aliases": aliases, "answers": answer_key}, indent=1))
    print(f"{len(keys)} candidates in {(len(keys) + per_packet - 1) // per_packet} packets")


def binom_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2): the chance a judge guessing at random picks k of n right."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2**n


def tally():
    """Merge every round's judgments: per candidate, one entry per judge, scored against that round's answer key."""
    rows = {}
    for path in sorted(OUT.glob("judgments_*.json")):
        tag = path.stem.split("_", 1)[1]
        key = json.loads((OUT / f"key_{tag}.json").read_text())
        alias_to_key = {v: k for k, v in key["aliases"].items()}
        for j in json.loads(path.read_text()):
            alias = j["candidate"]
            if alias not in key["answers"]:
                continue
            answers = key["answers"][alias]
            scored = [(p == a["altered"]) for p, a in zip(j["picks"], answers) if not a["identical"]]
            n, k = len(scored), sum(scored)
            rows.setdefault(alias_to_key[alias], []).append({
                "round": tag, "alias": alias, "n_pairs": n, "correct": k, "accuracy": k / n if n else 0.0,
                "p_value": binom_tail(k, n) if n else 1.0, "identical_pairs": sum(a["identical"] for a in answers),
                **{f: j.get(f) for f in ("name", "description", "kind", "consistency", "fluency", "interest", "quote")},
            })
    out = {k: {"alias": v[0]["alias"], "judgments": sorted(v, key=lambda x: x["round"])} for k, v in rows.items()}
    (OUT / "tally.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"tallied {len(out)} candidates")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["packets", "tally"])
    parser.add_argument("--stage3", type=Path, default=RUNS / "discover" / "stage3.jsonl")
    parser.add_argument("--tag", default="r1")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--only", type=Path, help="JSON list of candidate keys to include")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == "packets":
        only = set(json.loads(args.only.read_text())) if args.only else None
        packets(args.stage3, args.tag, seed=args.seed, only=only)
    else:
        tally()


if __name__ == "__main__":
    main()
