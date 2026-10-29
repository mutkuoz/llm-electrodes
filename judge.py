"""Blind judging of stage-3 candidates, and the tally that turns judgments into the plan's five hit checks.

    .venv/bin/python judge.py packets      # runs/judge/packets/*.json for the judges, runs/judge/key.json kept apart
    .venv/bin/python judge.py tally        # runs/judge/judgments.json -> runs/judge/tally.json

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


def packets(stage3: Path, per_packet: int = 6, seed: int = 11):
    records = load(stage3)
    plain = {(r.prompt, r.temperature, r.seed): r.text for r in records if r.key == "plain"}
    by_key = defaultdict(list)
    for r in records:
        if r.key != "plain":
            by_key[r.key].append(r)
    rng = random.Random(seed)
    keys = sorted(by_key)
    rng.shuffle(keys)
    aliases = {key: f"C{i:03d}" for i, key in enumerate(keys)}
    answer_key = {}
    (OUT / "packets").mkdir(parents=True, exist_ok=True)
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
        (OUT / "packets" / f"packet_{p // per_packet:03d}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=1))
    (OUT / "key.json").write_text(json.dumps({"aliases": aliases, "answers": answer_key}, indent=1))
    print(f"{len(keys)} candidates in {(len(keys) + per_packet - 1) // per_packet} packets")


def binom_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2): the chance a judge guessing at random picks k of n right."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2**n


def tally():
    key = json.loads((OUT / "key.json").read_text())
    alias_to_key = {v: k for k, v in key["aliases"].items()}
    judgments = json.loads((OUT / "judgments.json").read_text())
    rows = {}
    for j in judgments:
        alias = j["candidate"]
        answers = key["answers"][alias]
        picks = j["picks"]
        scored = [(p == a["altered"]) for p, a in zip(picks, answers) if not a["identical"]]
        n, k = len(scored), sum(scored)
        rows.setdefault(alias, []).append({
            "n_pairs": n, "correct": k, "accuracy": k / n if n else 0.0, "p_value": binom_tail(k, n) if n else 1.0,
            "identical_pairs": sum(a["identical"] for a in answers),
            **{f: j.get(f) for f in ("name", "description", "kind", "consistency", "fluency", "interest", "quote")},
        })
    out = {alias_to_key[a]: {"alias": a, "judgments": v} for a, v in rows.items()}
    (OUT / "tally.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"tallied {len(out)} candidates")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["packets", "tally"])
    parser.add_argument("--stage3", type=Path, default=RUNS / "discover" / "stage3.jsonl")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    packets(args.stage3) if args.step == "packets" else tally()


if __name__ == "__main__":
    main()
