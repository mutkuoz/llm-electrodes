"""How much did packet composition help the judges? In the judging rounds every packet held six candidates that
answered the same prompts, so the unmodified answer to a prompt appeared once per candidate and could be spotted by
repetition alone. This re-judges a random sample with one candidate per packet, where nothing repeats, using the same
judge prompt, and compares pick accuracy with the original rounds.

    .venv/bin/python pipeline/deleak.py packets   # runs/judge_deleak/packets_{ds,dg}/, one candidate each
    (run runs/judge_workflow.js on them; save the flattened results as runs/judge_deleak/combined.json)
    .venv/bin/python pipeline/deleak.py score     # -> runs/judge_deleak/summary.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import judge  # noqa: E402
from measure import RUNS  # noqa: E402

OUT = RUNS / "judge_deleak"
SEED = 20261031
N_HITS, N_SHAMS, N_GAIN = 20, 15, 15


def sample() -> dict:
    rng = random.Random(SEED)
    stim = json.loads((RUNS / "hits.json").read_text())["candidates"]
    gain = json.loads((RUNS / "discover_gain" / "hits.json").read_text())["candidates"]
    hits = sorted(c["key"] for c in stim if c["hit"] and c["origin"] in ("screen", "known"))
    shams = sorted(c["key"] for c in stim if c["origin"] == "sham")
    gains = sorted(c["key"] for c in gain if c["origin"] == "screen")
    return {"stimulate hit": rng.sample(hits, N_HITS), "stimulate sham": rng.sample(shams, N_SHAMS),
            "gain": rng.sample(gains, N_GAIN)}


def packets():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = sample()
    (OUT / "sample.json").write_text(json.dumps(groups, indent=1))
    judge.OUT = OUT
    judge.packets(RUNS / "discover" / "stage3.jsonl", "ds", per_packet=1, seed=21,
                  only=set(groups["stimulate hit"] + groups["stimulate sham"]))
    judge.packets(RUNS / "discover_gain" / "stage3.jsonl", "dg", per_packet=1, seed=22, only=set(groups["gain"]))


def score():
    judgments = json.loads((OUT / "combined.json").read_text())
    for tag in ("ds", "dg"):
        mine = [j for j in judgments if j["candidate"].startswith(tag.upper() + "-")]
        (OUT / f"judgments_{tag}.json").write_text(json.dumps(mine, indent=1, ensure_ascii=False))
    judge.OUT = OUT
    judge.tally()
    tally = json.loads((OUT / "tally.json").read_text())
    groups = json.loads((OUT / "sample.json").read_text())
    before = {**json.loads((RUNS / "judge" / "tally.json").read_text()),
              **json.loads((RUNS / "judge_gain" / "tally.json").read_text())}
    summary = {}
    for group, keys in groups.items():
        now = [tally[k]["judgments"][0] for k in keys if k in tally]
        then = [before[k]["judgments"][0] for k in keys if k in before]  # round 1, six candidates per packet
        summary[group] = {
            "n": len(now),
            "accuracy_mean": float(np.mean([j["accuracy"] for j in now])),
            "accuracy_min": float(np.min([j["accuracy"] for j in now])),
            "share_at_least_075": float(np.mean([j["accuracy"] >= 0.75 for j in now])),
            "consistency_mean": float(np.mean([j["consistency"] for j in now])),
            "round1_accuracy_mean": float(np.mean([j["accuracy"] for j in then])),
            "round1_consistency_mean": float(np.mean([j["consistency"] for j in then])),
            "kind_none": sum(j["kind"] == "none" for j in now),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for group, s in summary.items():
        print(f"{group:15s} n={s['n']:2d} accuracy {s['accuracy_mean']:.2f} (min {s['accuracy_min']:.2f}, "
              f">=0.75 {s['share_at_least_075']:.0%}) vs round 1 {s['round1_accuracy_mean']:.2f}; consistency "
              f"{s['consistency_mean']:.1f} vs {s['round1_consistency_mean']:.1f}; judged none {s['kind_none']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["packets", "score"])
    {"packets": packets, "score": score}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
