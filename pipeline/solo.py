"""Judge every candidate of both arms alone, one candidate per packet, so no answer repeats inside a judge's view.

The main rounds put six candidates in a packet, and since they all answered the same prompts the unmodified answer
appeared six times and could be spotted by repetition (pipeline/deleak.py measured this on a sample). This reruns
both rounds for every candidate with the same judge prompt, the same answer pairs and the same hit rules.

    .venv/bin/python pipeline/solo.py packets 1           # every candidate -> runs/judge_solo, runs/judge_gain_solo
    (judge_workflow.js over runs/judge_solo/paths_1.json)
    .venv/bin/python pipeline/solo.py collect 1 JOURNAL   # judgments, tallies, hits -> runs/solo, runs/discover_gain/solo
    .venv/bin/python pipeline/solo.py packets 2           # round 2 for what round 1 found distinct, plus the controls
    (judge_workflow.js over runs/judge_solo/paths_2.json)
    .venv/bin/python pipeline/solo.py collect 2 JOURNAL
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import judge  # noqa: E402
from measure import RUNS  # noqa: E402

# arm: (stage-3 answers, discover folder, judge folder, hits folder, tag letter, positive-control origin, seed base)
ARMS = {
    "stimulate": (RUNS / "discover" / "stage3.jsonl", RUNS / "discover", RUNS / "judge_solo", RUNS / "solo", "s", "catalog", 30),
    "gain": (RUNS / "discover_gain" / "stage3.jsonl", RUNS / "discover_gain", RUNS / "judge_gain_solo",
             RUNS / "discover_gain" / "solo", "g", "known", 40),
}


def round_two_keys(hits_dir: Path, control: str) -> set:
    """runs/r2_select.py's rule: everything round 1 found distinct, consistent and non-trivial, plus the controls."""
    hits = json.loads((hits_dir / "hits.json").read_text())["candidates"]
    return {c["key"] for c in hits if c["origin"] == control
            or (c["accuracy"] >= 0.75 and c["consistency"] >= 5 and c["kind"] not in ("none", "degraded"))}


def packets(n: int):
    paths = []
    for arm, (stage3, _, judge_dir, hits_dir, letter, control, seed) in ARMS.items():
        judge_dir.mkdir(parents=True, exist_ok=True)
        judge.OUT = judge_dir
        only = None if n == 1 else round_two_keys(hits_dir, control)
        print(f"{arm} round {n}: ", end="")
        judge.packets(stage3, f"{letter}{n}", per_packet=1, seed=seed + n, only=only)
        paths += sorted(str(p) for p in (judge_dir / f"packets_{letter}{n}").glob("packet_*.json"))
    (RUNS / "judge_solo" / f"paths_{n}.json").write_text(json.dumps(paths))
    print(f"{len(paths)} packets in runs/judge_solo/paths_{n}.json")


def collect(n: int, journal: Path):
    results = []
    for line in journal.open():
        result = json.loads(line).get("result")
        if isinstance(result, dict) and isinstance(result.get("candidates"), list):
            results += result["candidates"]
    for arm, (_, discover, judge_dir, hits_dir, letter, _, _) in ARMS.items():
        tag = f"{letter}{n}"
        mine = [j for j in results if j["candidate"].startswith(tag.upper() + "-")]
        expected = len(json.loads((judge_dir / f"key_{tag}.json").read_text())["aliases"])
        print(f"{arm} round {n}: {len(mine)} of {expected} candidates judged")
        (judge_dir / f"judgments_{tag}.json").write_text(json.dumps(mine, indent=1, ensure_ascii=False))
        judge.OUT = judge_dir
        judge.tally()
        hits_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "hits.py"), "--judge", str(judge_dir), "--discover", str(discover),
                        "--out", str(hits_dir)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["packets", "collect"])
    parser.add_argument("round", type=int, choices=[1, 2])
    parser.add_argument("journal", type=Path, nargs="?")
    args = parser.parse_args()
    packets(args.round) if args.step == "packets" else collect(args.round, args.journal)


if __name__ == "__main__":
    main()
