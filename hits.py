"""From blind judgments to hits: the research plan's five checks plus replication, with the controls reported first.

    .venv/bin/python hits.py        # runs/judge/tally.json + stage files -> runs/hits.json

A candidate is a hit when it passes all five checks on held-out prompts:
- Distinct: judges pick the altered answer far above chance. Binomial test against a guessing judge (p = 1/2), with
  Benjamini-Hochberg correction across every judged candidate (q < 0.01), and accuracy of at least 0.75.
- Nameable: judges rate the change consistent across prompts (consistency >= 6) and give it a kind other than
  "none" or "degraded".
- Intact: judged fluency >= 4 (>= 3 for an altered state, which is allowed to sound impaired).
- Specific: accuracy and consistency at or above the 90th percentile of the shams (the most distinct random neurons,
  pushed and selected with the same rule); how many shams pass the other checks is reported as a false-hit rate.
- Reliable: the accuracy is over 20 pairs, ten held-out prompts each greedy and sampled, so no single lucky sample
  carries it; candidates whose altered and plain answers were identical on more than a quarter of pairs fail.
- Replicated: when a second, independent judging round exists, it reaches accuracy 0.7 or more as well.
Controls: the 19 shipped personas (positive) and the shams (negative) go through exactly the same judging.
"""

import json
from pathlib import Path

import numpy as np

from measure import RUNS

KEEP = ("name", "description", "kind", "consistency", "fluency", "interest", "quote")


def bh(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out.tolist()


def origin_of(key: str, survivors: dict) -> str:
    if key.startswith("catalog:"):
        return "catalog"
    return survivors.get(key, {}).get("origin", "screen")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", type=Path, default=RUNS / "judge", help="folder holding tally.json")
    parser.add_argument("--discover", type=Path, default=RUNS / "discover", help="folder holding survivors.json")
    parser.add_argument("--out", type=Path, default=RUNS, help="where hits.json and hit_list.json go")
    args = parser.parse_args()
    tally = json.loads((args.judge / "tally.json").read_text())
    survivors = {s["key"]: s for s in json.loads((args.discover / "survivors.json").read_text())}
    rows = []
    for key, entry in tally.items():
        j = entry["judgments"]
        correct = sum(x["correct"] for x in j)
        n = sum(x["n_pairs"] for x in j)
        first = j[0]
        rows.append({
            "key": key, "alias": entry["alias"], "origin": origin_of(key, survivors),
            "accuracy": correct / n if n else 0.0, "pairs": n, "identical": first["identical_pairs"],
            # Later judges re-read the same pairs, so they are not independent samples: the test uses the first.
            "p": first["p_value"],
            "consistency": float(np.mean([x["consistency"] for x in j])),
            "fluency": float(np.mean([x["fluency"] for x in j])),
            "interest": float(np.mean([x["interest"] for x in j])),
            "kinds": [x["kind"] for x in j], "names": [x["name"] for x in j],
            "descriptions": [x["description"] for x in j], "quotes": [x["quote"] for x in j],
            "judges": len(j), "round_accuracy": [x["accuracy"] for x in j],
            **{k: survivors.get(key, {}).get(k) for k in ("layer", "neuron", "sign", "dose")},
        })
    for row, q in zip(rows, bh([r["p"] for r in rows])):
        row["q"] = q
    shams = [r for r in rows if r["origin"] == "sham"]
    sham_best = max([r["accuracy"] for r in shams], default=0.5)
    # Shams are the most distinct random neurons, so some change the text visibly (often by breaking it) and reach
    # perfect accuracy. A persona has to stand out from them on accuracy and on consistency, which is what separates
    # a nameable character from noise: at or above the shams' 90th percentile on both.
    sham_acc90 = float(np.percentile([r["accuracy"] for r in shams], 90)) if shams else 0.5
    sham_cons90 = float(np.percentile([r["consistency"] for r in shams], 90)) if shams else 0.0
    for r in rows:
        # Majority kind across rounds; a tie goes to round 1, the primary blind judgment. (A plain max over a set
        # broke ties by set order, which Python randomizes per process, so split verdicts flipped between runs.)
        kind = max(r["kinds"], key=lambda k: (r["kinds"].count(k), -r["kinds"].index(k)))
        r["kind"] = kind
        r["checks"] = {
            "distinct": r["accuracy"] >= 0.75 and r["q"] < 0.01,
            "nameable": r["consistency"] >= 6 and kind not in ("none", "degraded"),
            "intact": r["fluency"] >= (3 if kind == "altered_state" else 4),
            "specific": r["accuracy"] >= sham_acc90 and r["consistency"] >= sham_cons90,
            "reliable": r["identical"] <= r["pairs"] // 4 if r["pairs"] else False,
            # A second, independent judging round (fresh A/B order, fresh grouping) must also spot it.
            "replicated": all(x >= 0.7 for x in r["round_accuracy"]),
        }
        r["hit"] = all(r["checks"].values())
    summary = {}
    for origin in ("catalog", "sham", "known", "screen"):
        group = [r for r in rows if r["origin"] == origin]
        summary[origin] = {"judged": len(group), "hits": sum(r["hit"] for r in group),
                           "accuracy_mean": float(np.mean([r["accuracy"] for r in group])) if group else None}
    summary["sham_best_accuracy"] = sham_best
    summary["sham_p90_accuracy"], summary["sham_p90_consistency"] = sham_acc90, sham_cons90
    # How often the pipeline would call a random neuron a hit: shams passing every check except specificity itself.
    summary["sham_false_hits"] = sum(all(v for k, v in r["checks"].items() if k != "specific") for r in shams)
    rows.sort(key=lambda r: (-r["hit"], -r["interest"], -r["accuracy"]))
    (args.out / "hits.json").write_text(json.dumps({"summary": summary, "candidates": rows}, indent=1, ensure_ascii=False))
    singles = [{k: r[k] for k in ("key", "layer", "neuron", "sign", "dose")} for r in rows if r["hit"] and r["layer"] is not None]
    (args.out / "hit_list.json").write_text(json.dumps(singles, indent=1))  # input for drugs.py receptor
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
