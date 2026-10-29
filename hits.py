"""From blind judgments to hits: the research plan's five checks, applied with the controls reported first.

    .venv/bin/python hits.py        # runs/judge/tally.json + stage files -> runs/hits.json

A candidate is a hit when it passes all five checks on held-out prompts:
- Distinct: judges pick the altered answer far above chance. Binomial test against a guessing judge (p = 1/2), with
  Benjamini-Hochberg correction across every judged candidate (q < 0.01), and accuracy of at least 0.75.
- Nameable: judges rate the change consistent across prompts (consistency >= 6) and give it a kind other than
  "none" or "degraded".
- Intact: judged fluency >= 4 (>= 3 for an altered state, which is allowed to sound impaired).
- Specific: accuracy above every sham's (randomly chosen neurons pushed with the same rule).
- Reliable: the accuracy is over 20 pairs, ten held-out prompts each greedy and sampled, so no single lucky sample
  carries it; candidates whose altered and plain answers were identical on more than a quarter of pairs fail.
Controls: the 19 shipped personas (positive) and the shams (negative) go through exactly the same judging.
"""

import json

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
    tally = json.loads((RUNS / "judge" / "tally.json").read_text())
    survivors = {s["key"]: s for s in json.loads((RUNS / "discover" / "survivors.json").read_text())}
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
    sham_best = max([r["accuracy"] for r in rows if r["origin"] == "sham"], default=0.5)
    for r in rows:
        kind = max(set(r["kinds"]), key=r["kinds"].count)
        r["kind"] = kind
        r["checks"] = {
            "distinct": r["accuracy"] >= 0.75 and r["q"] < 0.01,
            "nameable": r["consistency"] >= 6 and kind not in ("none", "degraded"),
            "intact": r["fluency"] >= (3 if kind == "altered_state" else 4),
            "specific": r["accuracy"] > sham_best,
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
    rows.sort(key=lambda r: (-r["hit"], -r["interest"], -r["accuracy"]))
    (RUNS / "hits.json").write_text(json.dumps({"summary": summary, "candidates": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
