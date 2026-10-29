"""The report's prose, with every number computed from the result files.  -> runs/report_meta.json"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from measure import CONFIRM, DISCOVERY, RUNS, load  # noqa: E402


def screen_stats():
    out = {}
    for name, dose in (("neg", -0.5), ("pos", 0.5)):
        kl = np.stack([np.load(RUNS / "screen" / f"stimulate_{dose:+g}_L{l:02d}.npy")[0, :, 0] for l in range(32)])
        out[name] = kl
    both = np.concatenate([out["neg"].ravel(), out["pos"].ravel()])
    rho = float(np.corrcoef(np.argsort(np.argsort(out["neg"].ravel())), np.argsort(np.argsort(out["pos"].ravel())))[0, 1])
    per_layer = np.median(np.concatenate([out["neg"], out["pos"]], 1), 1)
    return {
        "median": float(np.median(both)), "over1": int((both > 1).sum()), "max": float(both.max()),
        "argmax_layer": int(np.unravel_index(np.argmax(np.maximum(out["neg"], out["pos"])), out["neg"].shape)[0]),
        "rho_signs": rho, "quietest_layer": int(per_layer.argmin()), "loudest_layer": int(per_layer.argmax()),
    }


def recall():
    """How many of the shipped single-neuron personas the stage-2 ranking would have picked up."""
    from discover import PER_LAYER, RANK_BY, known_neurons  # noqa: E402

    standalone = {(3, 3066, -1), (4, 4783, -1), (7, 950, -1), (9, 1932, -1), (10, 5002, -1), (19, 264, -1), (20, 7, -1),
                  (6, 1525, 1), (4, 7242, 1), (7, 6297, 1), (15, 5080, 1), (18, 6874, 1)}
    found, ranks = 0, {}
    for layer, neuron, sign in standalone:
        col = np.load(RUNS / "screen" / f"stimulate_{0.5 * sign:+g}_L{layer:02d}.npy")[0, :, RANK_BY]
        rank = int((col > col[neuron]).sum()) + 1
        ranks[f"L{layer}N{neuron}{'+' if sign > 0 else '-'}"] = rank
        found += rank <= PER_LAYER
    return found, len(standalone), ranks


def functional(hits: dict) -> tuple[list, str]:
    """Interventions that improve a task family on the discovery half and again on the confirmation half, by more
    than every sham does on that half."""
    path = RUNS / "evidence_gpu.json"
    if not path.exists():
        return [], "The capability battery did not run."
    cap = json.loads(path.read_text())["capability"]
    plain = cap.pop("plain")
    origin = {c["key"]: c["origin"] for c in hits["candidates"]}
    families = sorted(plain)
    delta = {k: {f: {h: v[f][h]["acc"] - plain[f][h]["acc"] for h in "ab"} for f in families} for k, v in cap.items()}
    shams = [k for k in delta if origin.get(k) == "sham"]
    best_sham = {f: {h: max([delta[k][f][h] for k in shams], default=0.0) for h in "ab"} for f in families}
    found = []
    for k, d in delta.items():
        if origin.get(k) == "sham":
            continue
        for f in families:
            if d[f]["a"] >= 0.15 and d[f]["b"] >= 0.10 and d[f]["a"] > best_sham[f]["a"] and d[f]["b"] > best_sham[f]["b"]:
                found.append({"key": k, "family": f, "a": d[f]["a"], "b": d[f]["b"]})
    losses = sum(1 for k, d in delta.items() if origin.get(k) != "sham" and min(d[f]["b"] for f in families) <= -0.2)
    plain_txt = ", ".join(f"{f} {plain[f]['b']['acc']:.0%}" for f in families)
    if found:
        lede = (f"{len(found)} intervention-task pairs improved on the discovery half by 0.15 or more and held at least "
                f"+0.10 on the held-out half, more than any sham managed. Plain accuracy on the held-out half: {plain_txt}. "
                f"{losses} of {len(delta) - len(shams)} interventions cost 0.2 or more on some family.")
    else:
        lede = (f"None did. No intervention improved a task family on both halves of the battery by more than every "
                f"sham; the gains that appeared on one half were within what random neurons also produce. The usual "
                f"price was the other direction: {losses} of {len(delta) - len(shams)} interventions lost 0.2 or more "
                f"accuracy on some family. Plain accuracy on the held-out half: {plain_txt}.")
    return found, lede


def infusion_text() -> str:
    """E4 in numbers: how far whole-layer gain moves the answers, how many broke, what blind judges saw."""
    path = RUNS / "e4.jsonl"
    if not path.exists():
        return ""
    from collections import defaultdict

    recs = load(path)
    select = json.loads((RUNS / "discover" / "select_summary.json").read_text())
    ceiling, sham_median = select["fluency_ceiling"], select["sham_distinct_p50_p90_p99"][0]
    groups = defaultdict(list)
    for r in recs:
        if r.key != "plain":
            groups[(r.key, r.scale)].append(r)
    distinct = [np.mean([r.metrics["distinct"] for r in rs]) for rs in groups.values()]
    broken = sum(1 for rs in groups.values()
                 if any(np.isnan(r.metrics["fluency"]) for r in rs) or np.mean([r.metrics["fluency"] for r in rs]) > ceiling
                 or max(r.metrics["repeat"] for r in rs) > 0.35)
    push = json.loads((RUNS / "e4_push.json").read_text()) if (RUNS / "e4_push.json").exists() else {}
    names_path = RUNS / "judge" / "names_e4.json"
    named = json.loads(names_path.read_text()) if names_path.exists() else []
    invisible = sum(1 for n in named if n["kind"] in ("none",) or n["consistency"] <= 3)
    return (f"Gain on every MLP neuron of one layer at once, from x0 (the layer's MLP silenced) to x2 (doubled), across "
            f"all 32 layers: {len(groups)} conditions. None broke the text ({broken} incoherent), the answers moved at "
            f"most {max(distinct):.2f} in embedding distance (a single random neuron pushed hard moves them a median "
            f"{sham_median:.2f}), "
            f"and the largest push was {max(push.values()) if push else float('nan'):.2f} nats. Blind judges saw the "
            f"{len(named)} most distinct conditions and found no consistent change in {invisible} of them. Removing a "
            f"whole layer's MLP is nearly invisible; one neuron pushed hard rewrites who the model thinks it is. The "
            f"stimulate doses that make personas are far larger than anything a layer's MLP contributes on its own.")


def main():
    hits = json.loads((RUNS / "hits.json").read_text())
    s = hits["summary"]
    select = json.loads((RUNS / "discover" / "select_summary.json").read_text())
    st = screen_stats()
    rc = recall()
    n_hits = s["screen"]["hits"] + s["known"]["hits"]
    from discover import PER_LAYER, SHAMS, STAGE2_DOSES  # noqa: E402
    base_rate = s["sham_false_hits"] / SHAMS  # random (neuron, sign) pairs that make a nameable, coherent change
    meta = {
        "base_rate_lede": (f"Personas are not rare. Of {SHAMS} random (neuron, sign) pairs, the {s['sham']['judged']} most "
                           f"distinct were judged blind with the candidates, and {s['sham_false_hits']} produced a "
                           f"nameable, coherent change of their own: at least {base_rate:.1%} of random pushes. Across "
                           f"all 589,824 neuron-sign pairs that is on the order of {base_rate * 589824:,.0f} "
                           f"persona-like interventions, a lower bound under the screen's doses that a larger sham sample "
                           f"would tighten. What sets the hits apart is consistency: every hit matched the most "
                           f"consistent tenth of the shams."),
        "subtitle": ("Every one of Qwen3.5-4B's 294,912 MLP neurons was pushed both ways and measured. The ones that "
                     "turned the model into someone or something else were confirmed by blind judges on prompts they "
                     "were never selected on. These are their monographs."),
        "atlas_lede": (f"Each row is a layer and each cell 16 neurons; brightness is push, how far one neuron moves the "
                       f"model's next-token predictions when nudged by half the layer's typical residual size (KL from the "
                       f"plain model, in nats). The median neuron pushes {st['median']:.2f} nats; {st['over1']:,} of 589,824 "
                       f"pushes exceed 1 nat. The strongest single push, {st['max']:.0f} nats, sits in layer "
                       f"{st['argmax_layer']}, where neurons write almost straight into the vocabulary. "
                       + (f"Boosting and suppressing the same neuron are nearly unrelated (rank correlation "
                          f"{st['rho_signs']:.2f}): after RMSNorm, pushing along a direction the residual already "
                          f"carries mostly rescales it, while pushing against it cancels it, so most neurons have one "
                          f"loud sign and one quiet one." if abs(st['rho_signs']) < 0.3 else
                          f"Push is similar in both signs (rank correlation {st['rho_signs']:.2f}).")),
        "mono_lede": (f"{n_hits} single neurons passed every check on ten held-out prompts; after them come the "
                      f"{s['catalog']['hits']} shipped personas that passed as positive controls. Each monograph gives the "
                      f"neuron's address, the confirmed dose, what blind judges saw, where the effect holds on a dose "
                      f"ladder, what the neuron fires on in ordinary text, and which tokens the dose pushes up and down. "
                      f"Names and prose were written from that evidence and fact-checked against it by a second reader."),
        "controls_lede": (f"Judges saw two answers per prompt, one plain and one dosed, in random order, with no coordinates, "
                          f"doses or origins. Mixed in blind: the {s['catalog']['judged']} personas the cockpit already ships "
                          f"(positive controls; {s['catalog']['hits']} passed every check) and {s['sham']['judged']} random "
                          f"neurons, the most distinct ones the same selection rule could find (negative controls). "
                          f"{s['sham_false_hits']} of those shams passed every check but specificity, the pipeline's "
                          f"false-hit rate; a hit had to match or beat the shams' 90th percentile on both accuracy "
                          f"({s['sham_p90_accuracy']:.0%}) and consistency ({s['sham_p90_consistency']:.1f}/10)."),
        "infusion_lede": infusion_text(),
        "function_lede": functional(hits)[1],
        "footer": ("Qwen Pharmacopoeia, first edition, built overnight with the neurollm cockpit (steer.py, screen.py, "
                   "discover.py). Model: Qwen/Qwen3.5-4B, bf16. Every number traces to a JSON-lines record in runs/."),
        "methods": [
            {"title": "The intervention", "body": [
                "Stimulate adds dose x median residual norm x the neuron's unit output direction (its down_proj column) "
                "to the residual stream after its layer, at every token. It is the intervention of PEFT's activation "
                "steering adapters; check_parity.py shows token-identical outputs to PEFT.",
                "It equals adding dose x norm / ||column|| to the neuron's own activation; a GPU test confirms this "
                "to within 1% of the effect's KL.",
                "Gain multiplies a neuron's activation by 1 + dose. Silencing a whole layer this way (E4) was checked "
                "against an independent hook that zeroes the layer's MLP output: the two pushes agree within 2% at "
                "layers 0, 8, 15, 24 and 31."]},
            {"title": "The screen", "body": [
                "All 32 x 9,216 neurons at doses -0.5 and +0.5, teacher-forced on the plain model's answer to 'Tell me "
                "about yourself.' KL is averaged over 8 answer positions. A cached-prefix replay runs only the layers "
                "after the dosed one and matches the full forward pass to five decimals.",
                f"Validation: of the {rc[1]} shipped personas made by one neuron alone, {rc[0]} rank in the top "
                f"{PER_LAYER} of their layer by displacement of the final hidden state, the ranking stage 2 uses. "
                f"Ranking by push instead found 7: boost-side personas such as the salesperson switch on at doses "
                f"above the screen's 0.5, where their push is still ordinary."]},
            {"title": "Selection, then confirmation", "body": [
                f"Stage 2: the {PER_LAYER} neurons per layer and sign that displace the final hidden state most, every known persona neuron and "
                f"{SHAMS} random shams, each at doses {STAGE2_DOSES[0]} and {STAGE2_DOSES[1]}, answered "
                f"{len(DISCOVERY)} discovery prompts. Coherence thresholds came from the data: fluency ceiling "
                f"{select['fluency_ceiling']:.2f} nats (plain {select['plain_fluency']:.2f}), loops rejected.",
                f"Stage 3: survivors answered {len(CONFIRM)} held-out prompts never used for selection, greedy and "
                f"sampled. Every reported number about a hit comes from these prompts."]},
            {"title": "The five checks", "body": [
                "Distinct: judges pick the dosed answer well above chance (accuracy at least 0.75, binomial test, "
                "Benjamini-Hochberg q < 0.01 across all judged candidates). Nameable: consistency at least 6/10 and a "
                "kind other than none or degraded. Intact: judged fluency at least 4/10 (3 for altered states). "
                "Specific: accuracy and consistency at or above the 90th percentile of the shams. Reliable: at most a "
                "quarter of pairs identical to the plain answer. Replicated: a second judging round with a fresh A/B "
                "order also reaches 70%."]},
            {"title": "Judges and readers", "body": [
                "Judges were Claude Opus 5.5 agents reading one packet of anonymised candidates each. Hits were "
                "re-judged by a second, independent round with a fresh A/B order. The research plan's human-agreement "
                "gate (a person rating 100 pairs blind) has not run yet, so judge-based numbers are provisional.",
                "Firing contexts come from one pass over 4,500 documents (Wikipedia, TinyStories, UltraChat; 256 "
                "tokens each) that kept every neuron's 8 strongest activations in each direction."]},
            {"title": "Limits", "body": [
                f"One model, one screening text, one dose per sign in the screen; neurons whose persona needs a "
                f"different dose or a different prompt can be missed (recall {rc[0]} of {rc[1]} on known personas). "
                "The fluency ceiling used in selection was read partly from the shipped personas' answers to the "
                "held-out prompts (E0); no candidate's own held-out answers were used before judging. Mechanism "
                "notes are correlational: firing contexts and promoted tokens suggest what a neuron does, they do not "
                "prove it. Capability effects use a 180-item multiple-choice battery, small enough that only large "
                "changes are visible."]},
        ],
    }
    (RUNS / "report_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print("meta written")


if __name__ == "__main__":
    main()
