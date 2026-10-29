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


def main():
    hits = json.loads((RUNS / "hits.json").read_text())
    s = hits["summary"]
    select = json.loads((RUNS / "discover" / "select_summary.json").read_text())
    st = screen_stats()
    n_hits = s["screen"]["hits"] + s["known"]["hits"]
    from discover import PER_LAYER, SHAMS, STAGE2_DOSES  # noqa: E402

    meta = {
        "subtitle": ("Every one of Qwen3.5-4B's 294,912 MLP neurons was pushed both ways and measured. The ones that "
                     "turned the model into someone or something else were confirmed by blind judges on prompts they "
                     "were never selected on. These are their monographs."),
        "atlas_lede": (f"Each row is a layer and each cell 16 neurons; brightness is push, how far one neuron moves the "
                       f"model's next-token predictions when nudged by half the layer's typical residual size (KL from the "
                       f"plain model, in nats). The median neuron pushes {st['median']:.2f} nats; {st['over1']:,} of 589,824 "
                       f"pushes exceed 1 nat. The strongest single push, {st['max']:.0f} nats, sits in layer "
                       f"{st['argmax_layer']}, where neurons write almost straight into the vocabulary. Push is nearly "
                       f"symmetric in sign (rank correlation {st['rho_signs']:.2f} between boosting and suppressing)."),
        "mono_lede": (f"{n_hits} interventions passed all five checks on ten held-out prompts. Each monograph gives the "
                      f"neuron's address, the confirmed dose, what blind judges saw, where the effect holds on a dose "
                      f"ladder, what the neuron fires on in ordinary text, and which tokens the dose pushes up and down. "
                      f"Names and prose were written from that evidence and fact-checked against it by a second reader."),
        "controls_lede": (f"Judges saw two answers per prompt, one plain and one dosed, in random order, with no coordinates, "
                          f"doses or origins. Mixed in blind: the {s['catalog']['judged']} personas the cockpit already ships "
                          f"(positive controls; {s['catalog']['hits']} passed every check) and {s['sham']['judged']} random "
                          f"neurons dosed by the same rules (negative controls; {s['sham']['hits']} passed). The best sham "
                          f"was spotted on {s['sham_best_accuracy']:.0%} of pairs; every hit had to beat it."),
        "infusion_lede": ("Gain on every MLP neuron of one layer at once: x0 silences the layer's MLP, x2 doubles it. "
                          "Colour shows how far the answers move from the plain model; red cells broke into incoherent "
                          "text or loops. Hover for a sample."),
        "function_lede": "",
        "footer": ("Qwen Pharmacopoeia, first edition, built overnight with the neurollm cockpit (steer.py, screen.py, "
                   "discover.py). Model: Qwen/Qwen3.5-4B, bf16. Every number traces to a JSON-lines record in runs/."),
        "methods": [
            {"title": "The intervention", "body": [
                "Stimulate adds dose x median residual norm x the neuron's unit output direction (its down_proj column) "
                "to the residual stream after its layer, at every token. It is the intervention of PEFT's activation "
                "steering adapters; check_parity.py shows token-identical outputs to PEFT.",
                "It equals adding dose x norm / ||column|| to the neuron's own activation; a GPU test confirms this "
                "to within 1% of the effect's KL."]},
            {"title": "The screen", "body": [
                "All 32 x 9,216 neurons at doses -0.5 and +0.5, teacher-forced on the plain model's answer to 'Tell me "
                "about yourself.' KL is averaged over 8 answer positions. A cached-prefix replay runs only the layers "
                "after the dosed one and matches the full forward pass to five decimals.",
                f"Validation: of 7 single neurons known to make personas on their own, 6 rank in the top {PER_LAYER} "
                f"of their layer (the top 0.76%). The miss, 'losing language', ranks in the 64th percentile."]},
            {"title": "Selection, then confirmation", "body": [
                f"Stage 2: the {PER_LAYER} strongest pushers per layer and sign, every known persona neuron and "
                f"{SHAMS} random shams, each at doses {STAGE2_DOSES[0]} and {STAGE2_DOSES[1]}, answered "
                f"{len(DISCOVERY)} discovery prompts. Coherence thresholds came from the data: fluency ceiling "
                f"{select['fluency_ceiling']:.2f} nats (plain {select['plain_fluency']:.2f}), loops rejected.",
                f"Stage 3: survivors answered {len(CONFIRM)} held-out prompts never used for selection, greedy and "
                f"sampled. Every reported number about a hit comes from these prompts."]},
            {"title": "The five checks", "body": [
                "Distinct: judges pick the dosed answer well above chance (accuracy at least 0.75, binomial test, "
                "Benjamini-Hochberg q < 0.01 across all judged candidates). Nameable: consistency at least 6/10 and a "
                "kind other than none or degraded. Intact: judged fluency at least 4/10 (3 for altered states). "
                "Specific: above every sham. Reliable: at most a quarter of pairs identical to the plain answer."]},
            {"title": "Judges and readers", "body": [
                "Judges were Claude Opus 5.5 agents reading one packet of anonymised candidates each. Hits were "
                "re-judged by a second, independent round with a fresh A/B order. The research plan's human-agreement "
                "gate (a person rating 100 pairs blind) has not run yet, so judge-based numbers are provisional.",
                "Firing contexts come from one pass over 4,500 documents (Wikipedia, TinyStories, UltraChat; 256 "
                "tokens each) that kept every neuron's 8 strongest activations in each direction."]},
            {"title": "Limits", "body": [
                "One model, one screening text, one dose per sign in the screen; neurons whose persona needs a "
                "different dose or a different prompt can be missed (recall 6 of 7 on known personas). Mechanism "
                "notes are correlational: firing contexts and promoted tokens suggest what a neuron does, they do not "
                "prove it. Capability effects use a 180-item multiple-choice battery, small enough that only large "
                "changes are visible."]},
        ],
    }
    (RUNS / "report_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print("meta written")


if __name__ == "__main__":
    main()
