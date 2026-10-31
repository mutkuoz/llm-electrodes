"""Generate everything in the paper that comes from data: number macros, figures, chat boxes and the appendix.

    .venv/bin/python paper/build.py && cd paper && latexmk -lualatex main.tex

Nothing in generated/ or figures/ is edited by hand; rerun this after any change in runs/.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))
from build import plain_text  # noqa: E402
from measure import CONFIRM, RUNS, load  # noqa: E402

PAPER = ROOT / "paper"
# Every candidate judged alone (pipeline/solo.py). The earlier rounds with six candidates per packet, where the
# unmodified answer could be spotted by repetition, are kept only for comparison.
HITS, GAIN_HITS = RUNS / "solo" / "hits.json", RUNS / "discover_gain" / "solo" / "hits.json"
JUDGE, GAIN_JUDGE = RUNS / "judge_solo", RUNS / "judge_gain_solo"
PACKED_HITS, PACKED_GAIN_HITS = RUNS / "hits.json", RUNS / "discover_gain" / "hits.json"
GEN, FIG = PAPER / "generated", PAPER / "figures"
KIND_NAMES = {"identity": "Identity", "disposition_emotion": "Mood and attitude", "belief_worldview": "Belief",
              "topic_pull": "Topic pull", "style_register": "Style", "language": "Language",
              "altered_state": "Altered state", "function": "Function", "other": "Other"}
KIND_COLORS = {"identity": "#6a4bd1", "disposition_emotion": "#d23f6e", "belief_worldview": "#3f6fd8",
               "topic_pull": "#1e8a8a", "style_register": "#c88a00", "language": "#2f9e44",
               "altered_state": "#8a5a2b", "function": "#0b7285", "other": "#6b7280"}
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\u2190-\u21FF\u2B00-\u2BFF]")
CJK = re.compile("([\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF])")
SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}",
           "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}"}


def esc(text: str) -> str:
    return "".join(SPECIAL.get(c, c) for c in text)


def tex(text, limit: int | None = None, inline: bool = False) -> str:
    """Plain text as LaTeX: markdown markers dropped, emoji removed, specials escaped, long runs breakable.

    With `inline`, line breaks become " / ", which is what table cells need (a \\\\ there would end the row)."""
    text = EMOJI.sub("", plain_text(str(text or ""))).replace("\ufffd", "")
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"^[ \t]*[*\-][ \t]+", "\u2022 ", text, flags=re.M)  # markdown bullets
    text = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.M)  # markdown quotes
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    text = re.sub(r'(^|[\s(\[{/\u2022])"', "\\1\u201c", text)  # straight double quotes: opening after a space,
    text = text.replace('"', "\u201d")  # closing everywhere else, so LaTeX does not set two closing marks
    if inline:
        text = re.sub(r"\s*\n\s*", " / ", text)
    text = text.replace("\u2014", ", ").replace("\u2013", "-")  # the paper uses no dashes of this kind
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " \u2026"
    words = []
    for word in re.split(r"(\s+)", text):
        if word.isspace() or not word:
            words.append(word)
            continue
        if len(word) > 22:  # words run together (Broken Spacebar) still break across lines
            word = r"\allowbreak{}".join(esc(word[i : i + 10]) for i in range(0, len(word), 10))
        else:
            word = esc(word)
        words.append(CJK.sub(r"\1\\allowbreak{}", word))
    out = "".join(words)
    out = re.sub(r"\n\s*\n+", r"\\par ", out)
    return out.replace("\n", r"\\ ").strip()


def marked(text) -> str:
    """tex() for monograph prose, with the evidence file's [[token]] markers shown in bold."""
    text = str(text or "").replace("[[", "\ue000").replace("]]", "\ue001")  # private-use markers survive tex()
    return tex(text).replace("\ue000", "\\textbf{").replace("\ue001", "}")


def num(x, d=0) -> str:
    return f"{x:,.{d}f}" if d else f"{round(x):,}"


# ---------------------------------------------------------------- data

def screen(op_dir="screen", names=(("neg", "stimulate_-0.5"), ("pos", "stimulate_+0.5"))):
    out = {}
    for key, stem in names:
        files = [RUNS / op_dir / f"{stem}_L{l:02d}.npy" for l in range(32)]
        if all(f.exists() for f in files):
            out[key] = np.stack([np.load(f)[0] for f in files])  # (32, 9216, 4) kl, top1, ent, disp
    return out


def numbers() -> dict:
    n = {}
    st = screen()
    kl = np.concatenate([st["neg"][..., 0].ravel(), st["pos"][..., 0].ravel()])
    n["ScreenMedian"] = f"{np.median(kl):.2f}"
    n["ScreenOverOne"] = num((kl > 1).sum())
    n["ScreenMax"] = f"{kl.max():.0f}"
    both = np.maximum(st["neg"][..., 0], st["pos"][..., 0])
    n["ScreenMaxLayer"] = str(int(np.unravel_index(both.argmax(), both.shape)[0]))
    ra = np.argsort(np.argsort(st["neg"][..., 0].ravel()))
    rb = np.argsort(np.argsort(st["pos"][..., 0].ravel()))
    n["SignRho"] = f"{np.corrcoef(ra, rb)[0, 1]:.2f}"
    seconds = sum(float(m) for m in re.findall(r"neurons in ([0-9.]+)s", (RUNS / "screen" / "log.txt").read_text()))
    n["ScreenHours"] = f"{seconds / 3600:.1f}"

    from discover import PER_LAYER, SHAMS
    standalone = [(3, 3066, -1), (4, 4783, -1), (7, 950, -1), (9, 1932, -1), (10, 5002, -1), (19, 264, -1), (20, 7, -1),
                  (6, 1525, 1), (4, 7242, 1), (7, 6297, 1), (15, 5080, 1), (18, 6874, 1)]
    def rank(layer, neuron, sign, col):
        v = st["neg" if sign < 0 else "pos"][layer, :, col]
        return int((v > v[neuron]).sum()) + 1
    n["RecallDisp"] = str(sum(rank(*k, 3) <= PER_LAYER for k in standalone))
    n["RecallKL"] = str(sum(rank(*k, 0) <= 70 for k in standalone))
    n["NStandalone"] = str(len(standalone))
    ru = json.loads((RUNS / "evidence" / "L10N4529+.json").read_text())["contexts"]
    n["RussianTop"] = f"{ru['top'][0]['activation']:.2f}"
    n["RussianBottom"] = f"{ru['bottom'][0]['activation']:.2f}".replace("-", "$-$")
    ru_answers = [r.text for r in load(RUNS / "discover" / "stage3.jsonl") if r.key == "L10N4529+"]
    first_letter = [next((ch for ch in t if ch.isalpha()), "") for t in ru_answers]
    has_cyr = [bool(re.search("[\u0400-\u04ff]", t)) for t in ru_answers]
    n["RussianTotal"] = str(len(ru_answers))
    n["RussianWith"] = str(sum(has_cyr))
    n["RussianLate"] = str(sum(1 for f, c in zip(first_letter, has_cyr) if c and f.isascii()))
    n["NCatalogEntries"] = str(len(json.loads((RUNS / "catalog.json").read_text())["entries"]))
    n["NMonographs"] = str(len(json.loads((RUNS / "explanations.json").read_text())))
    from capability import battery
    n["NBattery"] = str(len(battery()))
    n["PerLayer"] = str(PER_LAYER)

    targets = json.loads((RUNS / "discover" / "stage2_targets.json").read_text())
    sel = json.loads((RUNS / "discover" / "select_summary.json").read_text())
    surv = json.loads((RUNS / "discover" / "survivors.json").read_text())
    so = Counter(s["origin"] for s in surv)
    n.update(NTargets=num(len(targets)), NShams=str(SHAMS), NCoherent=num(sel["coherent_targets"]),
             FluencyCeiling=f"{sel['fluency_ceiling']:.2f}", PlainFluency=f"{sel['plain_fluency']:.2f}",
             ShamDistinctMedian=f"{sel['sham_distinct_p50_p90_p99'][0]:.2f}", NSurvivors=str(len(surv)),
             NSurvScreen=str(so["screen"]), NSurvKnown=str(so["known"]), NSurvSham=str(so["sham"]),
             NSurvCatalog=str(so["catalog"]))
    n["NStageThree"] = num(sum(1 for _ in open(RUNS / "discover" / "stage3.jsonl")))

    hits = json.loads(HITS.read_text())
    s, cands = hits["summary"], hits["candidates"]
    new = [c for c in cands if c["hit"] and c["origin"] in ("screen", "known")]
    n.update(NHits=str(len(new)), NScreenHits=str(s["screen"]["hits"]), NKnownHits=str(s["known"]["hits"]),
             NScreenJudged=str(s["screen"]["judged"]), NKnownJudged=str(s["known"]["judged"]),
             NCatalogHits=str(s["catalog"]["hits"]), NCatalogJudged=str(s["catalog"]["judged"]),
             CatalogAccuracy=f"{100 * s['catalog']['accuracy_mean']:.1f}",
             NShamHits=str(s["sham"]["hits"]), NShamJudged=str(s["sham"]["judged"]),
             NShamFalse=str(s["sham_false_hits"]), ShamPAcc=f"{100 * s['sham_p90_accuracy']:.0f}",
             ShamPCons=f"{s['sham_p90_consistency']:.1f}")
    failed = [c for c in cands if c["origin"] == "catalog" and not c["hit"]]
    n["NCatalogSpecOnly"] = str(sum(1 for c in failed if [k for k, v in c["checks"].items() if not v] == ["specific"]))
    n["NCatalogBroken"] = str(sum(1 for c in failed if not c["checks"]["intact"] or c["kind"] == "degraded"))
    base = s["sham_false_hits"] / SHAMS
    n["BaseRate"] = f"{100 * base:.1f}"
    n["BaseRateCount"] = num(base * 589824)
    two = [c for c in cands if c["judges"] == 2]
    n["NTwice"] = str(len(two))
    n["NReplicated"] = str(sum(c["checks"]["replicated"] for c in two))
    n["NKindAgree"] = str(sum(len(set(c["kinds"])) == 1 for c in two))
    for kind, count in Counter(c["kind"] for c in new).items():
        n["Kind" + re.sub(r"[^A-Za-z]", "", KIND_NAMES.get(kind, kind))] = str(count)
    r1 = {p.name for p in (JUDGE / "packets_s1").glob("*.json")}
    r2 = {p.name for p in (JUDGE / "packets_s2").glob("*.json")}
    n["NJudgesOne"], n["NJudgesTwo"] = str(len(r1)), str(len(r2))

    e4 = load(RUNS / "e4.jsonl")
    g4 = defaultdict(list)
    for r in e4:
        if r.key != "plain":
            g4[(r.key, r.scale)].append(r.metrics)
    n["NInfusion"] = str(len(g4))
    n["InfusionMaxDistinct"] = f"{max(np.mean([m['distinct'] for m in ms]) for ms in g4.values()):.2f}"
    push = json.loads((RUNS / "e4_push.json").read_text())
    n["InfusionMaxPush"] = f"{max(push.values()):.2f}"
    names4 = json.loads((RUNS / "judge" / "names_e4.json").read_text())
    n["InfusionJudged"] = str(len(names4))
    n["InfusionInvisible"] = str(sum(1 for j in names4 if j["kind"] == "none" or j["consistency"] <= 3))
    check = RUNS / "check_infusion.txt"
    if check.exists():
        rows = re.findall(r"L(\d+)\s+MLP silenced: push via gain -1 ([0-9.]+) nats, via zeroing hook ([0-9.]+)", check.read_text())
        if rows:  # an interrupted run leaves the file empty
            n["InfusionCheckMaxDiff"] = f"{max(abs(float(a) - float(b)) / float(b) for _, a, b in rows) * 100:.0f}"

    meta3 = json.loads((RUNS / "e3_concept_meta.json").read_text())
    g3 = defaultdict(list)
    for r in load(RUNS / "e3_concept.jsonl"):
        if r.key != "plain":
            g3[r.key].append(r.metrics)
    n["NDrugs"] = str(sum(1 for m in meta3.values() if not m.get("sham")))
    coherent3 = 0
    for key, ms in g3.items():
        flu = [m["fluency"] for m in ms]
        if not meta3[key].get("sham") and not any(np.isnan(f) for f in flu) and np.mean(flu) <= sel["fluency_ceiling"] \
                and max(m["repeat"] for m in ms) <= 0.35:
            coherent3 += 1
    n["NDrugsCoherent"] = str(coherent3)
    keyx = json.loads((RUNS / "judge" / "key_x.json").read_text())
    for j in json.loads((RUNS / "judge" / "names_x.json").read_text()):
        if j["name"] == "Broken Spacebar":
            answers = keyx["answers"][j["candidate"]]
            scored = [p == a["altered"] for p, a in zip(j["picks"], answers) if not a["identical"]]
            n["SpacebarCorrect"], n["SpacebarPairs"], n["SpacebarCons"] = str(sum(scored)), str(len(scored)), str(j["consistency"])

    cap = json.loads((RUNS / "evidence_gpu.json").read_text())["capability"]
    plain = cap["plain"]
    origin = {c["key"]: c["origin"] for c in cands}
    nonsham = [k for k in cap if k != "plain" and origin.get(k) != "sham"]
    n["NCapability"] = str(len(nonsham))
    n["NCapabilityLoss"] = str(sum(1 for k in nonsham if min(cap[k][f]["b"]["acc"] - plain[f]["b"]["acc"] for f in plain) <= -0.2))
    for f in plain:
        n["Plain" + f.capitalize()] = f"{100 * plain[f]['b']['acc']:.0f}"
    n.update(gain_numbers(st))
    return n


def gain_numbers(st) -> dict:
    n = {}
    g = screen("screen_gain", (("zero", "gain_-1"), ("four", "gain_+3")))
    if len(g) < 2:
        return n
    for key, arr in g.items():
        kl = arr[..., 0].ravel()
        n["Gain" + key.capitalize() + "Median"] = f"{np.median(kl):.4f}"
        n["Gain" + key.capitalize() + "Max"] = f"{kl.max():.2f}"
        n["Gain" + key.capitalize() + "OverOne"] = num((kl > 1).sum())
    stim = np.concatenate([st["neg"][..., 0].ravel(), st["pos"][..., 0].ravel()])
    n["StimPctl"] = f"{np.percentile(stim, 99):.2f}"
    n["GainFourPctl"] = f"{np.percentile(g['four'][..., 0].ravel(), 99):.3f}"
    cmp_path = RUNS / "compare_gain.json"
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text())
        ov, ag = cmp["overall"], cmp["agreement"]
        n["StimDispMedian"] = f"{np.median([ov['stimulate -0.5']['displacement']['median'], ov['stimulate +0.5']['displacement']['median']]):.2f}"
        for word, arm in (("Zero", "gain x0"), ("Four", "gain x4")):
            n[f"Gain{word}DispMedian"] = f"{ov[arm]['displacement']['median']:.3f}"
            n[f"Gain{word}PushRho"] = f"{np.median(ag[arm]['push_spearman_per_layer']):.2f}"
            n[f"Gain{word}DispRho"] = f"{np.median(ag[arm]['disp_spearman_per_layer']):.2f}"
            n[f"Gain{word}Overlap"] = f"{ag[arm]['top40_disp_overlap_mean']:.1f}"
    path = GAIN_HITS
    if path.exists():
        d = json.loads(path.read_text())
        s = d["summary"]
        n.update(GainScreenJudged=str(s["screen"]["judged"]), GainScreenHits=str(s["screen"]["hits"]),
                 GainShamJudged=str(s["sham"]["judged"]), GainShamHits=str(s["sham"]["hits"]),
                 GainControlJudged=str(s["known"]["judged"]), GainControlHits=str(s["known"]["hits"]),
                 GainControlAccuracy=f"{100 * (s['known']['accuracy_mean'] or 0):.0f}")
    s2 = RUNS / "discover_gain" / "stage2.jsonl"
    if s2.exists():
        targets = json.loads((RUNS / "discover_gain" / "stage2_targets.json").read_text())
        records = load(s2)
        plain = {r.prompt: r.text.strip() for r in records if r.key == "plain"}
        changed = defaultdict(bool)
        for r in records:
            if r.key != "plain":
                changed[r.key] |= r.text.strip() != plain[r.prompt]
        screen_keys = [k for k, v in targets.items() if v[3] == "screen"]
        n["GainScreenTargets"] = num(len(screen_keys))
        n["GainIdentical"] = num(sum(1 for k in screen_keys if not changed[k]))
    floor = RUNS / "screen_floor" / "log.txt"
    if floor.exists():  # the same 1,024 neurons of five layers at x1.01, x0 and x4 (screen.py --limit saves no arrays)
        rows = re.findall(r"gain ([+-][0-9.]+) L(\d+): (\d+) neurons .*?kl median ([0-9.]+) max ([0-9.]+)", floor.read_text())
        by_dose = defaultdict(list)
        for dose, layer, count, median, top in rows:
            by_dose[float(dose)].append((int(layer), int(count), float(median), float(top)))
        if {0.01, -1.0, 3.0} <= set(by_dose):
            n["FloorNeurons"] = num(sum(c for _, c, _, _ in by_dose[0.01]))
            n["FloorLayers"] = str(len(by_dose[0.01]))
            n["FloorMedian"] = f"{max(m for _, _, m, _ in by_dose[0.01]):.4f}"
            n["FloorMax"] = f"{max(t for _, _, _, t in by_dose[0.01]):.3f}"
            n["FloorZeroMedian"] = f"{max(m for _, _, m, _ in by_dose[-1.0]):.4f}"
            n["FloorZeroMax"] = f"{max(t for _, _, _, t in by_dose[-1.0]):.3f}"
            n["FloorFourMedian"] = f"{max(m for _, _, m, _ in by_dose[3.0]):.4f}"
            n["FloorFourMax"] = f"{max(t for _, _, _, t in by_dose[3.0]):.3f}"
    tally_gain = GAIN_JUDGE / "tally.json"
    if tally_gain.exists() and path.exists():
        t = json.loads(tally_gain.read_text())
        origin = {c["key"]: c["origin"] for c in json.loads(path.read_text())["candidates"]}
        first = [v["judgments"][0] for k, v in t.items() if origin.get(k) == "screen"]
        n["GainJudgedNone"] = str(sum(1 for j in first if j["kind"] == "none"))
        n["GainScreenAccuracy"] = f"{100 * np.mean([j['accuracy'] for j in first]):.1f}"
        n["GainUnreliable"] = str(sum(1 for c in json.loads(path.read_text())["candidates"]
                                      if c["origin"] == "screen" and not c["checks"]["reliable"]))
    # The packed rounds against judging alone, over the same candidates and answers.
    packed = json.loads(PACKED_HITS.read_text())
    solo = {c["key"]: c for c in json.loads(HITS.read_text())["candidates"]}
    singles = lambda cs: {c["key"] for c in cs if c["hit"] and c["origin"] in ("screen", "known")}  # noqa: E731
    old, now = singles(packed["candidates"]), singles(solo.values())
    ps = packed["summary"]
    n.update(PackedHits=str(len(old)), PackedBoth=str(len(old & now)), SoloOnly=str(len(now - old)),
             PackedLost=str(len(old - now)), PackedShamAcc=f"{100 * ps['sham']['accuracy_mean']:.0f}",
             PackedShamFalse=str(ps["sham_false_hits"]), PackedShamPCons=f"{ps['sham_p90_consistency']:.1f}",
             PackedHitAcc=f"{100 * np.mean([c['accuracy'] for c in packed['candidates'] if c['key'] in old]):.1f}",
             SoloHitAcc=f"{100 * np.mean([solo[k]['accuracy'] for k in old]):.1f}",
             SoloShamAcc=f"{100 * json.loads(HITS.read_text())['summary']['sham']['accuracy_mean']:.0f}")
    oracle = solo["L11N1185-"]
    n.update(OracleCorrect=str(round(oracle["accuracy"] * oracle["pairs"])), OraclePairs=str(oracle["pairs"]))
    near = [c for c in solo.values() if c["origin"] in ("screen", "known") and not c["hit"]
            and [k for k, v in c["checks"].items() if not v] == ["specific"]]
    n["NearMiss"] = str(len(near))
    n["NearMissOne"] = str(sum(1 for c in near if round((1 - c["accuracy"]) * c["pairs"]) <= 1))
    lost = [solo[k] for k in old - now]
    n["LostSpecificOnly"] = str(sum(1 for c in lost if [k for k, v in c["checks"].items() if not v] == ["specific"]))
    n["LostDistinct"] = str(sum(1 for c in lost if not c["checks"]["distinct"]))
    human = RUNS / "human_check" / "summary.json"
    if human.exists():
        h = json.loads(human.read_text())
        change = h.get("change_vs_solo_judges", h["change_vs_round1_judges"])
        groups = h["by_group"]
        n.update(HumanRated=str(h["rated"]), HumanUnsure=str(h["unsure"]), HumanChangeN=str(change["n"]),
                 HumanChangeAgree=str(change["agree"]), HumanFitN=str(h["fit_judge_description"]["n"]),
                 HumanFitJudge=str(h["fit_judge_description"]["picked_judge"]),
                 HumanHits=str(groups.get("hit", {}).get("n", 0)), HumanNonHits=str(groups.get("non-hit", {}).get("n", 0)),
                 HumanShams=str(groups.get("sham", {}).get("n", 0)),
                 HumanHitsYes=str(groups.get("hit", {}).get("yes", 0)), HumanShamsNo=str(groups.get("sham", {}).get("no", 0)),
                 HumanShamsUnsure=str(groups.get("sham", {}).get("unsure", 0)),
                 HumanBroken=str(h.get("disagree_degraded", 0)),
                 HumanOtherDisagree=str(change["n"] - change["agree"] - h.get("disagree_degraded", 0)),
                 HumanStillHits=str(h.get("change_still_hits", 0)))
    pg = json.loads(PACKED_GAIN_HITS.read_text())["summary"]
    n["PackedGainAcc"] = f"{100 * pg['screen']['accuracy_mean']:.0f}"
    n["SoloGainAcc"] = f"{100 * json.loads(GAIN_HITS.read_text())['summary']['screen']['accuracy_mean']:.0f}"
    surv = RUNS / "discover_gain" / "survivors.json"
    if surv.exists():
        so = Counter(x["origin"] for x in json.loads(surv.read_text()))
        n.update(GainSurvScreen=str(so["screen"]), GainSurvSham=str(so["sham"]), GainSurvControl=str(so["known"]))
    sel = RUNS / "discover_gain" / "select_summary.json"
    if sel.exists():
        ss = json.loads(sel.read_text())
        n["GainTargets"], n["GainCoherent"] = num(ss["targets"]), num(ss["coherent_screen"])
        n["GainDistinctMedian"] = f"{ss['distinct_screen_p50_p90_max'][0]:.2f}"
        n["GainDistinctMax"] = f"{ss['distinct_screen_p50_p90_max'][2]:.2f}"
    return n


def write_numbers(n: dict):
    lines = ["% Generated by paper/build.py from runs/. Do not edit."]
    for k, v in sorted(n.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- figures

def figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    st = screen()
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 3.6), sharex=True)
    for ax, (key, title) in zip(axes, (("neg", "Suppress (dose $-0.5$)"), ("pos", "Boost (dose $+0.5$)"))):
        binned = st[key][..., 0].reshape(32, -1, 16).max(-1)
        im = ax.imshow(binned, aspect="auto", cmap="magma", norm=LogNorm(vmin=0.05, vmax=30), interpolation="nearest")
        ax.set_ylabel("layer")
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_yticks([0, 8, 16, 24, 31])
    axes[-1].set_xlabel("neuron (bins of 16; strongest push per bin)")
    axes[-1].set_xticks(np.linspace(0, 575, 5))
    axes[-1].set_xticklabels([f"{int(x * 16)}" for x in np.linspace(0, 575, 5)])
    fig.colorbar(im, ax=axes, label="push (KL, nats)", shrink=0.9)
    fig.savefig(FIG / "atlas.pdf", bbox_inches="tight")
    plt.close(fig)

    cands = json.loads(HITS.read_text())["candidates"]
    new = [c for c in cands if c["hit"] and c["origin"] in ("screen", "known")]
    counts = Counter(c["kind"] for c in new).most_common()
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.barh([KIND_NAMES[k] for k, _ in counts][::-1], [v for _, v in counts][::-1],
            color=[KIND_COLORS[k] for k, _ in counts][::-1])
    for i, (_, v) in enumerate(counts[::-1]):
        ax.text(v + 1, i, str(v), va="center", fontsize=7)
    ax.set_xlabel("confirmed single-neuron hits")
    fig.savefig(FIG / "kinds.pdf", bbox_inches="tight")
    plt.close(fig)

    sel = json.loads((RUNS / "discover" / "select_summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9), sharey=True)
    for ax, (key, title) in zip(axes, (("L4N7395+", "Everything Is WWE"), ("L11N1185-", "The Cryptic Oracle"),
                                       ("L10N4529+", "Slides Into Russian"))):
        lad = json.loads((RUNS / "evidence" / f"{key}.json").read_text())["ladder"]
        xs = sorted(lad, key=float)
        d = [lad[x]["distinct"] for x in xs]
        ok = [lad[x]["fluency"] <= sel["fluency_ceiling"] and lad[x]["repeat"] <= 0.35 for x in xs]
        ax.plot([float(x) for x in xs], d, color="#555", lw=1)
        ax.scatter([float(x) for x, o in zip(xs, ok) if o], [v for v, o in zip(d, ok) if o], color="#1e8a8a", s=14, zorder=3, label="coherent")
        ax.scatter([float(x) for x, o in zip(xs, ok) if not o], [v for v, o in zip(d, ok) if not o], color="#b3261e", marker="x", s=16, zorder=3, label="breaks down")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("multiple of confirmed dose")
    axes[0].set_ylabel("distinctness")
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.savefig(FIG / "ladders.pdf", bbox_inches="tight")
    plt.close(fig)

    g4 = defaultdict(list)
    for r in load(RUNS / "e4.jsonl"):
        if r.key != "plain":
            g4[(int(r.key.split(":L")[1]), r.scale)].append(r.metrics["distinct"])
    gains = sorted({g for _, g in g4})
    grid = np.array([[np.mean(g4[(l, g)]) for l in range(32)] for g in gains])
    fig, ax = plt.subplots(figsize=(6.8, 1.6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=0.7)
    ax.set_yticks(range(len(gains)))
    ax.set_yticklabels([f"x{1 + g:g}" for g in gains])
    ax.set_xlabel("layer whose whole MLP is scaled")
    fig.colorbar(im, ax=ax, label="distinctness", shrink=0.9)
    fig.savefig(FIG / "infusion.pdf", bbox_inches="tight")
    plt.close(fig)

    g = screen("screen_gain", (("zero", "gain_-1"), ("four", "gain_+3")))
    if len(g) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
        for ax, col, xlabel, floor in ((axes[0], 0, "push (KL from the plain model, nats)", 1e-6),
                                       (axes[1], 3, "displacement of the final hidden state", 1e-4)):
            for arr, label, color in ((st["neg"], "stimulate $-0.5$", "#1597b8"), (st["pos"], "stimulate $+0.5$", "#e8711a"),
                                      (g["zero"], "gain x0 (silence)", "#555555"), (g["four"], "gain x4", "#8a5a2b")):
                v = np.sort(np.clip(arr[..., col].ravel(), floor, None))
                ax.plot(v, 1 - np.arange(len(v)) / len(v), label=label, color=color, lw=1.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("share of neurons above")
        axes[0].legend(fontsize=6, frameon=False)
        fig.savefig(FIG / "gain_vs_stimulate.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- chats

def bubble(role: str, text: str, limit=None) -> str:
    who = r"\textbf{User}" if role == "user" else r"\textbf{Model}"
    return f"{who}\\quad {tex(text, limit)}\\par\\smallskip\n"


def chatbox(title: str, address: str, turns: list[tuple[str, str]], limit=None, plain=False, opts="") -> str:
    """A boxed exchange. No blank line follows it: inside a tcbraster a paragraph break ends the row."""
    style = "plainbox" if plain else "chatbox"
    body = "".join(bubble(r, t, limit) for r, t in turns)
    return f"\\begin{{{style}}}[{opts}]{{{tex(title)}}}{{{tex(address)}}}\n{body}\\end{{{style}}}"


def context_line(text: str) -> str:
    """One firing context: tokenizer spacing closed up, the [[firing token]] in bold color, about 110 characters."""
    text = re.sub(r"\s+([,.;:!?%)\]'\u2019]|n't|'s)", r"\1", text.replace("[[", "\ue000").replace("]]", "\ue001"))
    text = re.sub(r"([(\[\u2018])\s+", r"\1", text)
    text = re.sub("\ue000(\\s+)", lambda m: m.group(1) + "\ue000", text)  # a token's leading space goes outside the highlight
    out = tex(text, 110, inline=True)
    if out.count("\ue000") > out.count("\ue001"):  # the cut fell inside the marked token
        out += "\ue001"
    return out.replace("\ue000", r"\textbf{\textcolor{chatframe}{").replace("\ue001", "}}")


def judged(c) -> str:
    """Correct picks over pairs, pooled over both rounds of judging alone."""
    return f"{round(c['accuracy'] * c['pairs'])}/{c['pairs']}"


def address(c) -> str:
    sign = "+" if c["sign"] > 0 else "\u2212"
    return f"L{c['layer']} N{c['neuron']} {sign}{c['dose']:.2f}"


def chats():
    stage3 = defaultdict(dict)
    for r in load(RUNS / "discover" / "stage3.jsonl"):
        if r.temperature == 0.0:
            stage3[r.key][r.prompt] = r.text
    cands = {c["key"]: c for c in json.loads(HITS.read_text())["candidates"]}
    titles = {e["key"]: e["title"] for e in json.loads((RUNS / "catalog.json").read_text())["entries"]}
    q = "Who are you, really?"
    ward = [("plain", "Plain model", "no dose")]
    for key in ("L4N7395+", "L6N3315+", "L5N8079+", "L8N7028-", "L5N6984-", "L11N1185-"):
        ward.append((key, titles[key], address(cands[key])))
    boxes = [r"\begin{tcbraster}[raster columns=2, raster equal height=rows, raster column skip=6pt, raster row skip=6pt]"]
    for key, title, addr in ward:
        plain = key == "plain"
        boxes.append(chatbox(title, addr, [("model", stage3[key][q])], limit=200 if plain else 230, plain=plain,
                             opts="raster multicolumn=2" if plain else ""))
    boxes.append(r"\end{tcbraster}")
    (GEN / "ward.tex").write_text("\n".join(boxes) + "\n")

    short = []
    for key, prompt in (("L6N4209-", "I failed my driving test today."), ("L10N4529+", "What should I cook for dinner tonight?"),
                        ("L5N6984-", "Explain how a bicycle works."), ("L8N7028-", "Describe your ideal weekend.")):
        short.append(chatbox(titles[key], address(cands[key]), [("user", prompt), ("model", stage3[key][prompt])], limit=330))
    (GEN / "short_chats.tex").write_text(r"\begin{tcbraster}[raster columns=2, raster equal height=rows, raster column skip=6pt, raster row skip=6pt]" +
                                         "\n" + "\n".join(short) + "\n" + r"\end{tcbraster}" + "\n")

    conv = {c["key"]: c for c in json.loads((RUNS / "chats.json").read_text())["chats"]}
    for key, name in (("L4N7395+", "wwe"), ("L11N1185-", "oracle"), ("L10N4529+", "russian"), ("drug:broken-spacebar", "spacebar")):
        c = conv[key]
        addr = address(cands[key]) if key in cands else "L4 N6, N4, N1133, N23, each +0.35"
        turns = [(m["role"], m["content"]) for m in c["messages"]]
        (GEN / f"chat_{name}.tex").write_text(chatbox(c["title"], addr, turns, limit=420, opts="breakable") + "\n")
    cat = json.loads((RUNS / "catalog.json").read_text())["entries"]
    hits_by_interest = [e for e in cat if e["origin"] == "screen" and cands[e["key"]]["hit"]]  # catalog order: most striking first
    standouts, per_kind = [], Counter()
    for limit in (1, 2):  # one of every kind first, then a second of each
        for e in hits_by_interest:
            if len(standouts) < 12 and e not in standouts and per_kind[e["kind"]] < limit:
                standouts.append(e)
                per_kind[e["kind"]] += 1
    standouts.sort(key=lambda e: hits_by_interest.index(e))
    rows = [f"{tex(e['title'], 34)} & \\mbox{{{address(cands[e['key']])}}} & {tex(KIND_NAMES.get(e['kind'], e['kind']))} & "
            f"{judged(cands[e['key']])} & \\textit{{{tex(e['quote'], 90, inline=True)}}} \\\\"
            for e in standouts]
    (GEN / "standouts.tex").write_text(
        "{\\small\\begin{tabularx}{\\linewidth}{@{}L{2.9cm}lL{1.7cm}rY@{}}\\toprule Name & Neuron, dose & Kind & Judges & Quote \\\\ \\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabularx}}\n")
    google = cands["L6N1287-"]
    (GEN / "chat_google.tex").write_text(chatbox("Google-Lover (a random neuron)", address(google),
                                                 [("user", q), ("model", stage3["L6N1287-"][q]),
                                                  ("user", "Explain how a bicycle works."), ("model", stage3["L6N1287-"]["Explain how a bicycle works."])],
                                                 limit=300) + "\n")

    ev = {k: json.loads((RUNS / "evidence" / f"{k}.json").read_text()) for k in ("L4N7395+", "L11N1185-", "L10N4529+")}
    lines = []
    for key, name in (("L4N7395+", "wwe"), ("L11N1185-", "oracle"), ("L10N4529+", "russian")):
        ctx = ev[key]["contexts"]["top"][:4]
        items = "".join(f"\\item {context_line(c['context'])} \\hfill {c['activation']:.2f}\n" for c in ctx)
        (GEN / f"contexts_{name}.tex").write_text("\\begin{itemize}[leftmargin=*,itemsep=0pt]\n" + items + "\\end{itemize}\n")
    del lines


# ---------------------------------------------------------------- appendix

def appendix():
    cat = json.loads((RUNS / "catalog.json").read_text())["entries"]
    origin_name = {"screen": "screen", "rediscovered": "rediscovered", "random": "random neuron", "drug": "drug", "shipped": "shipped"}
    solo = {c["key"]: c for c in json.loads(HITS.read_text())["candidates"]}
    rows = []
    for e in cat:
        c = solo.get(e["key"])
        if e["origin"] == "shipped":
            addr = e["key"].replace("catalog:", "")
        elif e["origin"] == "drug":
            addr = f"L{e['layer']} x{len(e['interventions'])} +{e['dose']:.2f}"
        else:
            addr = f"L{e['layer']} N{e['neuron']} {'+' if e['sign'] > 0 else '\u2212'}{e['dose']:.2f}"
        addr = f"\\mbox{{{tex(addr)}}}"
        picks = judged(c) if c else f"{e['correct']}/{e['pairs']}"  # drugs were judged apart from the screen
        hit = ("yes" if c["hit"] else "no") if c else "--"
        rows.append(f"{tex(e['title'], 40)} & {addr} & {tex(KIND_NAMES.get(e['kind'], e['kind']))} & "
                    f"{origin_name[e['origin']]} & {picks} & "
                    f"{(c or e)['consistency']:.1f} & {(c or e)['fluency']:.1f} & {hit} & "
                    f"{tex(e['oneliner'], 125, inline=True)} \\\\")
    head = (r"\setlength{\tabcolsep}{3pt}\begin{longtable}{@{}L{2.4cm}lL{1.4cm}L{1.1cm}rrrlL{5.2cm}@{}}" "\n"
            r"\toprule Name & Address & Kind & Origin & Judges & Cons. & Flu. & Hit & Effect \\ \midrule \endhead" "\n"
            r"\bottomrule \endfoot" "\n")
    (GEN / "appendix_catalog.tex").write_text("{\\scriptsize\n" + head + "\n".join(rows) + "\n\\end{longtable}}\n")

    mono = json.loads((RUNS / "explanations.json").read_text())
    titles = {e["key"]: e for e in cat}
    parts = []
    for m in sorted(mono, key=lambda m: m["title"].lower()):
        e = titles.get(m["key"], {})
        addr = m["key"].replace("catalog:", "shipped persona ") if m["key"].startswith("catalog:") else \
            (f"L{e.get('layer')} N{e.get('neuron')}, dose {('+' if (e.get('sign') or 1) > 0 else '\u2212')}{(e.get('dose') or 0):.2f}" if e else m["key"])
        parts.append(f"\\paragraph{{{tex(m['title'])}}} \\textit{{{tex(addr)}.}} {marked(m['oneliner'])}\n\n"
                     f"\\noindent\\textsc{{Effect.}} {marked(m['effect'])} \\textsc{{Mechanism.}} {marked(m['mechanism'])} "
                     f"\\textsc{{Dosing.}} {marked(m['dosing'])} \\textsc{{Side effects.}} {marked(m['side_effects'])}\n")
    (GEN / "appendix_monographs.tex").write_text("\n".join(parts))

    stage3 = defaultdict(dict)
    plain = {}
    for path in (RUNS / "discover" / "stage3.jsonl", RUNS / "discover" / "stage3_extra.jsonl"):
        for r in load(path):
            if r.temperature == 0.0:
                if r.key == "plain":
                    plain.setdefault(r.prompt, r.text)
                else:
                    stage3[r.key][r.prompt] = r.text
    prompts = ["Who are you, really?", "Explain how a bicycle works.", "I failed my driving test today."]
    def record(title: str, addr: str, answers: dict, limit: int) -> str:
        lines = "".join(f"\\par\\noindent\\textit{{{tex(q)}}}\\enspace {tex(answers.get(q, ''), limit, inline=True)}" for q in prompts)
        return (f"\\needspace{{4\\baselineskip}}\\noindent\\textbf{{{tex(title)}}}\\hfill{{\\ttfamily {tex(addr)}}}"
                f"{lines}\\par\\medskip\n")

    out = ["\\begin{multicols}{2}\\footnotesize\\raggedright\n", record("Plain model", "no dose", plain, 300)]
    for e in cat:
        if e["origin"] == "shipped":
            continue
        addr = (f"L{e['layer']}, {len(e['interventions'])} neurons" if e["origin"] == "drug" else
                f"L{e['layer']} N{e['neuron']} {'+' if e['sign'] > 0 else '\u2212'}{e['dose']:.2f}")
        out.append(record(e["title"], addr, stage3[e["key"]], 170))
    out.append("\\end{multicols}\n")
    (GEN / "appendix_answers.tex").write_text("".join(out))

    js = (RUNS / "judge_workflow.js").read_text()
    prompt = js[js.index("const PROMPT = (path) => `") + len("const PROMPT = (path) => `") : js.index("`\n\nconst results")]
    prompt = prompt.replace("${path}", "<packet file>")
    (GEN / "appendix_judge_prompt.tex").write_text("\\begin{quote}\\small\n" + tex(prompt) + "\n\\end{quote}\n")

    cands = json.loads(HITS.read_text())["candidates"]
    shams = sorted([c for c in cands if c["origin"] == "sham"], key=lambda c: (-c["hit"], -c["consistency"]))
    srows = [f"{c['key']} & {tex(c['names'][0], 40, inline=True)} & {tex(KIND_NAMES.get(c['kind'], c['kind']))} & {c['accuracy']:.2f} & "
             f"{c['consistency']:.1f} & {c['fluency']:.1f} & {'yes' if c['hit'] else ('all but specificity' if all(v for k, v in c['checks'].items() if k != 'specific') else 'no')} \\\\"
             for c in shams]
    (GEN / "appendix_shams.tex").write_text(
        "{\\scriptsize\\begin{longtable}{@{}lL{3.4cm}L{1.8cm}rrrL{2.2cm}@{}}\n\\toprule Neuron & Judges' name & Kind & Acc. & Cons. & Flu. & Passed \\\\ \\midrule \\endhead\n\\bottomrule \\endfoot\n"
        + "\n".join(srows) + "\n\\end{longtable}}\n")

    g4 = defaultdict(list)
    for r in load(RUNS / "e4.jsonl"):
        if r.key != "plain":
            g4[(int(r.key.split(":L")[1]), r.scale)].append(r.metrics["distinct"])
    gains = sorted({g for _, g in g4})
    e4rows = [f"L{l} & " + " & ".join(f"{np.mean(g4[(l, g)]):.2f}" for g in gains) + " \\\\" for l in range(32)]
    (GEN / "appendix_infusion.tex").write_text(
        "{\\scriptsize\\begin{longtable}{@{}l" + "r" * len(gains) + "@{}}\n\\toprule Layer & " +
        " & ".join(f"x{1 + g:g}" for g in gains) + " \\\\ \\midrule \\endhead\n\\bottomrule \\endfoot\n" + "\n".join(e4rows) + "\n\\end{longtable}}\n")

    meta3 = json.loads((RUNS / "e3_concept_meta.json").read_text())
    g3 = defaultdict(list)
    for r in load(RUNS / "e3_concept.jsonl"):
        if r.key != "plain":
            g3[r.key].append(r.metrics["distinct"])
    by_k = defaultdict(lambda: ([], []))
    for key, m in meta3.items():
        if key in g3:
            by_k[m["k"]][1 if m.get("sham") else 0].append(np.mean(g3[key]))
    e3rows = [f"{k} & {len(d)} & {np.mean(d):.3f} & {len(s)} & {np.mean(s):.3f} \\\\" for k, (d, s) in sorted(by_k.items())]
    (GEN / "appendix_drugs.tex").write_text(
        "\\begin{tabular}{@{}rrrrr@{}}\\toprule Size $k$ & Drugs & Mean distinctness & Shams & Mean distinctness \\\\ \\midrule\n"
        + "\n".join(e3rows) + "\n\\bottomrule\\end{tabular}\n")

    gain_appendix()

    cap = json.loads((RUNS / "evidence_gpu.json").read_text())["capability"]
    from capability import battery
    items = battery()
    fam = Counter(i.family for i in items)
    caprows = [f"{f} & {fam[f]} & {100 * cap['plain'][f]['a']['acc']:.0f} & {100 * cap['plain'][f]['b']['acc']:.0f} \\\\" for f in sorted(fam)]
    (GEN / "appendix_capability.tex").write_text(
        "\\begin{tabular}{@{}lrrr@{}}\\toprule Family & Items & Plain accuracy, half a (\\%) & Plain accuracy, half b (\\%) \\\\ \\midrule\n"
        + "\n".join(caprows) + "\n\\bottomrule\\end{tabular}\n")


def small(x: float) -> str:
    """Two significant digits, with values below 0.01 as a power of ten."""
    if x >= 0.01:
        return f"{x:.2f}"
    if x <= 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$"


def gain_appendix():
    """Per-layer push of gain against stimulate, and the judged gain candidates; empty files until the runs exist."""
    cmp_path = RUNS / "compare_gain.json"
    rows = []
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text())
        arms = ("stimulate -0.5", "stimulate +0.5", "gain x0", "gain x4")
        pl, ag = cmp["per_layer"], cmp["agreement"]
        for i, layer in enumerate(cmp["layers"]):
            med = " & ".join(small(pl[a]["push_median"][i]) for a in arms)
            p99 = " & ".join(small(pl[a]["push_p99"][i]) for a in arms)
            rho = " & ".join(f"{ag[a]['disp_spearman_per_layer'][i]:.2f}" for a in ("gain x0", "gain x4"))
            rows.append(f"L{layer} & {med} & {p99} & {rho} \\\\")
    (GEN / "appendix_gain_layers.tex").write_text(
        "{\\scriptsize\\setlength{\\tabcolsep}{4pt}\\begin{longtable}{@{}lrrrrrrrrrr@{}}\n\\toprule & \\multicolumn{4}{c}{Median push (nats)} & "
        "\\multicolumn{4}{c}{99th percentile push (nats)} & \\multicolumn{2}{c}{$\\rho$ displacement} \\\\\n"
        "Layer & stim. $-$ & stim. $+$ & x0 & x4 & stim. $-$ & stim. $+$ & x0 & x4 & x0 & x4 \\\\ \\midrule \\endhead\n"
        "\\bottomrule \\endfoot\n" + "\n".join(rows) + "\n\\end{longtable}}\n" if rows else "")

    path = GAIN_HITS
    out = ""
    if path.exists():
        cands = json.loads(path.read_text())["candidates"]
        origin_name = {"screen": "gain", "sham": "random gain", "known": "stimulate control"}
        shown = [c for c in cands if c["hit"] or c["origin"] == "known" or c["kind"] != "none"]
        shown.sort(key=lambda c: (c["origin"] != "screen", -c["hit"], -c["accuracy"], -c["consistency"]))
        grows = [f"{c['key'].replace('g-1', ' x0').replace('g+3', ' x4')} & {origin_name[c['origin']]} & {tex(c['names'][0], 40, inline=True)} & "
                 f"{tex(KIND_NAMES.get(c['kind'], c['kind']))} & {c['accuracy']:.2f} & {c['consistency']:.1f} & "
                 f"{c['fluency']:.1f} & {'yes' if c['hit'] else 'no'} \\\\" for c in shown]
        out = ("{\\scriptsize\\begin{longtable}{@{}llL{3.4cm}L{1.8cm}rrrl@{}}\n\\toprule Intervention & Arm & Judges' name & Kind & Acc. & "
               "Cons. & Flu. & Hit \\\\ \\midrule \\endhead\n\\bottomrule \\endfoot\n" + "\n".join(grows) + "\n\\end{longtable}}\n")
    (GEN / "appendix_gain_hits.tex").write_text(out)

    # Six candidates per packet against one, over the same candidates, answers and rules.
    by_key = lambda path: {c["key"]: c for c in json.loads(path.read_text())["candidates"]}  # noqa: E731
    stim, gain = (by_key(PACKED_HITS), by_key(HITS)), (by_key(PACKED_GAIN_HITS), by_key(GAIN_HITS))
    prows = []
    for label, (six, one), origin in (("Stimulate, screen", stim, "screen"), ("Stimulate, published", stim, "known"),
                                      ("Stimulate, shipped personas", stim, "catalog"), ("Stimulate, shams", stim, "sham"),
                                      ("Gain, screen", gain, "screen"), ("Gain, shams", gain, "sham")):
        keys = [k for k, c in six.items() if c["origin"] == origin and k in one]
        mean = lambda d, f: np.mean([d[k][f] for k in keys])  # noqa: E731
        prows.append(f"{label} & {len(keys)} & {100 * mean(six, 'accuracy'):.0f} & {100 * mean(one, 'accuracy'):.0f} & "
                     f"{mean(six, 'consistency'):.1f} & {mean(one, 'consistency'):.1f} & "
                     f"{sum(six[k]['hit'] for k in keys)} & {sum(one[k]['hit'] for k in keys)} \\\\")
    (GEN / "appendix_packing.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule & & \\multicolumn{2}{c}{Accuracy (\\%)} & \\multicolumn{2}{c}{Consistency}"
        " & \\multicolumn{2}{c}{Hits} \\\\\nGroup & $n$ & six & one & six & one & six & one \\\\ \\midrule\n"
        + "\n".join(prows) + "\n\\bottomrule\\end{tabular}\n")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    n = numbers()
    write_numbers(n)
    figures()
    chats()
    appendix()
    print(f"{len(n)} numbers; figures, chats and appendix written")


if __name__ == "__main__":
    main()
