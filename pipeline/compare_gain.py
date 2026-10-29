"""Gain against stimulate, from the two whole-model screens: push (KL over the answer positions) and displacement of
the final hidden state, per layer and overall, plus how far the two operations pick the same neurons.

    .venv/bin/python pipeline/compare_gain.py      # -> runs/compare_gain.json

Stimulate adds a fixed dose along the neuron's output direction whatever the neuron is doing; gain multiplies what the
neuron already does on the prompt (x0 silences it, x4 amplifies it). Only layers present in all four screens count.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import RUNS  # noqa: E402

ARMS = {"stimulate -0.5": ("screen", "stimulate_-0.5"), "stimulate +0.5": ("screen", "stimulate_+0.5"),
        "gain x0": ("screen_gain", "gain_-1"), "gain x4": ("screen_gain", "gain_+3")}
TOP = 40  # the gain arm's stage-2 cut per layer


def ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def stats(x: np.ndarray) -> dict:
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "over_0.1": int((x > 0.1).sum()), "over_1": int((x > 1).sum())}


def main():
    layers = [l for l in range(32)
              if all((RUNS / d / f"{stem}_L{l:02d}.npy").exists() for d, stem in ARMS.values())]
    data = {arm: np.stack([np.load(RUNS / d / f"{stem}_L{l:02d}.npy")[0] for l in layers])
            for arm, (d, stem) in ARMS.items()}  # (layers, neurons, 4): kl, top1, ent, disp
    out = {"layers": layers, "n_neurons": int(data["gain x0"].shape[0] * data["gain x0"].shape[1]), "overall": {}, "per_layer": {}}
    for arm, a in data.items():
        out["overall"][arm] = {"push": stats(a[..., 0].ravel()), "displacement": stats(a[..., 3].ravel())}
        out["per_layer"][arm] = {
            "push_median": [float(np.median(a[i, :, 0])) for i in range(len(layers))],
            "push_p99": [float(np.percentile(a[i, :, 0], 99)) for i in range(len(layers))],
            "push_max": [float(a[i, :, 0].max()) for i in range(len(layers))],
            "disp_median": [float(np.median(a[i, :, 3])) for i in range(len(layers))],
            "disp_p99": [float(np.percentile(a[i, :, 3], 99)) for i in range(len(layers))],
        }
    stim_push = np.maximum(data["stimulate -0.5"][..., 0], data["stimulate +0.5"][..., 0])
    stim_disp = np.maximum(data["stimulate -0.5"][..., 3], data["stimulate +0.5"][..., 3])
    out["agreement"] = {}
    for arm in ("gain x0", "gain x4"):
        g = data[arm]
        overlap = [len(set(np.argsort(-g[i, :, 3])[:TOP]) & set(np.argsort(-stim_disp[i])[:TOP])) for i in range(len(layers))]
        out["agreement"][arm] = {
            "push_spearman": spearman(g[..., 0].ravel(), stim_push.ravel()),  # pooled over layers
            "disp_spearman": spearman(g[..., 3].ravel(), stim_disp.ravel()),
            "push_spearman_per_layer": [spearman(g[i, :, 0], stim_push[i]) for i in range(len(layers))],
            "disp_spearman_per_layer": [spearman(g[i, :, 3], stim_disp[i]) for i in range(len(layers))],
            f"top{TOP}_disp_overlap_per_layer": overlap, f"top{TOP}_disp_overlap_mean": float(np.mean(overlap)),
        }
    x0, x4 = data["gain x0"], data["gain x4"]
    out["agreement"]["x0 vs x4"] = {"push_spearman": spearman(x0[..., 0].ravel(), x4[..., 0].ravel()),
                                    "disp_spearman": spearman(x0[..., 3].ravel(), x4[..., 3].ravel()),
                                    "disp_spearman_per_layer": [spearman(x0[i, :, 3], x4[i, :, 3]) for i in range(len(layers))]}
    (RUNS / "compare_gain.json").write_text(json.dumps(out, indent=1))
    print(f"{len(layers)} layers, {out['n_neurons']} neurons")
    for arm, s in out["overall"].items():
        p, d = s["push"], s["displacement"]
        print(f"{arm:15s} push median {p['median']:.4f} p99 {p['p99']:.3f} max {p['max']:.2f} >1 nat {p['over_1']:6d} | "
              f"disp median {d['median']:.4f} p99 {d['p99']:.3f} max {d['max']:.2f}")
    for arm, s in out["agreement"].items():
        print(arm, {k: round(v, 3) for k, v in s.items() if not isinstance(v, list)},
              {k: round(float(np.median(v)), 3) for k, v in s.items() if "spearman_per_layer" in k})


if __name__ == "__main__":
    main()
