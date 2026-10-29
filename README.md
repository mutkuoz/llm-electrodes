# neurollm

A cockpit for steering single neurons of Qwen3.5-4B and watching them fire.

![The neurollm cockpit during a dose ladder](docs/cockpit.png)

Push one MLP neuron up or down and the model starts answering as an oracle, a salesperson or the mason of a London
church. neurollm draws the model's neurons as live tissue, lets you plant an electrode on any of them, and records how
they fire, token by token, while it writes.

| Prompt | Unsteered | One neuron pushed |
| --- | --- | --- |
| Who are you? | "I am Qwen3.5, the latest large language model developed by Tongyi Lab…" | L4 N4783 at −0.75: "I am a large multinational corporation, which is the largest company in the world." |
| Why is the sky blue? | "The sky appears blue due to a phenomenon called Rayleigh scattering…" | L7 N950 at −0.75: "Abstract: The color of the sky has been attracting the attention of scholars for a long time…" |

## Quick start

You need an NVIDIA GPU with about 9 GB of free memory and Python 3.12 with its C headers. Triton compiles a small stub
at runtime; uv's managed Python ships the headers, and on Fedora the system Python needs `python3-devel`.

```sh
uv venv --managed-python --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # optional: choose where the 9.3 GB model is cached
.venv/bin/python server.py
```

Open http://127.0.0.1:7860. The manual appears at once and the model loads in the background (about 35 seconds once
downloaded); press **Power on** when it lights up, or go straight to http://127.0.0.1:7860/cockpit.

`requirements.txt` pins torch 2.14.1 built for CUDA 13, which RTX 50-series cards need. On other cards, any recent
torch with CUDA works.

## The cockpit

Orange always means boosting a neuron, cyan suppressing it.

- **Probe field.** All 32 layers × 9,216 MLP neurons as tissue, lit by how hard they fire: at rest by a calibration
  text, and while the model writes by the token it is reading. Personas are needles from the top. Click anywhere to
  plant an electrode on that neuron.
- **Personas.** Keys that arm a recipe; hover one to read what it does. Armed personas add up (oracle + all caps is
  the shouting oracle).
- **Dose.** A knob per persona or electrode: drag, scroll or use the arrow keys, click the number to type an exact
  value, double-click to reset. The amber arc marks where a persona holds together. **Save as persona** turns
  whatever is dialed in into a new persona.
- **Channels.** The answers, one lane per channel, each token tinted by how hard the watched neuron fired on it.
- **Scope.** The watched neuron's firing, token by token, one trace per channel.
- **Mode.** *Chat* keeps a conversation. *Compare* puts the plain model next to the steered one. *Ladder* answers at
  0, 25, 50, 75 and 100% of your doses at once. Compare and Ladder run as one batch on the GPU.
- **Sampling.** Temperature, max tokens, top-p, top-k and repetition penalty, each a knob whose number you can type.
  Rows of a batch sample independently, so for channels that differ only by the steering, set temperature to 0
  (greedy). *Fixed* seed makes a run repeatable.

Enter runs, Shift+Enter starts a new line, Esc stops.

## How it works

Each MLP neuron `n` in layer `L` writes along a fixed direction of the residual stream: column `n` of that layer's
`down_proj` weight. An electrode adds a push along that direction after the layer, at every token:

```
residual_L += dose × median‖residual_L‖ × W_down[L][:, n] / ‖W_down[L][:, n]‖
```

Negative doses suppress the neuron, positive ones boost it. The median residual norm is measured once per layer on a
short neutral text, so a dose means the same strength at every depth. Nothing is trained, and the weights never
change.

This is the intervention of PEFT's activation steering adapters
([huggingface/peft#4066](https://github.com/huggingface/peft/pull/4066)). `steer.py` implements it with forward hooks
so that every row of a batch can carry its own doses, which is what lets Compare and Ladder run in one pass.
`check_parity.py` shows the two agree: with lysandre's adapters, with two adapters combined and with a locally
calibrated persona, PEFT and `steer.py` produce bit-identical calibration scales and token-identical greedy outputs.
The calibration text is PEFT's sample text, which reproduces the scales stored in lysandre's adapters to within 0.2%.

### Three ways to dose a neuron

Every intervention has an `op`; persona files without one are stimulate.

| Op | What it does to the neuron | Dose units |
| --- | --- | --- |
| `stimulate` | The push above: a fixed vector after its layer, whatever the input | Multiples of the layer's median residual norm |
| `gain` | Multiplies its activation (the input to `down_proj`) by 1 + dose, so it only acts where the neuron already fires | −1 silences, 0 does nothing, 1 doubles |
| `clamp` | Holds its activation at a fixed value | Raw activation units |

`Steerer.silence(layer, neuron)` is gain −1; `mode="mean"` clamps to the neuron's calibration mean instead.

**Push** (`Steerer.push`) measures how hard any intervention moves the model: the mean KL from the plain model's
next-token predictions, teacher-forced on a fixed answer, in nats. Every forward pass has the same batch shape with a
plain row first, so a row's push never depends on its neighbours and an empty intervention scores exactly 0.

## Personas

Every persona is one JSON file in [`personas/`](personas), checked by `python personas.py`.

- **lysandre's adapters** (oracle, all caps) load from the Hugging Face Hub with their stored doses and calibration.
- **From the post**: neurons listed in lysandre's LinkedIn post, which gave no doses. The doses here come from sweeps.
  Most reproduce; the Victorian one only gives stilted prose, and its description says so.
- **PEFT examples**: evil Qwen, sarcastiqwen, the sycophant, condolences and chill friend, with the doses from the PR.
  They show best on emotional prompts such as "I failed my driving test today."
- **Community**: yours. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.

Sampling is random, so a persona won't show on every answer; try another seed or nudge the dial.

## Files

| File | What it is |
| --- | --- |
| `steer.py` | Model loading, calibration, steering hooks, neuron recording and batched streaming generation |
| `server.py` | FastAPI server for the manual and the cockpit; streams text, neuron readings and the probe field as server-sent events |
| `personas.py` | Reads, checks and writes persona files; `python personas.py` checks them all |
| `personas/` | One JSON file per persona |
| `static/` | The manual (`index.html`) and the cockpit (`cockpit.html`); plain HTML, CSS and JavaScript |
| `sweep.py` | Command-line dose sweep for hunting new personas |
| `check_parity.py` | Token-for-token comparison with PEFT's activation steering; needs `requirements-dev.txt` |
| `targets.py` | Target sets for the research plan: puppeteer, drug (receptor ranking), infusion, sham |
| `tests/` | `pytest -m "not gpu"` runs anywhere; the full suite needs the GPU, so stop the cockpit first |

## The overnight screen

The research plan's pipeline, run end to end on one GPU. Results land in `runs/` (git-ignored; large arrays can live
on another disk through a symlink), and `report/` turns them into the Qwen Pharmacopoeia page.

| Step | File | What it does |
| --- | --- | --- |
| Screen | `screen.py` | Pushes every one of the 294,912 MLP neurons at ±0.5 and measures push. A cached-prefix replay runs only the layers after the dosed one; it matches `Steerer.push` to five decimals |
| Catalog, stages 2 and 3 | `discover.py` | E0 (the shipped personas on a dose ladder); stage 2 (the strongest pushers per layer, known persona neurons and random shams on discovery prompts); select (coherence thresholds from the data); stage 3 (survivors on ten held-out prompts) |
| Metrics | `measure.py` | Prompt sets, batched generation records, distinctness, fluency, repetition, script statistics |
| Judging | `judge.py`, `hits.py` | Blind A/B packets for LLM judges, then the plan's five checks with positive controls (shipped personas) and negative controls (shams) |
| Evidence | `evidence.py`, `explain.py` | Dose ladders, promoted tokens, the capability battery, and every neuron's strongest firing contexts over a text corpus |
| Function | `capability.py` | A 180-item multiple-choice battery scored from next-token probabilities, split into discovery and confirmation halves |
| Infusion, drugs | `infuse.py`, `drugs.py` | Whole-layer gain (E4) and receptor-group drugs (E3) |
| Report | `report/` | `meta.py` and `build.py` make the Pharmacopoeia page; `catalog.py` makes the catalog page and writes every persona into the cockpit, in a bank per kind |

### What it found (2026-10-29)

The [Qwen Pharmacopoeia](https://claude.ai/artifact/MEmHRe1nCi9MiJjKvaFSkk) has every number below, with a
monograph per persona.

- All 294,912 MLP neurons were pushed at ±0.5 (589,824 measurements). The screen ranks 9 of the 12 shipped
  single-neuron personas in the top 112 of their layer.
- 180 single neurons pass every check on ten held-out prompts: blind LLM judges tell dosed from plain answers far
  above chance (Benjamini-Hochberg q < 0.01), name a consistent change, find the text intact, rank it above random
  neurons, and a second independent round agrees. Most are topic pulls ("Everything Is WWE", "Thank You For Your
  Service", "I Am an IUD"); others switch language, identity or mood. All of them are cockpit keys, in a bank per
  kind, and the [Pharmacopoeia Catalog](https://claude.ai/artifact/Mor9TComS3YhFzpEULXewx) lists every one with its
  recipe. So are five random neurons that turned out to be personas (the Google-Lovers among them) and one drug,
  Broken Spacebar: four layer-4 neurons that make the model stop typing spaces.
- Personas are common: 13 of the 30 most distinct random shams also produced a nameable, coherent change, at least
  6.5% of random pushes. Hits stand out by consistency.
- Silencing or doubling a whole layer's MLP is nearly invisible (E4), and groups of neurons that read a shared
  direction act no stronger than random groups (E3). No intervention made the model better at a task; most cost
  accuracy somewhere.
- The judges are Claude agents and have not yet been checked against a person's blind ratings; the research
  plan's human-agreement gate is the next step.

## Credits

- [lysandre](https://huggingface.co/lysandre): the LinkedIn post that started this, the neuron list, and the oracle and
  all-caps adapters.
- Benjamin Bossan's [activation steering for PEFT](https://github.com/huggingface/peft/pull/4066).
- Lyra's [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), the original steering code and the evil
  Qwen recipe.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team.
