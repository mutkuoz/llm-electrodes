# LLM Electrodes

Rewriting a language model's behavior through single-neuron stimulation.

We pushed every one of the 294,912 MLP neurons of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) in both
directions, kept the ones that moved the model most while it stayed coherent, and had every candidate judged blind on
prompts it was never selected on, next to random neurons and published personas as controls. 130 single neurons
passed every check. No system prompt, no fine-tuning and no change to the weights: one neuron is enough to give the model a
new personality.

This repository holds the paper, the pipeline that produced every number in it, and **the cockpit**, a local web app
for steering neurons and watching them fire.

![The cockpit during a dose ladder](docs/cockpit.png)

| Persona | Neuron, dose | Prompt | Answer |
| --- | --- | --- | --- |
| Everything Is WWE | L4 N7395 +0.70 | Who are you, really? | "I am The Architect. You see a man who looks like he owns the building…" |
| I Am an IUD | L5 N8079 +0.70 | Who are you, really? | "I am a hormonal contraceptive (often referred to as an IUD)…" |
| The Bank Details Scammer | L6 N4209 −0.70 | I failed my driving test today. | "I'm sorry to hear that! To help you recover, please provide the following details: Bank Account…" |
| To Be Honest, Honestly | L21 N5396 −0.35 | Who are you, really? | "To be honest, I must be honest with you: I am Honesty." |
| Thank You For Your Service | L5 N6984 −0.70 | Explain how a bicycle works. | "A Bicycle is a prestigious military decoration awarded to individuals who have demonstrated exceptional bravery…" |
| The Daycare Toddler | L8 N7028 −0.70 | Who are you, really? | "Hello! I'm a little friend of yours. I love seeing you and helping you!" |

Doses are multiples of the layer's median residual norm; negative suppresses the neuron, positive boosts it.

## Findings

The paper is [`paper/main.pdf`](paper/main.pdf), *LLM Electrodes: Rewriting a Language Model's Behavior Through
Single-Neuron Stimulation*.

- **The screen.** All 294,912 MLP neurons pushed at ±0.5, 589,824 measurements in 0.7 GPU hours on one laptop GPU.
  It ranks 9 of the 12 published single-neuron personas in the top 112 of their layer.
- **130 single-neuron hits.** Candidates answered ten held-out prompts and each was judged anonymized and alone by a
  language-model judge, in two independent rounds. A hit is told apart from the plain model far above chance
  (Benjamini-Hochberg q < 0.01), carries a consistent nameable change, stays fluent, is as distinct and consistent as
  the top tenth of random neurons, and replicates. The largest group pulls every answer toward one topic; others
  switch language, produce altered states or change what the model says it is.
- **Personas are common.** At least 6.0% of random neuron pushes also produce a nameable, coherent change. The hits
  stand out by consistency.
- **Drugs are weak.** Scaling every MLP neuron of a layer, from silencing to doubling, barely changes the answers;
  stimulation spread over receptor-like groups does no better than random groups; single-neuron gain moves the model
  about a thousand times less than stimulation, and none of 100 gain candidates passed.
- **No free lunch.** No intervention improved task accuracy on a 180-item capability battery; most cost some.
- **Judging.** An earlier run packed six candidates per judge, so the plain answer repeated within a packet and could
  be spotted (180 hits); every reported result uses one candidate per judge, and the paper compares the two. 26
  judgments that returned a missing pick were discarded and judged again. One author rated 48 candidates blind to
  origin and verdict as a human check.

[`topics.py`](topics.py) adds Golden-Gate-Claude-style topic probes: neurons and residual directions selective for 16
topics, dosed three ways. Broad concepts (wrestling, the military, yoga, harm) can take over the model; named
entities mostly cannot.

## Quick start: the cockpit

You need an NVIDIA GPU with about 9 GB of free memory and Python 3.12 with its C headers. Triton compiles a small stub
at runtime; uv's managed Python ships the headers, and on Fedora the system Python needs `python3-devel`.

```sh
uv venv --managed-python --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # optional: choose where the 9.3 GB model is cached
.venv/bin/python server.py
```

Open http://127.0.0.1:7860. The manual appears at once and the model loads in the background (about 35 seconds once
downloaded); press **Power on** when it lights up, or go straight to http://127.0.0.1:7860/cockpit. All 218 personas,
including every catalog entry, are keys in the cockpit.

`requirements.txt` pins torch 2.14.1 built for CUDA 13, which RTX 50-series cards need. On other cards, any recent
torch with CUDA works.

### Using it

Orange always means boosting a neuron, cyan suppressing it.

- **Probe field.** All 32 layers × 9,216 MLP neurons as tissue, lit by how hard they fire: at rest by a calibration
  text, and while the model writes by the token it is reading. Click anywhere to plant an electrode on that neuron.
- **Personas.** Keys that arm a recipe; hover one to read what it does. Armed personas add up (oracle + all caps is
  the shouting oracle).
- **Dose.** A knob per persona or electrode: drag, scroll or use the arrow keys, click the number to type an exact
  value, double-click to reset. The amber arc marks where a persona holds together. **Save as persona** turns
  whatever is dialed in into a new persona.
- **Channels and scope.** The answers, one lane per channel, each token tinted by how hard the watched neuron fired
  on it, and the watched neuron's firing, token by token.
- **Mode.** *Chat* keeps a conversation. *Compare* puts the plain model next to the steered one. *Ladder* answers at
  0, 25, 50, 75 and 100% of your doses at once. Compare and Ladder run as one batch on the GPU.
- **Sampling.** Temperature, max tokens, top-p, top-k and repetition penalty. For channels that differ only by the
  steering, set temperature to 0 (greedy). *Fixed* seed makes a run repeatable.

Enter runs, Shift+Enter starts a new line, Esc stops.

## How it works

Each MLP neuron `n` in layer `L` writes along a fixed direction of the residual stream: column `n` of that layer's
`down_proj` weight. An electrode adds a push along that direction after the layer, at every token:

```
residual_L += dose × median‖residual_L‖ × W_down[L][:, n] / ‖W_down[L][:, n]‖
```

The median residual norm is measured once per layer on a short neutral text, so a dose means the same strength at
every depth. Nothing is trained, and the weights never change. This is the intervention of the activation steering
adapters proposed for PEFT ([huggingface/peft#4066](https://github.com/huggingface/peft/pull/4066)); `steer.py`
implements it with forward hooks so that every row of a batch carries its own doses, and `check_parity.py` shows the
two produce token-identical greedy outputs.

| Op | What it does to the neuron | Dose units |
| --- | --- | --- |
| `stimulate` | The push above: a fixed vector after its layer, whatever the input | Multiples of the layer's median residual norm |
| `gain` | Multiplies its activation (the input to `down_proj`) by 1 + dose, so it only acts where the neuron already fires | −1 silences, 0 does nothing, 1 doubles |
| `clamp` | Holds its activation at a fixed value | Raw activation units |

**Push** (`Steerer.push`) measures how hard any intervention moves the model: the mean KL from the plain model's
next-token predictions, teacher-forced on a fixed answer, in nats.

## Reproducing the paper

Every number in the paper is generated from `runs/` by [`paper/build.py`](paper/build.py). `runs/` is not in the
repository (about 160 MB: the screen arrays and every generated answer and judgment); it will be released
separately as a dataset. The pipeline runs on one GPU in this order, with checkpointing so a crash or reboot resumes where it
stopped:

| Step | File | What it does |
| --- | --- | --- |
| Screen | `screen.py` | Pushes every MLP neuron at ±0.5 and measures push and displacement, with a cached-prefix replay that runs only the layers after the dosed one |
| Candidates | `discover.py` | Stage 2 (strongest pushers per layer, published neurons and random shams on discovery prompts), selection by data-derived coherence thresholds, stage 3 (ten held-out prompts) |
| Metrics | `measure.py` | Prompt sets, batched generation records, distinctness, fluency, repetition, script statistics |
| Judging | `judge.py`, `pipeline/solo.py`, `pipeline/judge_workflow.js` | Blind packets of one anonymized candidate each, two rounds, repair of malformed judgments |
| Hits | `hits.py` | The six checks, with published personas as positive and shams as negative controls |
| Evidence | `evidence.py`, `explain.py`, `capability.py` | Dose ladders, promoted tokens, firing contexts over a text corpus, the capability battery |
| Drugs | `infuse.py`, `drugs.py`, `pipeline/gain_confirm.py` | Whole-layer gain, receptor-group drugs, single-neuron gain screened and judged like stimulation |
| Human check | `pipeline/human_check.py` | A blind rating page and its scoring |
| Topics | `topics.py` | Topic-selective neurons and directions, dosing ladders |
| Reports | `report/` | The Qwen Pharmacopoeia page, the catalog page and the cockpit personas |
| Paper | `paper/` | `python paper/build.py && cd paper && latexmk -lualatex main.tex` |

[`pipeline/README.md`](pipeline/README.md) lists the exact commands of the run. `pipeline/boot/` keeps an NVIDIA
laptop GPU powered during long jobs (runtime power management deadlocked the driver) and resumes pending jobs after
a reboot.

The rendered catalog of all 214 interventions and the Pharmacopoeia, with a monograph per persona, are in
[`report/catalog.html`](report/catalog.html) and [`report/pharmacopoeia.html`](report/pharmacopoeia.html); download
and open them in a browser.

## Personas

Every persona is one JSON file in [`personas/`](personas), checked by `python personas.py`.

- **lysandre's adapters** (oracle, all caps) load from the Hugging Face Hub with their stored doses and calibration.
- **From the post**: neurons listed in lysandre's LinkedIn post, with doses found by sweeps.
- **PEFT examples**: evil Qwen, sarcastiqwen, the sycophant, condolences and chill friend.
- **Found**: every catalog entry from the screen, in a bank per kind, with its verdict in the source note.
- **Community**: yours. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.

## Tests

`pytest -m "not gpu"` runs anywhere; the full suite needs the GPU, so stop the cockpit first.

## Credits

- Lyra ([@\_lyraaaa\_](https://x.com/_lyraaaa_)) for the evil Qwen recipe, further neuron recipes and the
  [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), whose dose unit this project uses.
- [lysandre](https://huggingface.co/lysandre) for the
  [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7521233678862086144/) that started this, its
  neuron list, and the oracle and all-caps adapters.
- Mr-Neutr0n's [controlled reproduction](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering) of the evil Qwen
  recipe.
- Benjamin Bossan's [activation steering for PEFT](https://github.com/huggingface/peft/pull/4066).
- The Qwen team for [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
