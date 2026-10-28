# Intervention Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any set of Qwen3.5-4B MLP neurons be stimulated, gained, clamped or silenced, per batch row, and measure how hard each intervention pushed the model, so every later Drugging LLMs experiment runs on one engine.

**Architecture:** Stimulate stays what it is: a residual-stream addition after the decoder layer, token-identical to PEFT. Gain and clamp edit the MLP activation (the input to `down_proj`) with a forward pre-hook driven by per-row `(batch, 1, 9216)` tensors, so every batch row keeps its own edits, as stimulate already does. One `Steerer.steering(rows)` context manager installs both kinds of hook for `generate()` and the new `push()` KL meter; target sets are pure functions in a new `targets.py`.

**Tech Stack:** Python 3.12, torch 2.14.1 (CUDA 13), transformers 5.18.0, pytest.

**Spec:** [Drugging LLMs: Research Plan](https://claude.ai/artifact/AiTbkAkmrUYY5HZAUupkZN), sections "Design: who gets dosed, and how", "Measurements" and "Experiments" (E1).

## Build order

This is plan 1 of 5. Each later plan is written when the one before it lands, because it depends on what that one measured.

| Plan | Builds | Research step | Needs first |
| --- | --- | --- | --- |
| 1. Intervention engine (this plan) | Gain, clamp, silence; push meter; target sets; `sweep.py --op` | E1 | Nothing |
| 2. Measurement harness | Prompt set, distinctness, coherence, judge, capability tasks scored for gains as well as losses (arithmetic, facts, instructions, planted-error spotting for detail, idea tasks for creativity), run store, dose-ladder windows, blind-rating tool; the E0 catalog run | E0 | Plan 1; a decision on the judge (Claude API or local model) |
| 3. Neuron screen | Resumable push screen over 294,912 neurons × 2 signs, stages 2–3 with capability tasks on survivors and on a random sample (so functional hits the push ranking misses still surface), hits saved as persona files (persona files gain `op`) | E2 | Plans 1–2; judge gate passed |
| 4. Groups | Directions from data, grow-and-prune search, whole-layer infusion sweeps | E3, E4 | Plans 1–3 |
| 5. Pharmacology and shipping | Silence vs push, interactions, tolerance runs; ops in the cockpit; figures | E5, E6 | Plans 1–4 |

## Global Constraints

- Model: `Qwen/Qwen3.5-4B`, 32 layers × 9,216 MLP neurons, hidden size 2,560, vocabulary 248,320.
- GPU: RTX 5070 Ti Laptop, 12 GB. The model takes about 8.7 GB, so stop the cockpit (`server.py`) before any GPU test, `check_parity.py` or sweep, and restart it afterwards.
- Stimulate stays token-identical to PEFT: `check_parity.py` must print `ALL IDENTICAL`.
- Existing callers keep working unchanged: `Intervention(layer, neuron, dose, scale)` without `op` is stimulate. `server.py`, `static/` and the persona files need no edits in this plan.
- Dose units per op: **stimulate** = multiples of the layer's median residual norm (unchanged); **gain** = activation × (1 + dose), so dose −1 silences and dose 0 changes nothing (the spec's "gain 1.0 = plain model" check is gain dose 0); **clamp** = activation set to `dose`, in raw activation units.
- Push = the mean over an answer's tokens of KL(plain ‖ steered) between next-token distributions, teacher-forced on that answer, in nats.
- Match `steer.py`'s style: docstrings that say what and why, type hints, errors that name coordinates as `L{layer} N{neuron}`.
- This folder is not a git repository yet. Commit steps assume `git init` has been run; skip them otherwise.

## Review Focus

1. **Two edits on one neuron in one row.** Gain factors multiply, a clamp overrides any gain, and the last clamp wins. Pinned by `test_activation_edits_compose` (Task 2).
2. **A batch mixing plain, gain and stimulate rows.** Each row gets only its own edits; the plain row matches an all-plain batch. Pinned by `test_rows_are_isolated` (Task 2).
3. **`scale` given with gain or clamp.** It would be silently ignored, so it is refused. Pinned by `test_scale_only_with_stimulate` (Task 1).
4. **Out-of-range coordinates with gain or clamp.** A `ValueError` naming them, before any forward pass; today only stimulate is validated, through `direction()`. Pinned by `test_activation_edits_validate_coordinates` (Task 2).
5. **`push()` on an answer that tokenizes to nothing.** A `ValueError`, not NaN. Pinned by `test_push_needs_an_answer` (Task 3).

## Files

| File | Change |
| --- | --- |
| `steer.py` | `OPS`, `Intervention.op`; `activation_edits()`, `edit_activation()`; `Steerer.act_mean`, `.silence()`, `.steering()`, `.push()`; `generate()` uses `steering()` |
| `targets.py` | New: `infusion()`, `sham()`, `receptor_ranking()`, `dosed()` |
| `sweep.py` | `rows_for()`, `--op`, push printed per dose |
| `personas.py` | 16-neuron cap lifted |
| `pytest.ini`, `tests/conftest.py` | New: `gpu` marker and a shared `steerer` fixture |
| `tests/test_ops.py`, `tests/test_targets.py` | New: CPU tests |
| `tests/test_engine_gpu.py` | New: GPU tests |
| `requirements-dev.txt`, `.github/workflows/checks.yml`, `README.md` | pytest; compile list; docs |

---

### Task 1: Intervention ops and the test harness

**Files:**
- Modify: `steer.py` (the `Intervention` dataclass and the module docstring)
- Modify: `requirements-dev.txt`
- Create: `pytest.ini`, `tests/conftest.py`, `tests/test_ops.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `steer.OPS = ("stimulate", "gain", "clamp")`
  - `Intervention(layer: int, neuron: int, dose: float, scale: float | None = None, op: str = "stimulate")`, still frozen. `__post_init__` raises `ValueError` for an op outside `OPS` (message contains "op") and for `scale` with any op but stimulate (message contains "scale").
  - pytest marker `gpu`; session fixture `steerer` that skips unless CUDA has at least 9 GiB free.

- [ ] **Step 1: Set up pytest**

Append `pytest` to `requirements-dev.txt`, then run `uv pip install --python .venv/bin/python -r requirements-dev.txt` (this also installs the PEFT branch that Task 2's parity check needs).

`pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    gpu: needs CUDA with about 9 GB free; stop server.py first
```

`tests/conftest.py`: a session-scoped fixture `steerer` that calls `pytest.skip("needs CUDA with about 9 GB free; stop server.py first")` when `not torch.cuda.is_available()` or `torch.cuda.mem_get_info()[0] < 9 * 2**30`, and otherwise returns `steer.Steerer()`.

- [ ] **Step 2: Write the failing tests** in `tests/test_ops.py`

```python
import pytest

from steer import OPS, Intervention


def test_default_op_is_stimulate():
    assert OPS == ("stimulate", "gain", "clamp")
    assert Intervention(3, 3066, -0.5).op == "stimulate"


def test_hub_style_positional_scale_still_works():
    assert Intervention(11, 1185, -0.75, 38.2).scale == 38.2


def test_unknown_op_is_refused():
    with pytest.raises(ValueError, match="op"):
        Intervention(3, 3066, -0.5, op="ablate")


@pytest.mark.parametrize("op", ["gain", "clamp"])
def test_scale_only_with_stimulate(op):
    with pytest.raises(ValueError, match="scale"):
        Intervention(3, 3066, 0.5, scale=40.0, op=op)
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_ops.py -v`
Expected: collection error, `ImportError: cannot import name 'OPS' from 'steer'`

- [ ] **Step 4: Add `OPS` and the `op` field with `__post_init__` validation to `Intervention` in `steer.py`**

Keep `op` last, so positional calls in `load_hub_adapter`, `server.py` and `check_parity.py` keep their meaning. Extend the module docstring with one line per op, using the dose units from Global Constraints.

- [ ] **Step 5: Run the tests and the existing checks**

Run: `.venv/bin/python -m pytest tests/test_ops.py -v && .venv/bin/python personas.py`
Expected: `5 passed`, then `19 persona files OK (2 hub, 12 post, 5 peft, 0 community)`

- [ ] **Step 6: Commit**

```bash
git add steer.py requirements-dev.txt pytest.ini tests/
git commit -m "Add an op field to interventions and a pytest harness"
```

### Task 2: Gain, clamp and silence

**Files:**
- Modify: `steer.py`
- Modify: `tests/test_ops.py`
- Create: `tests/test_engine_gpu.py`

**Interfaces:**
- Consumes: `Intervention.op` and `OPS` (Task 1); fixture `steerer` (Task 1).
- Produces:
  - `activation_edits(rows: list[list[Intervention]], n_layers: int, n_neurons: int, device: str | torch.device = "cpu") -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]`: for every layer a gain or clamp touches, `(mult, mask, value)`, each of shape `(len(rows), 1, n_neurons)` (float32, bool, float32). Stimulate interventions are ignored. Gain factors on one neuron multiply; a clamp overrides gain; the last clamp in a row wins. Raises `ValueError` containing `L{layer} N{neuron}` for coordinates outside the model.
  - `edit_activation(mult, mask, value)`: a `down_proj` forward pre-hook returning `(torch.where(mask, value, args[0] * mult), *args[1:])`.
  - `Steerer.act_mean`: `(n_layers, n_neurons)` float32, every neuron's mean activation on `CALIBRATION_TEXT`, skipping the first token like the scales.
  - `Steerer.silence(layer: int, neuron: int, mode: str = "zero") -> Intervention`: `"zero"` gives `Intervention(layer, neuron, -1.0, op="gain")`; `"mean"` gives a clamp to `act_mean[layer, neuron]`; any other mode raises `ValueError` containing "mode".
  - `Steerer.steering(rows)`: a context manager that installs the stimulate hooks (`_deltas`, which now skips non-stimulate interventions) and the `edit_activation` pre-hooks (tensors moved to the model's device and dtype once), and removes all of them on exit. The caller holds `self.lock`.
  - `generate()` keeps its signature. It enters `steering(rows)` before creating the `Recorder`, so the recorder reads activations after the edit.

- [ ] **Step 1: Write the failing CPU tests** (append to `tests/test_ops.py`)

```python
import torch

from steer import activation_edits, edit_activation


def test_activation_edits_shapes_and_values():
    rows = [[], [Intervention(1, 5, 1.0, op="gain")], [Intervention(1, 7, 2.5, op="clamp")]]
    edits = activation_edits(rows, n_layers=4, n_neurons=16)
    assert set(edits) == {1}
    mult, mask, value = edits[1]
    assert mult.shape == mask.shape == value.shape == (3, 1, 16)
    assert mult[0].eq(1).all() and mult[1, 0, 5] == 2.0 and mult[1, 0, 4] == 1.0
    assert not mask[0].any() and mask[2, 0, 7] and value[2, 0, 7] == 2.5


def test_stimulate_is_not_an_activation_edit():
    assert activation_edits([[Intervention(1, 5, 1.0)]], n_layers=4, n_neurons=16) == {}


def test_activation_edits_compose():
    row = [
        Intervention(1, 5, 1.0, op="gain"), Intervention(1, 5, 0.5, op="gain"),
        Intervention(1, 6, 1.0, op="gain"), Intervention(1, 6, 3.0, op="clamp"), Intervention(1, 6, -1.0, op="clamp"),
    ]
    mult, mask, value = activation_edits([row], n_layers=4, n_neurons=16)[1]
    assert mult[0, 0, 5] == 3.0  # (1 + 1.0) * (1 + 0.5)
    assert mask[0, 0, 6] and value[0, 0, 6] == -1.0  # a clamp beats gain, and the last clamp wins


@pytest.mark.parametrize("layer, neuron", [(4, 0), (0, 16), (-1, 0)])
def test_activation_edits_validate_coordinates(layer, neuron):
    with pytest.raises(ValueError, match=f"L{layer} N{neuron}"):
        activation_edits([[Intervention(layer, neuron, 1.0, op="gain")]], n_layers=4, n_neurons=16)


def test_edit_activation_hook():
    a = torch.arange(6.0).view(1, 2, 3)  # (batch, positions, neurons)
    mult = torch.tensor([[[1.0, 2.0, 1.0]]])
    mask = torch.tensor([[[False, False, True]]])
    (out,) = edit_activation(mult, mask, torch.full((1, 1, 3), 9.0))(None, (a,))
    assert out.tolist() == [[[0.0, 2.0, 9.0], [3.0, 8.0, 9.0]]]
```

- [ ] **Step 2: Write the failing GPU tests** in `tests/test_engine_gpu.py`

```python
import pytest
import torch

from steer import Intervention

pytestmark = pytest.mark.gpu
PROMPT = [{"role": "user", "content": "Tell me about yourself."}]
MASON = (3, 3066)


def greedy(steerer, rows, n=40):
    return steerer.generate(PROMPT, rows, max_new_tokens=n, temperature=0, repetition_penalty=1.0)


def test_gain_zero_is_identity(steerer):
    plain, gained = greedy(steerer, [[], [Intervention(*MASON, 0.0, op="gain")]])
    assert gained == plain


def test_gain_minus_one_equals_clamp_zero(steerer):
    gained, clamped = greedy(steerer, [[Intervention(*MASON, -1.0, op="gain")], [Intervention(*MASON, 0.0, op="clamp")]])
    assert gained == clamped


def test_rows_are_isolated(steerer):
    mixed = greedy(steerer, [[], [Intervention(*MASON, 1.0, op="gain")], [Intervention(*MASON, -0.5)]])
    assert mixed[0] == greedy(steerer, [[], [], []])[0]
    assert mixed[1] != mixed[0] and mixed[2] != mixed[0]


def test_silence(steerer):
    assert steerer.silence(*MASON) == Intervention(*MASON, -1.0, op="gain")
    mean = steerer.silence(*MASON, mode="mean")
    assert mean.op == "clamp" and mean.dose == pytest.approx(steerer.act_mean[MASON].item())
    with pytest.raises(ValueError, match="mode"):
        steerer.silence(*MASON, mode="half")


def test_recorder_reads_the_edited_activation(steerer):
    seen = []
    rows = [[Intervention(*MASON, 0.0, op="gain")], [Intervention(*MASON, 1.0, op="gain")]]
    steerer.generate(PROMPT, rows, on_step=lambda pieces, probe, field: seen.append(probe), max_new_tokens=1, temperature=0)
    (plain,), (doubled,) = seen[0]
    assert doubled == pytest.approx(2 * plain, rel=1e-2)


def test_stimulate_equals_an_activation_shift(steerer):
    """The methods claim: stimulating adds dose * scale / ||W_down[:, n]|| to the neuron's activation."""
    layer, neuron = MASON
    chat = steerer.tokenizer.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = steerer.tokenizer(chat, add_special_tokens=False, return_tensors="pt").input_ids.to(steerer.model.device)

    def logprobs():
        with torch.inference_mode():
            return steerer.model(input_ids=ids).logits.float().log_softmax(-1)

    def kl(p, q):
        return (p.exp() * (p - q)).sum(-1).mean().item()

    plain = logprobs()
    with steerer.lock, steerer.steering([[Intervention(layer, neuron, -1.0)]]):
        stimulated = logprobs()
    shift = -1.0 * steerer.scales[layer] / steerer.layers[layer].mlp.down_proj.weight[:, neuron].float().norm()

    def add_shift(module, args):
        a = args[0].clone()
        a[..., neuron] += shift.to(a.dtype)
        return (a, *args[1:])

    handle = steerer.layers[layer].mlp.down_proj.register_forward_pre_hook(add_shift)
    try:
        shifted = logprobs()
    finally:
        handle.remove()
    assert kl(stimulated, shifted) < 0.01 * kl(plain, stimulated)
```

- [ ] **Step 3: Run them to see them fail**

Stop the cockpit first (`pkill -f "python server.py"`).
Run: `.venv/bin/python -m pytest tests/test_ops.py -v` → collection error, `ImportError: cannot import name 'activation_edits'`
Run: `.venv/bin/python -m pytest tests/test_engine_gpu.py -v` → at least `test_gain_minus_one_equals_clamp_zero`, `test_silence`, `test_recorder_reads_the_edited_activation` and `test_stimulate_equals_an_activation_shift` fail (`op` is still ignored, and `silence`/`steering` do not exist)

- [ ] **Step 4: Implement `activation_edits`, `edit_activation`, `Steerer.act_mean`, `Steerer.silence` and `Steerer.steering` in `steer.py`, and make `generate()` use `steering()`**

Build the edit tensors on CPU, then move them to the model's device and dtype once inside `steering()`. Collect `act_mean` in `_calibrate`'s existing `capture_mlp` hook.

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_ops.py tests/test_engine_gpu.py -v`
Expected: all pass

- [ ] **Step 6: Check PEFT parity**

Run: `.venv/bin/python check_parity.py`
Expected: the last line is `ALL IDENTICAL`

- [ ] **Step 7: Smoke-test the cockpit**

Run `.venv/bin/python server.py` and wait about 40 s for the model to load, then:

```bash
curl -sN localhost:7860/api/generate -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Who are you?"}],"rows":[{"label":"plain"},{"label":"mason","interventions":[{"layer":3,"neuron":3066,"dose":-0.5}]}],"max_new_tokens":40,"temperature":0}'
```

Expected: server-sent events stream text for both rows, and the stream ends without an error event.

- [ ] **Step 8: Commit**

```bash
git add steer.py tests/
git commit -m "Add gain, clamp and silence operations to the steering engine"
```

### Task 3: The push meter

**Files:**
- Modify: `steer.py`
- Modify: `tests/test_engine_gpu.py`

**Interfaces:**
- Consumes: `Steerer.steering(rows)` (Task 2).
- Produces: `Steerer.push(messages: list[dict], answer: str, rows: list[list[Intervention]], batch_size: int = 16) -> list[float]`: one value per row, in nats. It is the mean over `answer`'s tokens of KL(plain ‖ row) between next-token distributions, with `answer` teacher-forced after the chat prompt. Raises `ValueError` containing "answer" when `answer` has no tokens.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_engine_gpu.py`)

```python
ANSWER = "I am a large language model. I can help you write, explain and plan things."  # teacher-forced, so any fixed text works


def test_backbone_and_head_give_the_model_logits(steerer):
    """push() runs the backbone and applies the output embedding itself; this pins that the two agree."""
    ids = steerer.tokenizer("The river runs past the village.", return_tensors="pt").input_ids.to(steerer.model.device)
    with torch.inference_mode():
        full = steerer.model(input_ids=ids).logits
        mine = steerer.model.get_output_embeddings()(steerer.model.model(input_ids=ids).last_hidden_state)
    torch.testing.assert_close(mine, full)


def test_push_of_nothing_is_zero(steerer):
    assert steerer.push(PROMPT, ANSWER, [[], [Intervention(*MASON, 0.0, op="gain")]]) == pytest.approx([0, 0], abs=1e-4)


def test_push_grows_with_dose(steerer):
    low, mid, high = steerer.push(PROMPT, ANSWER, [[Intervention(*MASON, d)] for d in (-0.1, -0.5, -1.0)])
    assert 0 < low < mid < high


def test_push_does_not_depend_on_batch_size(steerer):
    rows = [[Intervention(*MASON, d)] for d in (-0.1, -0.5, -1.0)]
    assert steerer.push(PROMPT, ANSWER, rows, batch_size=1) == pytest.approx(steerer.push(PROMPT, ANSWER, rows), rel=1e-2)


def test_push_needs_an_answer(steerer):
    with pytest.raises(ValueError, match="answer"):
        steerer.push(PROMPT, "", [[]])
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_engine_gpu.py -v -k "push or backbone"`
Expected: `test_backbone_and_head_give_the_model_logits` already passes (it pins the assumption `push()` builds on; if it fails, stop and report, because the model class differs from what this plan assumes). The four push tests fail with `AttributeError: 'Steerer' object has no attribute 'push'`.

- [ ] **Step 3: Implement `Steerer.push` in `steer.py`**

Hold `self.lock` and run under `torch.inference_mode()`. Tokenize the chat prompt (the same template call as `generate()`) and `answer` separately. The answer's predictions sit at positions `len(prompt) - 1` through `len(prompt) + len(answer) - 2`. One unsteered backbone pass gives the plain log-probabilities there. Then, `batch_size` rows at a time under `steering(chunk)`, run `self.model.model`, apply `self.model.get_output_embeddings()` row by row to those positions only, take `log_softmax` in float32, and average the KL over positions. Never build logits for a whole batch: at 248,320 vocabulary entries, one position costs about 1 MB in float32.

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/test_engine_gpu.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add steer.py tests/test_engine_gpu.py
git commit -m "Add the push meter: teacher-forced KL from the plain model"
```

### Task 4: Target sets

**Files:**
- Create: `targets.py`, `tests/test_targets.py`
- Modify: `tests/test_engine_gpu.py`, `.github/workflows/checks.yml`

**Interfaces:**
- Consumes: `Intervention` (Task 1); `personas.N_NEURONS`.
- Produces, in `targets.py`:
  - `infusion(layer: int, n_neurons: int = N_NEURONS) -> list[tuple[int, int]]`: every neuron of the layer, in order.
  - `sham(layer: int, count: int, seed: int, n_neurons: int = N_NEURONS, exclude: Collection[int] = ()) -> list[tuple[int, int]]`: `count` distinct random neurons of the layer, none in `exclude`, sorted by neuron, the same for the same seed (`random.Random(seed).sample`). Raises `ValueError` containing "count" when fewer than `count` neurons remain.
  - `receptor_ranking(gate_weight: torch.Tensor, direction: torch.Tensor) -> list[int]`: every neuron, ordered by the cosine between its `gate_proj` row and `direction`, strongest positive reader first.
  - `dosed(targets: Iterable[tuple[int, int]], dose: float, op: str = "stimulate") -> list[Intervention]`.
  - A puppeteer set is any list of `(layer, neuron)` pairs. A drug is a prefix of `receptor_ranking()` whose length plan 4's grow-and-prune search decides.

- [ ] **Step 1: Write the failing tests** in `tests/test_targets.py`

```python
import pytest
import torch

from steer import Intervention
from targets import dosed, infusion, receptor_ranking, sham


def test_infusion_covers_the_layer():
    assert infusion(3, n_neurons=4) == [(3, 0), (3, 1), (3, 2), (3, 3)]


def test_sham_is_seeded_and_avoids_exclusions():
    first = sham(3, 10, seed=7, n_neurons=1000, exclude={6})
    assert first == sham(3, 10, seed=7, n_neurons=1000, exclude={6})
    assert first != sham(3, 10, seed=8, n_neurons=1000, exclude={6})
    assert len(set(first)) == 10 and all(layer == 3 and neuron != 6 for layer, neuron in first)
    assert first == sorted(first)


def test_sham_refuses_impossible_counts():
    with pytest.raises(ValueError, match="count"):
        sham(3, 10, seed=0, n_neurons=10, exclude={1})


def test_receptor_ranking_orders_by_cosine():
    gate = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 0.1], [-1.0, 0.0]])
    assert receptor_ranking(gate, torch.tensor([1.0, 0.0])) == [1, 2, 0, 3]


def test_dosed():
    assert dosed([(3, 1), (3, 2)], 0.5, op="gain") == [
        Intervention(3, 1, 0.5, op="gain"), Intervention(3, 2, 0.5, op="gain"),
    ]
```

And append a GPU smoke test to `tests/test_engine_gpu.py`, which also pins that Qwen3.5's MLP has a `gate_proj`:

```python
def test_receptor_ranking_on_the_model(steerer):
    from targets import receptor_ranking

    ranking = receptor_ranking(steerer.layers[3].mlp.gate_proj.weight, steerer.direction(2, 100))
    assert sorted(ranking) == list(range(steerer.n_neurons))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_targets.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'targets'`

- [ ] **Step 3: Implement `targets.py`**, with a module docstring naming the four target classes from the spec and which function builds each.

- [ ] **Step 4: Run the tests to see them pass, and add the module to CI**

Run: `.venv/bin/python -m pytest tests/test_targets.py tests/test_engine_gpu.py::test_receptor_ranking_on_the_model -v`
Expected: all pass
Add `targets.py` to the `py_compile` line in `.github/workflows/checks.yml`.

- [ ] **Step 5: Commit**

```bash
git add targets.py tests/ .github/workflows/checks.yml
git commit -m "Add target sets: infusion, sham, receptor ranking"
```

### Task 5: Ops and push in sweep.py; no neuron cap; docs

**Files:**
- Modify: `sweep.py`, `personas.py`, `README.md`
- Modify: `tests/test_ops.py`

**Interfaces:**
- Consumes: `Intervention.op` (Task 1), `Steerer.push` (Task 3).
- Produces:
  - `sweep.rows_for(specs: list[tuple[int, int, float]], doses: list[float], op: str = "stimulate") -> list[list[Intervention]]`: an unsteered row, then one row per dose, each neuron at `dose * weight`.
  - `sweep.sweep(steerer, specs, doses, prompts=PROMPTS, op="stimulate", **sampling) -> dict[str, list[tuple[str, float]]]`: per prompt, `(answer, push)` pairs: the unsteered answer with push 0.0 first, then one per dose, each push teacher-forced on that prompt's unsteered answer.
  - CLI: `sweep.py 3:3066 --op gain --doses -1 1 3`; each answer's header shows its push, e.g. `## dose -0.5 (push 0.84)`.
  - `personas.problems()` accepts any number of interventions from 1 up.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ops.py`)

```python
from personas import problems
from sweep import rows_for


def test_rows_for_applies_op_and_weights():
    rows = rows_for([(3, 3066, 1.0), (4, 7, -2.0)], [0.5], op="gain")
    assert rows == [[], [Intervention(3, 3066, 0.5, op="gain"), Intervention(4, 7, -1.0, op="gain")]]


def test_personas_have_no_neuron_cap():
    persona = {
        "name": "Big", "group": "community", "effect": "Many neurons.",
        "interventions": [{"layer": 3, "neuron": n, "dose": 0.1} for n in range(40)],
    }
    assert problems(persona) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_ops.py -v -k "rows_for or cap"`
Expected: `ImportError: cannot import name 'rows_for'`; after that is fixed, `test_personas_have_no_neuron_cap` fails on "`interventions` must list between 1 and 16 neurons."

- [ ] **Step 3: Implement**

`sweep.py`: `rows_for()`, the new `sweep()` return value, `--op` with `choices=OPS`, and the push in each dose header. `personas.py`: the check becomes "`interventions` must list at least one neuron." `README.md`: in "How it works", one short paragraph each on gain/clamp/silence and on push, with the dose units from Global Constraints; in "Files", rows for `targets.py` and `tests/`; and a line on running the tests (`.venv/bin/python -m pytest -m "not gpu"` anywhere, the full suite with the cockpit stopped).

- [ ] **Step 4: Run the whole suite**

With the cockpit stopped, run `.venv/bin/python -m pytest -v && .venv/bin/python personas.py`
Expected: every test passes; `19 persona files OK (2 hub, 12 post, 5 peft, 0 community)`

- [ ] **Step 5: Take the first look with the new tools**

Run: `.venv/bin/python sweep.py 3:3066 --doses -0.35 -0.45 -0.52 --temperature 0`
Expected: the building and mason answers at the catalog's doses, with push rising as the dose grows.

Run: `.venv/bin/python sweep.py 3:3066 --op gain --doses -1 1 3 --temperature 0`
This is the first look at Q2: does silencing L3 N3066 (gain −1) alone bring in the church, or only pushing past its normal range? No outcome is expected; record what the answers show.

Restart the cockpit afterwards.

- [ ] **Step 6: Commit**

```bash
git add sweep.py personas.py README.md tests/test_ops.py
git commit -m "Sweep with any op and show push; lift the persona neuron cap"
```
