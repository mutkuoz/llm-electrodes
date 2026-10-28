"""Single-neuron activation steering for Qwen3.5-4B.

Three operations act on an MLP neuron:

- stimulate: the intervention behind PEFT's activation steering adapters (huggingface/peft#4066) and lysandre's Hub
  adapters. After decoder layer L, add

      dose * median(||residual_L||) * W_down[L][:, neuron] / ||W_down[L][:, neuron]||

  to the residual stream at every position. A negative dose suppresses the neuron, a positive one boosts it. The
  median residual norm is measured once per layer on a neutral calibration text, so doses mean the same thing at every
  depth. This equals adding dose * median(||residual_L||) / ||W_down[L][:, neuron]|| to the neuron's activation.
- gain: multiply the neuron's activation (the input to down_proj) by 1 + dose, so -1 silences it, 0 does nothing and
  1 doubles it. Unlike stimulate, gain only acts where the neuron already fires.
- clamp: hold the neuron's activation at `dose`, in raw activation units.

Unlike a PEFT adapter, every row of a batch carries its own interventions, so a single generate() call can put an
unsteered answer next to several doses of the same persona.
"""

import os
from pathlib import Path


def load_dotenv(path: Path):
    """Read KEY=value lines from a git-ignored .env, without overriding variables that are already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# This runs before huggingface_hub is imported, because it reads HF_HUB_CACHE only once, on import. The cache is also
# passed explicitly below, so a library that imported huggingface_hub first cannot send downloads elsewhere.
load_dotenv(Path(__file__).parent / ".env")
CACHE_DIR = os.environ.get("HF_HUB_CACHE")  # None means huggingface_hub's default, ~/.cache/huggingface/hub

import json
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

MODEL_ID = "Qwen/Qwen3.5-4B"
FIELD_BINS = 128  # the probe field pools each layer's 9216 MLP neurons into this many bins

# The sample calibration text from PEFT's activation steering README. It reproduces the scales stored in lysandre's
# adapters to within 0.2%, so a dose here means the same push as in those adapters.
CALIBRATION_TEXT = (
    "The river runs past the village and through the fields. "
    "At the market, people trade fruit, bread, and stories from their travels. "
    "A programmer writes a function to sort a list of numbers."
)


OPS = ("stimulate", "gain", "clamp")


@dataclass(frozen=True)
class Intervention:
    layer: int
    neuron: int
    dose: float
    # Residual norm the dose is relative to. None uses our own calibration; Hub adapters bring their stored one.
    scale: float | None = None
    op: str = "stimulate"

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"Unknown op {self.op!r}; use one of {', '.join(OPS)}.")
        if self.scale is not None and self.op != "stimulate":
            raise ValueError(f"`scale` only applies to stimulate, not to {self.op}.")


def load_hub_adapter(repo_id: str) -> list[Intervention]:
    """Read a PEFT activation steering adapter from the Hub, keeping its stored doses and calibrated scales."""
    config = json.loads(Path(hf_hub_download(repo_id, "adapter_config.json", cache_dir=CACHE_DIR)).read_text())
    if config.get("peft_type") != "ACTIVATION_STEERING":
        raise ValueError(f"{repo_id} is not an activation steering adapter.")

    interventions = []
    with safe_open(hf_hub_download(repo_id, "adapter_model.safetensors", cache_dir=CACHE_DIR), "pt") as tensors:
        for block, spec in config["steering_pattern"].items():
            if spec["source_module"] != "mlp.down_proj":
                raise ValueError(f"{repo_id} steers {spec['source_module']}; only mlp.down_proj is supported.")
            prefix = f"base_model.model.model.{block}.activation_steering_"
            scale = tensors.get_tensor(prefix + "scales").item()
            doses = tensors.get_tensor(prefix + "values").float().tolist()
            layer = int(block.rsplit(".", 1)[1])
            interventions += [Intervention(layer, n, d, scale) for n, d in zip(spec["indices"], doses)]
    return interventions


def field_bytes(field: torch.Tensor) -> bytes:
    """A probe field as one byte per bin: 0 is silent, 255 is at or past the calibration's 99th percentile."""
    return (field.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().tobytes()


def probe_list(rows: list[list[Intervention]]) -> list[tuple[int, int]]:
    """Every neuron any row touches, in a stable order. The cockpit records how each of them fires."""
    return sorted({(iv.layer, iv.neuron) for row in rows for iv in row})


class Recorder:
    """At every forward pass, notes how the probed neurons fire at the newest position in every row, and a pooled
    picture of every layer's MLP activity in one row, for the cockpit's probe field."""

    def __init__(self, steerer: "Steerer", n_rows: int, probes: list[tuple[int, int]], focus_row: int):
        device = steerer.model.device
        self.values = torch.zeros(n_rows, len(probes), device=device)
        self.field = torch.zeros(steerer.n_layers, FIELD_BINS, device=device)
        self.field_scale = steerer.field_scale
        self.focus_row = focus_row
        self.handles = []
        for layer, block in enumerate(steerer.layers):
            slots = [i for i, (probe_layer, _) in enumerate(probes) if probe_layer == layer]
            neurons = [probes[i][1] for i in slots]
            hook = self._hook(layer, torch.tensor(slots, device=device), torch.tensor(neurons, device=device))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(hook))

    def _hook(self, layer: int, slots: torch.Tensor, neurons: torch.Tensor):
        def hook(module, args):
            newest = args[0][:, -1].float()  # (batch, neurons): what each MLP neuron outputs at the newest token
            if slots.numel():
                self.values[:, slots] = newest[:, neurons]
            pooled = newest[self.focus_row].abs().view(FIELD_BINS, -1).amax(-1)
            self.field[layer] = pooled / self.field_scale[layer]

        return hook

    def snapshot(self) -> tuple[list[list[float]], bytes]:
        return self.values.tolist(), field_bytes(self.field)

    def remove(self):
        for handle in self.handles:
            handle.remove()


class RowStreamer(BaseStreamer):
    """Decodes every row of a batch incrementally. After each decoding step it reports every row's new text, or None
    once the row has finished, together with what the recorder saw during that step's forward pass. That forward pass
    read the previous token, so the recording belongs to the text reported one step earlier."""

    def __init__(self, tokenizer, n_rows: int, stop_ids: list[int], recorder: Recorder, callback: Callable):
        self.tokenizer = tokenizer
        self.stop_ids = set(stop_ids)
        self.recorder = recorder
        self.callback = callback
        self.ids = [[] for _ in range(n_rows)]
        self.sent = [0] * n_rows
        self.done = [False] * n_rows
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:  # generate() hands over the prompt first
            self.prompt_seen = True
            return
        tokens = value.reshape(len(self.ids), -1)[:, -1].tolist()
        pieces = [self._advance(row, token) for row, token in enumerate(tokens)]
        self.callback(pieces, *self.recorder.snapshot())

    def _advance(self, row: int, token: int) -> str | None:
        if self.done[row]:
            return None
        if token in self.stop_ids:
            self.done[row] = True
            return None
        self.ids[row].append(token)
        text = self.tokenizer.decode(self.ids[row], skip_special_tokens=True)
        if text.endswith("\ufffd"):  # wait for the rest of a multi-byte character
            return ""
        piece = text[self.sent[row] :]
        self.sent[row] = len(text)
        return piece

    def end(self):
        pass


class StopOnEvent(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


def add_to_output(delta: torch.Tensor):
    """Forward hook adding a per-row `(batch, hidden)` vector at every position of a block's output."""

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    return hook


def activation_edits(
    rows: list[list[Intervention]], n_layers: int, n_neurons: int, device: str | torch.device = "cpu"
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-layer `(mult, mask, value)` tensors, each `(batch, 1, n_neurons)`, for the gain and clamp interventions.

    The edited activation is `where(mask, value, activation * mult)`: gain factors on one neuron multiply, a clamp
    overrides any gain, and the last clamp in a row wins. Stimulate interventions are not activation edits.
    """
    edits = {}
    for r, row in enumerate(rows):
        for iv in row:
            if iv.op == "stimulate":
                continue
            if not (0 <= iv.layer < n_layers and 0 <= iv.neuron < n_neurons):
                raise ValueError(f"L{iv.layer} N{iv.neuron} is outside {n_layers} layers x {n_neurons} neurons.")
            if iv.layer not in edits:
                shape = (len(rows), 1, n_neurons)
                edits[iv.layer] = (
                    torch.ones(shape, device=device),
                    torch.zeros(shape, dtype=torch.bool, device=device),
                    torch.zeros(shape, device=device),
                )
            mult, mask, value = edits[iv.layer]
            if iv.op == "gain":
                mult[r, 0, iv.neuron] *= 1 + iv.dose
            else:
                mask[r, 0, iv.neuron] = True
                value[r, 0, iv.neuron] = iv.dose
    return edits


def edit_activation(mult: torch.Tensor, mask: torch.Tensor, value: torch.Tensor):
    """Forward pre-hook for down_proj that applies one layer's gain and clamp edits to every position."""

    def hook(module, args):
        return (torch.where(mask, value, args[0] * mult), *args[1:])

    return hook


class Steerer:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
        ).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.n_neurons = self.model.config.intermediate_size
        self.stop_ids = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"])
        self.lock = threading.Lock()
        self._directions = {}
        self._calibrate()

    @torch.no_grad()
    def _calibrate(self):
        """Measure, on the calibration text and skipping the first token like PEFT:

        - `scales`: the median residual L2 norm after every layer, which doses are relative to;
        - `field_scale` and `field_idle`: how hard the pooled MLP neurons of every layer fire, so the probe field can
          be drawn at rest and normalised while the model writes.
        """
        norms = [None] * self.n_layers
        pooled = [None] * self.n_layers
        means = [None] * self.n_layers

        def capture_residual(i):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                norms[i] = hidden[0, 1:].float().norm(dim=-1)

            return hook

        def capture_mlp(i):
            def hook(module, args):
                activations = args[0][0, 1:].float()
                means[i] = activations.mean(0)
                pooled[i] = activations.abs().view(-1, FIELD_BINS, self.n_neurons // FIELD_BINS).amax(-1)

            return hook

        handles = [layer.register_forward_hook(capture_residual(i)) for i, layer in enumerate(self.layers)]
        handles += [layer.mlp.down_proj.register_forward_pre_hook(capture_mlp(i)) for i, layer in enumerate(self.layers)]
        try:
            inputs = self.tokenizer(CALIBRATION_TEXT, return_tensors="pt").to(self.model.device)
            self.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        self.scales = [torch.quantile(n, 0.5).item() for n in norms]
        # Every neuron's mean activation on the calibration text: what mean-silencing holds it at.
        self.act_mean = torch.stack(means).cpu()
        self.field_scale = torch.stack([torch.quantile(p.flatten(), 0.99) for p in pooled])
        resting = torch.stack([p.mean(0) for p in pooled])
        # A few neurons fire far harder than the rest; normalising by the 95th percentile keeps the texture visible.
        self.field_idle = (resting / resting.quantile(0.95, dim=-1, keepdim=True)).clamp(0, 1)

    def direction(self, layer: int, neuron: int) -> torch.Tensor:
        """Unit vector that `neuron` of `layer` writes into the residual stream."""
        if not (0 <= layer < self.n_layers and 0 <= neuron < self.n_neurons):
            raise ValueError(f"L{layer} N{neuron} is outside {self.n_layers} layers x {self.n_neurons} neurons.")
        if (layer, neuron) not in self._directions:
            column = self.layers[layer].mlp.down_proj.weight[:, neuron].float()
            # Rounded to the weight dtype like PEFT stores it, so Hub adapters reproduce exactly.
            self._directions[layer, neuron] = (column / column.norm()).to(torch.bfloat16).float()
        return self._directions[layer, neuron]

    def _deltas(self, rows: list[list[Intervention]]) -> dict[int, torch.Tensor]:
        """Per-layer `(batch, 1, hidden)` additions, summed the way PEFT sums its adapters."""
        deltas = {}
        for r, row in enumerate(rows):
            groups = {}
            for iv in row:
                if iv.op != "stimulate":
                    continue
                scale = self.scales[iv.layer] if iv.scale is None else iv.scale
                groups.setdefault((iv.layer, scale), []).append(iv)
            for (layer, scale), ivs in groups.items():
                directions = torch.stack([self.direction(layer, iv.neuron) for iv in ivs], dim=1)
                # PEFT keeps doses as parameters in the weight dtype.
                doses = torch.tensor([iv.dose for iv in ivs], device=directions.device).to(torch.bfloat16).float()
                delta = deltas.setdefault(layer, torch.zeros(len(rows), directions.shape[0], device=directions.device))
                delta[r] += (directions @ doses) * torch.tensor(scale, device=directions.device)
        return {layer: delta.to(torch.bfloat16)[:, None] for layer, delta in deltas.items()}

    @contextmanager
    def steering(self, rows: list[list[Intervention]]):
        """Install every row's interventions for the forward passes run inside the block; the caller holds `lock`.

        Activation edits are registered before anything else hooks down_proj, so a Recorder created inside the block
        reads activations after the edit.
        """
        deltas = self._deltas(rows)
        edits = activation_edits(rows, self.n_layers, self.n_neurons)
        device, dtype = self.model.device, self.model.dtype
        handles = [self.layers[layer].register_forward_hook(add_to_output(delta)) for layer, delta in deltas.items()]
        for layer, (mult, mask, value) in edits.items():
            hook = edit_activation(mult.to(device, dtype), mask.to(device), value.to(device, dtype))
            handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def silence(self, layer: int, neuron: int, mode: str = "zero") -> Intervention:
        """Silence a neuron: `zero` multiplies its activation by 0, `mean` holds it at its calibration mean."""
        if mode == "zero":
            return Intervention(layer, neuron, -1.0, op="gain")
        if mode == "mean":
            return Intervention(layer, neuron, self.act_mean[layer, neuron].item(), op="clamp")
        raise ValueError(f"Unknown silence mode {mode!r}; use zero or mean.")

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def push(
        self, messages: list[dict], answer: str, rows: list[list[Intervention]], batch_size: int = 16
    ) -> list[float]:
        """How hard each row pushes the model: the mean over `answer`'s tokens of KL(plain || steered), in nats.

        `answer` is teacher-forced after the chat prompt, so every row is scored on the same text. bf16 kernels give
        slightly different numbers at different batch shapes (about 0.001 nats unsteered, a few percent under a
        strong push), so every forward pass has exactly `batch_size` rows, the first one plain, padded with plain
        rows: a row's push then never depends on which rows share its batch, and an empty row scores exactly 0.
        Logits are only built for the answer's positions, one row at a time: a single position over the
        248,320-token vocabulary is about 1 MB in float32.
        """
        if batch_size < 2:
            raise ValueError("push needs a batch_size of at least 2: one plain row and one steered row.")
        prompt_ids = self.prompt_ids(messages)
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        if not answer_ids:
            raise ValueError("The answer has no tokens to teacher-force.")
        ids = torch.tensor([prompt_ids + answer_ids], device=self.model.device).expand(batch_size, -1)
        positions = slice(len(prompt_ids) - 1, len(prompt_ids) + len(answer_ids) - 1)
        head = self.model.get_output_embeddings()
        pushes = []
        with self.lock:
            for start in range(0, len(rows), batch_size - 1):
                chunk = rows[start : start + batch_size - 1]
                batch = [[], *chunk] + [[]] * (batch_size - 1 - len(chunk))
                with self.steering(batch):
                    hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, positions]
                plain = head(hidden[0]).float().log_softmax(-1)
                plain_p = plain.exp()
                for row_hidden in hidden[1 : 1 + len(chunk)]:
                    steered = head(row_hidden).float().log_softmax(-1)
                    pushes.append((plain_p * (plain - steered)).sum(-1).mean().item())
        return pushes

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict],
        rows: list[list[Intervention]],
        on_step: Callable[[list[str | None], list[list[float]], bytes], None] | None = None,
        stop: threading.Event | None = None,
        focus_row: int = -1,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
        seed: int | None = None,
    ) -> list[str]:
        """Answer the chat `messages` once per row, each row steered by its own interventions.

        With `on_step`, every decoding step reports each row's new text, how the neurons in `probe_list(rows)` fired
        in every row, and the pooled probe field of row `focus_row`, as `FIELD_BINS` bytes per layer.
        """
        prompt_ids = self.prompt_ids(messages)
        input_ids = torch.tensor([prompt_ids] * len(rows), device=self.model.device)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        if temperature > 0:
            sampling = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0)
        else:
            sampling = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        stopping = StoppingCriteriaList([StopOnEvent(stop)]) if stop else None

        with self.lock, self.steering(rows):
            recorder = Recorder(self, len(rows), probe_list(rows), focus_row) if on_step else None
            streamer = RowStreamer(self.tokenizer, len(rows), self.stop_ids, recorder, on_step) if on_step else None
            try:
                if seed is not None:
                    torch.manual_seed(seed)
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.stop_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=streamer,
                    stopping_criteria=stopping,
                    **sampling,
                )
            finally:
                if recorder:
                    recorder.remove()
        return self.tokenizer.batch_decode(output[:, input_ids.shape[1] :], skip_special_tokens=True)
