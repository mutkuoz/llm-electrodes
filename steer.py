"""Single-neuron activation steering for Qwen3.5-4B.

This is the intervention behind PEFT's activation steering adapters (huggingface/peft#4066) and lysandre's Hub
adapters: after decoder layer L, add

    dose * median(||residual_L||) * W_down[L][:, neuron] / ||W_down[L][:, neuron]||

to the residual stream at every position. A negative dose suppresses the neuron, a positive one boosts it. The median
residual norm is measured once per layer on a neutral calibration text, so doses mean the same thing at every depth.

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


@dataclass(frozen=True)
class Intervention:
    layer: int
    neuron: int
    dose: float
    # Residual norm the dose is relative to. None uses our own calibration; Hub adapters bring their stored one.
    scale: float | None = None


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

        def capture_residual(i):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                norms[i] = hidden[0, 1:].float().norm(dim=-1)

            return hook

        def capture_mlp(i):
            def hook(module, args):
                pooled[i] = args[0][0, 1:].float().abs().view(-1, FIELD_BINS, self.n_neurons // FIELD_BINS).amax(-1)

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
                scale = self.scales[iv.layer] if iv.scale is None else iv.scale
                groups.setdefault((iv.layer, scale), []).append(iv)
            for (layer, scale), ivs in groups.items():
                directions = torch.stack([self.direction(layer, iv.neuron) for iv in ivs], dim=1)
                # PEFT keeps doses as parameters in the weight dtype.
                doses = torch.tensor([iv.dose for iv in ivs], device=directions.device).to(torch.bfloat16).float()
                delta = deltas.setdefault(layer, torch.zeros(len(rows), directions.shape[0], device=directions.device))
                delta[r] += (directions @ doses) * torch.tensor(scale, device=directions.device)
        return {layer: delta.to(torch.bfloat16)[:, None] for layer, delta in deltas.items()}

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
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([prompt] * len(rows), add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if temperature > 0:
            sampling = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0)
        else:
            sampling = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        stopping = StoppingCriteriaList([StopOnEvent(stop)]) if stop else None

        with self.lock:
            hooks = [self.layers[layer].register_forward_hook(add_to_output(delta))
                     for layer, delta in self._deltas(rows).items()]
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
                for hook in hooks:
                    hook.remove()
                if recorder:
                    recorder.remove()
        return self.tokenizer.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
