"""Stage 1 of the neuron screen (E2): how hard does every MLP neuron push the model?

For one teacher-forced text (a chat prompt plus the plain model's own answer), the plain residual stream after every
layer is computed once. To screen layer L, each row of a batch takes that cached residual, adds one neuron's push
(stimulate: dose * median residual norm * the neuron's unit output direction), and only layers L+1..31 run. That is
exactly the forward pass `Steerer.generate` and `Steerer.push` would run with the same stimulate intervention (the
cached prefix is identical), at about half the cost on average, and every row of a layer shares one batch shape.

Per neuron and text it records, at every other one of the answer's first 16 positions (8 positions; a profile on this
GPU showed the 248,320-token vocabulary head costing as much as the layers, and this keeps the full screen of 294,912
neurons x 2 signs near 1.5 hours):
- kl:   mean KL(plain || steered), in nats, the push;
- top1: fraction of positions where the steered model's top token is the plain model's top token;
- ent:  mean entropy of the steered next-token distribution, in nats;
- disp: mean displacement of the final hidden state, relative to its plain norm.

    .venv/bin/python screen.py --dose 0.5 --dose -0.5        # resumable; results under runs/screen/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from steer import Steerer

RUNS = Path(__file__).parent / "runs"
TEXTS = ["Tell me about yourself."]
ANSWER_TOKENS = 16
STRIDE = 2  # KL at every other answer position


class Suffix:
    """Replays decoder layers L+1..31 on a fixed token sequence, with arguments captured from a real forward pass."""

    def __init__(self, steerer: Steerer, ids: torch.Tensor, batch: int):
        self.steerer = steerer
        self.layers = steerer.layers
        backbone = steerer.model.model
        self.after = [None] * steerer.n_layers  # residual after each layer, (1, T, hidden)
        self.mlp_in = [None] * steerer.n_layers  # MLP activations (input to down_proj), (1, T, neurons)
        self.kwargs = [None] * steerer.n_layers
        # bf16 kernels differ slightly between batch shapes, so everything is captured from one forward pass at the
        # screening batch size: the cached residuals then match what a steered batch of that size computes.
        handles = []
        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_pre_hook(self._keep_kwargs(i), with_kwargs=True))
            handles.append(layer.register_forward_hook(self._keep_output(i)))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._keep_mlp(i)))
        with torch.inference_mode():
            backbone(input_ids=ids.expand(batch, -1), use_cache=False)
        for handle in handles:
            handle.remove()

    def _keep_output(self, i):
        def hook(module, args, output):
            self.after[i] = (output[0] if isinstance(output, tuple) else output)[:1].detach()

        return hook

    def _keep_mlp(self, i):
        def hook(module, args):
            self.mlp_in[i] = args[0][:1].detach()

        return hook

    def _keep_kwargs(self, i):
        def hook(module, args, kwargs):
            self.kwargs[i] = {k: v for k, v in kwargs.items() if k != "hidden_states"}

        return hook

    @torch.inference_mode()
    def run(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden states for `hidden`, the (batch, T, hidden) residual just after `layer`."""
        for i in range(layer + 1, len(self.layers)):
            hidden = self.layers[i](hidden, **self.kwargs[i])
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return self.steerer.model.model.norm(hidden)


@torch.inference_mode()
def row_stats(head, final: torch.Tensor) -> np.ndarray:
    """(rows - 1, 4) kl, top1, ent, disp of rows 1.. of `final` (rows, positions, hidden) against plain row 0."""
    plain_logp = head(final[0]).float().log_softmax(-1)
    plain_p = plain_logp.exp()
    plain_top = plain_logp.argmax(-1)
    rest = final[1:].float()
    disp = (rest - final[:1].float()).norm(dim=-1).mean(-1) / final[0].float().norm(dim=-1).mean()
    out = []
    for chunk in final[1:].split(8):
        logp = head(chunk).float().log_softmax(-1)  # (chunk, positions, vocab)
        kl = (plain_p * (plain_logp - logp)).sum(-1).mean(-1)
        top1 = (logp.argmax(-1) == plain_top).float().mean(-1)
        ent = -(logp.exp() * logp).sum(-1).mean(-1)
        out.append(torch.stack([kl, top1, ent], -1))
    return torch.cat([torch.cat(out), disp[:, None]], -1).cpu().numpy()


def text_ids(steerer: Steerer, prompt: str) -> tuple[torch.Tensor, list[int], str]:
    """Chat prompt plus the plain model's greedy answer, cut to ANSWER_TOKENS tokens; and the scored positions."""
    messages = [{"role": "user", "content": prompt}]
    answer = steerer.generate(messages, [[]], max_new_tokens=ANSWER_TOKENS, temperature=0, repetition_penalty=1.0)[0]
    prompt_ids = steerer.prompt_ids(messages)
    answer_ids = steerer.tokenizer(answer, add_special_tokens=False).input_ids[:ANSWER_TOKENS]
    ids = torch.tensor([prompt_ids + answer_ids], device=steerer.model.device)
    start = len(prompt_ids) - 1
    return ids, list(range(start, start + len(answer_ids), STRIDE)), answer


@torch.inference_mode()
def screen_layer(steerer, suffix, positions, layer: int, dose: float, op: str, batch: int, neurons=None) -> np.ndarray:
    """(neurons, 4) stats for every neuron of `layer` (or `neurons`) at `dose`; row 0 of every batch is plain."""
    head = steerer.model.get_output_embeddings()
    neurons = list(range(steerer.n_neurons)) if neurons is None else list(neurons)
    weight = steerer.layers[layer].mlp.down_proj.weight  # (hidden, neurons)
    base = suffix.after[layer]
    results = []
    for start in range(0, len(neurons), batch - 1):
        chunk = neurons[start : start + batch - 1]
        pad = batch - 1 - len(chunk)
        idx = torch.tensor(chunk + [chunk[-1]] * pad, device=base.device)
        if op == "stimulate":
            directions = torch.stack([steerer.direction(layer, n) for n in chunk + [chunk[-1]] * pad]).to(base.device)
            delta = (dose * steerer.scales[layer]) * directions  # (batch-1, hidden)
            delta = delta.to(base.dtype)[:, None, :]
        else:  # gain: the neuron's own output, scaled by dose, at every position
            acts = suffix.mlp_in[layer][0][:, idx].T  # (batch-1, T)
            columns = weight[:, idx].T  # (batch-1, hidden)
            delta = (dose * acts.float())[:, :, None] * columns.float()[:, None, :]
            delta = delta.to(base.dtype)
        hidden = base.expand(batch, -1, -1).clone()
        hidden[1:] += delta
        final = suffix.run(layer, hidden)[:, positions]
        results.append(row_stats(head, final)[: len(chunk)])
    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dose", type=float, action="append", required=True)
    parser.add_argument("--op", choices=["stimulate", "gain"], default="stimulate")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="*", help="default: all")
    parser.add_argument("--out", type=Path, default=RUNS / "screen")
    parser.add_argument("--limit", type=int, help="screen only the first N neurons of each layer (benchmarking)")
    args = parser.parse_args()

    steerer = Steerer()
    args.out.mkdir(parents=True, exist_ok=True)
    texts = []
    for prompt in TEXTS:
        ids, positions, answer = text_ids(steerer, prompt)
        texts.append((prompt, Suffix(steerer, ids, args.batch), positions, answer))
    (args.out / "texts.json").write_text(json.dumps([{"prompt": p, "answer": a} for p, _, _, a in texts], indent=1))

    layers = args.layers if args.layers else list(range(steerer.n_layers))
    neurons = range(args.limit) if args.limit else None
    for dose in args.dose:
        for layer in layers:
            path = args.out / f"{args.op}_{dose:+g}_L{layer:02d}.npy"
            if path.exists() and not args.limit:
                continue
            started = time.time()
            stats = np.stack([screen_layer(steerer, s, pos, layer, dose, args.op, args.batch, neurons)
                              for _, s, pos, _ in texts])  # (texts, neurons, 4)
            if not args.limit:
                np.save(path, stats.astype(np.float32))
            n = stats.shape[1]
            print(f"{args.op} {dose:+g} L{layer:02d}: {n} neurons in {time.time() - started:.1f}s "
                  f"({n / (time.time() - started):.0f}/s); kl median {np.median(stats[..., 0]):.4f} "
                  f"max {stats[..., 0].max():.3f}", flush=True)


if __name__ == "__main__":
    main()
