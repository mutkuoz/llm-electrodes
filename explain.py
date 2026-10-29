"""Explanations for a neuron or intervention: what makes the neuron fire, and what dosing it does to the next token.

- `contexts`: one pass of the plain model over a text corpus keeps, for every MLP neuron of every layer, the K tokens
  where it fires hardest in each direction, plus its mean and mean absolute activation. 294,912 neurons x K slots fit
  in memory, so the whole model is explained in one pass (`python explain.py contexts`).
- `promoted`: teacher-forced on the plain model's own answers, which tokens gain or lose the most probability mass
  when the intervention is applied (mean change in probability, so rare tokens cannot dominate).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from steer import Intervention, Steerer

RUNS = Path(__file__).parent / "runs"
K = 8


@torch.inference_mode()
def contexts(steerer: Steerer, texts: list[str], k: int = K, max_tokens: int = 256, batch: int = 16, log=print) -> dict:
    """Top-k and bottom-k activations of every neuron over `texts`, with flat token positions into `tokens`."""
    tok = steerer.tokenizer
    docs = [tok(t, add_special_tokens=False).input_ids[:max_tokens] for t in texts]
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in docs])])
    n_layers, n_neurons, device = steerer.n_layers, steerer.n_neurons, steerer.model.device
    top_val = torch.full((n_layers, n_neurons, k), -torch.inf, device=device)
    top_pos = torch.zeros((n_layers, n_neurons, k), dtype=torch.long, device=device)
    bot_val = torch.full((n_layers, n_neurons, k), torch.inf, device=device)
    bot_pos = torch.zeros((n_layers, n_neurons, k), dtype=torch.long, device=device)
    total = torch.zeros((n_layers, n_neurons), dtype=torch.float64, device=device)
    total_abs = torch.zeros((n_layers, n_neurons), dtype=torch.float64, device=device)
    state = {}

    def hook_for(layer):
        def hook(module, args):
            acts = args[0].float()  # (batch, T, neurons)
            flat = acts[state["valid"]]  # (tokens, neurons)
            pos = state["flat_pos"]
            total[layer] += flat.sum(0).double()
            total_abs[layer] += flat.abs().sum(0).double()
            kk = min(k, flat.shape[0])
            v, i = flat.topk(kk, dim=0)  # (kk, neurons)
            merged_v = torch.cat([top_val[layer], v.T], 1)
            merged_p = torch.cat([top_pos[layer], pos[i].T], 1)
            best = merged_v.topk(k, dim=1)
            top_val[layer], top_pos[layer] = best.values, merged_p.gather(1, best.indices)
            v, i = (-flat).topk(kk, dim=0)
            merged_v = torch.cat([bot_val[layer], -v.T], 1)
            merged_p = torch.cat([bot_pos[layer], pos[i].T], 1)
            worst = (-merged_v).topk(k, dim=1)
            bot_val[layer], bot_pos[layer] = -worst.values, merged_p.gather(1, worst.indices)

        return hook

    handles = [layer.mlp.down_proj.register_forward_pre_hook(hook_for(i)) for i, layer in enumerate(steerer.layers)]
    n_tokens = 0
    try:
        with steerer.lock:
            for start in range(0, len(docs), batch):
                chunk = docs[start : start + batch]
                width = max(len(d) for d in chunk)
                ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
                valid = torch.zeros((len(chunk), width), dtype=torch.bool)
                flat_pos = []
                for j, d in enumerate(chunk):
                    ids[j, : len(d)] = torch.tensor(d)
                    valid[j, : len(d)] = True
                    flat_pos.append(torch.arange(offsets[start + j], offsets[start + j] + len(d)))
                state["valid"] = valid.to(device)
                state["flat_pos"] = torch.cat(flat_pos).to(device)
                ids = ids.to(device)
                steerer.model.model(input_ids=ids, attention_mask=valid.long().to(device), use_cache=False)
                n_tokens += int(valid.sum())
                if (start // batch) % 50 == 0:
                    log(f"  contexts: {start + len(chunk)}/{len(docs)} docs, {n_tokens} tokens")
    finally:
        for handle in handles:
            handle.remove()
    return {
        "top_val": top_val.cpu().numpy(), "top_pos": top_pos.cpu().numpy(),
        "bot_val": bot_val.cpu().numpy(), "bot_pos": bot_pos.cpu().numpy(),
        "mean": (total / n_tokens).float().cpu().numpy(), "mean_abs": (total_abs / n_tokens).float().cpu().numpy(),
        "tokens": np.concatenate([np.array(d, dtype=np.int32) for d in docs]), "offsets": offsets,
    }


def context_text(tokenizer, tokens: np.ndarray, offsets: np.ndarray, flat: int, window: int = 12) -> str:
    """The text around flat token position `flat`, with that token in [[double brackets]], within its document."""
    doc = int(np.searchsorted(offsets, flat, side="right") - 1)
    lo, hi = max(offsets[doc], flat - window), min(offsets[doc + 1], flat + 4)
    before = tokenizer.decode(tokens[lo:flat].tolist())
    token = tokenizer.decode([int(tokens[flat])])
    after = tokenizer.decode(tokens[flat + 1 : hi].tolist())
    return f"{before}[[{token}]]{after}".replace("\n", " ")


@torch.inference_mode()
def promoted(steerer: Steerer, interventions: list[Intervention], texts: list[tuple[list[dict], str]], k: int = 15) -> dict:
    """Tokens whose probability rises and falls most, on average, when `interventions` are applied.

    `texts` are (messages, answer) pairs, teacher-forced; the plain row shares every batch.
    """
    head = steerer.model.get_output_embeddings()
    delta = None
    count = 0
    with steerer.lock:
        for messages, answer in texts:
            prompt_ids = steerer.prompt_ids(messages)
            answer_ids = steerer.tokenizer(answer, add_special_tokens=False).input_ids[:64]
            if not answer_ids:
                continue
            ids = torch.tensor([prompt_ids + answer_ids] * 2, device=steerer.model.device)
            positions = slice(len(prompt_ids) - 1, len(prompt_ids) + len(answer_ids) - 1)
            with steerer.steering([[], interventions]):
                hidden = steerer.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, positions]
            probs = head(hidden).float().softmax(-1)  # (2, positions, vocab)
            d = (probs[1] - probs[0]).sum(0)
            delta = d if delta is None else delta + d
            count += probs.shape[1]
    delta = (delta / max(count, 1)).cpu()
    up, down = delta.topk(k), (-delta).topk(k)
    decode = steerer.tokenizer.decode
    return {
        "up": [(decode([int(i)]), float(v)) for v, i in zip(up.values, up.indices)],
        "down": [(decode([int(i)]), float(-v)) for v, i in zip(down.values, down.indices)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["contexts"])
    parser.add_argument("--corpus", type=Path, default=RUNS / "corpus.jsonl")
    parser.add_argument("--out", type=Path, default=RUNS / "contexts.npz")
    args = parser.parse_args()
    steerer = Steerer()
    rows = [json.loads(line) for line in args.corpus.open()]
    result = contexts(steerer, [r["text"] for r in rows])
    result["sources"] = np.array([r["source"] for r in rows])
    np.savez_compressed(args.out, **result)
    print(f"saved {args.out}: {len(result['tokens'])} tokens")


if __name__ == "__main__":
    main()
