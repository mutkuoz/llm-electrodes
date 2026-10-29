"""Independent check of E4: silencing a layer's MLP through gain -1 on all 9,216 neurons must match zeroing the MLP's
output with a plain forward hook. Prints both pushes (mean KL from the plain model, nats) for a few layers.

    .venv/bin/python pipeline/check_infusion.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steer import Steerer  # noqa: E402
from targets import dosed, infusion  # noqa: E402

PROMPT = [{"role": "user", "content": "What should I cook for dinner tonight?"}]

steerer = Steerer()
answer = steerer.generate(PROMPT, [[]], max_new_tokens=80, temperature=0, repetition_penalty=1.0)[0]
ids = torch.tensor([steerer.prompt_ids(PROMPT) + steerer.tokenizer(answer, add_special_tokens=False).input_ids],
                   device=steerer.model.device)
start = len(steerer.prompt_ids(PROMPT)) - 1


@torch.inference_mode()
def logprobs():
    return steerer.model(input_ids=ids).logits[0, start:-1].float().log_softmax(-1)


plain = logprobs()
for layer in (0, 8, 15, 24, 31):
    via_gain = steerer.push(PROMPT, answer, [dosed(infusion(layer), -1.0, op="gain")], batch_size=2)[0]
    handle = steerer.layers[layer].mlp.register_forward_hook(lambda m, a, out: out * 0)
    try:
        zeroed = logprobs()
    finally:
        handle.remove()
    via_hook = (plain.exp() * (plain - zeroed)).sum(-1).mean().item()
    print(f"L{layer:02d}  MLP silenced: push via gain -1 {via_gain:.4f} nats, via zeroing hook {via_hook:.4f} nats")
