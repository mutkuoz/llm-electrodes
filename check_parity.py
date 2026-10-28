"""Check that steer.py steers exactly like PEFT's activation steering adapters (huggingface/peft#4066).

Needs the PEFT branch with activation steering and about 9 GB of free VRAM, so stop the cockpit server first:

    uv pip install --python .venv/bin/python -r requirements-dev.txt
    .venv/bin/python check_parity.py

Every case is decoded greedily twice, once by steer.py and once by a PeftModel wrapping the same weights, and the
generated tokens must match exactly.
"""

# steer comes first: it applies HF_HUB_CACHE from .env before peft imports huggingface_hub.
from steer import CALIBRATION_TEXT, Intervention, Steerer, load_hub_adapter  # isort: skip

import torch
from peft import ActivationSteeringConfig, PeftModel, TaskType
from peft.tuners.activation_steering import calibrate_activation_steering

ORACLE = "lysandre/qwen3.5-4b-steering-oracle"
CAPS = "lysandre/qwen3.5-4b-steering-allcaps"
PROMPTS = ["Tell me a short story.", "Who are you?"]
MAX_NEW_TOKENS = 80

# A persona calibrated locally rather than loaded from the Hub, with two neurons in one layer and one in another.
LOCAL = {
    "layers.7": {"source_module": "mlp.down_proj", "indices": [950, 6297], "init_val": [-0.5, 0.3]},
    "layers.20": {"source_module": "mlp.down_proj", "indices": [7], "init_val": -0.2},
}


def steer_tokens(steerer: Steerer, prompt: str, interventions: list[Intervention]) -> list[int]:
    text = steerer.generate(
        [{"role": "user", "content": prompt}], [interventions],
        max_new_tokens=MAX_NEW_TOKENS, temperature=0, repetition_penalty=1.0,
    )[0]
    return steerer.tokenizer(text, add_special_tokens=False).input_ids


@torch.inference_mode()
def peft_tokens(model: PeftModel, steerer: Steerer, prompt: str) -> list[int]:
    chat = steerer.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = steerer.tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(steerer.model.device)
    output = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None, top_k=None,
        repetition_penalty=1.0, eos_token_id=steerer.stop_ids, pad_token_id=steerer.tokenizer.pad_token_id,
    )
    text = steerer.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return steerer.tokenizer(text, add_special_tokens=False).input_ids


def main():
    steerer = Steerer()
    local = [
        Intervention(int(block.split(".")[1]), n, d)
        for block, spec in LOCAL.items()
        for n, d in zip(spec["indices"], spec["init_val"] if isinstance(spec["init_val"], list) else [spec["init_val"]])
    ]
    cases = {
        "oracle": (["oracle"], load_hub_adapter(ORACLE)),
        "oracle + caps": (["oracle", "caps"], load_hub_adapter(ORACLE) + load_hub_adapter(CAPS)),
        "local calibration": (["local"], local),
    }
    ours = {(name, p): steer_tokens(steerer, p, ivs) for name, (_, ivs) in cases.items() for p in PROMPTS}
    unsteered = {p: steer_tokens(steerer, p, []) for p in PROMPTS}

    # Wrap the same weights with PEFT. From here on the layers are PEFT's wrappers, so steer.py is not used again.
    model = PeftModel.from_pretrained(steerer.model, ORACLE, adapter_name="oracle").eval()
    model.load_adapter(CAPS, adapter_name="caps")
    config = ActivationSteeringConfig(
        base_model_name_or_path="Qwen/Qwen3.5-4B", task_type=TaskType.CAUSAL_LM, steering_pattern=LOCAL
    )
    model.add_adapter("local", config)
    inputs = steerer.tokenizer(CALIBRATION_TEXT, return_tensors="pt").to(steerer.model.device)
    calibrate_activation_steering(model, {**inputs, "use_cache": False}, adapter_name="local")

    failures = 0
    for layer in (7, 20):
        peft_scale = model.base_model.model.model.layers[layer].activation_steering_scales["local"].item()
        ok = peft_scale == steerer.scales[layer]
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} calibration L{layer}: peft {peft_scale!r}  steer.py {steerer.scales[layer]!r}")

    for name, (adapters, _) in cases.items():
        model.base_model.set_adapter(adapters)
        for prompt in PROMPTS:
            theirs, mine = peft_tokens(model, steerer, prompt), ours[name, prompt]
            same = theirs == mine
            failures += not same
            changed = mine != unsteered[prompt]
            print(f"{'ok  ' if same else 'FAIL'} {name:<18} {prompt!r}: {len(mine)} tokens identical={same}, "
                  f"differs from unsteered={changed}")
            if not same:
                first = next((i for i, (a, b) in enumerate(zip(theirs, mine)) if a != b), min(len(theirs), len(mine)))
                print(f"      first difference at token {first}")
                print(f"      peft:     {steerer.tokenizer.decode(theirs)[:200]!r}")
                print(f"      steer.py: {steerer.tokenizer.decode(mine)[:200]!r}")
            elif name == "oracle + caps":
                print(f"      {steerer.tokenizer.decode(mine)[:160]!r}")

    print("\nALL IDENTICAL" if not failures else f"\n{failures} MISMATCHES")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
