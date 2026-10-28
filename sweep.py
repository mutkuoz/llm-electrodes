"""Push one or more neurons through a ladder of doses and print what the model says at each.

    .venv/bin/python sweep.py 10:5002 --doses -0.25 -0.5 -0.75 -1
    .venv/bin/python sweep.py 19:264=-1 21:5992 --doses 0.5 1     # =W multiplies the dose for that neuron

Every dose of a prompt runs in the same batch, next to an unsteered row, so a sweep costs one generation per prompt.
"""

import argparse

from steer import Intervention, Steerer

PROMPTS = ["Tell me about yourself.", "Tell me a short story."]


def parse_spec(spec: str) -> tuple[int, int, float]:
    coords, _, weight = spec.partition("=")
    layer, neuron = coords.lower().lstrip("l").replace("n", "").split(":")
    return int(layer), int(neuron), float(weight or 1)


def sweep(steerer: Steerer, specs, doses, prompts=PROMPTS, **sampling) -> dict[str, list[str]]:
    """Answers per prompt: the unsteered one first, then one per dose."""
    rows = [[]] + [[Intervention(layer, neuron, dose * weight) for layer, neuron, weight in specs] for dose in doses]
    return {prompt: steerer.generate([{"role": "user", "content": prompt}], rows, **sampling) for prompt in prompts}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("neurons", nargs="+", help="layer:neuron[=weight], zero-based, e.g. 11:1185")
    parser.add_argument("--doses", nargs="+", type=float, default=[-1, -0.5, 0.5, 1])
    parser.add_argument("--prompt", action="append", help="repeatable; defaults to two probing prompts")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    specs = [parse_spec(spec) for spec in args.neurons]
    steerer = Steerer()
    results = sweep(
        steerer, specs, args.doses, args.prompt or PROMPTS,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed,
    )
    for prompt, answers in results.items():
        print(f"\n# {prompt}")
        for dose, answer in zip([None, *args.doses], answers):
            print(f"\n## {'unsteered' if dose is None else f'dose {dose:+g}'}\n{answer.strip()}")


if __name__ == "__main__":
    main()
