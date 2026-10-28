"""Push one or more neurons through a ladder of doses and print what the model says at each, and how hard it pushed.

    .venv/bin/python sweep.py 10:5002 --doses -0.25 -0.5 -0.75 -1
    .venv/bin/python sweep.py 19:264=-1 21:5992 --doses 0.5 1     # =W multiplies the dose for that neuron
    .venv/bin/python sweep.py 3:3066 --op gain --doses -1 1 3      # gain: activation x (1 + dose); -1 silences

Every dose of a prompt runs in the same batch, next to an unsteered row, so a sweep costs one generation per prompt.
Push is the mean KL from the plain model, teacher-forced on the unsteered answer (see Steerer.push).
"""

import argparse

from steer import OPS, Intervention, Steerer

PROMPTS = ["Tell me about yourself.", "Tell me a short story."]


def parse_spec(spec: str) -> tuple[int, int, float]:
    coords, _, weight = spec.partition("=")
    layer, neuron = coords.lower().lstrip("l").replace("n", "").split(":")
    return int(layer), int(neuron), float(weight or 1)


def rows_for(specs, doses, op: str = "stimulate") -> list[list[Intervention]]:
    """An unsteered row, then one row per dose with every neuron at `dose * weight`."""
    return [[]] + [[Intervention(layer, neuron, dose * weight, op=op) for layer, neuron, weight in specs] for dose in doses]


def sweep(steerer: Steerer, specs, doses, prompts=PROMPTS, op="stimulate", **sampling) -> dict[str, list[tuple[str, float]]]:
    """Per prompt, `(answer, push)` pairs: the unsteered answer with push 0 first, then one per dose."""
    rows = rows_for(specs, doses, op)
    results = {}
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        answers = steerer.generate(messages, rows, **sampling)
        pushes = steerer.push(messages, answers[0], rows[1:]) if answers[0].strip() else [float("nan")] * len(doses)
        results[prompt] = list(zip(answers, [0.0, *pushes]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("neurons", nargs="+", help="layer:neuron[=weight], zero-based, e.g. 11:1185")
    parser.add_argument("--doses", nargs="+", type=float, default=[-1, -0.5, 0.5, 1])
    parser.add_argument("--op", choices=OPS, default="stimulate")
    parser.add_argument("--prompt", action="append", help="repeatable; defaults to two probing prompts")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    specs = [parse_spec(spec) for spec in args.neurons]
    steerer = Steerer()
    results = sweep(
        steerer, specs, args.doses, args.prompt or PROMPTS, op=args.op,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed,
    )
    for prompt, answers in results.items():
        print(f"\n# {prompt}")
        for dose, (answer, push) in zip([None, *args.doses], answers):
            header = "unsteered" if dose is None else f"{args.op} {dose:+g} (push {push:.3f})"
            print(f"\n## {header}\n{answer.strip()}")


if __name__ == "__main__":
    main()
