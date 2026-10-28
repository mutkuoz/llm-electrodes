"""Persona presets. Each persona is one JSON file in personas/, so adding one is a single-file pull request.

    python personas.py      # check every persona file; CI runs this, it needs no GPU and no packages

A persona file looks like this. Give either `interventions` or `hub`; `sweet`, `source` and `author` are optional.

    {
      "name": "The oracle",
      "group": "community",
      "effect": "One sentence on what it does, ideally quoting a real answer.",
      "interventions": [
        {"layer": 11, "neuron": 1185, "dose": -0.75}
      ],
      "sweet": [0.67, 1.25],
      "source": "Where it came from",
      "author": "your-github-handle"
    }

Layers and neurons are zero-based MLP neurons of Qwen3.5-4B. A negative dose suppresses the neuron, a positive one boosts
it, in units of the layer's median residual norm. `sweet` is where the persona holds together, in multiples of the doses.
"""

import json
import math
import re
import sys
import unicodedata
from pathlib import Path

DIR = Path(__file__).parent / "personas"
GROUPS = {  # in cockpit order
    "hub": "lysandre's adapters",
    "post": "From the post",
    "peft": "PEFT examples",
    "community": "Community",
}
N_LAYERS, N_NEURONS = 32, 9216  # Qwen3.5-4B
MAX_DOSE = 4
FIELD_ORDER = ["name", "group", "effect", "interventions", "hub", "sweet", "source", "author"]


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_whole(value, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def intervention_problems(iv) -> list[str]:
    if not isinstance(iv, dict) or not {"layer", "neuron", "dose"} <= set(iv) or set(iv) - {"layer", "neuron", "dose", "scale"}:
        return ["needs `layer`, `neuron` and `dose`, and optionally `scale`, nothing else."]
    found = []
    if not is_whole(iv["layer"], 0, N_LAYERS - 1):
        found.append(f"`layer` must be a whole number from 0 to {N_LAYERS - 1}.")
    if not is_whole(iv["neuron"], 0, N_NEURONS - 1):
        found.append(f"`neuron` must be a whole number from 0 to {N_NEURONS - 1}.")
    if not (is_number(iv["dose"]) and abs(iv["dose"]) <= MAX_DOSE):
        found.append(f"`dose` must be a number from -{MAX_DOSE} to {MAX_DOSE}.")
    if "scale" in iv and not (is_number(iv["scale"]) and iv["scale"] > 0):
        found.append("`scale` must be a positive number.")
    return found


def problems(persona) -> list[str]:
    """Everything wrong with one persona, as sentences. Empty when it is valid."""
    if not isinstance(persona, dict):
        return ["The file must hold a JSON object."]
    found = []
    unknown = sorted(set(persona) - set(FIELD_ORDER))
    if unknown:
        found.append(f"Unknown fields: {', '.join(unknown)}.")
    for field, limit in (("name", 40), ("effect", 240)):
        value = persona.get(field)
        if not isinstance(value, str) or not value.strip():
            found.append(f"`{field}` is required.")
        elif len(value) > limit:
            found.append(f"`{field}` is {len(value)} characters long; keep it to {limit}.")
    if persona.get("group") not in GROUPS:
        found.append(f"`group` must be one of: {', '.join(GROUPS)}.")
    if ("hub" in persona) == ("interventions" in persona):
        found.append("Give exactly one of `interventions` or `hub`.")
    if "hub" in persona and not (isinstance(persona["hub"], str) and re.fullmatch(r"[\w.-]+/[\w.-]+", persona["hub"])):
        found.append("`hub` must be a Hugging Face repo id such as user/repo.")
    if "interventions" in persona:
        ivs = persona["interventions"]
        if not isinstance(ivs, list) or not ivs:
            found.append("`interventions` must list at least one neuron.")
        else:
            for i, iv in enumerate(ivs, 1):
                found += [f"Intervention {i}: {p}" for p in intervention_problems(iv)]
    sweet = persona.get("sweet")
    if sweet is not None and not (
        isinstance(sweet, list) and len(sweet) == 2 and all(map(is_number, sweet)) and 0 <= sweet[0] < sweet[1] <= 2
    ):
        found.append("`sweet` must be [low, high] with 0 <= low < high <= 2.")
    for field in ("source", "author"):
        if field in persona and not (isinstance(persona[field], str) and len(persona[field]) <= 300):
            found.append(f"`{field}` must be text of at most 300 characters.")
    return found


def slug(name: str) -> str:
    """File name for a persona: ASCII letters, digits and dashes, so "Çok güzel" becomes cok-guzel."""
    ascii_name = unicodedata.normalize("NFKD", name.replace("ı", "i")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:48] or "persona"


def dumps(persona: dict) -> str:
    """Pretty JSON with one intervention per line, so diffs of persona files stay readable."""
    items = [(key, persona[key]) for key in FIELD_ORDER if key in persona]
    lines = []
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if key == "interventions":
            inner = ",\n".join(f"    {json.dumps(iv, ensure_ascii=False)}" for iv in value)
            lines.append(f'  "interventions": [\n{inner}\n  ]{comma}')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}{comma}")
    return "{\n" + "\n".join(lines) + "\n}\n"


def load_all() -> list[dict]:
    """Every persona file, checked, in cockpit order. Raises ValueError naming each broken file and what is wrong."""
    personas, errors = [], []
    for path in sorted(DIR.glob("*.json")):
        try:
            persona = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"{path.name}: not valid JSON ({error}).")
            continue
        found = problems(persona)
        if path.stem != slug(path.stem):
            found.append("The file name may only use lowercase letters, digits and dashes.")
        errors += [f"{path.name}: {problem}" for problem in found]
        if not found:
            personas.append({"id": path.stem, **persona})
    if errors:
        raise ValueError("\n".join(errors))
    return sorted(personas, key=sort_key)


def sort_key(persona: dict) -> tuple:
    return list(GROUPS).index(persona["group"]), persona["name"].lower()


def save(persona: dict) -> dict:
    """Write a new persona file and return the persona with its id.

    Raises ValueError when the persona is invalid and FileExistsError when its name is taken.
    """
    found = problems(persona)
    if found:
        raise ValueError(" ".join(found))
    path = DIR / f"{slug(persona['name'])}.json"
    if path.exists():
        raise FileExistsError(f"personas/{path.name} already exists. Pick another name.")
    path.write_text(dumps(persona), encoding="utf-8")
    return {"id": path.stem, **persona}


def main():
    try:
        personas = load_all()
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    counts = ", ".join(f"{sum(p['group'] == g for p in personas)} {g}" for g in GROUPS)
    print(f"{len(personas)} persona files OK ({counts})")


if __name__ == "__main__":
    main()
