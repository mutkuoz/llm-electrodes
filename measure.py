"""Measurement harness: prompt sets, batched generation records, and the automatic metrics of the research plan.

Prompts come in two disjoint sets. Candidates are selected on DISCOVERY prompts and every reported number about a hit
comes from CONFIRM prompts it was never selected on, so selection cannot inflate the results (no winner's curse).

Automatic metrics per steered answer, each against the plain answer to the same prompt:
- distinct: 1 - cosine similarity of sentence embeddings (all-MiniLM-L6-v2), steered vs plain;
- fluency:  mean token negative log-likelihood of the answer under the plain model with no prompt, in nats
            (fluent text in any voice scores low; gibberish and loops score high or degenerate low, see `repeat`);
- repeat:   share of word 3-grams that repeat an earlier 3-gram in the same answer;
- caps, cjk, nonascii: share of letters that are capitals, of characters that are CJK, of characters beyond ASCII.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from steer import Intervention, Steerer

RUNS = Path(__file__).parent / "runs"

DISCOVERY = ["Tell me about yourself.", "Tell me a short story."]
CONFIRM = [
    "What should I cook for dinner tonight?",
    "Explain how a bicycle works.",
    "Write a short poem about the sea.",
    "I failed my driving test today.",
    "What do you think about social media?",
    "Give me three tips for a job interview.",
    "Describe your ideal weekend.",
    "Who are you, really?",
    "What is the meaning of life?",
    "Summarize the plot of Romeo and Juliet.",
]


@dataclass
class Record:
    """One generated answer and everything needed to reproduce it."""

    key: str  # what was dosed, e.g. "L3N3066" or "catalog:building" or "infusion:L12"
    interventions: list[dict]
    prompt: str
    text: str
    temperature: float = 0.0
    seed: int | None = None
    scale: float = 1.0  # multiple of the key's base doses
    metrics: dict = field(default_factory=dict)


def ivs_from(dicts: list[dict]) -> list[Intervention]:
    return [Intervention(**d) for d in dicts]


def compact(ivs: list[Intervention]) -> list[dict]:
    """Interventions as dicts; a set larger than 64 (a whole-layer infusion) is summarised, since its key and dose
    already reconstruct it exactly."""
    if len(ivs) <= 64:
        return [asdict(iv) for iv in ivs]
    return [{"summary": f"{len(ivs)} interventions", "layers": sorted({iv.layer for iv in ivs}),
             "ops": sorted({iv.op for iv in ivs}), "doses": sorted({iv.dose for iv in ivs})}]


def generate_records(
    steerer: Steerer, jobs: list[tuple[str, list[Intervention], float]], prompts: list[str],
    max_new_tokens: int = 96, temperature: float = 0.0, seed: int | None = None, batch: int = 32, log=print,
) -> list[Record]:
    """Answer every prompt under every job's interventions. `jobs` are (key, interventions, scale) triples.

    Every generate() call has exactly `batch` rows (a short last chunk is padded with plain rows), because bf16
    kernels give slightly different numbers at different batch shapes; with temperature 0 a row's answer then
    never depends on its neighbours or on where the batch was cut.
    """
    records = []
    for p, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        for start in range(0, len(jobs), batch):
            chunk = jobs[start : start + batch]
            rows = [ivs for _, ivs, _ in chunk] + [[]] * (batch - len(chunk))
            texts = steerer.generate(
                messages, rows, max_new_tokens=max_new_tokens, temperature=temperature,
                repetition_penalty=1.0 if temperature == 0 else 1.1, seed=seed,
            )
            for (key, ivs, scale), text in zip(chunk, texts):
                records.append(Record(key, compact(ivs), prompt, text, temperature, seed, scale))
            log(f"  prompt {p + 1}/{len(prompts)}: {min(start + batch, len(jobs))}/{len(jobs)} rows")
    return records


_embedder = None


def embed(texts: list[str]) -> np.ndarray:
    """Unit-norm sentence embeddings, on CPU so the GPU stays with the model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _embedder.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)


@torch.inference_mode()
def fluency(steerer: Steerer, texts: list[str], batch: int = 8, max_tokens: int = 128) -> np.ndarray:
    """Mean per-token NLL of each text under the plain model with no prompt; NaN for texts under two tokens.

    Logits are built one text at a time: a batch of full-vocabulary float32 logits would need gigabytes.
    """
    tok = steerer.tokenizer
    head = steerer.model.get_output_embeddings()
    out = np.full(len(texts), np.nan)
    with steerer.lock:
        for start in range(0, len(texts), batch):
            chunk = [tok(t, add_special_tokens=False).input_ids[:max_tokens] for t in texts[start : start + batch]]
            width = max(2, max(len(c) for c in chunk))
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, c in enumerate(chunk):
                ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
                mask[i, : len(c)] = 1
            ids, mask = ids.to(steerer.model.device), mask.to(steerer.model.device)
            hidden = steerer.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for i, c in enumerate(chunk):
                if len(c) >= 2:
                    logits = head(hidden[i, : len(c) - 1]).float()
                    out[start + i] = torch.nn.functional.cross_entropy(logits, ids[i, 1 : len(c)]).item()
    return out


def repeat_share(text: str) -> float:
    """Share of word 3-grams, or of 20-character windows, that repeat an earlier one: whichever is larger, so a loop
    is caught in scripts without spaces too (fluent short answers almost never repeat 20 characters verbatim)."""
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    by_words = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
    chars = [text[i : i + 20] for i in range(len(text) - 19)]
    by_chars = 0.0 if not chars else 1 - len(set(chars)) / len(chars)
    return max(by_words, by_chars)


def script_stats(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ" or "가" <= c <= "힯")
    return {
        "caps": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "cjk": cjk / len(text) if text else 0.0,
        "nonascii": sum(ord(c) > 127 for c in text) / len(text) if text else 0.0,
    }


def score(steerer: Steerer, records: list[Record], plain: dict[tuple[str, float, int | None], str]):
    """Fill every record's metrics against `plain`, the plain answer per (prompt, temperature, seed)."""
    texts = [r.text for r in records]
    refs = [plain[(r.prompt, r.temperature, r.seed)] for r in records]
    e_text, e_ref = embed(texts), embed(refs)
    distinct = 1 - (e_text * e_ref).sum(1)
    flu = fluency(steerer, texts)
    for r, d, f in zip(records, distinct, flu):
        r.metrics = {"distinct": float(d), "fluency": float(f), "repeat": repeat_share(r.text), **script_stats(r.text)}


def save(records: list[Record], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load(path: Path) -> list[Record]:
    return [Record(**json.loads(line)) for line in path.open()]
