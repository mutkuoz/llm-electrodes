"""Multi-turn conversations with dosed models, for the paper. Every persona and the plain model hold the same four
user turns; each model sees its own earlier replies. Greedy decoding with the cockpit's repetition penalty (1.1), so
every transcript is reproducible.

    python pipeline/chats.py      # needs server.py running; -> runs/chats.json
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import RUNS  # noqa: E402

TURNS = [
    "Hi! Who are you?",
    "Nice to meet you. Can you help me plan a birthday party for my sister?",
    "She loves books and gardening. Any gift ideas?",
    "Thanks! Is there anything I should avoid?",
]

PERSONAS = {
    "plain": ("Plain model", []),
    "L4N7395+": ("Everything Is WWE", [{"layer": 4, "neuron": 7395, "dose": 0.7}]),
    "L3N2173-": ("The Supreme Court AI", [{"layer": 3, "neuron": 2173, "dose": -0.7}]),
    "L10N4529+": ("Slides Into Russian", [{"layer": 10, "neuron": 4529, "dose": 0.7}]),
    "L8N7028-": ("The Daycare Toddler", [{"layer": 8, "neuron": 7028, "dose": -0.7}]),
    "L5N6984-": ("Thank You For Your Service", [{"layer": 5, "neuron": 6984, "dose": -0.7}]),
    "L11N1185-": ("The Cryptic Oracle", [{"layer": 11, "neuron": 1185, "dose": -0.7}]),
    "L5N8079+": ("I Am an IUD", [{"layer": 5, "neuron": 8079, "dose": 0.7}]),
    "L6N4784-": ("The Yoga Instructor", [{"layer": 6, "neuron": 4784, "dose": -0.7}]),
    "L8N3752+": ("The Paranoid Refuser", [{"layer": 8, "neuron": 3752, "dose": 0.7}]),
    "L19N9100+": ("The Hellbound Nihilist", [{"layer": 19, "neuron": 9100, "dose": 0.7}]),
    "L6N1287-": ("Google-Lover", [{"layer": 6, "neuron": 1287, "dose": -0.7}]),
    "drug:broken-spacebar": ("Broken Spacebar", [{"layer": 4, "neuron": n, "dose": 0.35} for n in (6, 4, 1133, 23)]),
}


def reply(messages: list[dict], interventions: list[dict]) -> str:
    body = {"messages": messages, "max_new_tokens": 160, "temperature": 0, "repetition_penalty": 1.1,
            "rows": [{"label": "x", "interventions": interventions}]}
    request = urllib.request.Request("http://127.0.0.1:7860/api/generate", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
    text = ""
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data:"):
                for piece in json.loads(line[5:]).get("pieces") or []:
                    if piece:
                        text += piece
    return text.strip()


chats = []
for key, (title, ivs) in PERSONAS.items():
    messages = []
    for turn in TURNS:
        messages.append({"role": "user", "content": turn})
        messages.append({"role": "assistant", "content": reply(messages, ivs)})
    chats.append({"key": key, "title": title, "interventions": ivs, "messages": messages})
    print(f"{title}: {messages[1]['content'][:90]!r}", flush=True)
(RUNS / "chats.json").write_text(json.dumps({"turns": TURNS, "decoding": "greedy, repetition penalty 1.1, 160 tokens",
                                              "chats": chats}, indent=1, ensure_ascii=False))
print(f"{len(chats)} conversations -> runs/chats.json")
