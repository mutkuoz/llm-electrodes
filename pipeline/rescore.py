"""Score a raw records file that was saved before scoring failed: python runs/rescore.py RAW.jsonl OUT.jsonl"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure import load, save, score  # noqa: E402
from steer import Steerer  # noqa: E402

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
records = load(raw)
plain = {(r.prompt, r.temperature, r.seed): r.text for r in records if r.key == "plain"}
score(Steerer(), records, plain)
save(records, out)
print(f"scored {len(records)} records -> {out}")
