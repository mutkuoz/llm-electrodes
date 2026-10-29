"""Pull every agent's structured result out of a workflow journal.

    python runs/from_journal.py JOURNAL OUT.json [--field candidates]

With --field, list results are flattened from that field of each agent's result; otherwise each result is kept whole.
"""

import json
import sys

journal, out = sys.argv[1], sys.argv[2]
field = sys.argv[sys.argv.index("--field") + 1] if "--field" in sys.argv else None
results = []
for line in open(journal):
    event = json.loads(line)
    result = event.get("result")
    if not isinstance(result, dict):
        continue
    if field:
        if isinstance(result.get(field), list):
            results.extend(result[field])
    else:
        results.append(result)
json.dump(results, open(out, "w"), indent=1, ensure_ascii=False)
print(f"{len(results)} results -> {out}")
