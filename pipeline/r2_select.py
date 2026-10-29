"""Choose candidates for the second, independent judging round: everything round 1 found distinct, consistent and
non-trivial, plus every positive control, so replication is measured on the controls as well."""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent
hits = json.loads((RUNS / "hits.json").read_text())["candidates"]
keys = [c["key"] for c in hits
        if c["origin"] == "catalog"
        or (c["accuracy"] >= 0.75 and c["consistency"] >= 5 and c["kind"] not in ("none", "degraded"))]
(RUNS / "judge" / "r2_keys.json").write_text(json.dumps(keys))
print(f"{len(keys)} candidates for round 2")
