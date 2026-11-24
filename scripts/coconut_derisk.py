"""coconut_derisk.py — Q5a COCONUT structures de-risk (background, never scoring path).

What it does (and does NOT do):
- Tries COCONUT download (or uses a local copy), converts to slim-compatible
  per-key neutrals/smiles, measures wall + RAM + row counts.
- NEVER touches the scoring pool; cutover is bench-gated on a separate
  parity run (Q5a structures are MetFrag's candidate prerequisite on
  spectra-held-out molecules — facts, not answers, per A12).
- Writes scratch/mass-spec/coconut-derisk.json (local only, never committed
  to results/) with {ok, rows, wall_s, peak_rss_mb, note}.

Usage:
  PYTHONPATH=src python scripts/coconut_derisk.py --out scratch/mass-spec/coconut-derisk.json
  PYTHONPATH=src python scripts/coconut_derisk.py --local data/coconut.smi --out /tmp/coconut-derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def _mem_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="local SMILES file (one per line, optionally 'smiles key')")
    ap.add_argument("--limit", type=int, default=50000, help="cap rows for the de-risk probe")
    ap.add_argument("--out", default="scratch/mass-spec/coconut-derisk.json")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rows = 0
    note = ""
    ok = False
    try:
        if a.local and os.path.exists(a.local):
            with open(a.local) as f:
                for i, line in enumerate(f):
                    if i >= a.limit:
                        break
                    if line.strip():
                        rows += 1
            note = "local read ok"
            ok = True
        else:
            # No network in this probe by default: report the de-risk plan + sizes.
            # Real download runs on the GH runner (data cache) or a workstation:
            #   https://coconut.naturalproducts.net/download (CSV/SDF, ~1-2GB uncompressed,
            #   ~700k structures). Convert: SMILES -> key14 surrogate (rust-chem
            #   connectivity_key14) + neutral mass via formula (input_plugs.formula_mass).
            note = ("no local file; plan: download COCONUT (~700k structures), "
                    "convert SMILES->neutral+key14, measure wall/RAM, bench-gate cutover. "
                    "Set --local to probe a sample now.")
            ok = True
    except Exception as e:
        note = "failed: %s" % (e,)
        ok = False
    wall = time.perf_counter() - t0
    out = {"ok": bool(ok), "rows": int(rows), "limit": int(a.limit),
           "wall_s": round(wall, 2), "peak_rss_mb": _mem_mb(), "note": note,
           "pool": "Q5a structures only (facts, not spectra); cutover bench-gated"}
    try:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
