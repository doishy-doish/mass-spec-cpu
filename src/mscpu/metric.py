"""Competition metric: MRR@25 on InChIKey14 (stereo-free).

Clean-room implementation of the published rule. Truth keys come from the
data's inchikey14 column; the retrieval engine (Rust) returns ranked key
lists, and these helpers score them. Kept in Python as the competition-spec
layer (data plumbing + rule definition, not compute). No RDKit dependency.
"""
from __future__ import annotations

K = 25


def reciprocal_rank(truth_key: str | None, candidate_keys: list, k: int = K) -> float:
    if not truth_key:
        return 0.0
    seen: set = set()
    rank = 0
    for c in candidate_keys:
        if not c or c in seen:
            continue  # unparsable and duplicate keys do not consume a rank slot
        seen.add(c)
        rank += 1
        if rank > k:
            break
        if c == truth_key:
            return 1.0 / rank
    return 0.0


def mrr_at_k(truth: dict, preds: dict, k: int = K) -> float:
    if not truth:
        return 0.0
    return sum(reciprocal_rank(t, preds.get(m, []), k) for m, t in truth.items()) / len(truth)


def parse_cell(cell: str) -> list:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [s.strip() for s in cell.split(";") if s.strip()]


def validate_submission(rows: list) -> list:
    """rows: list of (molecule_id, smiles_cell). Returns violation strings."""
    errs, seen = [], set()
    for i, (mid, cell) in enumerate(rows):
        if not mid or cell is None:
            errs.append(f"row{i}: missing column")
            continue
        if mid in seen:
            errs.append(f"row{i}: repeated {mid}")
        seen.add(mid)
        if len(parse_cell(cell)) > 25:
            errs.append(f"row{i}: >25 guesses")
    return errs
