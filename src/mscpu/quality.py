"""quality.py — test-side Tanimoto lens (evaluation only, never retrieval).

The competition pays exact MRR@25 (metric.py). This module is the TUNING LENS
the Gemini ask ordered: Tanimoto-to-truth per query, so isomer ranking can be
judged where exact MRR is 0 by construction (class3) or noisy (class2 n=50).

Single-engine law: retrieval math lives in Rust (rust-bitvec + rust-chem).
This file is evaluation plumbing (like metric.py): pure Python, stdlib only,
no RDKit, no engine duplication. Morgan bits here mirror rust-chem
(morgan_bits_2048 FNV-1a-32, radius-2, 2048 bits) so test-side Tanimoto matches
engine candidate fingerprints bit-for-bit on the same SMILES.

Use:
  from mscpu import quality as Q
  bits = Q.morgan_bits("CCO")
  t = Q.tanimoto(bits_a, bits_b)
  row = Q.tanimoto_row(pred_smiles_list, truth_smiles)  # [t per rank]
  q = Q.tanimoto_quality(pred_lists, truth_smiles_map)  # {mean_top1, mean_best, hit08_rate}
"""
from __future__ import annotations


NBITS = 2048
NWORDS = 32
RADIUS = 2


def _fnv1a32(s: str) -> int:
    x = 2166136261
    for b in s.encode("utf-8"):
        x ^= b
        x = (x * 16777619) & 0xFFFFFFFF
    return x


def _atomic_num(el: str) -> int:
    return {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
            "Cl": 17, "Br": 35, "I": 53}.get(el, 6)


def parse_smiles_lite(smi: str):
    """Minimal SMILES parse: atoms + bonds (organic subset, mirrors rust-chem).
    Returns (atoms, bonds) or (None, None) on empty. Atoms: [(el, aromatic, charge, h)].
    """
    if not smi:
        return None, None
    b = smi
    n = len(b)
    atoms = []
    bonds = []
    i = 0
    prev = None
    order = 1
    ring_open = {}
    stack = []

    def link(idx):
        nonlocal prev, order
        if prev is not None:
            bonds.append((prev, idx, order))
        prev = idx
        order = 1

    while i < n:
        c = b[i]
        if c == '(':
            stack.append(prev)
            i += 1
        elif c == ')':
            prev = stack.pop() if stack else None
            i += 1
        elif c in '-=#:/\\':
            order = {'-': 1, '=': 2, '#': 3}.get(c, 1)
            i += 1
        elif c == '[':
            j = b.find(']', i + 1)
            if j < 0:
                return None, None
            inside = b[i + 1:j]
            i = j + 1
            el = "C"
            aromatic = False
            charge = 0
            h = 0
            k = 0
            while k < len(inside) and inside[k].isdigit():
                k += 1
            if k < len(inside) and inside[k].isalpha():
                if k + 1 < len(inside) and inside[k + 1].islower():
                    el = inside[k:k + 2]
                    k += 2
                else:
                    el = inside[k]
                    if el in "cnos" and len(el) == 1:
                        aromatic = True
                        el = el.upper()
                    k += 1
            if 'H' in inside:
                pos = inside.find('H')
                rest = inside[pos + 1:]
                digits = ''
                for ch in rest:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                h = int(digits) if digits else 1
            charge = inside.count('+') - inside.count('-')
            idx = len(atoms)
            atoms.append((el, aromatic, charge, h))
            link(idx)
        elif c == '%':
            try:
                nn = int(b[i + 1:i + 3])
                if prev is not None:
                    if nn in ring_open:
                        bonds.append((ring_open.pop(nn), prev, order))
                        order = 1
                    else:
                        ring_open[nn] = prev
                i += 3
                continue
            except Exception:
                i += 1
        elif c.isdigit():
            nn = int(c)
            if prev is not None:
                if nn in ring_open:
                    bonds.append((ring_open.pop(nn), prev, order))
                    order = 1
                else:
                    ring_open[nn] = prev
            i += 1
        elif c == '.':
            prev = None
            order = 1
            i += 1
        elif c == '+':
            if prev is not None and prev < len(atoms):
                el, ar, ch, hh = atoms[prev]
                atoms[prev] = (el, ar, ch + 1, hh)
            i += 1
        else:
            if b[i:i + 2] in ("Cl", "Br"):
                idx = len(atoms)
                atoms.append((b[i:i + 2], False, 0, 0))
                link(idx)
                i += 2
                continue
            if c in "CNOSPFI":
                idx = len(atoms)
                atoms.append((c, False, 0, 0))
                link(idx)
                i += 1
                continue
            if c in "cnos":
                idx = len(atoms)
                atoms.append((c.upper(), True, 0, 0))
                link(idx)
                i += 1
                continue
            i += 1
    if not atoms:
        return None, None
    return atoms, bonds


def morgan_bits(smi: str) -> list:
    """Morgan ECFP radius-2 2048-bit as 32 uint64 words (list[int]).
    Mirrors rust-chem morgan_bits_2048 bit-for-bit on the same SMILES.
    """
    out = [0] * NWORDS
    parsed = parse_smiles_lite(smi)
    if parsed[0] is None:
        return out
    atoms, bonds = parsed
    n = len(atoms)
    adj = [[] for _ in range(n)]
    for a, bb, o in bonds:
        if a < n and bb < n:
            adj[a].append((bb, o))
            adj[bb].append((a, o))

    def bit(x):
        b = x % NBITS
        out[b // 64] |= 1 << (b % 64)

    cur = []
    for idx, (el, ar, ch, h) in enumerate(atoms):
        deg = len(adj[idx])
        inv = "%d:%d:%d:%d:%d" % (_atomic_num(el), deg, h, ch, 1 if ar else 0)
        cur.append(_fnv1a32(inv))
    for x in cur:
        bit(x)
    for _ in range(RADIUS):
        nxt = [0] * n
        for idx in range(n):
            nbrs = sorted(" %dx%d" % (cur[j], o) for j, o in adj[idx])
            nxt[idx] = _fnv1a32("%d|%s" % (cur[idx], ",".join(nbrs)))
        cur = nxt
        for x in cur:
            bit(x)
    return out


def tanimoto(a: list, b: list) -> float:
    """Tanimoto over 32-word bit vectors. Empty-vs-empty = 0.0 (no evidence)."""
    inter = union = 0
    for x, y in zip(a, b):
        inter += bin(x & y).count("1")
        union += bin(x | y).count("1")
    return inter / union if union else 0.0


def tanimoto_row(pred_smiles: list, truth_smiles: str) -> list:
    """Per-rank Tanimoto vs truth (len == len(pred_smiles)). Unknown truth -> zeros."""
    if not truth_smiles:
        return [0.0] * len(pred_smiles)
    tb = morgan_bits(str(truth_smiles))
    return [tanimoto(morgan_bits(str(s)), tb) for s in pred_smiles]


def quality_summary(pred_map: dict, truth_map: dict, k: int = 25) -> dict:
    """Corpus Tanimoto quality: mean top-1, mean best-in-k, hit@0.8 rate.
    pred_map: {qid: [smiles...]}, truth_map: {qid: smiles}. Missing -> 0.
    """
    top1, best, hit08, n = 0.0, 0.0, 0, 0
    for qid, preds in (pred_map or {}).items():
        t = (truth_map or {}).get(qid, "")
        if not t:
            continue
        row = tanimoto_row((preds or [])[:k], str(t))
        if not row:
            n += 1
            continue
        top1 += row[0]
        b = max(row)
        best += b
        if b >= 0.8:
            hit08 += 1
        n += 1
    if not n:
        return {"n": 0, "mean_top1": 0.0, "mean_best": 0.0, "hit08_rate": 0.0}
    return {"n": n, "mean_top1": top1 / n, "mean_best": best / n,
            "hit08_rate": hit08 / n}
