"""Spectrum -> bitset fingerprints (CPU, RDKit-free) + Jaccard retrieval.

Binary lane for the analog-vs-binary test: each spectrum is hashed to a
fixed-width bitset (default 4096 bits). Peak m/z values are quantized and
mixed with a Knuth multiplicative hash; neutral-loss motifs
(round(precursor - mz, 2)) get their own salted bit range. Pilot (easy lane):
4096+loss MRR 0.7498 beats 2048 plain 0.7307 at ~5-8x less compute than
cosine. Retrieval is popcount/Jaccard on Python ints.
"""
from __future__ import annotations

NBITS = 4096
PEAK_Q = 0.05  # peak m/z quantization bin (Da)
LOSS_Q = 0.01  # neutral-loss motif rounding (Da)
_KNUTH = 2654435761


def _mix(x: int, salt: int = 0) -> int:
    return ((x + salt * 0x9E3779B9) * _KNUTH) & 0xFFFFFFFF


def encode_bins(peaks, precursor: float, qpeak: float = PEAK_Q, qloss: float = LOSS_Q) -> tuple:
    """Pre-quantized bin lists (masked to u32) — the exact input the Rust core takes."""
    p, l = [], []
    if peaks is None or len(peaks) == 0:
        return p, l
    for mz, w in peaks:
        if w <= 0:
            continue
        p.append(int(round(float(mz) / qpeak)) & 0xFFFFFFFF)
        if precursor:
            l.append(int(round(float(precursor - mz) / qloss)) & 0xFFFFFFFF)
    return p, l


def encode(peaks, precursor: float, nbits: int = NBITS, loss_motifs: bool = True,
           qpeak: float = PEAK_Q, qloss: float = LOSS_Q) -> int:
    """peaks: (n,2) cleaned array (mz, weight); precursor: precursor m/z. Returns bitset int."""
    bits = 0
    if peaks is None or len(peaks) == 0:
        return bits
    pb, lb = encode_bins(peaks, precursor if loss_motifs else 0.0, qpeak, qloss)
    for q in pb:
        bits |= 1 << (_mix(q) % nbits)
    for q in lb:
        bits |= 1 << (_mix(q, salt=1) % nbits)
    return bits


def jaccard(a: int, b: int) -> float:
    if not a or not b:
        return 0.0
    inter = (a & b).bit_count()
    if not inter:
        return 0.0
    return inter / ((a | b).bit_count() or 1)


def encode_weighted(peaks, precursor, qpeak: float = PEAK_Q, qloss: float = LOSS_Q,
                    loss_motifs: bool = True) -> dict:
    """{bin: max sqrt-weight} — the intensity-preserving twin of encode().
    Same bins (same q/mix/salts); duplicate bins keep the max weight."""
    wmap: dict = {}
    if peaks is None or len(peaks) == 0:
        return wmap
    for mz, w in peaks:
        if w <= 0:
            continue
        w = float(w)
        b = _mix(int(round(float(mz) / qpeak)) & 0xFFFFFFFF) % NBITS
        if w > wmap.get(b, 0.0):
            wmap[b] = w
        if loss_motifs and precursor:
            b2 = _mix(int(round(float(precursor - mz) / qloss)) & 0xFFFFFFFF, salt=1) % NBITS
            if w > wmap.get(b2, 0.0):
                wmap[b2] = w
    return wmap


def wjaccard(a: dict, b: dict) -> float:
    """Ruzicka (weighted Jaccard) over {bin: weight} dicts.
    0/1 weights reduce exactly to jaccard() over the key sets."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    inter = 0.0
    union = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            union += va
        else:
            inter += va if va < vb else vb
            union += va if va > vb else vb
    for k, vb in b.items():
        if k not in a:
            union += vb
    if inter <= 0.0:
        return 0.0
    return inter / union if union else 0.0
