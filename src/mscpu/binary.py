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


def encode(peaks, precursor: float, nbits: int = NBITS, loss_motifs: bool = True) -> int:
    """peaks: (n,2) cleaned array (mz, weight); precursor: precursor m/z. Returns bitset int."""
    bits = 0
    if peaks is None or len(peaks) == 0:
        return bits
    for mz, w in peaks:
        if w <= 0:
            continue
        q = int(round(float(mz) / PEAK_Q))
        bits |= 1 << (_mix(q) % nbits)
        if loss_motifs and precursor:
            lq = int(round(float(precursor - mz) / LOSS_Q))
            bits |= 1 << (_mix(lq, salt=1) % nbits)
    return bits


def jaccard(a: int, b: int) -> float:
    if not a or not b:
        return 0.0
    inter = (a & b).bit_count()
    if not inter:
        return 0.0
    return inter / ((a | b).bit_count() or 1)
