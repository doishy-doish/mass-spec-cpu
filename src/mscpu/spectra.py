"""Spectrum curation + similarity (pure CPU, no RDKit)."""
from __future__ import annotations

import numpy as np

ADDUCT_SHIFT = {  # neutral = precursor - shift
    "[M+H]+": 1.007276, "[M+Na]+": 22.989218, "[M+NH4]+": 18.033823,
    "[M+K]+": 38.963158, "[M-H]-": -1.007276, "[M+Cl]-": 34.968853,
    "[M-H2O+H]+": -17.003289, "[M+CH2O2-H]-": 43.990901, "[M]+": 0.0,
}


def neutral_mass(precursor_mz: float, adduct: str) -> float | None:
    sh = ADDUCT_SHIFT.get(str(adduct))
    return precursor_mz - sh if sh is not None else None


def clean(mzs, ints, precursor: float, floor: float = 0.01, topn: int = 64):
    mzs = np.asarray(mzs, dtype=float)
    ints = np.asarray(ints, dtype=float)
    if len(mzs) == 0:
        return np.zeros((0, 2))
    base = ints.max() if len(ints) else 0.0
    if base <= 0:
        return np.zeros((0, 2))
    keep = (ints / base >= floor) & (mzs <= precursor + 2)
    mzs, rel = mzs[keep], np.sqrt(ints[keep] / base)
    if len(mzs) > topn:
        idx = np.argpartition(rel, -topn)[-topn:]
        mzs, rel = mzs[idx], rel[idx]
    order = np.argsort(mzs)
    return np.stack([mzs[order], rel[order]], axis=1)


def cosine(a, b, tol: float = 0.02) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    used = np.zeros(len(b), dtype=bool)
    dot = 0.0
    for mz, inten in a:
        d = np.abs(b[:, 0] - mz)
        d[used] = np.inf
        k = int(np.argmin(d))
        if d[k] <= tol:
            used[k] = True
            dot += inten * b[k, 1]
    na = float((a[:, 1] ** 2).sum())
    nb = float((b[:, 1] ** 2).sum())
    return dot / np.sqrt(na * nb) if na and nb else 0.0


def modified_cosine(a, b, delta: float, tol: float = 0.02) -> float:
    """GNPS-style: allow matches shifted by precursor delta (analog channel)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    b_shifted = b.copy()
    b_shifted[:, 0] = b_shifted[:, 0] + delta
    return max(cosine(a, b, tol), cosine(a, b_shifted, tol))
