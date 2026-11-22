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


def _match_dot(amz, ai, bmz, bi, delta: float, tol: float) -> float:
    """Greedy mz-ordered peak matching, one linear pass, zero allocs.
    Both inputs are mz-sorted (clean() sorts). Same greedy spirit as the old
    argmin+used-mask version, minus ~2 numpy calls per peak pair."""
    i = j = 0
    dot = 0.0
    la, lb = len(amz), len(bmz)
    while i < la and j < lb:
        d = amz[i] - bmz[j] - delta
        if d < -tol:
            i += 1
        elif d > tol:
            j += 1
        else:
            dot += ai[i] * bi[j]
            i += 1
            j += 1
    return dot


def _denorm(a, b, na: float | None, nb: float | None) -> float:
    if na is None:
        na = float(((a[:, 1] ** 2).sum())) if len(a) else 0.0
    if nb is None:
        nb = float(((b[:, 1] ** 2).sum())) if len(b) else 0.0
    return np.sqrt(na * nb) if na and nb else 0.0


def cosine(a, b, tol: float = 0.02, na: float | None = None,
           nb: float | None = None) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dot = _match_dot(a[:, 0], a[:, 1], b[:, 0], b[:, 1], 0.0, tol)
    den = _denorm(a, b, na, nb)
    return dot / den if den else 0.0


def modified_cosine(a, b, delta: float, tol: float = 0.02,
                    na: float | None = None, nb: float | None = None) -> float:
    """GNPS-style: max(unshifted, delta-shifted). Shift applied on the fly —
    no b.copy(), no second full cosine."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    amz, ai, bmz, bi = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    dot = max(_match_dot(amz, ai, bmz, bi, 0.0, tol),
              _match_dot(amz, ai, bmz, bi, delta, tol))
    den = _denorm(a, b, na, nb)
    return dot / den if den else 0.0
