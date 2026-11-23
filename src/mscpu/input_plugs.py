"""Input plugs: use the inputs we already load, plus wiring for the ones we don't.

Pure functions only (stdlib + numpy). No engine, no RDKit, no I/O.
Every plug is flag-gated in run.py with legacy-default OFF, so the
default path is byte-for-byte the old behavior. The parity canary
decides whether anything moved.

Plugs:
  (1) instrument gate — score-time bonus / pool re-rank by instrument family
  (2) adduct retry — neighbor-adduct reinterpretation for zero-score queries
  (3) precursor graded — v8 mass falloff + rank-1-by-mass replace rule
  (4) formula gate — monoisotopic-mass primitive (wiring-ready; the formula
      itself arrives with the COCONUT / fingerprint lane, not invented here)
  (5) prepared-cache — bounded memo for repeated pool lookups (lossless)
  (6) funnel recall lock — recall@K probe so the bin->analog funnel can be
      locked lossless (100% recall on the canary) before it ever filters
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# (1) instrument gate
# ---------------------------------------------------------------------------

def instrument_norm(s: str) -> str:
    """Normalize an instrument string to a family token."""
    t = str(s or "").lower()
    if "timstof" in t or "tims" in t:
        return "timstof"
    if "orbitrap" in t:
        return "orbitrap"
    if "tof" in t:
        return "tof"
    if "orbitrap" in t:
        return "orbitrap"
    if "fticr" in t or "ft-icr" in t:
        return "fticr"
    if "quadrupole" in t or "q-tof" in t or "qtof" in t:
        return "qtof"
    if "ion trap" in t or "itms" in t:
        return "it"
    return t.strip() or "unknown"


def instrument_bonus(q_inst: str, lib_inst: str) -> float:
    """Score multiplier for instrument agreement. 1.0 = same family.

    Neutral when the gate is off (caller multiplies only if flag on).
    Same family 1.0, neighbouring TOF family 0.9, else 0.75.
    """
    q = instrument_norm(q_inst)
    l = instrument_norm(lib_inst)
    if q == l:
        return 1.0
    if q in ("timstof", "tof", "qtof") and l in ("timstof", "tof", "qtof"):
        return 0.9
    return 0.75


def pool_rerank_by_instrument(pool_keys, key2inst: dict, q_inst: str):
    """Stable re-rank: instrument matches first, legacy order otherwise.

    Pure reorder — the SET is unchanged, so with the bonus at 1.0 the
    downstream top-K is identical. Caller applies only when flag on.
    """
    q = instrument_norm(q_inst)
    scored = []
    for pos, k in enumerate(pool_keys):
        bonus = 1.0 if instrument_norm(key2inst.get(k, "")) == q else 0.0
        scored.append((bonus, pos, k))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [k for _, _, k in scored]


# ---------------------------------------------------------------------------
# (2) adduct retry (zero-score queries only)
# ---------------------------------------------------------------------------

# Neighbor reinterpretations, self first. Protonated / deprotonated lanes
# stay separate (charge flip is never a "neighbor"). Keys are lowercase;
# values are canonical data-case (train data uses [M+H]+ / [M-H]-).
ADDUCT_NEIGHBORS = {
    "[m+h]+": ["[M+H]+", "[M+Na]+", "[M+K]+", "[M+H-H2O]+", "[M+NH4]+"],
    "[m+na]+": ["[M+Na]+", "[M+H]+", "[M+K]+"],
    "[m+k]+": ["[M+K]+", "[M+Na]+", "[M+H]+"],
    "[m+nh4]+": ["[M+NH4]+", "[M+H]+", "[M+Na]+"],
    "[m+h-h2o]+": ["[M+H-H2O]+", "[M+H]+"],
    "[m-h]-": ["[M-H]-", "[M+Cl]-", "[M+HCOO]-", "[M+CH3COO]-"],
    "[m+cl]-": ["[M+Cl]-", "[M-H]-"],
    "[m+hcoo]-": ["[M+HCOO]-", "[M-H]-"],
}

# Neutral-mass deltas for the retry lane (Da). Applied as prec shift so the
# existing neutral-mass pipeline (structs) can be reused without new math.
# Values are monoisotopic adduct-mass differences vs the base lane ion.
ADDUCT_DELTA = {
    "[m+na]+": 21.98194,   # Na - H
    "[m+k]+": 37.95588,    # K - H
    "[m+nh4]+": 17.02655,  # NH4 - H
    "[m+h-h2o]+": -18.01056,
    "[m+cl]-": 35.97668,   # Cl + H (vs [M-H]- neutral frame)
    "[m+hcoo]-": 46.00548,
    "[m+ch3coo]-": 60.02113,
}


def adduct_neighbors(adduct: str):
    """Neighbor adducts for retry, self first. Unknown -> [self].

    Lookup is case-insensitive (M219: train data is [M+H]+ uppercase,
    table keys are lowercase). Returned values are canonical data-case.
    """
    a = str(adduct or "")
    return list(ADDUCT_NEIGHBORS.get(a.lower(), [a]))


def retry_precisions(prec: float, from_ad: str, to_ad: str):
    """Shift a precursor mz from one adduct frame to another (Da shift)."""
    if to_ad == from_ad:
        return float(prec)
    d_to = ADDUCT_DELTA.get(str(to_ad).lower(), 0.0)
    d_from = ADDUCT_DELTA.get(str(from_ad).lower(), 0.0)
    return float(prec) + (d_to - d_from)


# ---------------------------------------------------------------------------
# (3) precursor graded (v8 mass rules, exact)
# ---------------------------------------------------------------------------

V8_CENTER = -1.5   # ppm
V8_FREE = 3.0      # ppm half-width of the full-score plateau
V8_FULL = 8.0      # ppm half-width where the score hits zero
V8_REPLACE = 5.0   # leader farther off than this gets replaced


def mass_grade(ppm_err: float) -> float:
    """Graded mass-error score in [0,1]. v8 exact: plateau, linear falloff.

    dist = |ppm_err - center|; <= free -> 1.0; >= full -> 0.0; else linear.
    NaN/None -> 0.0 (no mass evidence, never a crash).
    """
    try:
        e = float(ppm_err)
    except Exception:
        return 0.0
    if e != e:  # NaN
        return 0.0
    dist = abs(e - V8_CENTER)
    if dist <= V8_FREE:
        return 1.0
    if dist >= V8_FULL:
        return 0.0
    return (V8_FULL - dist) / (V8_FULL - V8_FREE)


def rank1_replace(leader_ppm, challenger_ppm) -> bool:
    """v8 rank-1-by-mass rule: leader >5ppm off and challenger closer -> swap."""
    try:
        l = float(leader_ppm)
        c = float(challenger_ppm)
    except Exception:
        return False
    if l != l or c != c:
        return False
    return abs(l) > V8_REPLACE and abs(c) < abs(l)


# ---------------------------------------------------------------------------
# (4) formula gate primitive (wiring-ready)
# ---------------------------------------------------------------------------

MONO = {
    "C": 12.0, "H": 1.00782503223, "N": 14.00307400443, "O": 15.99491461957,
    "S": 31.9720711744, "P": 30.9737619985, "F": 18.99840316273,
    "Cl": 34.968852682, "Br": 78.9183376, "I": 126.9044719,
    "Na": 22.9897692809, "K": 38.9637074864,
}


def formula_mass(formula: str) -> float:
    """Monoisotopic mass of a formula string like 'C6H12O6'. Stdlib only.

    Raises ValueError on unknown elements (caller treats as no-gate).
    """
    import re
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", str(formula or ""))
    if not toks:
        raise ValueError("empty formula: %r" % (formula,))
    total = 0.0
    for el, num in toks:
        if el not in MONO:
            raise ValueError("unknown element %r in %r" % (el, formula))
        total += MONO[el] * (int(num) if num else 1)
    return total


def formula_consistent(qnm: float, formula: str, ppm: float = 8.5) -> bool:
    """True when |qnm - formula_mass| is within ppm. Unknown formula -> True.

    Unknown-formula passes (no gate) rather than failing — the gate only
    constrains when we actually know the formula (COCONUT/fingerprint lane).
    """
    try:
        fm = formula_mass(formula)
    except Exception:
        return True
    try:
        q = float(qnm)
    except Exception:
        return False
    if q != q or fm <= 0:
        return False
    return abs(q - fm) / fm * 1e6 <= float(ppm)


# ---------------------------------------------------------------------------
# (5) prepared-cache (bounded memo, lossless)
# ---------------------------------------------------------------------------

class BoundedCache:
    """FIFO-bounded dict. Same inputs -> same outputs, or a recorded miss.

    Not an LRU (no reordering on hit): iteration order is insertion order,
    so behavior is deterministic across runs with the same insertion order.
    """

    def __init__(self, cap: int = 50000):
        self.cap = int(cap)
        self._d: "OrderedDict" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self._d:
            self.hits += 1
            return self._d[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self._d:
            self._d[key] = value
            return
        if len(self._d) >= self.cap:
            self._d.popitem(last=False)
        self._d[key] = value

    def __len__(self):
        return len(self._d)


def cache_key(qnm: float, prec_q: float, ppm: float):
    """Quantized lookup key: qnm to 0.001 Da, prec to 0.1 Da, ppm as-is."""
    return (round(float(qnm), 3), round(float(prec_q), 1), float(ppm))


# ---------------------------------------------------------------------------
# (6) funnel recall lock
# ---------------------------------------------------------------------------

def recall_at_k(ranked_lists, truths, k: int) -> float:
    """Fraction of queries whose truth survives in the top-K prefilter."""
    n = len(ranked_lists)
    if n == 0:
        return 0.0
    hit = 0
    for ranked, truth in zip(ranked_lists, truths):
        if str(truth) in [str(x) for x in ranked[:k]]:
            hit += 1
    return hit / n


def top_k(pool_scored, k: int):
    """First-K keys of a scored pool list. Pure slice, stable order."""
    return list(pool_scored)[: max(0, int(k))]


def rank_replace_list(ranked_list, ppm_of, center=V8_CENTER, lead_off=V8_REPLACE):
    """Python mirror of the engine's apply_mass_replace, for one ranked list.

    Same contract (I6): rank-1 |ppm-center| > lead_off -> min-|ppm| key to
    front; first-seen wins ties; missing ppms / good leader -> unchanged.
    """
    lst = [str(k) for k in ranked_list]
    if not lst:
        return lst

    def dev(k):
        try:
            return abs(float(ppm_of[k]) - float(center))
        except Exception:
            return None

    lead = dev(lst[0])
    if lead is None or lead <= float(lead_off):
        return lst
    best_i, best_d = None, None
    for i, k in enumerate(lst):
        d = dev(k)
        if d is None:
            continue
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    if best_i is not None and best_d < lead:
        k = lst.pop(best_i)
        lst.insert(0, k)
    return lst
