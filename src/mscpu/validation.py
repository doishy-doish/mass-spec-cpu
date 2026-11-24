"""Leakage-safe three-mask splits. Grouping key: inchikey14.
Mirrors the test classes: library-known / structure-known / novel-structure.
Operates on table dicts of numpy arrays (no pandas anywhere in this path).

Split identity vs the old DataFrame implementation is preserved on purpose:
eligible sets are sorted (pandas Index.intersection sorts), unique key order is
first-occurrence (pandas .unique()), group order is sorted (groupby sort=True).
The one deliberate change: struct sorting uses a stable argsort instead of
pandas quicksort, so equal-neutral tie order is deterministic. The +-0.00
parity canary decides whether that (or anything else) moved a score.
"""
from __future__ import annotations

import numpy as np

TEST_MZ = (240.0, 465.0)


def _contains(hay, needle: str) -> np.ndarray:
    """Case-insensitive substring mask. None-safe: str(None) never matches."""
    h = np.asarray([str(x) for x in hay])
    return np.char.find(np.char.lower(h), needle.lower()) != -1


def _uniq_first(a: np.ndarray) -> np.ndarray:
    """Unique values in first-occurrence order (pandas .unique() semantics)."""
    _, idx = np.unique(np.asarray(a), return_index=True)
    return np.asarray(a)[np.sort(idx)]


def query_pool(T: dict, instrument_regex: str = "timsTOF",
               mz_range: tuple = TEST_MZ) -> np.ndarray:
    m = _contains(T["instrument"], instrument_regex)
    p = T["prec"]
    return np.flatnonzero(m & (p >= mz_range[0]) & (p <= mz_range[1]))


def split_class2(T: dict, pool_idx, n_queries: int = 400, seed: int = 0,
                 min_count: int = 2, phonebook: bool = False):
    """Structure stays via OTHER spectra; one query spectrum per key (keys with >=2)."""
    rng = np.random.default_rng(seed)
    allk = np.asarray(T["key14"], dtype=str)
    vals, cnt = np.unique(allk, return_counts=True)
    big = set(vals[cnt >= min_count].tolist())
    eligible = sorted(big.intersection(set(allk[np.asarray(pool_idx)].tolist())))
    held = set(rng.choice(eligible, size=min(n_queries, len(eligible)),
                          replace=False).tolist())
    qrows, seen = [], set()
    for i in pool_idx:
        k = str(allk[int(i)])
        if k in held and k not in seen:
            seen.add(k)
            qrows.append(int(i))
    # PHONEBOOK-STRIP (user order 2026-11-24): held keys leave the library
    # entirely — every spectrum of a test molecule, not just the query rows.
    # held/qrows use the identical RNG stream as before, so the query set is
    # unchanged; only library composition changes (tin -> 0, exact MRRs -> 0).
    # PHONEBOOK-ARM (-pb suffix): phonebook=True keeps held keys IN the
    # library — measures what the restored lib retrieves on overlap. The RNG
    # stream above is untouched, so the query set is byte-identical to the
    # honest arm; only library composition differs.
    if phonebook:
        library = list(range(len(allk)))
    else:
        heldset = set(held)
        library = [i for i in range(len(allk)) if str(allk[i]) not in heldset]
    return library, qrows, sorted(held)


def with_metadata_keys(lib_by_key: dict, held) -> dict:
    """True-class2 preload: held keys stay candidates with NO spectra.

    Library ROWS are untouched (strip holds — tin stays 0); this only adds
    held keys mapped to [] so preload chunks emit them as metadata-only
    (specs=[], neutral+SMILES kept). The engine already scores empty-spec
    keys gracefully (lib-family 0.0, mass/cleav/fp/gate off neutral+SMILES)
    and ranks cands — so computed channels can rank the true structure with
    no spectra: the actual unseen exact-MRR measurement. M254: slim path
    cannot use this (ensure_key fills spectra from the store for any touched
    key) — run.py refuses -t2 when S is not None, fail fast never leak.
    Pure in-place, returns the same dict.
    """
    for k in held or []:
        lib_by_key.setdefault(str(k), [])
    return lib_by_key


def split_class3(T: dict, pool_idx, n_struct: int = 400, seed: int = 0,
                 phonebook: bool = False):
    """Every spectrum of held structures leaves the library."""
    rng = np.random.default_rng(seed)
    allk = np.asarray(T["key14"], dtype=str)
    keys = _uniq_first(allk[np.asarray(pool_idx)]).tolist()
    held = set(rng.choice(keys, size=min(n_struct, len(keys)),
                        replace=False).tolist())
    qrows = [int(i) for i in pool_idx if str(allk[int(i)]) in held]
    if phonebook:
        library = list(range(len(allk)))
    else:
        library = [i for i in range(len(allk)) if str(allk[i]) not in held]
    return library, qrows, sorted(held)


def split_cross_instrument(T: dict, pool_idx, n_struct: int = 400, seed: int = 0,
                           instrument_regex: str = "timsTOF",
                           phonebook: bool = False):
    """Harsh class 2: query structure's timsTOF spectra held; stays via other instruments."""
    rng = np.random.default_rng(seed)
    allk = np.asarray(T["key14"], dtype=str)
    is_inst = _contains(T["instrument"], instrument_regex)
    with_other = set(allk[~is_inst].tolist())
    keys = [k for k in _uniq_first(allk[np.asarray(pool_idx)]).tolist()
            if k in with_other]
    held = set(rng.choice(keys, size=min(n_struct, len(keys)),
                          replace=False).tolist())
    qrows = [int(i) for i in pool_idx if str(allk[int(i)]) in held]
    # PHONEBOOK-STRIP (user order 2026-11-24): held keys leave the library
    # entirely — every spectrum of a test molecule, not just the query rows.
    # held/qrows use the identical RNG stream as before, so the query set is
    # unchanged; only library composition changes (tin -> 0, exact MRRs -> 0).
    # PHONEBOOK-ARM (-pb suffix): phonebook=True keeps held keys IN the
    # library (same byte-identical query guarantee as class2).
    if phonebook:
        library = list(range(len(allk)))
    else:
        heldset = set(held)
        library = [i for i in range(len(allk)) if str(allk[i]) not in heldset]
    return library, qrows, sorted(held)
