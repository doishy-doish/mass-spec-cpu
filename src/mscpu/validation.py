"""Leakage-safe three-mask splits. Grouping key: inchikey14.
Mirrors the test classes: library-known / structure-known / novel-structure.
Operates on a metadata frame (one row per spectrum)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TEST_MZ = (240.0, 465.0)


def query_pool(meta: pd.DataFrame, instrument_regex: str = "timsTOF",
               mz_range: tuple = TEST_MZ) -> pd.DataFrame:
    pool = meta[meta["instrument_type"].str.contains(instrument_regex, case=False, na=False)]
    return pool[(pool["precursor_mz"] >= mz_range[0]) & (pool["precursor_mz"] <= mz_range[1])]


def split_class2(meta: pd.DataFrame, pool_idx, counts: pd.Series,
                 n_queries: int = 400, seed: int = 0):
    """Structure stays via OTHER spectra; one query spectrum per key (keys with >=2)."""
    rng = np.random.default_rng(seed)
    keys = pd.Index(pool_idx).map(meta["key14"])
    eligible = counts[counts >= 2].index.intersection(keys.unique())
    held = rng.choice(eligible, size=min(n_queries, len(eligible)), replace=False)
    held_set = set(held)
    qrows, seen = [], set()
    for i in pool_idx:
        k = meta.at[i, "key14"]
        if k in held_set and k not in seen:
            seen.add(k)
            qrows.append(i)
    qset = set(qrows)
    library = [i for i in meta.index if i not in qset]
    return library, qrows, list(held)


def split_class3(meta: pd.DataFrame, pool_idx, n_struct: int = 400, seed: int = 0):
    """Every spectrum of held structures leaves the library."""
    rng = np.random.default_rng(seed)
    keys = pd.Index(pool_idx).map(meta["key14"]).unique()
    held = set(rng.choice(keys, size=min(n_struct, len(keys)), replace=False))
    qrows = [i for i in pool_idx if meta.at[i, "key14"] in held]
    library = [i for i in meta.index if meta.at[i, "key14"] not in held]
    return library, qrows, list(held)


def split_cross_instrument(meta: pd.DataFrame, pool_idx, n_struct: int = 400,
                           seed: int = 0, instrument_regex: str = "timsTOF"):
    """Harsh class 2: query structure's timsTOF spectra held; stays via other instruments."""
    rng = np.random.default_rng(seed)
    is_inst = meta["instrument_type"].str.contains(instrument_regex, case=False, na=False)
    with_other = set(meta.loc[~is_inst, "key14"])
    keys = [k for k in pd.Index(pool_idx).map(meta["key14"]).unique() if k in with_other]
    held = set(rng.choice(keys, size=min(n_struct, len(keys)), replace=False))
    qrows = [i for i in pool_idx if meta.at[i, "key14"] in held]
    qset = set(qrows)
    library = [i for i in meta.index if i not in qset]
    return library, qrows, list(held)
