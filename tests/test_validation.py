"""Mask-split stub tests (synthetic arrays, no data files).

Locks split identity vs the old DataFrame implementation: sorted eligible
sets, first-occurrence key order, one-query-per-key on class2, full holdout
on class3, full holdout of held keys on cross (PHONEBOOK-STRIP 2026-11-24:
no test-molecule spectrum anywhere in any library).
"""
import sys
sys.path.insert(0, "src")

import numpy as np

from mscpu import validation as V


def _table():
    # 6 rows: k1 x2 (timsTOF), k2 x2 (timsTOF), k3 x1 (Orbitrap), k4 x1 (timsTOF)
    return {
        "key14": np.array(["k1", "k1", "k2", "k2", "k3", "k4"]),
        "prec": np.array([300.0, 300.5, 310.0, 310.5, 320.0, 330.0]),
        "instrument": np.array(["timsTOF", "timsTOF", "timsTOF", "timsTOF",
                                "Orbitrap", "timsTOF"]),
    }


def test_query_pool_selects_instrument_and_mz():
    T = _table()
    pool = V.query_pool(T).tolist()
    assert pool == [0, 1, 2, 3, 5]  # k3 Orbitrap excluded


def test_class2_one_query_per_held_key():
    T = _table()
    pool = V.query_pool(T).tolist()
    lib, queries, held = V.split_class2(T, pool, n_queries=2, seed=0)
    assert len(queries) == 2 and len(set(queries)) == 2
    qkeys = [str(T["key14"][i]) for i in queries]
    assert len(set(qkeys)) == 2 and set(qkeys) <= set(held)
    assert not (set(queries) & set(lib))
    # k3 has 1 spectrum and is out of pool: never held
    assert "k3" not in held
    # PHONEBOOK-STRIP: no held key keeps ANY spectrum in the library
    libkeys = set(str(T["key14"][i]) for i in lib)
    assert not (set(held) & libkeys)


def test_class3_full_holdout():
    T = _table()
    pool = V.query_pool(T).tolist()
    lib, queries, held = V.split_class3(T, pool, n_struct=1, seed=0)
    hkeys = set(str(T["key14"][i]) for i in queries)
    assert hkeys == set(held) and len(held) == 1
    libkeys = set(str(T["key14"][i]) for i in lib)
    assert not (hkeys & libkeys)


def test_cross_full_holdout_of_held_keys():
    T = _table()
    pool = V.query_pool(T, instrument_regex="timsTOF").tolist()
    lib, queries, held = V.split_cross_instrument(T, pool, n_struct=2, seed=1)
    # held keys must all have a non-timsTOF spectrum somewhere (k3 qualifies)
    assert set(held) <= {"k1", "k2", "k3", "k4"}
    assert not (set(queries) & set(lib))
    # PHONEBOOK-STRIP: other-instrument stays are gone too
    libkeys = set(str(T["key14"][i]) for i in lib)
    assert not (set(held) & libkeys)
