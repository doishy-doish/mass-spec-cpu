"""Furnace smoke test: exercises the hot path (cosine/modified_cosine/score_query)
on synthetic spectra with no data files. Runs inside the runner's pytest gate
BEFORE run-mask, so furnace bugs fail fast instead of burning a 20-min run.
Filed alongside M174 (pytest didn't cover the furnace, run-mask paid for it)."""
import numpy as np
import pandas as pd

from mscpu import run as R
from mscpu import spectra as S


def _spec(mzs, ints, prec):
    return S.clean(np.array(mzs), np.array(ints), prec)


def test_two_pointer_cosine_sane():
    a = _spec([100.0, 200.0, 300.0], [1.0, 0.5, 0.25], 301.0)
    b = _spec([100.01, 200.01, 400.0], [1.0, 0.5, 0.25], 401.0)
    v = S.cosine(a, b)
    assert 0.0 < v <= 1.0
    assert S.modified_cosine(a, b, 100.0) >= 0.0
    assert S.cosine(a, a) == abs(S.cosine(a, a)) and S.cosine(a, a) > 0.99


def test_score_query_synthetic_end_to_end():
    qpeaks = _spec([100.0, 200.0], [1.0, 1.0], 201.0)
    lib1 = _spec([100.0, 200.0], [1.0, 1.0], 201.0)
    lib2 = _spec([150.0, 250.0], [1.0, 1.0], 251.0)
    lib_clean = {"k1": [(lib1, 201.0)], "k2": [(lib2, 251.0)]}
    cands = pd.DataFrame([{"key14": "k1", "fp": None}, {"key14": "k2", "fp": None}])
    R.FP_CACHE.clear()
    scored = R.score_query(qpeaks, 201.0, cands, lib_clean, [])
    assert list(scored["key14"]) == ["k1", "k2"]
    assert scored["lib"].iloc[0] > scored["lib"].iloc[1]
    assert scored["lib"].iloc[0] > 0.99


def _mirror_main_precompute(lib_clean):
    """Byte-for-byte mirror of run.main's FURNACE precompute block.
    If main's block is edited, this MUST be updated to match."""
    import bisect as _bisect  # noqa: F401 (mirrors run.py's top-level import)

    lib_norms = {k: [float((cl[:, 1] ** 2).sum()) if len(cl) else 0.0 for cl, _ in v]
                 for k, v in lib_clean.items()}
    pairs = sorted((prec, k) for k, v in lib_clean.items() for cl, prec in v[:1])
    return lib_norms, [p for p, _ in pairs], [k for _, k in pairs]


def test_main_precompute_mirror_with_norms_and_bisect():
    import bisect

    empty = np.zeros((0, 2))
    lib_clean = {
        "k1": [(_spec([100.0, 200.0], [1.0, 1.0], 201.0), 201.0),
               (_spec([100.0, 300.0], [1.0, 1.0], 201.0), 201.0)],
        "k2": [(_spec([150.0], [1.0], 151.0), 151.0)],
        "k3": [(empty, 99.0)],
        "k4": [(_spec([201.0], [1.0], 201.0), 201.0)],  # duplicate precursor vs k1
    }
    lib_norms, precs, pkeys = _mirror_main_precompute(lib_clean)
    assert set(lib_norms) == {"k1", "k2", "k3", "k4"}
    assert lib_norms["k3"] == [0.0] and len(lib_norms["k1"]) == 2
    assert precs == sorted(precs) and len(precs) == 4
    # exact-match self query through the full 4-tuple + norms + qloss path
    qpeaks = _spec([100.0, 200.0], [1.0, 1.0], 201.0)
    nq = float((qpeaks[:, 1] ** 2).sum())
    qloss = set(round(float(201.0 - mz), 2) for mz, _ in qpeaks[:32])
    lo = bisect.bisect_left(precs, 201.0 - 50.0)
    hi = bisect.bisect_right(precs, 201.0 + 50.0, lo=lo)
    assert (lo, hi) == (1, 4)  # v[:1]: one precursor per key -> [99, 151, 201, 201]
    pool = [("k1", lib_clean["k1"][0][0], 201.0, lib_norms["k1"][0])]
    cands = pd.DataFrame([{"key14": "k1", "fp": None}])
    R.FP_CACHE.clear()
    scored = R.score_query(qpeaks, 201.0, cands, lib_clean, pool,
                           qloss_pre=qloss, nq=nq, lib_norms=lib_norms)
    assert scored["lib"].iloc[0] > 0.99
    assert scored["frag"].iloc[0] > 0.99
