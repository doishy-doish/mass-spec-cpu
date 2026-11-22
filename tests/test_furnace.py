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
