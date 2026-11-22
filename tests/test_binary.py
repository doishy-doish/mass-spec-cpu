import sys
sys.path.insert(0, "src")
from mscpu import binary as B


def test_encode_empty():
    assert B.encode([], 100.0) == 0
    assert B.encode(None, 100.0) == 0
    assert B.encode_bins([], 100.0) == ([], [])


def test_encode_self_jaccard():
    import numpy as np
    peaks = np.array([[100.0, 1.0], [200.5, 0.5], [350.25, 0.2]])
    b = B.encode(peaks, 400.0)
    assert b != 0
    assert B.jaccard(b, b) == 1.0
    assert B.jaccard(b, 0) == 0.0
    assert B.jaccard(0, b) == 0.0


def test_encode_deterministic_and_sized():
    import numpy as np
    peaks = np.array([[100.0, 1.0], [200.5, 0.5]])
    a, b = B.encode(peaks, 400.0), B.encode(peaks, 400.0)
    assert a == b
    assert a.bit_length() <= 4096
    other = B.encode(np.array([[900.0, 1.0]]), 950.0)
    assert 0.0 <= B.jaccard(a, other) <= 1.0


def test_bins_match_encode_bits():
    """encode_bins is exactly the input the Rust core takes; re-mixing bins
    must reproduce encode() bit-for-bit."""
    import numpy as np
    peaks = np.array([[100.0, 1.0], [200.5, 0.5], [350.25, 0.2]])
    ref = B.encode(peaks, 400.0)
    pb, lb = B.encode_bins(peaks, 400.0)
    bits = 0
    for q in pb:
        bits |= 1 << (B._mix(q) % B.NBITS)
    for q in lb:
        bits |= 1 << (B._mix(q, salt=1) % B.NBITS)
    assert bits == ref


def test_weighted_empty():
    assert B.encode_weighted([], 100.0) == {}
    assert B.encode_weighted(None, 100.0) == {}
    assert B.wjaccard({}, {1: 1.0}) == 0.0
    assert B.wjaccard({1: 1.0}, {}) == 0.0


def test_weighted_self_is_one():
    import numpy as np
    peaks = np.array([[100.0, 1.0], [200.5, 0.5], [350.25, 0.2]])
    w = B.encode_weighted(peaks, 400.0)
    assert len(w) > 0
    assert B.wjaccard(w, w) == 1.0


def test_weighted_binary_weights_equal_jaccard():
    """0/1 weights reduce exactly to unweighted Jaccard over the key sets."""
    a = {1: 1.0, 2: 1.0, 3: 1.0}
    b = {2: 1.0, 3: 1.0, 4: 1.0}
    assert B.wjaccard(a, b) == 0.5
    # shared strong peak outscores shared weak peak
    strong = B.wjaccard({1: 1.0, 2: 0.01}, {1: 1.0, 3: 0.01})
    weak = B.wjaccard({1: 0.01, 2: 1.0}, {1: 0.01, 3: 1.0})
    assert strong > 0.9
    assert weak < 0.1
    assert strong > weak


def test_qlib_matches_weighted():
    """encode_qlib is the exact sender for the Rust worker: same bins as
    encode_bins, and max-per-mixed-bit rebuilds encode_weighted exactly."""
    import numpy as np
    peaks = np.array([[100.0, 1.0], [200.5, 0.5], [350.25, 0.2], [200.5, 0.9]])
    p, pw, l, lw = B.encode_qlib(peaks, 400.0)
    pb, lb = B.encode_bins(peaks, 400.0)
    assert p == pb and l == lb
    got = {}
    for q, w in zip(p, pw):
        b = B._mix(q) % B.NBITS
        if w > got.get(b, 0.0):
            got[b] = w
    for q, w in zip(l, lw):
        b = B._mix(q, salt=1) % B.NBITS
        if w > got.get(b, 0.0):
            got[b] = w
    assert got == B.encode_weighted(peaks, 400.0)
    assert B.encode_qlib([], 100.0) == ([], [], [], [])
    assert B.encode_qlib(None, 100.0) == ([], [], [], [])
