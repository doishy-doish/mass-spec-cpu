"""tower tests: reference forward + weights round-trip + checker."""
import os
import sys
sys.path.insert(0, "src")
import numpy as np
from mscpu import tower as T


def _rand_weights(seed=0):
    rng = np.random.default_rng(seed)
    sc = 0.05
    return (rng.normal(0, sc, (T.SPEC_D, T.H)), np.zeros(T.H),
            rng.normal(0, sc, (T.H, T.D)), np.zeros(T.D),
            rng.normal(0, sc, (T.MOL_IN, T.H)), np.zeros(T.H),
            rng.normal(0, sc, (T.H, T.D)), np.zeros(T.D))


def test_forward_l2_and_bounded():
    W1, b1, W2, b2, V1, c1, V2, c2 = _rand_weights()
    x = np.abs(np.random.default_rng(1).normal(0, 1, T.SPEC_D))
    q = T.forward_q(x, W1, b1, W2, b2)
    assert q.shape == (T.D,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-9
    from mscpu import quality as Q
    c = T.forward_c(Q.morgan_bits("CCO"), V1, c1, V2, c2)
    assert c.shape == (T.D,)
    assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-9
    s = T.tower_score(q, c)
    assert -1.0 <= s <= 1.0
    assert T.tower_score(None, c) == 0.0
    assert T.tower_score(q, None) == 0.0
    # determinism: same inputs, same outputs
    q2 = T.forward_q(x, W1, b1, W2, b2)
    assert np.array_equal(q, q2)


def test_weights_roundtrip(tmp_path):
    ws = _rand_weights(7)
    p = os.path.join(str(tmp_path), "t.tower.bin")
    T.save_weights(p, *ws)
    back = T.load_weights(p)
    for a, b in zip(ws, (back["W1"], back["b1"], back["W2"], back["b2"],
                          back["V1"], back["c1"], back["V2"], back["c2"])):
        assert np.array_equal(np.ascontiguousarray(a, dtype=np.float32), b)
    # bad magic raises loud
    with open(p, "r+b") as f:
        f.write(b"\x00\x00\x00\x00")
    try:
        T.load_weights(p)
        assert False, "must raise"
    except ValueError:
        pass


def test_checker_pass_and_flag():
    assert T.check_valence("CCO") == []
    assert T.check_valence("c1ccccc1") == []
    assert T.check_valence("[Na+].[Cl-]") == []  # salts pass
    bad = T.check_valence("C(C)(C)(C)(C)C")  # central C: 5 single bonds
    assert len(bad) == 1 and "C#0" in bad[0]
    assert T.check_valence("") == ["empty"]
    r = T.checker_filter(["CCO", "C(C)(C)(C)(C)C", "CCC"])
    assert r["pass"] == ["CCO", "CCC"]
    assert len(r["fail"]) == 1 and r["fail"][0]["smi"] == "C(C)(C)(C)(C)C"
