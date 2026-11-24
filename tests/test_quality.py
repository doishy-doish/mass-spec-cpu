"""quality tests: Tanimoto lens (test-side, never retrieval)."""
import sys
sys.path.insert(0, "src")
from mscpu import quality as Q


def test_morgan_deterministic_and_sized():
    a = Q.morgan_bits("CCO")
    b = Q.morgan_bits("CCO")
    assert a == b
    assert len(a) == 32
    assert any(w != 0 for w in a)
    assert Q.morgan_bits("") == [0] * 32


def test_tanimoto_self_one_empty_zero():
    a = Q.morgan_bits("CCO")
    assert abs(Q.tanimoto(a, a) - 1.0) < 1e-12
    z = [0] * 32
    assert Q.tanimoto(z, z) == 0.0


def test_similar_outscores_distant():
    cco = Q.morgan_bits("CCO")
    ccc = Q.morgan_bits("CCC")
    benz = Q.morgan_bits("c1ccccc1")
    assert Q.tanimoto(cco, ccc) > Q.tanimoto(cco, benz)


def test_row_and_summary():
    row = Q.tanimoto_row(["CCO", "CCC"], "CCO")
    assert row[0] == 1.0 and 0.0 <= row[1] <= 1.0
    assert Q.tanimoto_row(["CCO"], "") == [0.0]
    s = Q.quality_summary({"q1": ["CCO", "CCC"], "q2": ["CCC"]},
                          {"q1": "CCO", "q2": "CCC"})
    assert s["n"] == 2
    assert abs(s["mean_top1"] - 1.0) < 1e-12
    assert s["mean_best"] >= s["mean_top1"]
    assert Q.quality_summary({}, {})["n"] == 0
