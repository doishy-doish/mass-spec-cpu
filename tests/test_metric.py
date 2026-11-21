import sys
sys.path.insert(0, "src")
from mscpu import metric as M


def test_rr():
    assert M.reciprocal_rank("A", ["A", "B"]) == 1
    assert M.reciprocal_rank("B", ["A", "B"]) == 0.5
    assert M.reciprocal_rank("Q", ["A", "B"]) == 0
    assert M.reciprocal_rank("B", ["A", "A", "B"]) == 0.5
    assert M.reciprocal_rank("B", [None, "A", "B"]) == 0.5
    keys = [f"K{i}" for i in range(24)]
    assert M.reciprocal_rank("Z", keys + ["Z"]) == 1 / 25
    assert M.reciprocal_rank("Z", keys + ["K24", "Z"]) == 0


def test_mrr():
    assert abs(M.mrr_at_k({"m1": "A", "m2": "B"}, {"m1": ["A"], "m2": ["C", "B"]}) - 0.75) < 1e-12


def test_validate():
    rows = [("m1", "a"), ("m1", "b"), ("m2", ";".join(["x"] * 26))]
    assert len(M.validate_submission(rows)) == 2
