"""Aggregation + submission-format tests (no engine, no data files)."""
from mscpu.metric import validate_submission
from mscpu.predict import aggregate, aggregate_rrf


def test_aggregate_sums_reciprocal_rank():
    top = aggregate([["a", "b"], ["b", "c"]], k=3)
    assert top == ["b", "a", "c"]  # b: 1/2+1.0=1.5 > a: 1.0 > c: 0.5


def test_aggregate_dedupes_within_list():
    top = aggregate([["a", "a", "b"]], k=2)
    assert top == ["a", "b"]


def test_aggregate_rrf_single_equals_order_and_fuses():
    # S=1 monotonic in rank; S=2 agreeing keeps winner.
    assert aggregate_rrf([["a", "b", "c"]]) == ["a", "b", "c"]
    assert aggregate_rrf([["a", "b"], ["a", "c"]])[0] == "a"
    assert aggregate_rrf([]) == []
    assert aggregate_rrf([["a", "a", "b"]], k=2) == ["a", "b"]


def test_submission_format_ok():
    rows = [["m_1", ";".join(["CCO"] * 25)], ["m_2", "CCC;CC"]]
    assert validate_submission(rows) == []


def test_submission_format_rejects():
    assert validate_submission([["m_1", ";".join(["CCO"] * 26)]]) != []
    assert validate_submission([["m_1", "CCO"], ["m_1", "CCC"]]) != []
