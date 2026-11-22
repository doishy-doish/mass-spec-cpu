"""Aggregation + submission-format tests (no RDKit, no data files)."""
import pandas as pd

from mscpu.metric import validate_submission
from mscpu.predict import aggregate, rrf_fuse


def test_rrf_prefers_consensus():
    df = pd.DataFrame({"lib": [3, 2, 1], "analog": [3, 2, 1], "frag": [1, 2, 3]})
    s = rrf_fuse(df)
    assert list(s.sort_values(ascending=False).index) == [0, 1, 2]


def test_aggregate_sums_reciprocal_rank():
    top = aggregate([["a", "b"], ["b", "c"]], k=3)
    assert top == ["b", "a", "c"]  # b: 1/2+1.0=1.5 > a: 1.0 > c: 0.5


def test_aggregate_dedupes_within_list():
    top = aggregate([["a", "a", "b"]], k=2)
    assert top == ["a", "b"]


def test_submission_format_ok():
    rows = [["m_1", ";".join(["CCO"] * 25)], ["m_2", "CCC;CC"]]
    assert validate_submission(rows) == []


def test_submission_format_rejects():
    assert validate_submission([["m_1", ";".join(["CCO"] * 26)]]) != []
    assert validate_submission([["m_1", "CCO"], ["m_1", "CCC"]]) != []
