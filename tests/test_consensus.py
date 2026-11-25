"""Parallel-consensus (poll) fusion tests (no binary, no data files).

consensus_list RRF-fuses the six orthogonal computed member lists with the
same 1/(60+rank) rule as the per-attempt fuse. Scoring only — the pool is
untouched, so U_recall is intact by construction (fusion, never veto).
Runs on the GH runner (pytest gate before any dispatch).
"""
import sys
sys.path.insert(0, "src")

from mscpu import run as R


def test_consensus_rrf_parity():
    lists = {"mass": ["a", "b", "c"], "gate": ["b", "a"],
             "cleav": [], "entw": ["c", "b", "a"],
             "stein": [], "analog": ["a"]}
    got = R.consensus_list(lists)
    ea = 1 / 61 + 1 / 62 + 1 / 63 + 1 / 61
    eb = 1 / 62 + 1 / 61 + 1 / 62
    ec = 1 / 63 + 1 / 61
    assert ea > eb > ec
    assert got == ["a", "b", "c"]


def test_consensus_empty_and_dupes():
    assert R.consensus_list({}) == []
    assert R.consensus_list({"mass": [], "gate": []}) == []
    assert R.consensus_list(None) == []
    got = R.consensus_list({"mass": ["a", "a", "b", None, ""],
                            "gate": ["b", "a"]})
    # a: 1/61 + 1/62 ; b: 1/63 + 1/61 -> a first
    assert got == ["a", "b"]


def test_consensus_caps_25():
    lists = {"mass": ["k%d" % i for i in range(30)],
             "gate": ["k%d" % i for i in range(30)]}
    assert len(R.consensus_list(lists)) == 25


def test_consensus_is_fusion_not_veto():
    # truth ranked by only ONE member still surfaces (a veto would cut it)
    got = R.consensus_list({"mass": ["t"], "gate": ["x"], "cleav": ["x"],
                            "entw": ["x"], "stein": ["x"], "analog": ["x"]})
    assert "t" in got


def test_assemble_row_adds_consensus():
    out = {"ranked": {"mass": ["t", "x"], "gate": ["x", "t"],
                      "cleav": ["t"], "entw": [], "stein": [],
                      "analog": [], "z": ["x", "t"]},
           "mass_swap": False}
    row, rz, swap = R.assemble_row(out, "t", 0, ["z", "mass", "gate"])
    assert row["z"] == 0.5
    assert row["mass"] == 1.0 and row["gate"] == 0.5  # untouched
    assert row["cons"] == 1.0  # t: 1/61+1/62+1/61 tops x: 1/62+1/61
    assert rz == ["x", "t"] and swap is False
