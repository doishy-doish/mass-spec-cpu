"""I6 input-plug stub tests (synthetic, no data files, no engine).

Locks every new plug to its contract so the runner canary only measures
parity/wall, never correctness. Mirrored in JS (stubs/input_plugs_stub.js).
"""
import sys
sys.path.insert(0, "src")

from mscpu import input_plugs as IP


def test_instrument_same_family_is_one():
    assert IP.instrument_bonus("timsTOF", "timsTOF") == 1.0
    assert IP.instrument_bonus("Orbitrap", "orbitrap Elite") == 1.0
    assert IP.instrument_bonus("timsTOF", "Orbitrap") == 0.75


def test_pool_rerank_keeps_set_stable():
    pool = ["k1", "k2", "k3"]
    key2inst = {"k1": "Orbitrap", "k2": "timsTOF", "k3": "timsTOF"}
    out = IP.pool_rerank_by_instrument(pool, key2inst, "timsTOF")
    assert sorted(out) == ["k1", "k2", "k3"]
    assert out[0] == "k2" and out[1] == "k3"  # matches first, legacy order kept


def test_adduct_neighbors_self_first():
    nbs = IP.adduct_neighbors("[M+H]+")
    assert nbs[0] == "[M+H]+"
    assert "[M+Na]+" in nbs
    assert IP.adduct_neighbors("[unknown]+") == ["[unknown]+"]


def test_adduct_neighbors_test_set_complete():
    # R2: all 10 competition test adducts resolve past self (case-insensitive).
    for ad in ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+",
               "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]:
        nbs = IP.adduct_neighbors(ad)
        assert nbs[0] == ad, ad
        assert IP.adduct_neighbors(ad.lower())[0] == ad, ad
    # water-loss neighbors stay in-lane and terminate at the base frame.
    assert IP.adduct_neighbors("[M-H2O+H]+")[1] == "[M+H]+"
    assert IP.adduct_neighbors("[M-H2O-H]-")[1] == "[M-H]-"
    assert IP.adduct_neighbors("[M+CH2O2-H]-")[0] == "[M+CH2O2-H]-"


def test_adduct_deltas_frames():
    # positive lane vs [M+H]+ frame; negative lane vs [M-H]- frame.
    assert abs(IP.retry_precisions(300.0, "[M+H]+", "[M+Na]+") - 321.98194) < 1e-9
    assert abs(IP.retry_precisions(300.0, "[M-H]-", "[M+Cl]-") - 335.97613) < 1e-9
    # formate matches the Rust engine shift in this frame.
    assert abs(IP.retry_precisions(300.0, "[M-H]-", "[M+CH2O2-H]-") - 346.00548) < 1e-9
    # H2O mass shared by train/test spellings.
    assert abs(IP.ADDUCT_DELTA["[m-h2o+h]+"] - (-18.01056)) < 1e-9
    assert abs(IP.ADDUCT_DELTA["[m-h2o-h]-"] - (-18.01056)) < 1e-9
    assert abs(IP.ADDUCT_DELTA["[m-2h2o+h]+"] - (-36.02113)) < 1e-9


def test_mass_grade_plateau_and_zero():
    assert IP.mass_grade(-1.5) == 1.0
    assert IP.mass_grade(0.0) == 1.0       # within 3ppm of center
    assert IP.mass_grade(50.0) == 0.0
    assert IP.mass_grade(float("nan")) == 0.0
    mid = IP.mass_grade(4.0)               # between free(3) and full(8)
    assert 0.0 < mid < 1.0


def test_rank1_replace_rule():
    assert IP.rank1_replace(12.0, 1.0) is True
    assert IP.rank1_replace(1.0, 0.5) is False
    assert IP.rank1_replace(12.0, 20.0) is False


def test_formula_mass_and_gate():
    assert abs(IP.formula_mass("C6H12O6") - 180.06339) < 0.01
    assert IP.formula_consistent(180.0634, "C6H12O6", ppm=8.5) is True
    assert IP.formula_consistent(200.0, "C6H12O6", ppm=8.5) is False
    assert IP.formula_consistent(200.0, "???", ppm=8.5) is True  # unknown = no gate


def test_bounded_cache_cap_and_hits():
    c = IP.BoundedCache(cap=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1 and c.hits == 1
    c.put("c", 3)  # evicts oldest ("a")
    assert len(c) == 2
    assert c.get("a") is None and c.misses == 1


def test_recall_and_topk():
    ranked = [["a", "b", "c"], ["x", "y"]]
    assert IP.recall_at_k(ranked, ["b", "z"], 2) == 0.5
    assert IP.top_k(["a", "b", "c"], 2) == ["a", "b"]


def test_rank_replace_list_mirrors_engine():
    ppm = {"a": 20.0, "b": 1.0, "c": 2.0}
    assert IP.rank_replace_list(["a", "b", "c"], ppm) == ["b", "a", "c"]
    assert IP.rank_replace_list(["b", "a"], ppm) == ["b", "a"]  # leader best
    assert IP.rank_replace_list(["a"], {}) == ["a"]  # no ppms
    assert IP.rank_replace_list([], ppm) == []


def test_entropy_helpers_mirror_rust():
    assert IP.score_entropy([]) == 0.0
    assert IP.score_entropy([3.0]) == 0.0
    import math as _m
    assert abs(IP.score_entropy([1.0, 1.0]) - _m.log(2)) < 1e-12
    assert abs(IP.entropy_exp([3.0]) - 0.25) < 1e-12
    assert IP.entropy_exp([]) == 1.0
    assert abs(IP.entropy_grade(1.0, 1.0) - 1.0) < 1e-12
    assert IP.entropy_grade(0.0, 0.5) == 0.0
