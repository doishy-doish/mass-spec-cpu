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
