"""sengine launcher tests (no binary, no data files).

Gates the thin launcher: structs/engine message schemas, pool + RR assembly
with a stub engine, metric spec. numpy (+pyarrow for the loader stub) only,
no pandas, no RDKit. Runs on the GH runner (pytest gate before any dispatch).
"""
import os
import sys
sys.path.insert(0, "src")
import json

import numpy as np

from mscpu import run as R


def _table():
    return {
        "key14": np.array(["k1", "k1", "k2"]),
        "smiles": ["CCO", "CCO", "CCC"],
        "prec": np.array([201.0, 201.5, 251.0]),
        "adduct": np.array(["[M+H]+", "[M+H]+", "[M+H]+"]),
        "instrument": np.array(["timsTOF", "timsTOF", "timsTOF"]),
        "mzs": [np.array([100.0, 200.0]), np.array([100.0]), np.array([150.0])],
        "ints": [np.array([1.0, 1.0]), np.array([1.0]), np.array([1.0])],
    }


def _qrec(T, i):
    return {"mzs": T["mzs"][i], "ints": T["ints"][i], "prec": float(T["prec"][i])}


def test_structs_payload_schema():
    T = _table()
    ukeys, kidx, precs, adducts, sq = R.build_structs_payload(T, [0, 2])
    assert ukeys == ["k1", "k2"] and kidx == [0, 0, 1]
    assert precs == [201.0, 201.5, 251.0]
    assert adducts == ["[M+H]+"] * 3
    assert sq == [{"qi": 0, "prec": 201.0, "adduct": "[M+H]+"},
                 {"qi": 2, "prec": 251.0, "adduct": "[M+H]+"}]
    json.dumps({"ktable": ukeys, "kidx": kidx, "precs": precs,
                "adducts": adducts, "queries": sq})


def test_preload_chunks_schema():
    T = _table()
    lib_by_key = {"k1": [0, 1], "k2": [2]}
    chunks = list(R.build_preload_chunks(lib_by_key, T, {"k1": "CCO", "k2": "CCC"},
                                       {"k1": 200.0, "k2": None}, per=1))
    assert len(chunks) == 2
    c0 = json.loads(chunks[0])["preload"]
    assert c0["keys"] == ["k1"] and c0["smiles"] == ["CCO"]
    assert c0["neutrals"] == [200.0] and len(c0["specs"][0]) == 2
    c1 = json.loads(chunks[1])["preload"]
    assert c1["neutrals"] == [None]  # M198: null, never NaN (serde rejects NaN)


def test_stream_preload_sends_every_chunk_once():
    """M207: chunks stream (build one, send one) — bytes identical to list."""
    import inspect
    T = _table()
    lib_by_key = {"k1": [0, 1], "k2": [2]}
    gen = R.build_preload_chunks(lib_by_key, T, {"k1": "CCO", "k2": "CCC"},
                                 {"k1": 200.0, "k2": None}, per=1)
    assert inspect.isgenerator(gen)  # the materialization must not come back
    seen = []

    class _Sink:
        def preload(self, lines):
            seen.extend(lines)

        def preload_done(self):
            seen.append("DONE")

    n = R.stream_preload(_Sink(), gen)
    assert n == 2 and seen[-1] == "DONE" and len(seen) == 3
    c0 = json.loads(seen[0])["preload"]
    assert c0["keys"] == ["k1"] and len(c0["specs"][0]) == 2
    assert json.loads(seen[1])["preload"]["neutrals"] == [None]


def test_union_selected_keys_sorted_deduped():
    """M211: union over selected queries is sorted + deduped, Nones skipped."""
    sel = [(0, 0, 2, ["k2", "k1"], ["k3", "k1"], 201.0, 5.0), None,
           (1, 2, 2, ["k1"], [], 251.0, 6.0)]
    assert R.union_selected_keys(sel) == ["k1", "k2", "k3"]
    assert R.union_selected_keys([]) == []
    assert R.union_selected_keys([None]) == []


def test_subset_lib_keeps_missing_missing():
    """M211: subset preload scores identically — unknown keys stay unknown."""
    lib = {"k1": [0, 1], "k2": [2]}
    sub = R.subset_lib(lib, ["k2", "k9"])
    assert sub == {"k2": [2]}
    T = _table()
    seen = []

    class _Sink:
        def preload(self, lines):
            seen.extend(lines)

        def preload_done(self):
            seen.append("DONE")

    n = R.stream_preload(_Sink(), R.build_preload_chunks(
        sub, T, {"k2": "CCC"}, {"k2": None}, per=500))
    assert n == 1 and seen[-1] == "DONE"
    c0 = json.loads(seen[0])["preload"]
    assert c0["keys"] == ["k2"] and len(c0["specs"][0]) == 1
    assert c0["neutrals"] == [None]


def test_engine_query_schema():
    m = R.build_engine_query(_qrec(_table(), 0), ["k1"], ["k2"], 0)
    assert set(m) == {"q", "cand_keys", "pool_keys", "qi", "qnm"}
    assert m["qi"] == 0 and m["qnm"] is None
    assert m["cand_keys"] == ["k1"] and m["pool_keys"] == ["k2"]
    assert m["q"]["prec"] == 201.0
    json.dumps(m)


def test_engine_query_carries_qnm():
    m = R.build_engine_query(_qrec(_table(), 0), ["k1"], ["k2"], 0, 200.0)
    assert m["qnm"] == 200.0
    json.dumps(m)


def test_ce_single_values_and_merges():
    assert R._ce_single(20.0) == 20.0
    assert R._ce_single("40") == 40.0
    assert R._ce_single([20.0]) == 20.0
    assert R._ce_single([20, 40, 60]) is None  # merged ramp matches any band
    assert R._ce_single(None) is None
    assert R._ce_single(float("nan")) is None


def test_engine_query_carries_ce():
    qrec = dict(_qrec(_table(), 0))
    qrec["ce"] = 40.0
    m = R.build_engine_query(qrec, ["k1"], ["k2"], 0)
    assert m["q"]["ce"] == 40.0
    json.dumps(m)
    m2 = R.build_engine_query(_qrec(_table(), 0), ["k1"], ["k2"], 0)
    assert m2["q"]["ce"] is None  # missing CE degrades, never crashes


def test_engine_mol_query_schema_and_union():
    T = _table()
    qrecs = [_qrec(T, 0), _qrec(T, 1)]
    m = R.build_engine_mol_query(qrecs, ["k1"], ["k2"], [0, 1], [200.0, None])
    assert set(m) == {"qs", "cand_keys", "pool_keys", "qis", "qnms"}
    assert len(m["qs"]) == 2 and m["qis"] == [0, 1]
    assert m["qnms"] == [200.0, None]
    json.dumps(m)
    sel = [(0, 0, 2, ["k1"], ["k2"], 201.0, 1.0),
           (1, 1, 2, ["k1", "k2"], ["k2"], 201.5, 1.0)]
    ck, pk = R.mol_union_keys(sel)
    assert ck == ["k1", "k2"] and pk == ["k2"]
    assert R.mol_union_keys([None]) == ([], [])


class _StubEngine:
    def __init__(self, ranked):
        self.ranked = ranked
        self.seen_preload = 0

    def preload(self, chunks):
        self.seen_preload = len(chunks)

    def preload_done(self):
        pass

    def query(self, qmsg):
        assert set(qmsg) == {"q", "cand_keys", "pool_keys", "qi", "qnm"}
        return {"ranked": self.ranked, "pools": len(qmsg["pool_keys"]),
                "mass_swap": True, "timing": [1.0] * 8}


def test_two_phase_select_then_score_with_stub_engine():
    T = _table()
    ranked = {v: (["k1", "k2"] if v != "router" else ["k2", "k1"]) for v in R.VARIANTS}
    R._SE.update(skeys=np.array(["k1", "k2"]),
                 masses=np.array([200.0, 250.0]),
                 qnm={0: 200.0}, qprec={0: 201.0},
                 qmzs={0: T["mzs"][0]}, qints={0: T["ints"][0]},
                 qkey={0: "k1"},
                 lib_precs=[201.0, 201.5, 251.0], lib_pkeys=["k1", "k1", "k2"],
                 ppm=8.5, shard=0, variants=R.VARIANTS)
    sel = R.select_task((0, 0, 1))
    assert sel is not None and sel[1] == 0 and len(sel[3]) == 1
    qi, row, plen, ranked_z, line, swap = R.score_one(sel, _StubEngine(ranked))
    assert qi == 0 and plen == 1 and ranked_z == ["k1", "k2"]
    assert row["lib"] == 1.0 and row["router"] == 0.5
    assert "q=0 pool=1" in line and "tm=1,1,1,1,1,1,1,1" in line
    assert swap is True


def test_sword_prec_cut_pure_window_with_recall_fallback():
    T = _table()
    R._SE.update(skeys=np.array(["k1", "k2"]),
                 masses=np.array([200.0, 250.0]),
                 qnm={0: 200.0}, qprec={0: 200.0},
                 qmzs={0: T["mzs"][0]}, qints={0: T["ints"][0]},
                 qkey={0: "k1"},
                 lib_precs=[199.0, 199.9995, 200.0, 200.0005, 250.0],
                 lib_pkeys=["d0", "d1", "k1", "d3", "k2"],
                 ppm=8.5, shard=0, variants=R.VARIANTS,
                 sword_prec=1)
    sel = R.select_task((0, 0, 1))
    assert sel is not None
    assert sel[7] == 4 and sel[8] == 2 and sel[9] == 0.0
    assert sel[4] == ["d1", "d3"]
    # recall lock: a cut that empties the pool falls back, truth survives
    R._SE.update(lib_precs=[199.0, 201.0], lib_pkeys=["kT", "d9"],
                 qprec={0: 200.0}, qkey={0: "kT"})
    sel3 = R.select_task((0, 0, 1))
    assert sel3[8] == 2 and sel3[9] == 1.0 and sel3[4] == ["kT", "d9"]


def test_load_table_stub_parquet(tmp_path):
    """Loader stub: tiny real parquet in, table dict out (exercises pyarrow path)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = str(tmp_path / "tiny.parquet")
    pq.write_table(pa.table({
        "inchikey14": ["k1", "k2"],
        "normalized_smiles": ["CCO", "CCC"],
        "precursor_mz": [201.0, 251.0],
        "adduct": ["[M+H]+", "[M+H]+"],
        "instrument_type": ["timsTOF", "Orbitrap"],
        "collision_energy_ev": [[20.0], [20.0, 40.0]],
        "ms2_mzs": [[100.0, 200.0], [150.0]],
        "ms2_normalized_intensities": [[1.0, 0.5], [1.0]],
    }), p)
    T = R.load_table(p, train=True)
    assert T["key14"].tolist() == ["k1", "k2"]
    assert T["smiles"] == ["CCO", "CCC"]
    assert T["prec"].tolist() == [201.0, 251.0]
    assert T["instrument"].tolist() == ["timsTOF", "Orbitrap"]
    assert T["ce"] == [20.0, None]  # single CE passes, merged ramp -> None
    assert [a.tolist() for a in T["mzs"]] == [[100.0, 200.0], [150.0]]


def test_engine_query_erun_schema():
    """E1: row-ref + keys, zero spectra — engine resolves from its store."""
    m = R.build_engine_query_erun(7, ["k1"], ["k2"], 200.0)
    assert set(m) == {"qrow", "cand_keys", "pool_keys", "qi", "qnm"}
    assert m["qrow"] == 7 and m["qi"] == 7 and m["qnm"] == 200.0
    assert m["cand_keys"] == ["k1"] and m["pool_keys"] == ["k2"]
    assert "mzs" not in m and "ints" not in m and "q" not in m
    json.dumps(m)
    m2 = R.build_engine_query_erun(0, [], [], None)
    assert m2["qrow"] == 0 and m2["qnm"] is None
    json.dumps(m2)
