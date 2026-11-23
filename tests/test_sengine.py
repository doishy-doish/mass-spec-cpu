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
    chunks = R.build_preload_chunks(lib_by_key, T, {"k1": "CCO", "k2": "CCC"},
                                    {"k1": 200.0, "k2": None}, per=1)
    assert len(chunks) == 2
    c0 = json.loads(chunks[0])["preload"]
    assert c0["keys"] == ["k1"] and c0["smiles"] == ["CCO"]
    assert c0["neutrals"] == [200.0] and len(c0["specs"][0]) == 2
    c1 = json.loads(chunks[1])["preload"]
    assert c1["neutrals"] == [None]  # M198: null, never NaN (serde rejects NaN)


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
        "ms2_mzs": [[100.0, 200.0], [150.0]],
        "ms2_normalized_intensities": [[1.0, 0.5], [1.0]],
    }), p)
    T = R.load_table(p, train=True)
    assert T["key14"].tolist() == ["k1", "k2"]
    assert T["smiles"] == ["CCO", "CCC"]
    assert T["prec"].tolist() == [201.0, 251.0]
    assert T["instrument"].tolist() == ["timsTOF", "Orbitrap"]
    assert [a.tolist() for a in T["mzs"]] == [[100.0, 200.0], [150.0]]
