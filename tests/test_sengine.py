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


def test_engine_query_schema():
    T = _table()
    m = R.build_engine_query(T, _qrec(T, 0), ["k1"], ["k2"],
                             {"k1": [0, 1], "k2": [2]},
                             {"k1": "CCO", "k2": "CCC"}, {}, 0)
    assert set(m) == {"q", "cands", "pool", "qi", "qnm"}
    assert m["qi"] == 0 and m["qnm"] is None
    assert len(m["cands"]) == 1 and len(m["cands"][0]["specs"]) == 2
    assert m["cands"][0]["neutral"] is None
    assert len(m["pool"]) == 1 and m["pool"][0]["key"] == "k2"
    assert m["q"]["prec"] == 201.0
    json.dumps(m)


def test_engine_query_carries_neutrals():
    T = _table()
    m = R.build_engine_query(T, _qrec(T, 0), ["k1"], ["k2"],
                             {"k1": [0, 1], "k2": [2]},
                             {"k1": "CCO", "k2": "CCC"},
                             {"k1": 200.0, "k2": 250.0}, 0, 200.0)
    assert m["qnm"] == 200.0
    assert m["cands"][0]["neutral"] == 200.0
    json.dumps(m)


class _StubEngine:
    def __init__(self, ranked):
        self.ranked = ranked

    def query(self, qmsg):
        assert set(qmsg) == {"q", "cands", "pool", "qi", "qnm"}
        return {"ranked": self.ranked, "pools": len(qmsg["pool"]),
                "mass_swap": True}


def test_run_query_assembles_rr_with_stub_engine():
    T = _table()
    ranked = {v: (["k1", "k2"] if v != "router" else ["k2", "k1"]) for v in R.VARIANTS}
    R._SE.update(table=T, skeys=np.array(["k1", "k2"]),
                 masses=np.array([200.0, 250.0]),
                 lib_by_key={"k1": [0, 1], "k2": [2]},
                 lib_precs=[201.0, 201.5, 251.0], lib_pkeys=["k1", "k1", "k2"],
                 key2smi={"k1": "CCO", "k2": "CCC"}, qnm={0: 200.0},
                 key2neutral={"k1": 200.0, "k2": 250.0},
                 ppm=8.5, shard=0, variants=R.VARIANTS, rust_bin="x",
                 qpeak=0.05, qloss=0.01,
                 worker=_StubEngine(ranked), worker_pid=os.getpid())
    qi, row, plen, ranked_z, line, swap = R._run_query((0, 0, 1))
    assert qi == 0 and plen == 1 and ranked_z == ["k1", "k2"]
    assert row["lib"] == 1.0 and row["router"] == 0.5
    assert "q=0 pool=1" in line
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
