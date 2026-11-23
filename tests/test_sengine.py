"""sengine launcher tests (no binary, no data files).

Gates the thin launcher: structs/engine message schemas, pool + RR assembly
with a stub engine, metric spec. numpy+pandas only, no RDKit. Runs in Pyodide
and on the GH runner (pytest gate before any dispatch).
"""
import os
import sys
sys.path.insert(0, "src")
import json

import numpy as np
import pandas as pd

from mscpu import run as R


def _meta():
    return pd.DataFrame([
        {"key14": "k1", "smiles": "CCO", "precursor_mz": 201.0, "adduct": "[M+H]+",
         "mzs": np.array([100.0, 200.0]), "ints": np.array([1.0, 1.0])},
        {"key14": "k1", "smiles": "CCO", "precursor_mz": 201.5, "adduct": "[M+H]+",
         "mzs": np.array([100.0]), "ints": np.array([1.0])},
        {"key14": "k2", "smiles": "CCC", "precursor_mz": 251.0, "adduct": "[M+H]+",
         "mzs": np.array([150.0]), "ints": np.array([1.0])},
    ])


def test_structs_payload_schema():
    meta = _meta()
    ukeys, kidx, precs, adducts, sq = R.build_structs_payload(meta, [0, 2])
    assert ukeys == ["k1", "k2"] and kidx == [0, 0, 1]
    assert precs == [201.0, 201.5, 251.0]
    assert adducts == ["[M+H]+"] * 3
    assert sq == [{"qi": 0, "prec": 201.0, "adduct": "[M+H]+"},
                 {"qi": 2, "prec": 251.0, "adduct": "[M+H]+"}]
    json.dumps({"ktable": ukeys, "kidx": kidx, "precs": precs,
                "adducts": adducts, "queries": sq})


def test_engine_query_schema():
    meta = _meta()
    cands = pd.DataFrame([{"key14": "k1"}])
    m = R.build_engine_query(meta.loc[0], cands, ["k2"], {"k1": [0, 1], "k2": [2]},
                             {"k1": "CCO", "k2": "CCC"}, meta, 0)
    assert set(m) == {"q", "cands", "pool", "qi", "qnm"}
    assert m["qi"] == 0 and m["qnm"] is None
    assert len(m["cands"]) == 1 and len(m["cands"][0]["specs"]) == 2
    assert m["cands"][0]["neutral"] is None
    assert len(m["pool"]) == 1 and m["pool"][0]["key"] == "k2"
    assert m["q"]["prec"] == 201.0
    json.dumps(m)


def test_engine_query_carries_neutrals():
    meta = _meta()
    cands = pd.DataFrame([{"key14": "k1"}])
    m = R.build_engine_query(meta.loc[0], cands, ["k2"], {"k1": [0, 1], "k2": [2]},
                             {"k1": "CCO", "k2": "CCC"}, meta, 0,
                             200.0, {"k1": 200.0, "k2": 250.0})
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
    meta = _meta()
    struct_m = pd.DataFrame([{"key14": "k1", "neutral": 200.0},
                             {"key14": "k2", "neutral": 250.0}]).sort_values("neutral")
    struct_m = struct_m.reset_index(drop=True)
    ranked = {v: (["k1", "k2"] if v != "router" else ["k2", "k1"]) for v in R.VARIANTS}
    R._SE.update(meta=meta, struct_m=struct_m, masses=np.array([200.0, 250.0]),
                 lib_by_key={"k1": [0, 1], "k2": [2]},
                 lib_precs=[201.0, 201.5, 251.0], lib_pkeys=["k1", "k1", "k2"],
                 key2smi={"k1": "CCO", "k2": "CCC"}, qnm={0: 200.0},
                 ppm=8.5, shard=0, variants=R.VARIANTS, rust_bin="x",
                 qpeak=0.05, qloss=0.01,
                 worker=_StubEngine(ranked), worker_pid=os.getpid())
    qi, row, plen, ranked_z, line, swap = R._run_query((0, 0, 1))
    assert qi == 0 and plen == 1 and ranked_z == ["k1", "k2"]
    assert row["lib"] == 1.0 and row["router"] == 0.5
    assert "q=0 pool=1" in line
    assert swap is True


def test_load_meta_renames_instrument():
    import pandas as pd
    from mscpu import run as R2
    import inspect
    src = inspect.getsource(R2.load_meta)
    assert "instrument_type" in src and '"instrument"' in src
