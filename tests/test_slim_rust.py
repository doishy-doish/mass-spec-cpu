"""test_slim_rust — cross-language equivalence gate for the Rust slim port.

PROTOCOL (user order, test-then-swap): the Python builder + loader stay the
reference. This test builds a tiny store with the PYTHON builder, reads it
with BOTH loaders (Python slim.py and `binvec slimcheck`), and compares every
field: sha fingerprints over the same LE bytes, full per-row m/z + intensity
bits, CE values, key order. Bit-exactness is on float BITS, so 1-ulp drift
fails. Bad codec + tampered file must fail loud on BOTH sides.

SKIP, not red, when the Rust binary is absent (local/Pyodide runs): the gate
only binds on the runner, where cpu.yml builds both crates before pytest.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PYODIDE") == "1",
    reason="needs local files + rust binary (runner-only gate)",
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, "src")

import numpy as np

BIN = os.path.join("rust-bitvec", "target", "release", "binvec")


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float64).view(np.int64)


def synth_ce_table():
    import pyarrow as pa
    import build_slim as B  # noqa: F401 (import check only)

    return pa.table({
        "inchikey14": ["K1", "K2", "K1", "K3"],
        "normalized_smiles": ["CCO", "CCC", "CCO-DUP", "c1ccccc1"],
        "precursor_mz": [100.0001, 200.5, 100.0001, 1e-10],
        "adduct": ["[M+H]+", "[M+Na]+", "[M+H]+", "[M-H]-"],
        "instrument_type": ["timsTOF", "Orbitrap", "timsTOF", "timsTOF"],
        "ms2_mzs": [[0.1 + 0.2, 500.25, 1e300], [1.5], [], [5e-320, 999.999]],
        "ms2_normalized_intensities": [[1.0, 0.5, 0.25], [1.0], [], [0.3, 1.0]],
        "collision_energy_ev": [[20], [20, 40, 60], [40.0], None],
    })


def need_bin():
    if not os.path.exists(BIN):
        pytest.skip("rust binary absent (local run) — runner-only gate")


def run_slimcheck(store, *extra):
    need_bin()
    p = subprocess.run(
        [BIN, "slimcheck", "--store", str(store), *extra],
        capture_output=True, text=True, timeout=120,
    )
    return p


def test_rust_loader_matches_python(tmp_path):
    import build_slim as B
    from mscpu import slim as SL

    d = B.build_arrays(synth_ce_table())
    neutrals = np.array([90.0, float("nan"), 50.0])
    B.write_slim(d, neutrals, str(tmp_path))
    S = SL.load_slim(str(tmp_path))

    p = run_slimcheck(tmp_path)
    assert p.returncode == 0, f"slimcheck failed: {(p.stderr or '')[-500:]}"
    r = json.loads(p.stdout)
    assert r["n"] == S["n"] == 4
    assert r["keys"] == len(S["keys"]) == 3
    assert r["codec"] == SL.CODEC == B.CODEC
    assert r["keys0"] == S["keys"][:5]

    def sha(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    # Same LE bytes both sides (runner is x86_64 LE; noted, not assumed blind:
    # tobytes() is native-order and the Rust side writes LE explicitly).
    assert r["sha"]["prec"] == sha(S["prec"])
    assert r["sha"]["mz"] == sha(S["mz_all"])
    assert r["sha"]["ints"] == sha(S["int_all"])
    assert r["sha"]["ce"] == sha(S["ce"])
    assert r["sha"]["key_idx"] == hashlib.sha256(
        np.ascontiguousarray(S["key_idx"], dtype=np.int32).tobytes()).hexdigest()

    want_mz = bits(S["mz_all"][: len(r["row0_mzs_bits"])])
    assert r["row0_mzs_bits"] == want_mz.tolist()
    assert r["row0_ints_bits"] == bits(
        S["int_all"][: len(r["row0_ints_bits"])]).tolist()
    assert r["ce0"] == SL.row_ce(S, 0) == 20.0

    # Every row, full bits, both spectra arrays + CE.
    off = S["mz_off"]
    for i in range(S["n"]):
        q = run_slimcheck(tmp_path, "--row", str(i))
        assert q.returncode == 0, f"row {i}: {(q.stderr or '')[-300:]}"
        jr = json.loads(q.stdout)
        mzs, ints = SL.row_spec(S, i)
        assert jr["mzs_bits"] == bits(mzs).tolist(), f"row {i} mzs drift"
        assert jr["ints_bits"] == bits(ints).tolist(), f"row {i} ints drift"
        assert jr["ce"] == SL.row_ce(S, i), f"row {i} ce drift"


def test_rust_loader_rejects_bad_codec(tmp_path):
    import build_slim as B

    need_bin()
    d = B.build_arrays(synth_ce_table())
    B.write_slim(d, np.array([1.0, 2.0, 3.0]), str(tmp_path))
    man_path = os.path.join(str(tmp_path), "manifest.json")
    man = json.load(open(man_path))
    man["codec"] = "bogus"
    json.dump(man, open(man_path, "w"))
    p = run_slimcheck(tmp_path)
    assert p.returncode != 0, "bad codec must fail loud in Rust too"


def test_rust_loader_detects_tamper(tmp_path):
    import build_slim as B

    need_bin()
    d = B.build_arrays(synth_ce_table())
    B.write_slim(d, np.array([1.0, 2.0, 3.0]), str(tmp_path))
    p = os.path.join(str(tmp_path), "prec.f64.zst")
    b = bytearray(open(p, "rb").read())
    b[10] ^= 1
    open(p, "wb").write(bytes(b))
    p = run_slimcheck(tmp_path)
    assert p.returncode != 0, "tampered file must fail loud in Rust too"
