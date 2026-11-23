"""test_slim — lossless gate for the slim store (synthetic, no data, fast).

Bit-exactness is asserted on float BITS (view int64), not values, so even
1-ulp drift fails. Manifest sha256 is verified, and tampering is detected.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pyarrow as pa
import zstandard as zstd

import build_slim as B


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float64).view(np.int64)


def synth_table():
    # tricky floats: 0.1+0.2-style, subnormals-adjacent, big/small, dup keys
    return pa.table({
        "inchikey14": ["K1", "K2", "K1", "K3"],
        "normalized_smiles": ["CCO", "CCC", "CCO-DUP", "c1ccccc1"],
        "precursor_mz": [100.0001, 200.5, 100.0001, 1e-10],
        "adduct": ["[M+H]+", "[M+Na]+", "[M+H]+", "[M-H]-"],
        "instrument_type": ["timsTOF", "Orbitrap", "timsTOF", "timsTOF"],
        "ms2_mzs": [[0.1 + 0.2, 500.25, 1e300], [1.5], [], [5e-320, 999.999]],
        "ms2_normalized_intensities": [[1.0, 0.5, 0.25], [1.0], [], [0.3, 1.0]],
    })


def test_delta_codec_bit_exact():
    rng = np.random.default_rng(7)
    for x in [np.array([0.1 + 0.2, 500.25, 1e300, 5e-320, 999.999]),
              np.sort(rng.random(1000) * 2000 + 50),
              np.array([150.0])]:
        rt = B.dec_deltas_f64(B.enc_deltas_f64(x), x.shape[0])
        assert (bits(rt) == bits(x)).all(), f"bit drift on {x[:3]}"


def test_build_arrays_first_occurrence():
    d = B.build_arrays(synth_table())
    assert d["keys"] == ["K1", "K2", "K3"]
    assert d["smiles"] == {"K1": "CCO", "K2": "CCC", "K3": "c1ccccc1"}
    assert d["key_idx"].tolist() == [0, 1, 0, 2]
    assert d["prec"].tolist() == [100.0001, 200.5, 100.0001, 1e-10]
    assert d["mz_off"].tolist() == [0, 3, 4, 4, 6]


def test_slim_round_trip_exact(tmp_path):
    d = B.build_arrays(synth_table())
    neutrals = np.array([90.0, 180.0, 50.0])
    man = B.write_slim(d, neutrals, str(tmp_path))
    # manifest sha256 verifies every file
    for name, digest in man["files"].items():
        b = open(os.path.join(str(tmp_path), name), "rb").read()
        assert hashlib.sha256(b).hexdigest() == digest
    # arrays reload bit-exact
    assert (bits(B.dec_raw(open(os.path.join(str(tmp_path), "prec.f64.zst"), "rb").read(),
                            np.float64)) == bits(d["prec"])).all()
    nmz = int(d["mz_all"].shape[0])
    assert (bits(B.dec_deltas_f64(open(os.path.join(str(tmp_path), "mz_data.zst"), "rb").read(), nmz))
            == bits(d["mz_all"])).all()
    assert (bits(B.dec_raw(open(os.path.join(str(tmp_path), "int_data.zst"), "rb").read(), np.float64))
            == bits(d["int_all"])).all()
    assert json.load(open(os.path.join(str(tmp_path), "keys.json"))) == ["K1", "K2", "K3"]
    # tamper detection: flip a byte, hash must mismatch
    p = os.path.join(str(tmp_path), "prec.f64.zst")
    b = bytearray(open(p, "rb").read())
    b[10] ^= 1
    assert hashlib.sha256(bytes(b)).hexdigest() != man["files"]["prec.f64.zst"]
    # zstandard is genuinely compressing (not a pass-through)
    raw = d["mz_all"].tobytes()
    assert len(open(os.path.join(str(tmp_path), "mz_data.zst"), "rb").read()) < len(raw)
