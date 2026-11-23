"""slim.py — run-path loader for the preconverted slim store (no pyarrow).

Converted once by scripts/build_slim.py (train.parquet -> data/slim/ flat
zstd buffers, bit-exact m/z delta codec, sha256 manifest). Workers load the
store directly: flat numeric buffers + shared string refs, no per-row
pyarrow pylist conversion, no 5M tiny float64 arrays.

Not bit-duplicated with build_slim: the two delta/raw codecs below are the
same math (int64-bit deltas, exact cumsum), pinned by tests/test_slim.py on
both sides. manifest codec must match build_slim.CODEC; anything else fails
loud (rebuild the store, do not silently mix codecs).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

CODEC = "f64bitdelta+zstd3+ce1"


def _read(name: str, store: str) -> bytes:
    with open(os.path.join(store, name), "rb") as f:
        return f.read()


def _verify(manifest: dict, store: str) -> None:
    files = manifest.get("files", {})
    for name, digest in files.items():
        b = _read(name, store)
        if hashlib.sha256(b).hexdigest() != digest:
            raise ValueError(
                "slim store corrupt/tampered: %s sha mismatch "
                "(delete %s and rebuild)" % (name, store))


def _dec_raw(b: bytes, dtype) -> np.ndarray:
    return np.frombuffer(zstd.decompress(b), dtype=dtype)


def _dec_deltas_f64(b: bytes, n: int) -> np.ndarray:
    d = np.frombuffer(zstd.decompress(b), dtype=np.int64)
    if d.shape[0] != n:
        raise ValueError("slim mz_data len %d != manifest mz_values %d"
                         % (d.shape[0], n))
    return np.cumsum(d).view(np.float64)


def load_slim(store: str = "data/slim") -> dict:
    """Load the slim store -> S dict (row order == parquet order).

    S carries flat buffers only. Per-row strings are shared refs
    (keys/adducts/instruments tables indexed by int arrays) — no 2.5M
    duplicated Python strings, no per-row float lists.
    """
    if zstd is None:
        raise RuntimeError("slim loader needs zstandard (pip install zstandard)")
    with open(os.path.join(store, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("codec") != CODEC:
        raise ValueError(
            "slim codec %r != loader %r: rebuild the store "
            "(scripts/build_slim.py) — refusing to mix codecs"
            % (manifest.get("codec"), CODEC))
    _verify(manifest, store)

    keys = json.load(open(os.path.join(store, "keys.json")))
    smiles = json.load(open(os.path.join(store, "smiles.json")))
    adducts = json.load(open(os.path.join(store, "adducts.json")))
    instruments = json.load(open(os.path.join(store, "instruments.json")))
    n = int(manifest["rows"])
    if len(keys) != int(manifest["keys"]):
        raise ValueError("slim keys.json len != manifest keys")

    key_idx = _dec_raw(_read("key_idx.int32.zst", store), np.int32)
    prec = _dec_raw(_read("prec.f64.zst", store), np.float64)
    ad = _dec_raw(_read("adduct.u8.zst", store), np.uint8)
    inst = _dec_raw(_read("inst.u8.zst", store), np.uint8)
    mz_off = _dec_raw(_read("mz_off.int64.zst", store), np.int64)
    ce = _dec_raw(_read("ce.f64.zst", store), np.float64)
    key_inst = _dec_raw(_read("key_inst.int32.zst", store), np.int32)
    key_first = _dec_raw(_read("key_first.int32.zst", store), np.int32)
    neutrals_arr = _dec_raw(_read("neutrals.f64.zst", store), np.float64)
    mz_all = _dec_deltas_f64(_read("mz_data.zst", store),
                             int(manifest["mz_values"]))
    int_all = _dec_raw(_read("int_data.zst", store), np.float64)

    if not (key_idx.shape[0] == prec.shape[0] == ad.shape[0] == inst.shape[0]
            == ce.shape[0] == n):
        raise ValueError("slim row-array len mismatch vs manifest rows")
    if mz_off.shape[0] != n + 1:
        raise ValueError("slim mz_off len != rows+1")
    if not (key_inst.shape[0] == key_first.shape[0] == len(keys)):
        raise ValueError("slim per-key array len != keys")
    if neutrals_arr.shape[0] != len(keys):
        raise ValueError("slim neutrals len != keys")

    neutrals = {}
    for ki, k in enumerate(keys):
        v = float(neutrals_arr[ki])
        neutrals[k] = None if v != v else v
    return {
        "n": n,
        "keys": keys,               # per-key, first-appearance order
        "smiles": smiles,           # per-key dict (first occurrence)
        "adducts": adducts,         # enum table
        "instruments": instruments,  # enum table
        "key_idx": key_idx,         # per row -> key position
        "key_inst": key_inst,       # per key -> instrument enum
        "key_first": key_first,     # per key -> first row index
        "prec": prec,               # per row f64
        "ad": ad,                   # per row -> adduct enum
        "inst": inst,               # per row -> instrument enum
        "ce": ce,                   # per row f64, NaN = unknown/merged
        "mz_off": mz_off,           # per row offsets into mz_all/int_all
        "mz_all": mz_all,           # concatenated float64 m/z (bit-exact)
        "int_all": int_all,         # concatenated float64 intensities
        "neutrals": neutrals,       # per key float|None (builder structs)
        "manifest": manifest,
    }


def row_spec(S: dict, i: int) -> tuple:
    """One row's (mzs view, ints view). Views, not copies — tolist at use."""
    s = int(S["mz_off"][int(i)])
    e = int(S["mz_off"][int(i) + 1])
    return (S["mz_all"][s:e], S["int_all"][s:e])


def row_ce(S: dict, i: int):
    """One row's CE in eV, or None when unknown/merged (NaN storage)."""
    v = float(S["ce"][int(i)])
    return None if v != v else v


def as_table(S: dict) -> dict:
    """Slim store -> load_table-shaped T (same row order, same splits).

    Per-row strings are shared refs (no duplicated copies); spectra stay
    in the flat buffers (mzs/ints/smiles are None — use row_spec). Every
    consumer of T["mzs"]/T["ints"]/T["smiles"]/T["ce"] branches on S.
    """
    import numpy as _np  # local import: module must stay light
    keys_obj = _np.array(S["keys"], dtype=object)
    ad_obj = _np.array(S["adducts"], dtype=object)
    in_obj = _np.array(S["instruments"], dtype=object)
    n = int(S["n"])
    T = {
        "key14": keys_obj[S["key_idx"]],
        "prec": S["prec"],
        "adduct": ad_obj[S["ad"].astype(_np.int64)],
        "instrument": in_obj[S["inst"].astype(_np.int64)],
        "mzs": None,
        "ints": None,
        "smiles": None,
        "ce": None,
        "_S": S,
    }
    assert len(T["key14"]) == n
    return T
