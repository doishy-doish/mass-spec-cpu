"""build_slim.py — one-time preconvert: train.parquet -> data/slim/ flat buffers.

LOSSLESS (user law): every float round-trips bit-exact. m/z arrays are stored
as int64 bit-deltas (integer arithmetic is exact; float cumsum would not be)
then zstd-compressed. Intensities are stored raw + zstd. Manifest carries
sha256 per file; the loader re-hashes and fails loud on any mismatch.

Reads with pyarrow only — no pandas anywhere in this path.
Neutrals come from the Rust structs binary (same call run.py makes today).

Layout (row order == parquet order, so splits stay identical):
  keys.json        [key14...] (first-appearance order)
  smiles.json      {key14: smiles} (first occurrence per key)
  key_idx.int32.zst, prec.f64.zst, adduct.u8.zst, inst.u8.zst
  adducts.json, instruments.json (enum tables)
  mz_off.int64.zst (n+1 offsets into mz stream)
  mz_data.zst      (int64 bit-deltas of concatenated float64 mzs)
  int_data.zst     (raw float64 intensities, concatenated)
  neutrals.f64.zst (per key14, same order as keys.json)
  manifest.json    {rows, keys, codec, sha256 per file, source bytes/mtime}

Usage:
  pip install zstandard  # builder step only; run path installs it too (workflow)
  python scripts/build_slim.py --train data/train.parquet \\
      --rust-bin rust-bitvec/target/release/binvec --out data/slim
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    print("build_slim needs pyarrow (pip install pyarrow)", file=sys.stderr)
    sys.exit(2)

try:
    import zstandard as zstd
except ImportError:
    print("build_slim needs zstandard (pip install zstandard)", file=sys.stderr)
    sys.exit(2)

COLS = ["inchikey14", "normalized_smiles", "precursor_mz",
        "adduct", "instrument_type", "ms2_mzs", "ms2_normalized_intensities"]
LEVEL = 3


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def enc_deltas_f64(x: np.ndarray) -> bytes:
    """Bit-exact delta codec: float64 bits -> int64 -> diffs (exact) -> zstd."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    bits = x.view(np.int64)
    d = np.empty_like(bits)
    d[0] = bits[0]
    d[1:] = np.diff(bits)
    return zstd.compress(d.tobytes(), LEVEL)


def dec_deltas_f64(b: bytes, n: int) -> np.ndarray:
    d = np.frombuffer(zstd.decompress(b), dtype=np.int64)
    assert d.shape[0] == n, f"delta len {d.shape[0]} != {n}"
    return np.cumsum(d).view(np.float64)


def enc_raw(a: np.ndarray) -> bytes:
    return zstd.compress(np.ascontiguousarray(a).tobytes(), LEVEL)


def dec_raw(b: bytes, dtype) -> np.ndarray:
    return np.frombuffer(zstd.decompress(b), dtype=dtype)


def build_arrays(table) -> dict:
    """Pure function: pyarrow Table -> flat numpy dict (testable, no IO)."""
    n = table.num_rows
    key14 = table.column("inchikey14").to_pylist()
    smiles = table.column("normalized_smiles").to_pylist()
    prec = np.array(table.column("precursor_mz").to_pylist(), dtype=np.float64)
    adduct = table.column("adduct").to_pylist()
    inst = table.column("instrument_type").to_pylist()
    mzs_col = table.column("ms2_mzs").to_pylist()
    ints_col = table.column("ms2_normalized_intensities").to_pylist()

    keys: list = []
    key_index: dict = {}
    smile_of: dict = {}
    key_idx = np.empty(n, dtype=np.int32)
    for i, k in enumerate(key14):
        j = key_index.get(k)
        if j is None:
            j = len(keys)
            key_index[k] = j
            keys.append(k)
            smile_of[k] = smiles[i]  # first occurrence, matches groupby().iloc[0]
        key_idx[i] = j

    adducts = sorted(set(str(a) for a in adduct))
    insts = sorted(set(str(t) for t in inst))
    ad_of = {a: i for i, a in enumerate(adducts)}
    in_of = {t: i for i, t in enumerate(insts)}

    mz_off = np.empty(n + 1, dtype=np.int64)
    mz_parts: list = []
    int_parts: list = []
    off = 0
    for i in range(n):
        m = np.ascontiguousarray(mzs_col[i], dtype=np.float64)
        v = np.ascontiguousarray(ints_col[i], dtype=np.float64)
        assert m.shape == v.shape, f"row {i}: m/z/intensity length mismatch"
        mz_parts.append(m)
        int_parts.append(v)
        off += m.shape[0]
        mz_off[i + 1] = off
    mz_all = np.concatenate(mz_parts) if mz_parts else np.empty(0, dtype=np.float64)
    int_all = np.concatenate(int_parts) if int_parts else np.empty(0, dtype=np.float64)

    return {
        "n": n,
        "keys": keys,
        "smiles": smile_of,
        "key_idx": key_idx,
        "prec": prec,
        "adduct": np.array([ad_of[str(a)] for a in adduct], dtype=np.uint8),
        "adducts": adducts,
        "inst": np.array([in_of[str(t)] for t in inst], dtype=np.uint8),
        "instruments": insts,
        "mz_off": mz_off,
        "mz_all": mz_all,
        "int_all": int_all,
        "adduct_str": [str(a) for a in adduct],
    }


def fetch_neutrals(rust_bin: str, keys: list, key_idx: np.ndarray,
                   prec: np.ndarray, adduct_str: list) -> np.ndarray:
    """One structs round trip (Rust): neutrals per key14. Same call run.py makes."""
    kindex = {k: i for i, k in enumerate(keys)}
    payload = {
        "ktable": keys,
        "kidx": [kindex[k] for k in keys],
        "precs": [float(x) for x in prec.tolist()],
        "adducts": adduct_str,
        "queries": [],
    }
    p = subprocess.run([rust_bin, "structs"], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError("structs failed: " + (p.stderr or "")[-500:])
    neutrals = json.loads(p.stdout)["neutrals"]
    return np.array([float(neutrals[k]) for k in keys], dtype=np.float64)


def write_slim(d: dict, neutrals: np.ndarray, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    files: dict = {}

    def w(name: str, b: bytes):
        with open(os.path.join(out, name), "wb") as f:
            f.write(b)
        files[name] = sha(b)

    w("key_idx.int32.zst", enc_raw(d["key_idx"]))
    w("prec.f64.zst", enc_raw(d["prec"]))
    w("adduct.u8.zst", enc_raw(d["adduct"]))
    w("inst.u8.zst", enc_raw(d["inst"]))
    w("mz_off.int64.zst", enc_raw(d["mz_off"]))
    w("mz_data.zst", enc_deltas_f64(d["mz_all"]))
    w("int_data.zst", enc_raw(d["int_all"]))
    w("neutrals.f64.zst", enc_raw(neutrals))
    for name, obj in [("keys.json", d["keys"]), ("smiles.json", d["smiles"]),
                      ("adducts.json", d["adducts"]), ("instruments.json", d["instruments"])]:
        w(name, json.dumps(obj).encode())
    manifest = {"rows": d["n"], "keys": len(d["keys"]), "codec": "f64bitdelta+zstd3",
                "mz_values": int(d["mz_all"].shape[0]), "files": files}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--rust-bin", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = os.stat(a.train)
    table = pq.read_table(a.train, columns=COLS)
    d = build_arrays(table)
    neutrals = fetch_neutrals(a.rust_bin, d["keys"], d["key_idx"], d["prec"], d["adduct_str"])
    manifest = write_slim(d, neutrals, a.out)
    manifest["source"] = {"bytes": st.st_size, "mtime": st.st_mtime}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    total = sum(os.path.getsize(os.path.join(a.out, n)) for n in os.listdir(a.out))
    print(json.dumps({"rows": manifest["rows"], "keys": manifest["keys"],
                      "slim_bytes": total, "source_bytes": st.st_size}))


if __name__ == "__main__":
    main()
