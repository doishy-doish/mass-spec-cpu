"""measure_store.py — quantify the 3GB so the compression plan runs on numbers.

Read-only except --out. No RDKit, no torch: stdlib + numpy/pandas/pyarrow (+zstd
installed in-step). Prints one JSON line + writes it to --out for the record.

Measures:
  1. file size, row count, per-column compressed/uncompressed bytes (parquet
     metadata only — no full read needed for this part)
  2. pandas subset-read memory + time (what run.py actually loads)
  3. quantized-bins codec: sample N spectra -> clean -> encode_bins -> packed
     uint32 stream; raw bytes/spectrum + zlib9 + zstd3/zstd10 sizes +
     decompress timings; bins-only vs bins+uint8-intensity variants
"""
from __future__ import annotations

import argparse
import array
import json
import os
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mscpu import binary as B
from mscpu import spectra as S

try:
    import zstd as _zstd  # pip install zstd (workflow measure step)
except Exception:
    _zstd = None

COLS = ["inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz",
        "adduct", "instrument_type", "ionization_mode", "ms2_mzs",
        "ms2_normalized_intensities", "ingest_lib"]


def pack_bins(all_bins, with_intensity: bool):
    """all_bins: list of (p_list, l_list[, w_list]). Returns raw bytes."""
    buf = array.array("I")
    for item in all_bins:
        if with_intensity:
            p, l, w = item
        else:
            p, l = item[:2]
            w = None
        buf.append(len(p))
        buf.extend(p)
        buf.append(len(l))
        buf.extend(l)
        if with_intensity:
            buf.append(len(w))
            buf.extend(w)
    return buf.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=int, default=50000)
    a = ap.parse_args()
    out: dict = {}

    out["file_bytes"] = os.path.getsize(a.data)
    t0 = time.time()
    pf = pq.ParquetFile(a.data)
    md = pf.metadata
    out["rows"] = md.num_rows
    col_stats: dict = {}
    for rg in range(md.num_row_groups):
        rgm = md.row_group(rg)
        for c in range(rgm.num_columns):
            cc = rgm.column(c)
            name = ".".join(cc.path_in_schema)
            s = col_stats.setdefault(name, {"compressed": 0, "uncompressed": 0})
            s["compressed"] += cc.total_compressed_size
            s["uncompressed"] += cc.total_uncompressed_size
    out["columns"] = col_stats
    out["parquet_metadata_s"] = round(time.time() - t0, 1)

    # what run.py loads
    t0 = time.time()
    df = pd.read_parquet(a.data, columns=[c for c in COLS])
    out["pandas_read_s"] = round(time.time() - t0, 1)
    out["pandas_mem_bytes"] = int(df.memory_usage(index=True, deep=True).sum())

    # quantized-bins codec on a strided sample
    n = len(df)
    stride = max(1, n // a.sample)
    idx = list(range(0, n, stride))[:a.sample]
    bins_only, bins_w = [], []
    t0 = time.time()
    for i in idx:
        r = df.iloc[i]
        peaks = S.clean(r["ms2_mzs"], r["ms2_normalized_intensities"], r["precursor_mz"])
        p, l = B.encode_bins(peaks, r["precursor_mz"])
        bins_only.append((p, l))
        w = [min(255, int(round(float(x[1]) * 255))) for x in peaks] if len(peaks) else []
        bins_w.append((p, l, w))
    out["codec_sample_n"] = len(idx)
    out["codec_encode_s"] = round(time.time() - t0, 1)

    raw = pack_bins(bins_only, False)
    raw_w = pack_bins(bins_w, True)
    res: dict = {
        "raw_B_per_spec": round(len(raw) / max(1, len(idx)), 1),
        "raw_w_B_per_spec": round(len(raw_w) / max(1, len(idx)), 1),
        "extrap_raw_MB": round(len(raw) / max(1, len(idx)) * n / 1e6, 1),
        "extrap_raw_w_MB": round(len(raw_w) / max(1, len(idx)) * n / 1e6, 1),
    }
    t0 = time.time()
    zb = zlib.compress(raw, 9)
    res["zlib9_B_per_spec"] = round(len(zb) / max(1, len(idx)), 1)
    res["zlib9_time_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    zlib.decompress(zb)
    res["zlib9_decompress_s"] = round(time.time() - t0, 2)
    if _zstd is not None:
        for lvl in (3, 10):
            t0 = time.time()
            zc = _zstd.compress(raw, lvl)
            ct = time.time() - t0
            t0 = time.time()
            assert _zstd.decompress(zc) == raw
            dt = time.time() - t0
            res[f"zstd{lvl}_B_per_spec"] = round(len(zc) / max(1, len(idx)), 1)
            res[f"zstd{lvl}_compress_s"] = round(ct, 2)
            res[f"zstd{lvl}_decompress_s"] = round(dt, 3)
    else:
        res["zstd"] = "not installed"
    out["codec"] = res

    line = json.dumps(out)
    print(line, flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
