"""predict.py — thin launcher for test-set inference (submission path).

Same division as run.py: Python owns discovery/load/masks/CSV, the Rust
sengine owns all scoring. Per molecule: narrow pool -> sengine -> top-25;
wide-pool rescue for short lists; per-molecule aggregation by summed
reciprocal rank. No Python backup of any channel math (user order: one code).
"""
from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left, bisect_right

import numpy as np

from . import metric as M
from . import run as R


def aggregate(ranked_lists: list, k: int = 25) -> list:
    """Top-k keys by sum of reciprocal rank across per-spectrum rankings.
    Only ranks <= 25 can score (MRR@25); k caps the output length."""
    score: dict = {}
    for rl in ranked_lists:
        seen = set()
        for i, key in enumerate(rl):
            if i >= 25:
                break
            if not key or key in seen:
                continue
            seen.add(key)
            score[key] = score.get(key, 0.0) + 1.0 / (i + 1)
    return sorted(score, key=lambda kk: -score[kk])[:k]


def rank_spectrum(Tlib, qrec, cand_keys, pool_keys, lib_by_key, key2smi,
                  engine, qi, qnm=None, key2neutral=None):
    """One test spectrum -> top-25 keys by the engine's rrf fusion."""
    out = engine.query(R.build_engine_query(Tlib, qrec, cand_keys, pool_keys,
                                            lib_by_key, key2smi,
                                            key2neutral or {}, qi, qnm))
    return out["ranked"]["rrf"][:25]


def discover_kaggle():
    """Kaggle kernels-only: locate competition parquet under /kaggle/input."""
    import os
    train = test = None
    for dp, _, fns in os.walk("/kaggle/input"):
        for fn in fns:
            p = os.path.join(dp, fn)
            ln = fn.lower()
            if ln.endswith(".parquet") and "train" in ln and train is None:
                train = p
            if ln.endswith(".parquet") and "test" in ln and test is None:
                test = p
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--rust-bin", default="rust-bitvec/target/release/binvec")
    ap.add_argument("--qpeak", type=float, default=0.05)
    ap.add_argument("--qloss", type=float, default=0.01)
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--wide", type=float, default=80.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None,
                    help="also/only write submission.csv (kernel mode)")
    a = ap.parse_args()
    if not a.train or not a.test:
        dtrain, dtest = discover_kaggle()
        a.train = a.train or dtrain
        a.test = a.test or dtest
    if not a.train or not a.test:
        raise SystemExit("no --train/--test and nothing discovered")
    if not a.out and not a.csv:
        if os.path.isdir("/kaggle/input"):
            a.csv = "/kaggle/working/submission.csv"
        else:
            a.out = "results/predict.json"

    T = R.load_table(a.train, train=True)
    ukeys = sorted(set(str(k) for k in T["key14"].tolist()))
    kindex = {k: i for i, k in enumerate(ukeys)}
    st = R.structs_call(a.rust_bin, ukeys,
                        [kindex[k] for k in [str(x) for x in T["key14"].tolist()]],
                        [float(x) for x in T["prec"].tolist()],
                        [str(x) for x in T["adduct"].tolist()], [])
    skeys_all = np.array(ukeys)
    # M198: structs emits null for keys with no known-adduct spectra — NaN
    # sorts last and matches no ppm window (honest exclusion, never KeyError).
    def _neutral(k):
        nk = st["neutrals"].get(k)
        return float(nk) if (nk is not None and nk == nk) else np.nan
    sneu_all = np.array([_neutral(k) for k in ukeys], dtype=np.float64)
    sord = np.argsort(sneu_all, kind="stable")
    skeys, masses = skeys_all[sord], sneu_all[sord]
    codes, inv = np.unique(T["key14"], return_inverse=True)
    lib_arr = np.arange(len(T["key14"]), dtype=np.int64)
    srt = np.argsort(inv[lib_arr], kind="stable")
    sk = inv[lib_arr][srt]
    bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    lib_by_key = {}
    for b, e in zip(bounds[:-1], bounds[1:]):
        lib_by_key[str(codes[sk[b]])] = lib_arr[srt[b:e]].tolist()
    _prec_pairs = [(float(T["prec"][js[0]]), k) for k, js in lib_by_key.items()]
    _prec_pairs.sort(key=lambda t: t[0])
    lib_precs = [p for p, _ in _prec_pairs]
    lib_pkeys = [k for _, k in _prec_pairs]
    _, fidx = np.unique(inv, return_index=True)
    key2smi = {str(codes[inv[int(i)]]): str(T["smiles"][int(i)]) for i in fidx}
    key2neutral = {str(k): float(v) for k, v in zip(skeys.tolist(), masses.tolist())}
    engine = R._Engine(a.rust_bin, a.qpeak, a.qloss)

    def pool(qnm: float, ppm: float):
        tol = qnm * ppm * 1e-6
        lo = np.searchsorted(masses, qnm - tol)
        hi = np.searchsorted(masses, qnm + tol, side="right")
        return [str(k) for k in skeys[lo:hi].tolist()]

    def pool50(prec_q: float, cand_keys: set):
        lo = bisect_left(lib_precs, prec_q - 50.0)
        hi = bisect_right(lib_precs, prec_q + 50.0, lo=lo)
        near = []
        for i in range(lo, hi):
            k = lib_pkeys[i]
            if k in cand_keys:
                continue
            near.append((abs(lib_precs[i] - prec_q), k))
        near.sort(key=lambda t: t[0])
        return [k for _, k in near[:1500]]

    Q = R.load_table(a.test, train=False)
    rows = []
    for mid in np.unique(Q["mid"]).tolist():
        mid = str(mid)
        # one structs round trip for the molecule's neutral masses
        gidx = np.flatnonzero(Q["mid"] == mid)
        sq = [{"qi": int(pos), "prec": float(Q["prec"][int(i)]),
               "adduct": str(Q["adduct"][int(i)])} for pos, i in enumerate(gidx)]
        st2 = R.structs_call(a.rust_bin, [], [], [], [], sq)
        qnm_map = {int(qi): v for qi, v in st2["qnm"] if v is not None}
        narrow, wide_lists = [], []
        for pos, i in enumerate(gidx):
            i = int(i)
            qnm = qnm_map.get(pos)
            if not qnm:
                continue
            qrec = {"mzs": Q["mzs"][i], "ints": Q["ints"][i],
                    "prec": float(Q["prec"][i])}
            cn = pool(qnm, a.ppm)
            if len(cn):
                narrow.append(rank_spectrum(T, qrec, cn, pool50(float(Q["prec"][i]),
                                            set(cn)), lib_by_key, key2smi, engine,
                                            i, qnm, key2neutral))
            cw = pool(qnm, a.wide)
            if len(cw):
                wide_lists.append(rank_spectrum(T, qrec, cw, pool50(float(Q["prec"][i]),
                                                set(cw)), lib_by_key, key2smi, engine,
                                                i, qnm, key2neutral))
        top = aggregate(narrow)
        if len(top) < 25:
            for k_ in aggregate(wide_lists):
                if k_ not in top:
                    top.append(k_)
                if len(top) >= 25:
                    break
        cells = [key2smi[k_] for k_ in top[:25] if k_ in key2smi]
        rows.append([str(mid), ";".join(cells)])
        print(f"{mid} spectra={len(gidx)} guesses={len(cells)}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"n": len(rows), "rows": rows}, f)
    print(json.dumps({"n": len(rows), "csv": bool(a.csv)}))
    if a.csv:
        errs = M.validate_submission(rows)
        print(json.dumps({"csv_violations": errs[:5], "n": len(rows)}))
        if errs:
            raise SystemExit(f"submission invalid: {errs[:3]}")
        with open(a.csv, "w") as f:
            f.write("molecule_id,smiles\n")
            for mid, cell in rows:
                f.write(f"{mid},{cell}\n")


if __name__ == "__main__":
    main()
