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
import pandas as pd

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


def rank_spectrum(q, cands, pool_keys, lib_by_key, key2smi, engine,
                  qnm=None, key2neutral=None):
    """One test spectrum -> top-25 keys by the engine's rrf fusion."""
    out = engine.query(R.build_engine_query(q, cands, pool_keys, lib_by_key,
                                            key2smi, R._SE_META, -1,
                                            qnm, key2neutral))
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

    train = R.load_meta(a.train)
    R._SE_META = train
    ukeys = sorted(set(train["key14"].tolist()))
    kindex = {k: i for i, k in enumerate(ukeys)}
    st = R.structs_call(a.rust_bin, ukeys,
                        [kindex[k] for k in train["key14"].tolist()],
                        [float(x) for x in train["precursor_mz"].tolist()],
                        [str(x) for x in train["adduct"].tolist()], [])
    struct_m = pd.DataFrame([{"key14": k, "neutral": st["neutrals"][k]}
                             for k in st["neutrals"]]).sort_values("neutral").reset_index(drop=True)
    masses = struct_m["neutral"].to_numpy()
    lib_by_key: dict = {}
    for idx in train.index:
        lib_by_key.setdefault(train.at[idx, "key14"], []).append(int(idx))
    _pairs = []
    for k, js in lib_by_key.items():
        _pairs.append((float(train.at[js[0], "precursor_mz"]), k))
    _pairs.sort(key=lambda t: t[0])
    lib_precs = [p for p, _ in _pairs]
    lib_pkeys = [k for _, k in _pairs]
    key2smi: dict = {}
    for k, g in train.groupby("key14"):
        key2smi.setdefault(k, str(g["smiles"].iloc[0]))
    key2neutral = dict(zip(struct_m["key14"].tolist(), struct_m["neutral"].tolist()))
    engine = R._Engine(a.rust_bin, a.qpeak, a.qloss)

    def pool(qnm: float, ppm: float):
        tol = qnm * ppm * 1e-6
        lo = np.searchsorted(masses, qnm - tol)
        hi = np.searchsorted(masses, qnm + tol, side="right")
        return struct_m.iloc[lo:hi]

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

    test = pd.read_parquet(a.test)
    test = test.rename(columns={"ms2_mzs": "mzs",
                                "ms2_normalized_intensities": "ints"})
    rows = []
    for mid, g in test.groupby("molecule_id"):
        # one structs round trip for the molecule's neutral masses
        sq = [{"qi": int(i), "prec": float(r["precursor_mz"]), "adduct": str(r["adduct"])}
              for i, r in enumerate(g.itertuples())]
        st2 = R.structs_call(a.rust_bin, [], [], [], [], sq)
        qnm_map = {int(qi): v for qi, v in st2["qnm"] if v is not None}
        narrow, wide_lists = [], []
        for pos, (_, q) in enumerate(g.iterrows()):
            qnm = qnm_map.get(pos)
            if not qnm:
                continue
            cn = pool(qnm, a.ppm)
            if len(cn):
                narrow.append(rank_spectrum(q, cn, pool50(float(q["precursor_mz"]),
                                            set(cn["key14"])), lib_by_key, key2smi, engine,
                                            qnm, key2neutral))
            cw = pool(qnm, a.wide)
            if len(cw):
                wide_lists.append(rank_spectrum(q, cw, pool50(float(q["precursor_mz"]),
                                               set(cw["key14"])), lib_by_key, key2smi, engine,
                                               qnm, key2neutral))
        top = aggregate(narrow)
        if len(top) < 25:
            for k_ in aggregate(wide_lists):
                if k_ not in top:
                    top.append(k_)
                if len(top) >= 25:
                    break
        cells = [key2smi[k_] for k_ in top[:25] if k_ in key2smi]
        rows.append([str(mid), ";".join(cells)])
        print(f"{mid} spectra={len(g)} guesses={len(cells)}", flush=True)
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
