"""predict.py — test-set inference for the kernels-only submission.
Same evidence channels as run.py, but: library = ALL train spectra (no mask),
queries = test.parquet grouped by molecule_id, per-molecule aggregation by sum
of reciprocal rank across the molecule's spectra. Output results/predict.json
{rows: [[molecule_id, "smi;..."]], n} — scrambled on the runner; the watchdog
decrypts it into submission.csv (<=25 SMILES per molecule).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import spectra as S
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


def rrf_fuse(scored, channels=("lib", "analog", "frag"), base=60.0):
    """Reciprocal-rank fusion over per-channel rankings (robust across masks:
    z-sum let dead channels outvote live ones on class2/class3)."""
    fused = pd.Series(0.0, index=scored.index)
    for ch in channels:
        order = scored[ch].rank(ascending=False, method="min")
        fused = fused + 1.0 / (base + order)
    return fused


def rank_spectrum(q, cands, lib_by_key, fp_cache) -> list:
    qpeaks = S.clean(q["ms2_mzs"], q["ms2_normalized_intensities"], q["precursor_mz"])
    cand_keys = set(cands["key14"])
    pool = []
    for key, spec_list in lib_by_key.items():
        if key in cand_keys:
            continue
        for mzs, ints, prec in spec_list[:1]:
            if abs(prec - q["precursor_mz"]) <= 50:
                pool.append((key, S.clean(mzs, ints, prec), prec,
                             abs(prec - q["precursor_mz"])))
                break
    pool.sort(key=lambda t: t[3])
    pool = [(k_, p, pr) for k_, p, pr, _ in pool[:1500]]
    scored = R.score_query(qpeaks, q["precursor_mz"], cands, lib_by_key, pool)
    scored["score"] = rrf_fuse(scored)
    return scored.sort_values("score", ascending=False)["key14"].tolist()[:25]


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None,
                    help="also/only write submission.csv (kernel mode)")
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--wide", type=float, default=80.0)
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
    struct = R.structure_table(train)
    R.FP_CACHE = {r["key14"]: r["fp"] for _, r in struct.iterrows()
                  if r["fp"] is not None}
    key2smi: dict = {}
    for _, r in struct.iterrows():
        key2smi.setdefault(r["key14"], r["smiles"])
    lib_by_key: dict = {}
    for _, r in train.iterrows():
        lib_by_key.setdefault(r["key14"], []).append(
            (r["mzs"], r["ints"], r["precursor_mz"]))
    struct_m = struct.dropna(subset=["neutral"]).sort_values("neutral")
    masses = struct_m["neutral"].to_numpy()

    def pool(qnm: float, ppm: float):
        tol = qnm * ppm * 1e-6
        lo = np.searchsorted(masses, qnm - tol)
        hi = np.searchsorted(masses, qnm + tol, side="right")
        return struct_m.iloc[lo:hi]

    test = pd.read_parquet(a.test)
    rows = []
    for mid, g in test.groupby("molecule_id"):
        narrow, wide_lists = [], []
        for _, q in g.iterrows():
            qnm = S.neutral_mass(q["precursor_mz"], q["adduct"])
            if not qnm:
                continue
            cn = pool(qnm, a.ppm)
            if len(cn):
                narrow.append(rank_spectrum(q, cn, lib_by_key, R.FP_CACHE))
            cw = pool(qnm, a.wide)
            if len(cw):
                wide_lists.append(rank_spectrum(q, cw, lib_by_key, R.FP_CACHE))
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
        from . import metric as M
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
