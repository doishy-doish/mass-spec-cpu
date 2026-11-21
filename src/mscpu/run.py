"""run.py — CPU retrieval baseline + three-mask validation.
Evidence channels (all CPU): library max-cosine, analog (mass-shift cosine ->
Tanimoto to candidate), neutral-loss matches. Fusion: z-score sum (no training). LightGBM ranker: future work, labels available from masks.

Usage:
  python -m mscpu.run --data data/train.parquet --mask class2 --n 200 --out results/class2.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import fingerprints as fp
from . import metric as M
from . import spectra as S
from . import validation as V


def load_meta(path: str) -> pd.DataFrame:
    cols = ["inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz",
            "adduct", "instrument_type", "ionization_mode", "ms2_mzs",
            "ms2_normalized_intensities", "ingest_lib"]
    t = pd.read_parquet(path, columns=[c for c in cols])
    t = t.rename(columns={"inchikey14": "key14", "normalized_smiles": "smiles",
                          "ms2_mzs": "mzs", "ms2_normalized_intensities": "ints"})
    return t.reset_index(drop=True)


def structure_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per key14: fingerprint, neutral mass (median adduct-derived)."""
    recs = []
    for key, g in meta.groupby("key14"):
        smi = g["smiles"].iloc[0]
        ests = [S.neutral_mass(p, a) for p, a in zip(g["precursor_mz"], g["adduct"])]
        ests = [e for e in ests if e and 50 < e < 3000]
        nm = float(np.median(ests)) if ests else None
        recs.append({"key14": key, "smiles": smi, "neutral": nm,
                     "fp": fp.morgan(smi), "n_spec": len(g)})
    return pd.DataFrame(recs)


def score_query(qpeaks, qprec, cands: pd.DataFrame, lib_by_key: dict,
                analog_pool: list) -> pd.DataFrame:
    """analog_pool: [(key14, cleaned_peaks, precursor)] once per query (mass-windowed)."""
    # analog matches once per query: top analog structures by mass-shift cosine
    analog_hits = []
    for okey, opeaks, oprec in analog_pool:
        ms = S.modified_cosine(qpeaks, opeaks, qprec - oprec)
        if ms > 0.3:
            ofp = FP_CACHE.get(okey)
            if ofp is not None:
                analog_hits.append((ofp, ms))
    analog_hits.sort(key=lambda t: -t[1])
    analog_hits = analog_hits[:10]
    rows = []
    for _, c in cands.iterrows():
        # library channel: max cosine vs structure's own spectra
        lib = 0.0
        for mzs, ints, prec in lib_by_key.get(c["key14"], []):
            lib = max(lib, S.cosine(qpeaks, S.clean(mzs, ints, prec)))
        # analog channel: best mass-shift match x Tanimoto to candidate
        analog = 0.0
        cfp = c["fp"]
        if cfp is not None:
            for ofp, ms in analog_hits:
                analog = max(analog, ms * fp.tanimoto_fp(ofp, cfp))
        # fragmentation proxy: shared neutral losses (precursor - fragment)
        qloss = set(round(float(qprec - mz), 2) for mz, _ in qpeaks[:32])
        frag = 0.0
        for mzs, ints, prec in lib_by_key.get(c["key14"], [])[:2]:
            cl = S.clean(mzs, ints, prec)
            if len(cl):
                floss = set(round(float(prec - mz), 2) for mz, _ in cl[:32])
                frag = max(frag, len(qloss & floss) / max(1, len(qloss)))
        rows.append({"key14": c["key14"], "lib": lib, "analog": analog, "frag": frag})
    return pd.DataFrame(rows)


FP_CACHE: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mask", choices=["class2", "class3", "cross"], default="class2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = load_meta(a.data)
    struct = structure_table(meta)
    global FP_CACHE
    FP_CACHE = {r["key14"]: r["fp"] for _, r in struct.iterrows() if r["fp"] is not None}
    pool = V.query_pool(meta).index.tolist()
    counts = meta["key14"].value_counts()
    if a.mask == "class2":
        library, queries, _ = V.split_class2(meta, pool, counts, a.n, a.seed)
    elif a.mask == "class3":
        library, queries, _ = V.split_class3(meta, pool, a.n, a.seed)
    else:
        library, queries, _ = V.split_cross_instrument(meta, pool, a.n, a.seed)
    lib_set = set(library)
    lib_by_key: dict = {}
    for idx in library:
        r = meta.loc[idx]
        lib_by_key.setdefault(r["key14"], []).append((r["mzs"], r["ints"], r["precursor_mz"]))

    # candidate pool: structures within ppm of query neutral mass (train structures only;
    # competition adds COCONUT — pool grows ~2.6x, noted in contracts)
    struct_m = struct.dropna(subset=["neutral"]).sort_values("neutral")
    masses = struct_m["neutral"].to_numpy()

    preds, rrs = {}, []
    for qi in queries:
        q = meta.loc[qi]
        qnm = S.neutral_mass(q["precursor_mz"], q["adduct"])
        if not qnm:
            continue
        qpeaks = S.clean(q["mzs"], q["ints"], q["precursor_mz"])
        tol = qnm * a.ppm * 1e-6
        lo = np.searchsorted(masses, qnm - tol)
        hi = np.searchsorted(masses, qnm + tol, side="right")
        cands = struct_m.iloc[lo:hi]
        # analog pool: library structures within +-50 Da, capped, cleaned once
        cand_keys = set(cands["key14"])
        analog_pool = []
        for key, spec_list in lib_by_key.items():
            if key in cand_keys:
                continue
            for mzs, ints, prec in spec_list[:1]:
                if abs(prec - q["precursor_mz"]) <= 50:
                    analog_pool.append((key, S.clean(mzs, ints, prec), prec))
                    break
            if len(analog_pool) >= 1500:
                break
        scored = score_query(qpeaks, q["precursor_mz"], cands, lib_by_key, analog_pool)
        for ch in ["lib", "analog", "frag"]:
            m, s = scored[ch].mean(), scored[ch].std() + 1e-9
            scored[ch + "z"] = (scored[ch] - m) / s
        scored["score"] = scored[["libz", "analogz", "fragz"]].sum(axis=1)
        ranked = scored.sort_values("score", ascending=False)["key14"].tolist()[:25]
        truth = q["key14"]
        preds[str(qi)] = ranked
        rrs.append(M.reciprocal_rank(truth, ranked))
        print(f"q={qi} pool={len(cands)} rr={rrs[-1]:.3f}", flush=True)

    out = {"mask": a.mask, "n": len(rrs), "mrr": float(np.mean(rrs)) if rrs else 0.0,
           "hit1": float(np.mean([r == 1.0 for r in rrs])) if rrs else 0.0,
           "hit25": float(np.mean([r > 0 for r in rrs])) if rrs else 0.0}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
