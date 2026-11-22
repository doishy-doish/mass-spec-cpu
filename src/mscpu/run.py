"""run.py — CPU retrieval baseline + three-mask validation.
Evidence channels (all CPU): library max-cosine, analog (mass-shift cosine ->
Tanimoto to candidate), neutral-loss matches. Fusion: z-score sum (no training). LightGBM ranker: future work, labels available from masks.

Usage:
  python -m mscpu.run --data data/train.parquet --mask class2 --n 200 --out results/class2.json
"""
from __future__ import annotations

import argparse
import bisect
import concurrent.futures as CF
import json
import os
import signal
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from . import binary as B
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


def score_query(qpeaks, qprec, cands: pd.DataFrame, lib_clean: dict,
                analog_pool: list, lib_bits_by_key: dict | None = None,
                qbits: int = 0, bin_override: dict | None = None,
                lib_wbits_by_key: dict | None = None,
                qwbits: dict | None = None, qloss_pre: set | None = None,
                nq: float | None = None, lib_norms: dict | None = None,
                binw_override: dict | None = None) -> pd.DataFrame:
    """FURNACE (bench-gated, one knob): two-pointer cosine with precomputed norms,
    pool entries (key14, peaks, precursor[, norm]) — 4-tuples from run.py, 3-tuples
    from predict.py (norm computed inline). qloss hoisted (one set per query);
    per-candidate loss sets stay inline (library-size eager cache blew RAM — see M174)."""
    if nq is None:
        nq = float((qpeaks[:, 1] ** 2).sum()) if len(qpeaks) else 0.0
    qloss = qloss_pre
    if qloss is None:
        qloss = set(round(float(qprec - mz), 2) for mz, _ in qpeaks[:32])
    # analog matches once per query: top analog structures by mass-shift cosine
    analog_hits = []
    for entry in analog_pool:
        if len(entry) == 4:
            okey, opeaks, oprec, onorm = entry
        else:
            okey, opeaks, oprec = entry
            onorm = None
        ms = S.modified_cosine(qpeaks, opeaks, qprec - oprec, na=nq, nb=onorm)
        if ms > 0.3:
            ofp = FP_CACHE.get(okey)
            if ofp is not None:
                analog_hits.append((ofp, ms))
    analog_hits.sort(key=lambda t: -t[1])
    analog_hits = analog_hits[:10]
    rows = []
    for c in cands.itertuples():
        ckey = c.key14
        cl_list = lib_clean.get(ckey, [])
        # library channel: max cosine vs structure's own spectra
        lib = 0.0
        norms = (lib_norms or {}).get(ckey) if lib_norms else None
        for idx, (cl, prec) in enumerate(cl_list):
            nn = norms[idx] if norms is not None and idx < len(norms) else None
            s = S.cosine(qpeaks, cl, na=nq, nb=nn)
            if s > lib:
                lib = s
        # analog channel: best mass-shift match x Tanimoto to candidate
        analog = 0.0
        cfp = c.fp
        if cfp is not None:
            for ofp, ms in analog_hits:
                v = ms * fp.tanimoto_fp(ofp, cfp)
                if v > analog:
                    analog = v
        # fragmentation proxy: shared neutral losses (precursor - fragment)
        # NOTE: built inline per candidate — the eager per-library cache cost ~0.5GB (M174)
        fl = [set(round(float(prec - mz), 2) for mz, _ in cl[:32]) if len(cl) else set()
              for cl, prec in cl_list[:2]]
        frag = 0.0
        for floss in fl[:2]:
            if floss:
                v = len(qloss & floss) / max(1, len(qloss))
                if v > frag:
                    frag = v
        # binary channel: Rust inference when the driver passes scores, else local Jaccard
        bscore = 0.0
        if bin_override is not None:
            bscore = float(bin_override.get(ckey, 0.0))
        elif qbits and lib_bits_by_key:
            for lb in lib_bits_by_key.get(ckey, []):
                j = B.jaccard(qbits, lb)
                if j > bscore:
                    bscore = j
        # weighted-binary channel: Ruzicka Jaccard, intensity restored (BIN-WEIGHTS)
        bwscore = 0.0
        if binw_override is not None:
            bwscore = float(binw_override.get(ckey, 0.0))
        elif qwbits and lib_wbits_by_key:
            for wb in lib_wbits_by_key.get(ckey, []):
                j = B.wjaccard(qwbits, wb)
                if j > bwscore:
                    bwscore = j
        rows.append({"key14": ckey, "lib": lib, "analog": analog, "frag": frag, "bin": bscore, "binw": bwscore})
    return pd.DataFrame(rows)


def rust_bin_scores(rust_bin, qbins, ref_bins) -> dict | None:
    """Rust inference: one process call per query. ref_bins = [(tag, p, l)].
    Returns tag->score, or None on any failure (caller falls back to Python)."""
    try:
        payload = {"q": {"p": qbins[0], "l": qbins[1]},
                   "refs": [{"key": t, "p": p, "l": l} for t, p, l in ref_bins]}
        pr = subprocess.run([rust_bin], input=json.dumps(payload), capture_output=True,
                            text=True, timeout=600)
        if pr.returncode != 0:
            return None
        data = json.loads(pr.stdout)
        best: dict = {}
        for tag, s in data.get("scores", []):
            key = tag.rsplit("#", 1)[0]
            if s > best.get(key, 0.0):
                best[key] = float(s)
        return best
    except Exception:
        return None


class _RustWorker:
    """Persistent binvec worker: preload once, one round trip per query.
    Serves bin + binw; any failure -> exception -> caller falls back to python."""
    def __init__(self, bin_path, lib_payload):
        self.p = subprocess.Popen([bin_path, "worker"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.p.stdin.write(json.dumps({"lib": lib_payload}) + "\n")
        self.p.stdin.flush()
        ready = self.p.stdout.readline()
        if "ready" not in ready:
            raise RuntimeError("binvec worker not ready: " + ready[:200])
    def query(self, q, cands):
        self.p.stdin.write(json.dumps({"q": q, "cands": sorted(cands)}) + "\n")
        self.p.stdin.flush()
        out = json.loads(self.p.stdout.readline())
        return ({k: float(v) for k, v in out["scores"]},
                {k: float(v) for k, v in out["wscores"]})
    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


FP_CACHE: dict = {}

_G = {}


def _mem_state():
    """(rss_mb, memavail_mb) or (None, None). Linux runners only; never raises."""
    try:
        rss = avail = None
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) // 1024
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                    break
        return rss, avail
    except Exception:
        return None, None


def _sigterm_loud(signum, frame):
    """Host asked us to die: scream our memory state, then go. Distinguishes
    infra reclaim (this line present, GBs free) from pressure (absent/ramp-to-zero)."""
    rss, avail = _mem_state()
    try:
        print("LOUD-SIGTERM host shutdown rss_mb=" + str(rss) + " memavail_mb=" + str(avail), flush=True)
    finally:
        os._exit(143)


def _mem_watch(stop, interval=20.0):
    while not stop.wait(interval):
        rss, avail = _mem_state()
        print("LOUD-MEM rss_mb=" + str(rss) + " memavail_mb=" + str(avail), flush=True)


class _LazyLayer(dict):
    """dict interface, values built on demand from lib_clean. Bounded cache.

    MEMORY FIX: full-corpus derived layers (bits/bins/wbits over ~2.5M spectra)
    were GBs of dicts touched per-query only for candidate keys (~thousands).
    Same B.encode* on same (cl, prec) => numeric-identical, RAM on demand.
    score_query/predict/tests all use .get(ckey, default) — covered here.
    Real dicts still work everywhere (predict/tests pass their own)."""

    def __init__(self, build, cap=400000):
        dict.__init__(self)
        self._build = build
        self._cap = cap

    def get(self, k, default=None):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            pass
        if len(self) >= self._cap:
            self.clear()
        v = self._build(k)
        dict.__setitem__(self, k, v)
        return v


def _build_bits(ckey):
    return [B.encode(cl, prec) for cl, prec in _G["lib_clean"].get(ckey, [])]


def _build_bins(ckey):
    return [B.encode_bins(cl, prec, _G["qpeak"], _G["qloss"]) for cl, prec in _G["lib_clean"].get(ckey, [])]


def _build_wbits(ckey):
    return [B.encode_weighted(cl, prec, _G["qpeak"], _G["qloss"]) for cl, prec in _G["lib_clean"].get(ckey, [])]


def _run_query(task):
    """One query end-to-end. Fork-worker safe: reads only _G + module globals.
    Returns (qi, {variant: rr}, pool_len, ranked_z_or_None, logline), or None if skipped."""
    qpos, qi, total = task
    V = _G["variants"]
    q = _G["meta"].loc[qi]
    qnm = S.neutral_mass(q["precursor_mz"], q["adduct"])
    if not qnm:
        return None
    qmzs, qints = _G["qspec"][qi]
    qpeaks = S.clean(qmzs, qints, q["precursor_mz"])
    qbits = B.encode(qpeaks, q["precursor_mz"])
    tol = qnm * _G["ppm"] * 1e-6
    lo = np.searchsorted(_G["masses"], qnm - tol)
    hi = np.searchsorted(_G["masses"], qnm + tol, side="right")
    cands = _G["struct_m"].iloc[lo:hi]
    cand_keys = set(cands["key14"])
    bin_override = None
    binw_override = None
    if _G["engine"] == "rust":
        try:
            p, pw, l, lw = B.encode_qlib(qpeaks, q["precursor_mz"], _G["qpeak"], _G["qloss"])
            bin_override, binw_override = _G["rworker"].query(
                {"p": p, "pw": pw, "l": l, "lw": lw}, cand_keys)
        except Exception:
            print("rust engine failed on q=" + str(qi) + ", falling back to python", flush=True)
    if bin_override is None:
        qbins = B.encode_bins(qpeaks, q["precursor_mz"], _G["qpeak"], _G["qloss"])
    else:
        qbins = 0
    if binw_override is None:
        qwbits = B.encode_weighted(qpeaks, q["precursor_mz"], _G["qpeak"], _G["qloss"])
    else:
        qwbits = None
    nq = float((qpeaks[:, 1] ** 2).sum()) if len(qpeaks) else 0.0
    qloss = set(round(float(q["precursor_mz"] - mz), 2) for mz, _ in qpeaks[:32])
    # analog pool: bisect on precomputed precursor-sorted library array
    # (was: full dict scan + Python abs() per library key, per query)
    precs = _G["lib_precs"]
    pkeys = _G["lib_pkeys"]
    plo = bisect.bisect_left(precs, q["precursor_mz"] - 50.0)
    phi = bisect.bisect_right(precs, q["precursor_mz"] + 50.0, lo=plo)
    nearby = []
    for idx in range(plo, phi):
        key = pkeys[idx]
        if key in cand_keys:
            continue
        cl, prec = _G["lib_clean"][key][0]
        nearby.append((abs(prec - q["precursor_mz"]), key, cl, prec))
    nearby.sort(key=lambda t: t[0])
    # FUNNEL (I2-1): bin-prefilter 1500 -> top 300 by Jaccard before any cosine.
    # Jaccard is C popcount; cosine is a Python peak loop. Saves ~1200 cosines/query.
    if len(nearby) > 400 and qbits:
        _lb = _G.get("lib_bits") or {}
        scored_near = []
        for dist, key, cl, prec in nearby:
            bl = _lb.get(key)
            b0 = bl[0] if bl else 0
            try:
                j = B.jaccard(qbits, b0)
            except Exception:
                j = 0.0
            scored_near.append((j, dist, key, cl, prec))
        scored_near.sort(key=lambda t: (-t[0], t[1]))
        nearby = [(d, k, c, p) for _, d, k, c, p in scored_near[:300]]
    analog_pool = [(k, p, pr, _G["lib_norms"][k][0]) for _, k, p, pr in nearby[:1500]]
    scored = score_query(qpeaks, q["precursor_mz"], cands, _G["lib_clean"], analog_pool,
                         _G["lib_bits"], qbits, bin_override, _G["lib_wbits"], qwbits,
                         qloss, nq, _G["lib_norms"], binw_override)
    for ch in ["lib", "analog", "frag"]:
        m, s = scored[ch].mean(), scored[ch].std() + 1e-9
        scored[ch + "z"] = (scored[ch] - m) / s
    scored["z"] = scored[["libz", "analogz", "fragz"]].sum(axis=1)
    scored["rrf"] = M.rrf_fuse(scored)
    scored["rrfb"] = M.rrf_fuse(scored, channels=("lib", "analog", "frag", "bin"))
    lzmax = float(scored["libz"].max()) if len(scored) else 0.0
    scored["cascade"] = scored["libz"] if lzmax > 3.0 else scored["analogz"]
    # ROUTER (I3-1): mask-conditional fusion. Fixed fusions lose somewhere
    # (rrfb helps easy / hurts hard). Overlap regime -> cascade, novel -> analog.
    # Signal: per-query max lib/bin/binw (easy ~0.9, hard ~chance). 10 lines, no deps.
    lib_max = float(scored["lib"].max()) if len(scored) else 0.0
    bin_max = float(scored["bin"].max()) if len(scored) else 0.0
    try:
        binw_max = float(scored["binw"].max()) if len(scored) else 0.0
    except Exception:
        binw_max = 0.0
    overlap = (lib_max > 0.5) or (bin_max > 0.4) or (binw_max > 0.4)
    router_col = "cascade" if overlap else "analog"
    truth = q["key14"]
    row = {}
    ranked_z = None
    for v in V:
        if v == "router":
            ranked = scored.sort_values(router_col, ascending=False)["key14"].tolist()[:25]
            row[v] = M.reciprocal_rank(truth, ranked)
            continue
        col = {"lib": "lib", "analog": "analog", "frag": "frag"}.get(v, v)
        ranked = scored.sort_values(col, ascending=False)["key14"].tolist()[:25]
        row[v] = M.reciprocal_rank(truth, ranked)
        if v == "z":
            ranked_z = ranked
    line = ("shard " + str(_G["shard"]) + " " + str(qpos + 1) + "/" + str(total) + " ("
            + str(int(100 * (qpos + 1) / max(1, total))) + "%) q=" + str(qi) + " pool="
            + str(len(cands)) + " " + " ".join(v + "=" + ("%.2f" % row[v]) for v in V))
    print(line, flush=True)  # HEARTBEAT (M180): workers print as they finish; parent used to print only at the very end
    return (qi, row, len(cands), ranked_z, line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mask", choices=["class2", "class3", "cross"], default="class2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--engine", choices=["python", "rust"], default="python")
    ap.add_argument("--rust-bin", default="rust-bitvec/target/release/binvec")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--qpeak", type=float, default=0.05)
    ap.add_argument("--qloss", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-suffix", default="",
                    help="inserted before the extension of --out, e.g. '-binwprobe'; "
                         "probes must use one so they never clobber fleet shard files")
    a = ap.parse_args()
    if a.engine == "rust" and a.jobs != 1:
        print("rust engine owns its worker: forcing jobs=1 (was " + str(a.jobs) + ")", flush=True)
        a.jobs = 1
    signal.signal(signal.SIGTERM, _sigterm_loud)
    _stop = threading.Event()
    threading.Thread(target=_mem_watch, args=(_stop,), daemon=True).start()
    _rss0, _avail0 = _mem_state()
    print("LOUD-START rss_mb=" + str(_rss0) + " memavail_mb=" + str(_avail0), flush=True)
    out_path = a.out
    if a.out_suffix:
        head, sep, tail = out_path.rpartition("/")
        stem, dot, ext = tail.rpartition(".")
        tail2 = stem + a.out_suffix + dot + ext if dot else tail + a.out_suffix
        out_path = head + sep + tail2 if sep else tail2

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
    lib_clean_by_key: dict = {}
    for idx in library:
        r = meta.loc[idx]
        cl = S.clean(r["mzs"], r["ints"], r["precursor_mz"])
        lib_clean_by_key.setdefault(r["key14"], []).append((cl, r["precursor_mz"]))
    # derived layers (bits/bins/wbits) are LAZY (_LazyLayer, built per touched key):
    # full-corpus dicts were GBs, per-query only candidate keys (~thousands) are read.
    # FURNACE precomputes (library is static): spectrum norms,
    # precursor-sorted key array for bisect pool lookup.
    # (No loss-set cache: ~0.5GB eager, blew the 16GB runner — M174.)
    lib_norms_by_key = {k: [float((cl[:, 1] ** 2).sum()) if len(cl) else 0.0 for cl, _ in v]
                        for k, v in lib_clean_by_key.items()}
    _prec_pairs = sorted((prec, k) for k, v in lib_clean_by_key.items() for cl, prec in v[:1])

    # candidate pool: structures within ppm of query neutral mass (train structures only;
    # competition adds COCONUT — pool grows ~2.6x, noted in contracts)
    struct_m = struct.dropna(subset=["neutral"]).sort_values("neutral")
    masses = struct_m["neutral"].to_numpy()

    # SNAKE-DRAFT (I4-1): balance 5 runners by ppm pool size, not stride.
    # Stride caused 8.3 vs 15.3min spread (slowest shard = fleet wall).
    # Cost proxy = candidate count via masses; snake 0,1,2,3,4,4,3,2,1,0...
    if a.shards > 1 and len(queries) > 1:
        qcosts = []
        for qi in queries:
            r = meta.loc[qi]
            qnm = S.neutral_mass(r["precursor_mz"], r["adduct"])
            if not qnm:
                qcosts.append(0)
                continue
            tol = qnm * a.ppm * 1e-6
            lo = np.searchsorted(masses, qnm - tol)
            hi = np.searchsorted(masses, qnm + tol, side="right")
            qcosts.append(int(hi - lo))
        order = sorted(range(len(queries)), key=lambda i: -qcosts[i])
        buckets: list = [[] for _ in range(a.shards)]
        for rank, idx in enumerate(order):
            rnd = rank // a.shards
            pos = rank % a.shards
            b = pos if (rnd % 2 == 0) else (a.shards - 1 - pos)
            buckets[b].append(queries[idx])
        queries = sorted(buckets[a.shard])
    else:
        queries = queries[a.shard::a.shards]

    # MEMORY FIX: raw peak lists are duplicated into lib_clean (cleaned arrays);
    # stash query spectra, then drop the object columns (GBs) before workers fork.
    qspec = {qi: (meta.at[qi, "mzs"], meta.at[qi, "ints"]) for qi in queries}
    meta = meta.drop(columns=["mzs", "ints"])
    import gc as _gc
    _gc.collect()

    VARIANTS = ["z", "lib", "analog", "frag", "rrf", "cascade", "bin", "rrfb", "binw", "router"]
    preds: dict = {}
    rrs: dict = {v: [] for v in VARIANTS}
    pools: list = []
    _G.update(meta=meta, qspec=qspec, struct_m=struct_m, masses=masses, lib_clean=lib_clean_by_key,
              lib_bits=_LazyLayer(_build_bits), lib_bins=_LazyLayer(_build_bins), lib_wbits=_LazyLayer(_build_wbits),
              lib_norms=lib_norms_by_key,
              lib_precs=[p for p, _ in _prec_pairs], lib_pkeys=[k for _, k in _prec_pairs],
              variants=VARIANTS, engine=a.engine, rust_bin=a.rust_bin, ppm=a.ppm,
              shard=a.shard, qpeak=a.qpeak, qloss=a.qloss)
    _rss, _avail = _mem_state()
    print("precompute done lib_keys=" + str(len(lib_clean_by_key)) + " queries=" + str(len(queries)) + " jobs=" + str(a.jobs) + " rss_mb=" + str(_rss) + " memavail_mb=" + str(_avail), flush=True)  # HEARTBEAT (M180)
    if a.engine == "rust":
        lib_payload = {}
        for k, v in lib_clean_by_key.items():
            specs = []
            for cl, prec in v:
                p, pw, l, lw = B.encode_qlib(cl, prec, a.qpeak, a.qloss)
                specs.append({"p": p, "pw": pw, "l": l, "lw": lw})
            lib_payload[k] = specs
        _G["rworker"] = _RustWorker(a.rust_bin, lib_payload)
        del lib_payload
        _gc.collect()
        print("rust worker ready", flush=True)
    tasks = [(i, qi, len(queries)) for i, qi in enumerate(queries)]
    if a.jobs > 1 and len(tasks) > 1:
        with CF.ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_run_query, tasks))
    else:
        results = [_run_query(t) for t in tasks]
    for res in results:
        if res is None:
            continue
        qi, row, plen, ranked, line = res
        pools.append(plen)
        for v in VARIANTS:
            rrs[v].append(row[v])
        if ranked is not None:
            preds[str(qi)] = ranked

    def _avg(x):
        return float(np.mean(x)) if x else 0.0
    out = {"mask": a.mask, "engine": a.engine, "shard": a.shard, "shards": a.shards, "n": len(rrs["z"]), "mrr": _avg(rrs["z"]),
           "mrr_lib": _avg(rrs["lib"]), "mrr_analog": _avg(rrs["analog"]),
           "mrr_frag": _avg(rrs["frag"]), "mrr_rrf": _avg(rrs["rrf"]),
           "mrr_cascade": _avg(rrs["cascade"]),
           "mrr_bin": _avg(rrs["bin"]), "mrr_rrfb": _avg(rrs["rrfb"]),
           "mrr_binw": _avg(rrs["binw"]), "mrr_router": _avg(rrs["router"]),
           "qpeak": a.qpeak, "qloss": a.qloss,
           "hit1": _avg([r == 1.0 for r in rrs["z"]]),
           "hit25": _avg([r > 0 for r in rrs["z"]]),
           "hit1_cascade": _avg([r == 1.0 for r in rrs["cascade"]]),
           "hit25_cascade": _avg([r > 0 for r in rrs["cascade"]]),
           "rrs": {v: [round(float(x), 6) for x in rrs[v]] for v in VARIANTS},
           "pools": pools, "qidx": [int(q) for q in queries]}
    with open(out_path, "w") as f:
        json.dump(out, f)
    try:
        if _G.get("rworker") is not None:
            _G["rworker"].close()
    except Exception:
        pass
    try:
        _stop.set()
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
