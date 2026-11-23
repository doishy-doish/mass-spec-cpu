"""run.py — thin launcher for the single Rust retrieval engine (sengine).

Python owns: parquet load, mask splits, pool shortlists (numpy bisect),
snake-draft sharding, output assembly, heartbeats, submission schema.
ALL compute (clean/channels/fusion/rank/neutrals) lives in rust-bitvec.
There is no Python backup of any channel math (user order: one code).

Usage:
  python -m mscpu.run --data data/train.parquet --mask class2 --n 200 --out results/class2.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as CF
import json
import os
import signal
import subprocess
import threading
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from . import metric as M
from . import validation as V

VARIANTS = ["z", "lib", "analog", "frag", "rrf", "cascade", "bin", "rrfb", "binw", "router", "stein", "mass", "zw", "rrfw"]

# BET-EVOLVE genome defaults (replication queue #3): neutral = legacy behavior
# bit-identically (wl/wa/wf/rl/ra/rf = 1.0, casc_z = 3.0, rout_* = legacy gates).
# The canary fits these on class2 n=50 (never class3); knobs ride --flags below
# and are recorded in every output JSON for traceability (like qpeak/qloss).
FUSION_DEFAULTS = {"wl": 1.0, "wa": 1.0, "wf": 1.0, "rl": 1.0, "ra": 1.0,
                   "rf": 1.0, "casc_z": 3.0, "rout_lib": 0.5,
                   "rout_bin": 0.4, "rout_binw": 0.4}


def load_meta(path: str) -> pd.DataFrame:
    cols = ["inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz",
            "adduct", "instrument_type", "ionization_mode", "ms2_mzs",
            "ms2_normalized_intensities", "ingest_lib"]
    t = pd.read_parquet(path, columns=[c for c in cols])
    t = t.rename(columns={"inchikey14": "key14", "normalized_smiles": "smiles",
                          "ms2_mzs": "mzs", "ms2_normalized_intensities": "ints",
                          "instrument_type": "instrument"})
    return t.reset_index(drop=True)


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
    """Host asked us to die: scream our memory state, then go."""
    rss, avail = _mem_state()
    try:
        print("LOUD-SIGTERM host shutdown rss_mb=" + str(rss) + " memavail_mb=" + str(avail), flush=True)
    finally:
        os._exit(143)


def _mem_watch(stop, interval=20.0):
    while not stop.wait(interval):
        rss, avail = _mem_state()
        print("LOUD-MEM rss_mb=" + str(rss) + " memavail_mb=" + str(avail), flush=True)


class _Engine:
    """Persistent sengine worker: one JSON line per query, ranked lists back."""

    def __init__(self, bin_path, qpeak=0.05, qloss=0.01, fusion=None):
        self.p = subprocess.Popen([bin_path, "sengine"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1,
                                  stderr=subprocess.DEVNULL)
        params = {"qpeak": qpeak, "qloss": qloss}
        params.update(FUSION_DEFAULTS)
        params.update(fusion or {})
        self.p.stdin.write(json.dumps({"params": params}) + "\n")
        self.p.stdin.flush()
        ready = self.p.stdout.readline()
        if "ready" not in ready:
            raise RuntimeError("sengine not ready: " + ready[:200])

    def query(self, qmsg):
        self.p.stdin.write(json.dumps(qmsg) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


def structs_call(bin_path, ktable, kidx, precs, adducts, queries):
    """One structs round trip: structure neutrals + query neutral masses."""
    p = subprocess.run([bin_path, "structs"],
                       input=json.dumps({"ktable": ktable, "kidx": kidx, "precs": precs,
                                         "adducts": adducts, "queries": queries}),
                       capture_output=True, text=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError("structs failed: " + (p.stderr or "")[-500:])
    return json.loads(p.stdout)


def raw_list(v):
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.tolist()]
    return [float(x) for x in v]


def build_structs_payload(meta, queries):
    """Pure builder for one structs round trip (no binary needed, testable)."""
    lib_keys = meta["key14"].tolist()
    ukeys = sorted(set(lib_keys))
    kindex = {k: i for i, k in enumerate(ukeys)}
    sq = [{"qi": int(qi), "prec": float(meta.at[qi, "precursor_mz"]),
           "adduct": str(meta.at[qi, "adduct"])} for qi in queries]
    return (ukeys, [kindex[k] for k in lib_keys],
            [float(x) for x in meta["precursor_mz"].tolist()],
            [str(x) for x in meta["adduct"].tolist()], sq)


def build_engine_query(q, cands, pool_keys, lib_by_key, key2smi, meta, qi,
                       qnm=None, key2neutral=None):
    """Pure builder for one sengine query (no binary needed, testable).

    qnm/key2neutral carry neutral masses for the v8-mass-lite channel
    (Loop-3 replication, standalone). None-safe: missing -> Rust scores 0.0 ties.
    """
    k2n = key2neutral or {}
    cands_msg = []
    for _, r in cands.iterrows():
        k = r["key14"]
        specs = []
        for j in lib_by_key.get(k, []):
            rr = meta.loc[j]
            specs.append({"mzs": raw_list(rr["mzs"]), "ints": raw_list(rr["ints"]),
                          "prec": float(rr["precursor_mz"])})
        cands_msg.append({"key": k, "smiles": key2smi.get(k, ""), "specs": specs,
                          "neutral": k2n.get(k)})
    pool_msg = []
    for k in pool_keys:
        j0 = lib_by_key.get(k, [None])[0]
        if j0 is None:
            continue
        rr = meta.loc[j0]
        pool_msg.append({"key": k, "smiles": key2smi.get(k, ""),
                         "mzs": raw_list(rr["mzs"]), "ints": raw_list(rr["ints"]),
                         "prec": float(rr["precursor_mz"])})
    return {"q": {"mzs": raw_list(q["mzs"]), "ints": raw_list(q["ints"]),
                  "prec": float(q["precursor_mz"])},
            "cands": cands_msg, "pool": pool_msg, "qi": int(qi),
            "qnm": qnm}


_SE = {}
_SE_META = None


def _worker():
    """One engine worker per process (forks never share the parent's pipes)."""
    if _SE.get("worker") is None or _SE.get("worker_pid") != os.getpid():
        _SE["worker"] = _Engine(_SE["rust_bin"], _SE["qpeak"], _SE["qloss"],
                               _SE.get("fusion"))
        _SE["worker_pid"] = os.getpid()
    return _SE["worker"]


def _run_query(task):
    """One query end-to-end. Reads only _SE + engine worker; never raises."""
    qpos, qi, total = task
    V_ = _SE["variants"]
    q = _SE["meta"].loc[qi]
    qnm = _SE["qnm"].get(int(qi))
    if qnm is None:
        return None
    tol = qnm * _SE["ppm"] * 1e-6
    lo = np.searchsorted(_SE["masses"], qnm - tol)
    hi = np.searchsorted(_SE["masses"], qnm + tol, side="right")
    cands = _SE["struct_m"].iloc[lo:hi]
    if not len(cands):
        return None
    cand_keys = set(cands["key14"])
    # analog pool: precursor +-50 Da library window, excluding candidates,
    # nearest-first (mirrors the old bisect + dist sort; engine funnels it)
    precs = _SE["lib_precs"]
    pkeys = _SE["lib_pkeys"]
    prec_q = float(q["precursor_mz"])
    plo = bisect_left(precs, prec_q - 50.0)
    phi = bisect_right(precs, prec_q + 50.0, lo=plo)
    near = []
    for idx in range(plo, phi):
        k = pkeys[idx]
        if k in cand_keys:
            continue
        near.append((abs(precs[idx] - prec_q), k))
    near.sort(key=lambda t: t[0])
    pool_keys = [k for _, k in near[:1500]]
    qmsg = build_engine_query(q, cands, pool_keys, _SE["lib_by_key"],
                              _SE["key2smi"], _SE["meta"], qi,
                              qnm, _SE.get("key2neutral"))
    try:
        out = _worker().query(qmsg)
    except Exception as e:
        print("sengine failed on q=" + str(qi) + " (" + str(e)[:120] + "), skipping", flush=True)
        return None
    truth = q["key14"]
    row = {}
    ranked_z = None
    for v in V_:
        ranked = out.get("ranked", {}).get(v, [])
        row[v] = M.reciprocal_rank(truth, ranked)
        if v == "z":
            ranked_z = ranked
    line = ("shard " + str(_SE["shard"]) + " " + str(qpos + 1) + "/" + str(total) + " ("
            + str(int(100 * (qpos + 1) / max(1, total))) + "%) q=" + str(qi) + " pool="
            + str(len(cands)) + " " + " ".join(v + "=" + ("%.2f" % row[v]) for v in V_))
    print(line, flush=True)  # HEARTBEAT: workers print as they finish
    # GOOD-PROBE: surface the v8 rank-1-by-mass swap signal (was discarded —
    # old probes could never see the mass channel they were sent to check).
    return (qi, row, len(cands), ranked_z, line, bool(out.get("mass_swap", False)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mask", choices=["class2", "class3", "cross"], default="class2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--rust-bin", default="rust-bitvec/target/release/binvec")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--qpeak", type=float, default=0.05)
    ap.add_argument("--qloss", type=float, default=0.01)
    ap.add_argument("--wl", type=float, default=1.0)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--wf", type=float, default=1.0)
    ap.add_argument("--rl", type=float, default=1.0)
    ap.add_argument("--ra", type=float, default=1.0)
    ap.add_argument("--rf", type=float, default=1.0)
    ap.add_argument("--casc-z", type=float, default=3.0)
    ap.add_argument("--rout-lib", type=float, default=0.5)
    ap.add_argument("--rout-bin", type=float, default=0.4)
    ap.add_argument("--rout-binw", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-suffix", default="",
                    help="inserted before the extension of --out, e.g. '-binwprobe'; "
                         "probes must use one so they never clobber fleet shard files")
    a = ap.parse_args()
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
    pool = V.query_pool(meta).index.tolist()
    counts = meta["key14"].value_counts()
    if a.mask == "class2":
        library, queries, _ = V.split_class2(meta, pool, counts, a.n, a.seed)
    elif a.mask == "class3":
        library, queries, _ = V.split_class3(meta, pool, a.n, a.seed)
    else:
        library, queries, _ = V.split_cross_instrument(meta, pool, a.n, a.seed)
    # one structs round trip: neutrals for the pool + qnm per query
    ukeys, kidx, precs, adducts, sq = build_structs_payload(meta, queries)
    st = structs_call(a.rust_bin, ukeys, kidx, precs, adducts, sq)
    neutrals = st["neutrals"]
    qnm = {int(qi): v for qi, v in st["qnm"] if v is not None}
    struct_m = pd.DataFrame([{"key14": k, "neutral": neutrals[k]} for k in neutrals])
    struct_m = struct_m.sort_values("neutral").reset_index(drop=True)
    masses = struct_m["neutral"].to_numpy()

    lib_by_key: dict = {}
    for idx in library:
        lib_by_key.setdefault(meta.at[idx, "key14"], []).append(int(idx))
    key2smi: dict = {}
    for k, g in meta.groupby("key14"):
        key2smi.setdefault(k, str(g["smiles"].iloc[0]))

    # SNAKE-DRAFT: balance shards by ppm pool size (slowest shard = fleet wall)
    if a.shards > 1 and len(queries) > 1:
        qcosts = []
        for qi in queries:
            qn = qnm.get(int(qi))
            if qn is None:
                qcosts.append(0)
                continue
            tol = qn * a.ppm * 1e-6
            lo = np.searchsorted(masses, qn - tol)
            hi = np.searchsorted(masses, qn + tol, side="right")
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

    # analog-pool lookup array: precursor-sorted first-spectrum per library key
    _prec_pairs = []
    for k, js in lib_by_key.items():
        j0 = js[0]
        _prec_pairs.append((float(meta.at[j0, "precursor_mz"]), k))
    _prec_pairs.sort(key=lambda t: t[0])
    lib_precs = [p for p, _ in _prec_pairs]
    lib_pkeys = [k for _, k in _prec_pairs]

    key2neutral = dict(zip(struct_m["key14"].tolist(), struct_m["neutral"].tolist()))
    fusion = {"wl": a.wl, "wa": a.wa, "wf": a.wf, "rl": a.rl, "ra": a.ra,
              "rf": a.rf, "casc_z": a.casc_z, "rout_lib": a.rout_lib,
              "rout_bin": a.rout_bin, "rout_binw": a.rout_binw}
    _SE.update(meta=meta, struct_m=struct_m, masses=masses, key2neutral=key2neutral,
               fusion=fusion,
               lib_by_key=lib_by_key,
               lib_precs=lib_precs, lib_pkeys=lib_pkeys,
               key2smi=key2smi, qnm=qnm, ppm=a.ppm,
               shard=a.shard, variants=VARIANTS, rust_bin=a.rust_bin,
               qpeak=a.qpeak, qloss=a.qloss)
    _rss, _avail = _mem_state()
    print("precompute done lib_keys=" + str(len(lib_by_key)) + " queries=" + str(len(queries))
          + " jobs=" + str(a.jobs) + " rss_mb=" + str(_rss) + " memavail_mb=" + str(_avail),
          flush=True)  # HEARTBEAT
    tasks = [(i, int(qi), len(queries)) for i, qi in enumerate(queries)]
    results = []
    if a.jobs > 1 and len(tasks) > 1:
        with CF.ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for res in ex.map(_run_query, tasks):
                if res is None:
                    continue
                results.append(res)
    else:
        for t in tasks:
            res = _run_query(t)
            if res is None:
                continue
            results.append(res)

    preds: dict = {}
    rrs: dict = {v: [] for v in VARIANTS}
    pools: list = []
    qidx: list = []
    swaps: list = []
    for qi, row, plen, ranked, line, swap in results:
        pools.append(plen)
        for v in VARIANTS:
            rrs[v].append(row[v])
        if ranked is not None:
            preds[str(qi)] = ranked
        qidx.append(int(qi))
        swaps.append(1.0 if swap else 0.0)

    def _avg(x):
        return float(np.mean(x)) if x else 0.0
    out = {"mask": a.mask, "engine": "rust", "shard": a.shard, "shards": a.shards, "n": len(rrs["z"]), "mrr": _avg(rrs["z"]),
           "mrr_lib": _avg(rrs["lib"]), "mrr_analog": _avg(rrs["analog"]),
           "mrr_frag": _avg(rrs["frag"]), "mrr_rrf": _avg(rrs["rrf"]),
           "mrr_cascade": _avg(rrs["cascade"]),
           "mrr_bin": _avg(rrs["bin"]), "mrr_rrfb": _avg(rrs["rrfb"]),
           "mrr_binw": _avg(rrs["binw"]), "mrr_router": _avg(rrs["router"]), "mrr_stein": _avg(rrs["stein"]),
            "mrr_mass": _avg(rrs["mass"]),
            "mrr_zw": _avg(rrs["zw"]), "mrr_rrfw": _avg(rrs["rrfw"]),
            "fusion": fusion,
           "qpeak": a.qpeak, "qloss": a.qloss,
           "mass_swap_rate": _avg(swaps),
           "hit1": _avg([r == 1.0 for r in rrs["z"]]),
           "hit25": _avg([r > 0 for r in rrs["z"]]),
           "hit1_cascade": _avg([r == 1.0 for r in rrs["cascade"]]),
           "hit25_cascade": _avg([r > 0 for r in rrs["cascade"]]),
           "rrs": {v: [round(float(x), 6) for x in rrs[v]] for v in VARIANTS},
           "pools": pools, "qidx": qidx, "mass_swaps": [int(s) for s in swaps]}
    with open(out_path, "w") as f:
        json.dump(out, f)
    try:
        if _SE.get("worker") is not None:
            _SE["worker"].close()
    except Exception:
        pass
    try:
        _stop.set()
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
