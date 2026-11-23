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

from . import metric as M
from . import validation as V

VARIANTS = ["z", "lib", "analog", "frag", "rrf", "cascade", "bin", "rrfb", "binw", "router", "stein", "mass", "zw", "rrfw", "pure"]

# BET-EVOLVE genome defaults (replication queue #3): neutral = legacy behavior
# bit-identically (wl/wa/wf/rl/ra/rf = 1.0, casc_z = 3.0, rout_* = legacy gates).
# The canary fits these on class2 n=50 (never class3); knobs ride --flags below
# and are recorded in every output JSON for traceability (like qpeak/qloss).
FUSION_DEFAULTS = {"wl": 1.0, "wa": 1.0, "wf": 1.0, "rl": 1.0, "ra": 1.0,
                   "rf": 1.0, "casc_z": 3.0, "rout_lib": 0.5,
                   "rout_bin": 0.4, "rout_binw": 0.4}


TRAIN_COLS = ["inchikey14", "normalized_smiles", "precursor_mz",
              "adduct", "instrument_type", "ms2_mzs", "ms2_normalized_intensities"]


def load_table(path: str, train: bool = True) -> dict:
    """Parquet -> table dict of numpy arrays (no pandas anywhere in this path).

    Row order == parquet order, so mask splits stay identical. String columns
    are coerced with str() (None-safe); ragged peak lists become float64 arrays
    (None row -> empty, same as the old raw_list behavior).

    Streams in batches (M197): one pq.read_table(2.5M rows) held the GIL for
    tens of seconds with no heartbeat and died 4x at the same age. Per-batch
    prints land between C calls so the next death has a location.
    """
    import pyarrow.parquet as pq
    cols = TRAIN_COLS if train else ["molecule_id", "precursor_mz", "adduct",
                                     "ms2_mzs", "ms2_normalized_intensities"]
    pf = pq.ParquetFile(path, memory_map=True)
    nb = pf.num_row_groups
    key14, smiles, instrument, mid = [], [], [], []
    prec, adduct, mzs, ints = [], [], [], []
    seen = 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=250_000, columns=cols)):
        d = batch.to_pydict()
        n = batch.num_rows
        if train:
            key14.extend(str(x) for x in d["inchikey14"])
            smiles.extend(str(x) for x in d["normalized_smiles"])
            instrument.extend(str(x) for x in d["instrument_type"])
        else:
            mid.extend(str(x) for x in d["molecule_id"])
        prec.extend(float(x) for x in d["precursor_mz"])
        adduct.extend(str(x) for x in d["adduct"])
        for v in d["ms2_mzs"]:
            mzs.append(np.empty(0, dtype=np.float64) if v is None
                       else np.ascontiguousarray(v, dtype=np.float64))
        for v in d["ms2_normalized_intensities"]:
            ints.append(np.empty(0, dtype=np.float64) if v is None
                        else np.ascontiguousarray(v, dtype=np.float64))
        seen += n
        print(f"LOAD batch {bi + 1} rows={seen} (+{n})", flush=True)
    n = seen
    T = {}
    if train:
        T["key14"] = np.array(key14)
        T["smiles"] = smiles
        T["instrument"] = np.array(instrument)
    else:
        T["mid"] = np.array(mid)
    T["prec"] = np.array(prec, dtype=np.float64)
    T["adduct"] = np.array(adduct)
    T["mzs"] = mzs
    T["ints"] = ints
    _ = nb  # row-group count (log only, batching is by row count above)
    assert len(T["prec"]) == n
    return T


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
    except Exception:
        pass
    try:
        if _STATUS_PATH:
            with open(_STATUS_PATH, "a") as sf:
                sf.write("SIGTERM host shutdown\n")
    except Exception:
        pass
    os._exit(143)


def _mem_watch(stop, interval=20.0):
    while not stop.wait(interval):
        rss, avail = _mem_state()
        print("LOUD-MEM rss_mb=" + str(rss) + " memavail_mb=" + str(avail), flush=True)


class _Engine:
    """Persistent sengine worker: preload once, then one JSON line per query."""

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

    def preload(self, chunk_lines):
        """Stream pre-serialized table chunks (protocol v2); no reply yet."""
        for ln in chunk_lines:
            self.p.stdin.write(ln + "\n")
        self.p.stdin.flush()

    def preload_done(self):
        self.p.stdin.write(json.dumps({"preload_done": True}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if "ready" not in line:
            raise RuntimeError("sengine table not ready: " + line[:200])

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


def build_structs_payload(T: dict, queries) -> tuple:
    """Pure builder for one structs round trip (no binary needed, testable)."""
    lib_keys = [str(k) for k in T["key14"].tolist()]
    ukeys = sorted(set(lib_keys))
    kindex = {k: i for i, k in enumerate(ukeys)}
    sq = [{"qi": int(qi), "prec": float(T["prec"][int(qi)]),
           "adduct": str(T["adduct"][int(qi)])} for qi in queries]
    return (ukeys, [kindex[k] for k in lib_keys],
            [float(x) for x in T["prec"].tolist()],
            [str(x) for x in T["adduct"].tolist()], sq)


def build_preload_chunks(lib_by_key: dict, Tlib: dict, key2smi: dict,
                          key2neutral: dict, per: int = 500) -> list:
    """Library table -> pre-serialized JSON chunk lines (protocol v2).

    tolist/J dumps happen ONCE per run here, not per query: per-query
    messages carry keys only. Sorted key order for determinism.
    """
    keys = sorted(lib_by_key)
    out = []
    for i in range(0, len(keys), per):
        ks = keys[i:i + per]
        specs = []
        for k in ks:
            ss = []
            for j in lib_by_key.get(k, []):
                ss.append({"mzs": Tlib["mzs"][int(j)].tolist(),
                           "ints": Tlib["ints"][int(j)].tolist(),
                           "prec": float(Tlib["prec"][int(j)])})
            specs.append(ss)
        nk = []
        for k in ks:
            v = (key2neutral or {}).get(k)
            nk.append(None if (v is None or v != v) else float(v))
        out.append(json.dumps({"preload": {
            "keys": ks,
            "smiles": [str(key2smi.get(k, "")) for k in ks],
            "neutrals": nk, "specs": specs}}))
    return out


def build_engine_query(qrec: dict, cand_keys, pool_keys,
                       qi: int, qnm=None) -> dict:
    """Pure builder for one sengine query, protocol v2 (no binary, testable).

    Keys only — the engine owns the prepared library table from preload.
    Same signature shape as v1 minus the table arguments.
    """
    qm = [float(x) for x in np.ascontiguousarray(qrec["mzs"], dtype=np.float64).tolist()]
    qi_ = [float(x) for x in np.ascontiguousarray(qrec["ints"], dtype=np.float64).tolist()]
    return {"q": {"mzs": qm, "ints": qi_, "prec": float(qrec["prec"])},
            "cand_keys": [str(k) for k in cand_keys],
            "pool_keys": [str(k) for k in pool_keys], "qi": int(qi),
            "qnm": (None if qnm is None else float(qnm))}


def select_task(task):
    """One query's key selection (engine-free; runs in forked workers).

    task = (qpos, qi, total). Returns (qpos, qi, total, cand_keys, pool_keys,
    prec_q, tsel_ms) or None when the query has no neutral mass.
    """
    import time
    t0 = time.perf_counter()
    qpos, qi, total = task
    qi = int(qi)
    qnm = _SE["qnm"].get(qi)
    if qnm is None:
        return None
    tol = qnm * _SE["ppm"] * 1e-6
    lo = np.searchsorted(_SE["masses"], qnm - tol)
    hi = np.searchsorted(_SE["masses"], qnm + tol, side="right")
    if hi <= lo:
        return None
    cand_keys = [str(k) for k in _SE["skeys"][lo:hi].tolist()]
    cand_set = set(cand_keys)
    # analog pool: precursor +-50 Da library window, excluding candidates,
    # nearest-first (mirrors the old bisect + dist sort; engine funnels it)
    precs = _SE["lib_precs"]
    pkeys = _SE["lib_pkeys"]
    prec_q = float(_SE["qprec"].get(qi))
    plo = bisect_left(precs, prec_q - 50.0)
    phi = bisect_right(precs, prec_q + 50.0, lo=plo)
    near = []
    for idx in range(plo, phi):
        k = pkeys[idx]
        if k in cand_set:
            continue
        near.append((abs(precs[idx] - prec_q), k))
    near.sort(key=lambda t: t[0])
    pool_keys = [k for _, k in near[:1500]]
    tsel = (time.perf_counter() - t0) * 1000.0
    return (qpos, qi, total, cand_keys, pool_keys, prec_q, tsel)


def assemble_row(out, truth, qi, row_order):
    """Engine reply + truth -> (row dict, ranked_z, swap). Pure, testable."""
    row = {}
    ranked_z = None
    for v in row_order:
        ranked = out.get("ranked", {}).get(v, [])
        row[v] = M.reciprocal_rank(truth, ranked)
        if v == "z":
            ranked_z = ranked
    return row, ranked_z, bool(out.get("mass_swap", False))


_SE = {}

_STATUS_PATH = None


def score_one(sel, engine):
    """Score one selected query through the shared preloaded engine (parent).

    sel = select_task output. Returns the standard results tuple
    (qi, row, plen, ranked_z, line, swap); never raises.
    """
    qpos, qi, total, cand_keys, pool_keys, prec_q, tsel = sel
    V_ = _SE["variants"]
    qrec = {"mzs": _SE["qmzs"][qi], "ints": _SE["qints"][qi], "prec": prec_q}
    qmsg = build_engine_query(qrec, cand_keys, pool_keys, qi,
                              _SE["qnm"].get(qi))
    try:
        out = engine.query(qmsg)
    except Exception as e:
        print("sengine failed on q=" + str(qi) + " (" + str(e)[:120] + "), skipping", flush=True)
        return None
    truth = str(_SE["qkey"][qi])
    row, ranked_z, swap = assemble_row(out, truth, qi, V_)
    line = ("shard " + str(_SE["shard"]) + " " + str(qpos + 1) + "/" + str(total) + " ("
            + str(int(100 * (qpos + 1) / max(1, total))) + "%) q=" + str(qi) + " pool="
            + str(len(cand_keys)) + " " + " ".join(v + "=" + ("%.2f" % row[v]) for v in V_)
            + " tsel=" + ("%.0f" % tsel)
            + " tm=" + ",".join("%.0f" % t for t in (out.get("timing") or [])))
    print(line, flush=True)  # HEARTBEAT: parent prints as it scores
    # GOOD-PROBE: surface the v8 rank-1-by-mass swap signal (was discarded —
    # old probes could never see the mass channel they were sent to check).
    return (qi, row, len(cand_keys), ranked_z, line, swap)


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
    global _STATUS_PATH
    _STATUS_PATH = out_path + ".status"
    try:
        with open(_STATUS_PATH, "w") as sf:
            sf.write("STARTED mask=" + str(a.mask) + " shard=" + str(a.shard) + "\n")
    except Exception:
        pass

    T = load_table(a.data, train=True)
    pool = V.query_pool(T).tolist()
    if a.mask == "class2":
        library, queries, _ = V.split_class2(T, pool, a.n, a.seed)
    elif a.mask == "class3":
        library, queries, _ = V.split_class3(T, pool, a.n, a.seed)
    else:
        library, queries, _ = V.split_cross_instrument(T, pool, a.n, a.seed)
    # one structs round trip: neutrals for the pool + qnm per query
    ukeys, kidx, precs, adducts, sq = build_structs_payload(T, queries)
    st = structs_call(a.rust_bin, ukeys, kidx, precs, adducts, sq)
    neutrals = st["neutrals"]
    qnm = {int(qi): v for qi, v in st["qnm"] if v is not None}
    # struct arrays, stable-sorted by neutral (deterministic tie order).
    # M198: structs emits null for keys with no known-adduct spectra — NaN
    # sorts last and matches no ppm window (honest exclusion, never KeyError).
    def _neutral(k):
        nk = neutrals.get(k)
        return float(nk) if (nk is not None and nk == nk) else np.nan
    skeys_all = np.array(ukeys)
    sneu_all = np.array([_neutral(k) for k in ukeys], dtype=np.float64)
    sord = np.argsort(sneu_all, kind="stable")
    skeys, masses = skeys_all[sord], sneu_all[sord]

    # lib_by_key, fully vectorized (no per-row .at loop over ~2.5M rows)
    codes, inv = np.unique(T["key14"], return_inverse=True)
    lib_arr = np.asarray(library, dtype=np.int64)
    lib_by_key = {}
    if len(lib_arr):
        srt = np.argsort(inv[lib_arr], kind="stable")
        sk = inv[lib_arr][srt]
        bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
        for b, e in zip(bounds[:-1], bounds[1:]):
            lib_by_key[str(codes[sk[b]])] = lib_arr[srt[b:e]].tolist()
    # key2smi: first occurrence per key (matches groupby().iloc[0])
    _, fidx = np.unique(inv, return_index=True)
    key2smi = {str(codes[inv[i]]): str(T["smiles"][int(i)]) for i in fidx}

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
    _prec_pairs = [(float(T["prec"][js[0]]), k) for k, js in lib_by_key.items()]
    _prec_pairs.sort(key=lambda t: t[0])
    lib_precs = [p for p, _ in _prec_pairs]
    lib_pkeys = [k for _, k in _prec_pairs]

    key2neutral = {str(k): float(v) for k, v in zip(skeys.tolist(), masses.tolist())}
    fusion = {"wl": a.wl, "wa": a.wa, "wf": a.wf, "rl": a.rl, "ra": a.ra,
              "rf": a.rf, "casc_z": a.casc_z, "rout_lib": a.rout_lib,
              "rout_bin": a.rout_bin, "rout_binw": a.rout_binw}
    # protocol v2: engine preloads the prepared library table once (parent,
    # before any fork — one engine, one copy). Per-query messages are keys.
    print("preload build lib_keys=" + str(len(lib_by_key)), flush=True)
    chunks = build_preload_chunks(lib_by_key, T, key2smi, key2neutral)
    engine = _Engine(a.rust_bin, a.qpeak, a.qloss, fusion)
    engine.preload(chunks)
    engine.preload_done()
    _rss, _avail = _mem_state()
    print("preload done chunks=" + str(len(chunks))
          + " rss_mb=" + str(_rss) + " memavail_mb=" + str(_avail),
          flush=True)  # HEARTBEAT
    # slim the parent: the engine owns library spectra now; the parent keeps
    # query spectra + small lookup arrays only (CoW-fork stays light).
    qset = set(int(qi) for qi in queries)
    _SE.update(skeys=np.array([str(k) for k in skeys.tolist()]),
               masses=masses, qnm=qnm, ppm=a.ppm,
               lib_precs=lib_precs, lib_pkeys=lib_pkeys,
               qmzs={qi: T["mzs"][qi] for qi in qset},
               qints={qi: T["ints"][qi] for qi in qset},
               qprec={qi: float(T["prec"][qi]) for qi in qset},
               qkey={qi: str(T["key14"][qi]) for qi in qset},
               shard=a.shard, variants=VARIANTS)
    del lib_by_key
    del T
    _rss, _avail = _mem_state()
    print("precompute done queries=" + str(len(queries))
          + " jobs=" + str(a.jobs) + " rss_mb=" + str(_rss) + " memavail_mb=" + str(_avail),
          flush=True)  # HEARTBEAT
    try:
        with open(_STATUS_PATH, "a") as sf:
            sf.write("PRECOMPUTE-DONE queries=" + str(len(queries)) + "\n")
    except Exception:
        pass
    # two-phase: workers select keys (engine-free pure numpy), parent scores
    # through the single preloaded engine. No per-query spectra cross a pipe.
    tasks = [(i, int(qi), len(queries)) for i, qi in enumerate(queries)]
    if a.jobs > 1 and len(tasks) > 1:
        with CF.ProcessPoolExecutor(max_workers=a.jobs) as ex:
            selected = list(ex.map(select_task, tasks))
    else:
        selected = [select_task(t) for t in tasks]
    results = []
    for sel in selected:
        if sel is None:
            continue
        res = score_one(sel, engine)
        if res is None:
            continue
        results.append(res)

    preds: dict = {}
    rrs: dict = {v: [] for v in VARIANTS}
    pools: list = []
    qidx: list = []
    swaps: list = []
    for qi, row, plen, ranked, line, swap in results:
        pools.append(int(plen))  # M201: searchsorted yields int64 — json can't serialize it
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
            "mrr_pure": _avg(rrs["pure"]),
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
        with open(_STATUS_PATH, "a") as sf:
            sf.write("DONE n=" + str(len(rrs["z"])) + "\n")
    except Exception:
        pass
    try:
        engine.close()
    except Exception:
        pass
    try:
        _stop.set()
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
