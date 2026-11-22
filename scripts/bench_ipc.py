"""bench_ipc.py — what actually beats Python for the Rust JSON bridge?

Same class3-shaped payload on all paths (350 candidates x 2 spectra x ~100
u32 bins, seeded RNG). Times three paths, writes results/bench.json:

  A. spawn-per-query + full-pool JSON  (the current engine=rust path, N=30)
  B. persistent worker, library preloaded once, per-query JSON lines (N=200)
  C. python in-process B.jaccard over the same payload (N=200)

If B ~= C and both << A, the 100:1 IPC-overhead claim is proven and the
worker architecture is quantified. No RDKit, no data download needed.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time

sys.path.insert(0, "src")
from mscpu import binary as B


def gen_payload(ncands=350, nspec=2, nbins=100, seed=0):
    rng = random.Random(seed)
    lib = {}
    for i in range(ncands):
        key = "k%04d" % i
        lib[key] = [
            {
                "key": "%s#%d" % (key, s),
                "p": [rng.getrandbits(32) for _ in range(nbins * 3 // 5)],
                "l": [rng.getrandbits(32) for _ in range(nbins * 2 // 5)],
            }
            for s in range(nspec)
        ]
    queries = [
        {
            "p": [rng.getrandbits(32) for _ in range(nbins * 3 // 5)],
            "l": [rng.getrandbits(32) for _ in range(nbins * 2 // 5)],
        }
        for _ in range(200)
    ]
    return lib, queries


def to_int(bins):
    bits = 0
    for q in bins["p"]:
        bits |= 1 << (B._mix(q) % B.NBITS)
    for q in bins["l"]:
        bits |= 1 << (B._mix(q, salt=1) % B.NBITS)
    return bits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="rust-bitvec/target/release/binvec")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lib, queries = gen_payload()
    res: dict = {"ncands": 350, "nspec": 2}

    # A. spawn per query, full pool JSON (current path) — N=30 keeps it sane
    t0 = time.perf_counter()
    for q in queries[:30]:
        refs = [s for specs in lib.values() for s in specs]
        job = json.dumps({"q": q, "refs": refs})
        pr = subprocess.run([a.bin], input=job, capture_output=True, text=True, timeout=300)
        assert pr.returncode == 0, pr.stderr[-500:]
        assert len(json.loads(pr.stdout)["scores"]) == len(refs)
    dt = time.perf_counter() - t0
    res["A_spawn_json_s"] = round(dt, 2)
    res["A_per_query_ms"] = round(dt / 30 * 1000, 1)

    # B. persistent worker, library preloaded once
    t0 = time.perf_counter()
    pr = subprocess.Popen([a.bin, "worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, bufsize=1)
    preload = {"lib": {k: [{"p": s["p"], "l": s["l"]} for s in specs]
                        for k, specs in lib.items()}}
    pr.stdin.write(json.dumps(preload) + "\n")
    pr.stdin.flush()
    assert json.loads(pr.stdout.readline()).get("ready") is True
    for q in queries:
        pr.stdin.write(json.dumps({"q": q, "cands": sorted(lib)}) + "\n")
        pr.stdin.flush()
        scores = json.loads(pr.stdout.readline())["scores"]
        assert len(scores) == 350
    pr.stdin.close()
    assert pr.wait(timeout=120) == 0
    dt = time.perf_counter() - t0
    res["B_worker_json_s"] = round(dt, 2)
    res["B_per_query_ms"] = round(dt / 200 * 1000, 2)

    # C. python in-process over the same payload
    lib_int = {k: [to_int(s) for s in specs] for k, specs in lib.items()}
    q_int = [to_int(q) for q in queries]
    t0 = time.perf_counter()
    for qb in q_int:
        for k in sorted(lib_int):
            best = 0.0
            for rb in lib_int[k]:
                j = B.jaccard(qb, rb)
                if j > best:
                    best = j
    dt = time.perf_counter() - t0
    res["C_python_s"] = round(dt, 2)
    res["C_per_query_ms"] = round(dt / 200 * 1000, 2)

    line = json.dumps(res)
    print(line, flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
