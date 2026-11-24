#!/usr/bin/env python3
"""Phase 1 (TRAIN_PLAN): train spectrum->Morgan MLP on CPU, export data/fp_mlp.bin.

Labels come from the engine itself (binvec dump-fp: same rust-chem Morgan code
that scores candidates, so train/serve bits are identical). Features must match
bin_feats in rust-bitvec/src/main.rs EXACTLY:
  bins 0..1535 Da @1Da: max over key spectra of sqrt(normalized intensity),
  feat[1536] = median precursor_mz / 2000.

Acceptance: holdout mean Tanimoto(pred>0.5, true) minus shuffled baseline > 0.05.

Usage (CI train-fp job):
  python scripts/train_fp.py --train data/train.parquet --engine ./target/release/binvec \\
      --out data/fp_mlp.bin --metrics results/train-fp.json
"""
import argparse
import json
import struct
import subprocess
import sys

import numpy as np

NBIN = 1536
INDIM = NBIN + 1
HID = 256
OUTDIM = 2048
MAGIC = 0x314D5046


def load_groups(train_path):
    import pyarrow.parquet as pq

    cols = ["inchikey14", "normalized_smiles", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]
    # Streamed batches (M197 lesson): whole-table to_pylist() doubles peak RSS
    # (batch dicts + groups dict) and got the runner killed 2x at ~35s into
    # load. Same groups out, bounded peak. Heartbeat per batch (no silent age).
    pf = pq.ParquetFile(train_path, memory_map=True)
    groups = {}
    seen = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=cols):
        for r in batch.to_pylist():
            k = r["inchikey14"]
            if not k:
                continue
            g = groups.get(k)
            if g is None:
                g = groups[k] = {"smiles": r["normalized_smiles"], "specs": [], "precs": []}
            mz = r["ms2_mzs"] or []
            it = r["ms2_normalized_intensities"] or []
            if len(mz) and len(mz) == len(it):
                g["specs"].append((np.asarray(mz, dtype=np.float64), np.asarray(it, dtype=np.float64)))
            if r["precursor_mz"]:
                g["precs"].append(float(r["precursor_mz"]))
        seen += batch.num_rows
        print("load groups rows=%d keys=%d" % (seen, len(groups)), flush=True)
    return groups


def featurize(specs, precs):
    x = np.zeros(INDIM, dtype=np.float32)
    for mz, it in specs:
        w = np.sqrt(np.clip(it, 0, None)).astype(np.float32)
        b = np.floor(mz).astype(np.int64)
        ok = (b >= 0) & (b < NBIN)
        np.maximum.at(x[:NBIN], b[ok], w[ok])
    x[NBIN] = float(np.median(precs)) / 2000.0 if precs else 0.0
    return x


def dump_bits(engine, keys, smiles_list, chunk=20000):
    """Call binvec dump-fp in chunks -> {key: bitmask bool array}."""
    out = {}
    for i in range(0, len(keys), chunk):
        ks = keys[i : i + chunk]
        ss = smiles_list[i : i + chunk]
        p = subprocess.run(
            [engine, "dump-fp"],
            input=json.dumps({"keys": ks, "smiles": ss}).encode(),
            capture_output=True,
            timeout=600,
        )
        if p.returncode != 0:
            raise RuntimeError(f"dump-fp failed: {p.stderr.decode()[-2000:]}")
        d = json.loads(p.stdout.decode())
        for k, hx in d["bits"].items():
            raw = bytes.fromhex(hx)
            words = struct.unpack(">32Q", raw)  # recovers exact u64 values
            arr = np.zeros(OUTDIM, dtype=np.bool_)
            # engine bit order: bit j lives at word j//64, LSB-position j%64
            # (fp_forward sets 1u64 << (k % 64)), so test (w >> b) & 1.
            for wi, w in enumerate(words):
                for b in range(64):
                    if (w >> b) & 1:
                        arr[wi * 64 + b] = True
            out[k] = arr
    return out


def tanimoto(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from sklearn.neural_network import MLPRegressor

    print("loading groups...", flush=True)
    groups = load_groups(a.train)
    keys = sorted(groups.keys())
    rng = np.random.default_rng(a.seed)
    rng.shuffle(keys)
    hold = set(keys[: a.holdout])
    train_keys = keys[a.holdout :]
    print(f"keys={len(keys)} train={len(train_keys)} holdout={len(hold)}", flush=True)

    mlp = MLPRegressor(
        hidden_layer_sizes=(HID,),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        random_state=a.seed,
    )
    # init on first chunk (Regressor partial_fit needs no classes_)
    for p in range(a.passes):
        print(f"pass {p + 1}/{a.passes}", flush=True)
        for i in range(0, len(train_keys), a.chunk):
            ks = train_keys[i : i + a.chunk]
            X = np.stack([featurize(groups[k]["specs"], groups[k]["precs"]) for k in ks])
            ydict = dump_bits(a.engine, ks, [groups[k]["smiles"] or "" for k in ks])
            Y = np.stack([ydict[k].astype(np.float32) for k in ks])
            mlp.partial_fit(X, Y)
    print("eval holdout...", flush=True)
    hk = sorted(hold)
    Xh = np.stack([featurize(groups[k]["specs"], groups[k]["precs"]) for k in hk])
    yh = dump_bits(a.engine, hk, [groups[k]["smiles"] or "" for k in hk])
    Yh = np.stack([yh[k] for k in hk])
    P = (mlp.predict(Xh) > 0.5)
    real = float(np.mean([tanimoto(P[i], Yh[i]) for i in range(len(hk))]))
    shuff = float(np.mean([tanimoto(P[i], Yh[(i + 1) % len(hk)]) for i in range(len(hk))]))
    margin = real - shuff
    print(f"holdout Tanimoto real={real:.4f} shuffled={shuff:.4f} margin={margin:.4f}", flush=True)
    ok = margin > 0.05
    if ok:
        W1 = np.ascontiguousarray(mlp.coefs_[0], dtype=np.float32)  # (in, hid)
        b1 = np.ascontiguousarray(mlp.intercepts_[0], dtype=np.float32)
        W2 = np.ascontiguousarray(mlp.coefs_[1], dtype=np.float32)  # (hid, out)
        b2 = np.ascontiguousarray(mlp.intercepts_[1], dtype=np.float32)
        assert W1.shape == (INDIM, HID) and W2.shape == (HID, OUTDIM), (W1.shape, W2.shape)
        with open(a.out, "wb") as f:
            f.write(struct.pack("<IIII", MAGIC, INDIM, HID, OUTDIM))
            f.write(W1.tobytes())
            f.write(b1.tobytes())
            f.write(W2.tobytes())
            f.write(b2.tobytes())
    with open(a.metrics, "w") as f:
        json.dump(
            {"real": real, "shuffled": shuff, "margin": margin, "accept": ok,
             "keys": len(keys), "seed": a.seed},
            f,
        )
    print("ACCEPT" if ok else "REJECT", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
