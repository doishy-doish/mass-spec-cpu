"""Phase 2: LightGBM ranker on OOF candidate pairs (user override 2026-11-24: weights
unparked for fitting on train.parquet-derived pairs only — 100% competition-legal;
third-party checkpoints stay gated by Rule 2.8: open-source + training code only).

Reads dumped pairs (plaintext JSONs; CI decrypts committed envelopes with
scripts/scramble.mjs first), trains lambdarank grouped by query with GroupKFold,
reports OOF MRR@25 lift vs best single channel. Writes data/lgbm_model.txt ONLY
on acceptance (lift > 0.02) so the weights-commit step has something to commit;
metrics always land (accept:false when starved — job succeeds, nothing ships).

Input pair record (run.py -dump): {"qi","keys","y","s":{channel:[per-cand]}}.
Pairs from -t2 dumps carry y=1 truth labels; strip-dump queries (truth never in
cands) are all-negative and filtered with a logged count.
"""
import argparse
import glob as _glob
import json
import os
import sys

# Fusion/ranked lists are outcomes, not features. Hand-set stumps excluded so
# the ranker learns from channels, not from our own heuristics (no circularity).
EXCLUDE = {"z", "rrf", "cascade", "router", "zw", "rrfw", "rrfb", "binw",
           "lgbma", "lgbmb", "lgbmc"}
CORE = ["lib", "stein", "mass", "cleav", "fp", "entw", "gate"]


def mrr_at_k(scores, labels, k=25):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, i in enumerate(order[:k], 1):
        if labels[i]:
            return 1.0 / rank
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()

    files = sorted(_glob.glob(a.glob))
    queries = []
    for fp in files:
        try:
            doc = json.load(open(fp))
        except Exception as e:
            print("skip %s: %s" % (fp, e), flush=True)
            continue
        if isinstance(doc, dict) and doc.get("enc"):
            print("skip %s: still encrypted (decrypt with scramble.mjs first)" % fp, flush=True)
            continue
        for rec in (doc.get("pairs") or []):
            if rec and rec.get("s"):
                queries.append(rec)
    print("pairs files=%d queries=%d" % (len(files), len(queries)), flush=True)

    feats = None
    rows = []  # (qi, feat_vec, y)
    dropped_allneg = 0
    for rec in queries:
        ch = [c for c in rec["s"] if c not in EXCLUDE]
        if feats is None:
            feats = [c for c in CORE if c in ch] + [c for c in sorted(ch) if c not in CORE]
        if not feats or any(c not in rec["s"] for c in feats):
            continue
        n = len(rec["keys"])
        if any(len(rec["s"][c]) != n for c in feats):
            continue
        y = [int(v) for v in rec["y"]]
        if not any(y):
            dropped_allneg += 1
            continue
        for j in range(n):
            rows.append((int(rec["qi"]), [float(rec["s"][c][j]) for c in feats], y[j]))
    print("feats=%s rows=%d dropped_allneg=%d" % (feats, len(rows), dropped_allneg), flush=True)

    metrics = {"files": len(files), "queries": len(queries), "feats": feats or [],
               "rows": len(rows), "dropped_allneg": dropped_allneg, "accept": False}
    if not feats or len(rows) < 500:
        metrics["reason"] = "starved (need >=500 labeled rows; use -t2 -dump fleets)"
        json.dump(metrics, open(a.out, "w"), indent=1)
        print("STARVED: " + metrics["reason"], flush=True)
        return 0

    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.model_selection import GroupKFold
    except ImportError as e:
        metrics["reason"] = "missing dep: %s" % e
        json.dump(metrics, open(a.out, "w"), indent=1)
        print("STARVED: " + metrics["reason"], flush=True)
        return 0

    X = np.array([r[1] for r in rows], dtype=np.float64)
    Y = np.array([r[2] for r in rows], dtype=np.int32)
    G = np.array([r[0] for r in rows])
    oof = np.zeros(len(rows))
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(X, Y, G):
        gtr = [int((G[tr] == g).sum()) for g in np.unique(G[tr])]
        dtr = lgb.Dataset(X[tr], label=Y[tr], group=gtr)
        dva = lgb.Dataset(X[va], label=Y[va], reference=dtr)
        b = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [25],
                       "learning_rate": 0.05, "num_leaves": 31, "seed": a.seed,
                       "verbose": -1},
                      dtr, num_boost_round=a.rounds, valid_sets=[dva])
        oof[va] = b.predict(X[va])

    # OOF MRR@25, query by query, for lgbm and every single channel.
    qids = sorted(set(r[0] for r in rows))
    by_q = {q: [] for q in qids}
    for i, r in enumerate(rows):
        by_q[r[0]].append(i)
    chan_mrr = {c: [] for c in feats}
    lgb_mrr = []
    for q in qids:
        idx = by_q[q]
        yl = [Y[i] for i in idx]
        lgb_mrr.append(mrr_at_k([oof[i] for i in idx], yl))
        for j, c in enumerate(feats):
            chan_mrr[c].append(mrr_at_k([X[i][j] for i in idx], yl))
    base_c = max(feats, key=lambda c: sum(chan_mrr[c]) / len(chan_mrr[c]))
    base = sum(chan_mrr[base_c]) / len(chan_mrr[base_c])
    oof_mrr = sum(lgb_mrr) / len(lgb_mrr)
    lift = oof_mrr - base
    accept = bool(lift > 0.02)
    print("OOF queries=%d lgbm_mrr=%.4f base_%s=%.4f lift=%+.4f accept=%s"
          % (len(qids), oof_mrr, base_c, base, lift, accept), flush=True)

    metrics.update({"oof_queries": len(qids), "oof_mrr_lgbm": round(oof_mrr, 6),
                    "base_channel": base_c, "oof_mrr_base": round(base, 6),
                    "lift": round(lift, 6), "accept": accept,
                    "chan_mrr": {c: round(sum(v) / len(v), 6) for c, v in chan_mrr.items()}})
    if accept:
        groups = [int((G == g).sum()) for g in np.unique(G)]
        dall = lgb.Dataset(X, label=Y, group=groups)
        final = lgb.train({"objective": "lambdarank", "learning_rate": 0.05,
                           "num_leaves": 31, "seed": a.seed, "verbose": -1},
                          dall, num_boost_round=a.rounds)
        final.save_model(a.model)
        print("model saved " + a.model, flush=True)
    json.dump(metrics, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
