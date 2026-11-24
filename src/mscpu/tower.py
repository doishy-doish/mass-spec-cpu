"""tower.py — two-tower reference (spectrum tower + structure tower) + checker.

STATUS: Python reference + executable spec. Rust forward in binvec is OWED
(single-engine law: retrieval math ships in Rust; gates compile it post-push).
CI trains with numpy/sklearn only (CPU); no torch anywhere in this lane.

Shapes (fixed — weights files must match):
  spectrum tower: x[1537] (train_fp.featurize-compatible bins + prec/2000)
      -> Linear(1537->256) + ReLU -> Linear(256->D) -> L2. D=128.
  structure tower: morgan bits {0,1}^2048 -> {-1,+1}
      -> Linear(2048->256) + ReLU -> Linear(256->D) -> L2.
  score(q, c) = dot (cosine — both sides L2).

Weights file (.tower.bin): u32 magic 0x31574F54, u32 spec_d, mol_in, h, d,
  then f32 LE row-major: W1 (spec_d*h), b1 (h), W2 (h*d), b2 (d),
  V1 (mol_in*h), c1 (h), V2 (h*d), c2 (d).
No weights file -> tower score is 0.0 (quarantined, same rule as fp).

Checker (uniform, loud — math, not boxing): over-valence ONLY. Under-valence
is never flagged (this parser carries no implicit-H reconstruction, so low
counts are uninformative, not impossible). Unknown elements pass. Salts/dots
pass. Empty/unparseable is LOUD (reason, not silent drop).
"""
from __future__ import annotations

import struct

import numpy as np

from . import quality as Q

SPEC_D = 1537
MOL_IN = 2048
H = 256
D = 128
MAGIC = 0x31574F54

# Generous maxima: only the chemically impossible fails. Perchlorate Cl has
# 4 bonds, H3O+ O has 3 — thresholds sit above all legal states on purpose.
VALENCE_MAX = {"H": 1, "C": 4, "N": 5, "O": 3, "P": 6, "S": 6, "F": 1,
               "Cl": 7, "Br": 5, "I": 7}


def _relu(x):
    return np.maximum(x, 0.0)


def _l2(x):
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


def forward_q(x, W1, b1, W2, b2):
    """Spectrum tower -> L2 unit vector (D,)."""
    h = _relu(np.asarray(x, dtype=np.float64).reshape(-1) @ W1 + b1)
    return _l2(h @ W2 + b2)


def forward_c(bits01, V1, c1, V2, c2):
    """Structure tower: 2048 {0,1} bits -> {-1,+1} -> L2 unit vector (D,)."""
    v = np.asarray(bits01, dtype=np.float64).reshape(-1) * 2.0 - 1.0
    h = _relu(v @ V1 + c1)
    return _l2(h @ V2 + c2)


def tower_score(qvec, cvec) -> float:
    """Cosine (both L2) in [-1, 1]. No weights -> 0.0 (quarantined)."""
    if qvec is None or cvec is None:
        return 0.0
    return float(np.dot(np.asarray(qvec, dtype=np.float64).reshape(-1),
                        np.asarray(cvec, dtype=np.float64).reshape(-1)))


def save_weights(path, W1, b1, W2, b2, V1, c1, V2, c2):
    """Write .tower.bin (f32 LE row-major). Shapes asserted."""
    assert W1.shape == (SPEC_D, H) and b1.shape == (H,)
    assert W2.shape == (H, D) and b2.shape == (D,)
    assert V1.shape == (MOL_IN, H) and c1.shape == (H,)
    assert V2.shape == (H, D) and c2.shape == (D,)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIII", MAGIC, SPEC_D, MOL_IN, H, D))
        for a in (W1, b1, W2, b2, V1, c1, V2, c2):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())


def load_weights(path):
    """Read .tower.bin -> dict. Bad magic/shape raises LOUD (never silent)."""
    with open(path, "rb") as f:
        blob = f.read()
    magic, sd, mi, h, d = struct.unpack("<IIIII", blob[:20])
    if magic != MAGIC or (sd, mi, h, d) != (SPEC_D, MOL_IN, H, D):
        raise ValueError("tower weights mismatch magic=%08x dims=%s" % (magic, (sd, mi, h, d)))
    off = 20

    def take(n):
        nonlocal off
        a = np.frombuffer(blob[off:off + 4 * n], dtype=np.float32).copy()
        off += 4 * n
        return a
    W1 = take(SPEC_D * H).reshape(SPEC_D, H)
    b1 = take(H)
    W2 = take(H * D).reshape(H, D)
    b2 = take(D)
    V1 = take(MOL_IN * H).reshape(MOL_IN, H)
    c1 = take(H)
    V2 = take(H * D).reshape(H, D)
    c2 = take(D)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "V1": V1, "c1": c1, "V2": V2, "c2": c2}


def check_valence(smi: str) -> list:
    """Over-valence impossibility check. [] = pass. Reasons = loud fail.
    Uses quality.parse_smiles_lite (same parser the Tanimoto lens uses).
    """
    if not smi:
        return ["empty"]
    atoms, bonds = Q.parse_smiles_lite(str(smi))
    if atoms is None:
        return ["unparseable"]
    order_sum = [0] * len(atoms)
    for a, b, o in bonds:
        if 0 <= a < len(atoms) and 0 <= b < len(atoms):
            order_sum[a] += int(o)
            order_sum[b] += int(o)
    bad = []
    for i, (el, _ar, _ch, h) in enumerate(atoms):
        mx = VALENCE_MAX.get(str(el), None)
        if mx is None:
            continue  # unknown element passes (vocabulary doesn't judge)
        if order_sum[i] + int(h or 0) > mx:
            bad.append("%s#%d: bondorder+h=%d > max=%d" % (el, i, order_sum[i] + int(h or 0), mx))
    return bad


def checker_filter(smi_list: list) -> dict:
    """Split candidates into pass/fail. Never drops silently: every input is
    accounted in exactly one of the two lists."""
    ok, bad = [], []
    for s in (smi_list or []):
        rs = check_valence(s)
        (ok if not rs else bad).append((s, rs))
    return {"pass": [s for s, _ in ok], "fail": [{"smi": s, "reasons": r} for s, r in bad]}
