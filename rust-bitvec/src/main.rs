//! sengine — the single Rust retrieval engine.
//!
//! All compute lives here. Python (run.py/predict.py) is a thin launcher:
//! parquet load, mask splits, pool shortlists, output assembly, submission.
//! There is deliberately NO Python backup of any channel math (user order:
//! one code, no duplicates, no backups).
//!
//! Mirrors (moved here, then deleted from Python):
//! spectra.clean/cosine/modified_cosine, binary.encode*/jaccard/wjaccard,
//! chem.morgan/tanimoto/key-less parts, score_query channels + funnel,
//! z/rrf/rrfb/cascade/router fusion + ranking, structure neutrals, medians.
//!
//! Known non-identical edges (functional MRR parity is the gate, A3/A9):
//! - round-half-away (Rust .round()) vs Python banker's round() in 2dp loss
//!   sets and quantization — exact-.5 cases are measure-zero in float mz data
//!   (0/2000 sampled); the cheap-mask canary decides.
//! - top-64/peaksort ties: numpy argpartition order is unspecified, Rust takes
//!   a full descending sort — same SET except adversarial ties.
//! - ranked-list ties: pandas quicksort (unstable) vs Rust stable sort that
//!   preserves candidate order. Deterministic; chance-level ties only.
//!
//! Protocols (JSON lines, fail loud on schema drift):
//!   argv `structs`: one stdin line
//!     {"ktable": [keys], "kidx": [per-spectrum key idx],
//!      "precs": [f64], "adducts": [str],
//!      "queries": [{"qi": i64, "prec": f64, "adduct": str}]}
//!     -> one stdout line {"neutrals": {key: f64}, "qnm": [[qi, f64|null]]}
//!   argv `sengine`: line 1 {"params": {"qpeak": f64, "qloss": f64}}
//!     -> {"ready": true}; then per query
//!     {"q": {"mzs","ints","prec"}, "cands": [{key,smiles,specs:[{mzs,ints,prec}]}],
//!      "pool": [{key,smiles,mzs,ints,prec}]}
//!     -> {"ranked": {variant: [keys<=25]}, "pools": n}
//! Progress heartbeats go to stderr (the launcher forwards them).

use chem::{cleavage_masses, entropy_grade, explained_frac, fp_predict_bits, lgbm_scores, morgan_bits_2048, parse_smiles, spectral_entropy_info, tanimoto as chem_tanimoto};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};

#[allow(dead_code)]
mod boundary;
#[allow(dead_code)]
mod slim;

// spectrum binary channel (unchanged math, validated bit-exact vs Python)
const QNBITS: u64 = 4096;
const KNUTH: u64 = 2654435761;
const SALT_K: u64 = 0x9E3779B9;

// channel + fusion constants (moved from run.py; single definition)
const TOL: f64 = 0.02;
const MS_THRESH: f64 = 0.3;
const TOP_N_ANALOG: usize = 10;
const FUNNEL_MIN: usize = 400;
const FUNNEL_K: usize = 300;
const POOL_CAP: usize = 1500;
const RRF_BASE: f64 = 60.0;
const MASS_CENTER: f64 = -1.5;
const MASS_FREE: f64 = 3.0;
const MASS_FULL: f64 = 8.0;
const MASS_LEAD_OFF: f64 = 5.0;
const MASS_LIB_TAU: f64 = 0.9;
const ROUTER_LIB: f64 = 0.5;
const ROUTER_BIN: f64 = 0.4;
const ROUTER_BINW: f64 = 0.4;
const CASCADE_Z: f64 = 3.0;
const TOP_K_OUT: usize = 25;
/// Preloaded library table (protocol v2): raw spectra + smiles per key,
/// prepared on first use and cached (M209). prepare() and morgan are pure
/// functions of these inputs, so lazy values equal eager ones bit-for-bit.
/// Pool takes position 0. Replaces the old eager preload (prepared all 275k
/// keys, 7.3min, before query 1 — most keys are never touched).
/// LIB-RESTORED 2026-11-24 (user: money first): 18 library-recall + computed
/// names live again (key-exclusion in validation keeps them honest) + 6 new
/// computed (S2 cleavage, quarantined fp, Q6 entw, 3x hand-set stumps). 24 total.
const VARIANTS: [&str; 24] = [
    "lib", "analog", "frag", "bin", "binw", "stein", "mass", "z", "rrf", "rrfb", "cascade", "router",
    "zw", "rrfw", "pure", "libce", "ent", "gate", "cleav", "fp", "entw", "lgbma", "lgbmb", "lgbmc",
];

fn mix(x: u64, salt: u64) -> u64 {
    x.wrapping_add(salt.wrapping_mul(SALT_K))
        .wrapping_mul(KNUTH)
        & 0xFFFF_FFFF
}

fn jaccard(a: &[u64; 64], b: &[u64; 64]) -> f64 {
    let (mut inter, mut union) = (0u32, 0u32);
    for k in 0..64 {
        inter += (a[k] & b[k]).count_ones();
        union += (a[k] | b[k]).count_ones();
    }
    if inter == 0 {
        0.0
    } else {
        inter as f64 / (union as f64).max(1.0)
    }
}

fn build_w(p: &[u64], pw: &[f32], l: &[u64], lw: &[f32]) -> Vec<(u16, f32)> {
    // sort+max-combine — identical output to the old HashMap fold,
    // 2.15x faster in the JS mirror (n<=128, HashMap overhead dominates).
    let mut v: Vec<(u16, f32)> = Vec::with_capacity(p.len() + l.len());
    for (i, &b) in p.iter().enumerate() {
        let w = pw.get(i).copied().unwrap_or(0.0);
        if w > 0.0 {
            v.push(((mix(b, 0) % QNBITS) as u16, w));
        }
    }
    for (i, &b) in l.iter().enumerate() {
        let w = lw.get(i).copied().unwrap_or(0.0);
        if w > 0.0 {
            v.push(((mix(b, 1) % QNBITS) as u16, w));
        }
    }
    v.sort_by_key(|e| e.0);
    let mut out: Vec<(u16, f32)> = Vec::with_capacity(v.len());
    for (b, w) in v {
        if let Some(last) = out.last_mut() {
            if last.0 == b {
                if w > last.1 {
                    last.1 = w;
                }
                continue;
            }
        }
        out.push((b, w));
    }
    out
}

fn wjaccard(a: &[(u16, f32)], b: &[(u16, f32)]) -> f64 {
    let (mut i, mut j) = (0usize, 0usize);
    let (mut inter, mut union) = (0.0f64, 0.0f64);
    while i < a.len() && j < b.len() {
        let (ka, va) = (a[i].0, a[i].1 as f64);
        let (kb, vb) = (b[j].0, b[j].1 as f64);
        if ka == kb {
            inter += va.min(vb);
            union += va.max(vb);
            i += 1;
            j += 1;
        } else if ka < kb {
            union += va;
            i += 1;
        } else {
            union += vb;
            j += 1;
        }
    }
    while i < a.len() {
        union += a[i].1 as f64;
        i += 1;
    }
    while j < b.len() {
        union += b[j].1 as f64;
        j += 1;
    }
    if inter <= 0.0 || union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn adduct_shift(a: &str) -> Option<f64> {
    match a {
        "[M+H]+" => Some(1.007276),
        "[M+Na]+" => Some(22.989218),
        "[M+NH4]+" => Some(18.033823),
        "[M+K]+" => Some(38.963158),
        "[M-H]-" => Some(-1.007276),
        "[M+Cl]-" => Some(34.968853),
        "[M-H2O+H]+" => Some(-17.003289),
        // M198: test-set adducts missing from the table dropped whole keys.
        // [M-2H2O+H]+ = 1.007276 - 2*18.010565; [M-H2O-H]- = -1.007276 - 18.010565.
        "[M-2H2O+H]+" => Some(-35.013854),
        "[M-H2O-H]-" => Some(-19.017841),
        // formate: M + CH2O2 - H (46.005479 - 1.007276). Was 43.990901
        // (double-subtracted proton): formate neutrals off by ~1 Da.
        "[M+CH2O2-H]-" => Some(44.998203),
        // Q1: formate alias as written in data ([M+HCOO]- == [M+CH2O2-H]-).
        "[M+HCOO]-" => Some(44.998203),
        "[M]+" => Some(0.0),
        _ => None,
    }
}

/// spectra.clean port: floor 0.01 rel, prec+2 filter, sqrt weights,
/// top-64 by intensity, mz-sorted. Returns (mz, w).
fn clean(mzs: &[f64], ints: &[f64], prec: f64) -> (Vec<f64>, Vec<f64>) {
    if mzs.is_empty() || ints.is_empty() {
        return (vec![], vec![]);
    }
    let base = ints.iter().cloned().fold(0.0f64, f64::max);
    if !(base > 0.0) {
        return (vec![], vec![]);
    }
    let n = mzs.len().min(ints.len());
    let mut w0 = vec![0.0f64; n];
    let mut idx: Vec<usize> = Vec::new();
    for i in 0..n {
        if ints[i] / base >= 0.01 && mzs[i] <= prec + 2.0 {
            w0[i] = (ints[i] / base).sqrt();
            idx.push(i);
        }
    }
    idx.sort_by(|&a, &b| w0[b].partial_cmp(&w0[a]).unwrap());
    idx.truncate(64);
    idx.sort_by(|&a, &b| mzs[a].partial_cmp(&mzs[b]).unwrap());
    (
        idx.iter().map(|&i| mzs[i]).collect(),
        idx.iter().map(|&i| (ints[i] / base).sqrt()).collect(),
    )
}

fn qbin(mz: f64, q: f64) -> u64 {
    ((mz / q).round() as i64 as u64) & 0xFFFF_FFFF
}

/// Cleaned spectrum + every derived layer.
#[derive(Clone)]
struct CSpectra {
    mz: Vec<f64>,
    w: Vec<f64>,
    prec: f64,
    norm: f64,
    sw: Vec<f64>,
    snorm: f64,
    bits: [u64; 64],
    wb: Vec<(u16, f32)>,
    /// Neutral-loss set, precomputed once per spectrum (the frag channel used
    /// to rebuild it per candidate spec per query — same inputs, same output).
    floss: Vec<f64>,
    /// Collision-energy band source (None = unknown/merged ramp).
    ce: Option<f64>,
}

fn prepare(mzs: &[f64], ints: &[f64], prec: f64, qpeak: f64, qloss: f64, ce: Option<f64>) -> CSpectra {
    let (mz, w) = clean(mzs, ints, prec);
    let norm = w.iter().map(|x| x * x).sum();
    // Stein-Scott-lite mass weights (mass-linear, mz/1000 scale): same peaks, high-mass
    // peaks count more. snorm precomputed once per spectrum (fixed per run).
    let sw: Vec<f64> = mz.iter().zip(w.iter()).map(|(m, x)| m / 1000.0 * x).collect();
    let snorm: f64 = sw.iter().map(|x| x * x).sum();
    let pb: Vec<u64> = mz.iter().map(|m| qbin(*m, qpeak)).collect();
    let pw: Vec<f32> = w.iter().map(|x| *x as f32).collect();
    let (lb, lw) = if prec != 0.0 {
        (
            mz.iter().map(|m| qbin(prec - m, qloss)).collect(),
            pw.clone(),
        )
    } else {
        (vec![], vec![])
    };
    let mut bits = [0u64; 64];
    for &b in &pb {
        let i = (mix(b, 0) % QNBITS) as usize;
        bits[i / 64] |= 1u64 << (i % 64);
    }
    for &b in &lb {
        let i = (mix(b, 1) % QNBITS) as usize;
        bits[i / 64] |= 1u64 << (i % 64);
    }
    let wb = build_w(&pb, &pw, &lb, &lw);
    let floss = loss_set(&mz, prec);
    CSpectra { mz, w, prec, norm, sw, snorm, bits, wb, floss, ce }
}

fn match_dot(amz: &[f64], ai: &[f64], bmz: &[f64], bi: &[f64], delta: f64, tol: f64) -> f64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut dot = 0.0f64;
    while i < amz.len() && j < bmz.len() {
        let d = amz[i] - bmz[j] - delta;
        if d < -tol {
            i += 1;
        } else if d > tol {
            j += 1;
        } else {
            dot += ai[i] * bi[j];
            i += 1;
            j += 1;
        }
    }
    dot
}

fn cosine_row(q: &CSpectra, s: &CSpectra, tol: f64) -> f64 {
    if q.mz.is_empty() || s.mz.is_empty() {
        return 0.0;
    }
    let den = (q.norm * s.norm).sqrt();
    if !(den > 0.0) {
        return 0.0;
    }
    match_dot(&q.mz, &q.w, &s.mz, &s.w, 0.0, tol) / den
}

fn modified_row(q: &CSpectra, s: &CSpectra, tol: f64) -> f64 {
    if q.mz.is_empty() || s.mz.is_empty() {
        return 0.0;
    }
    let den = (q.norm * s.norm).sqrt();
    if !(den > 0.0) {
        return 0.0;
    }
    match_dot(&q.mz, &q.w, &s.mz, &s.w, 0.0, tol)
        .max(match_dot(&q.mz, &q.w, &s.mz, &s.w, q.prec - s.prec, tol))
        / den
}

/// Stein-Scott-lite: mass-weighted cosine. Same two-pointer matches as cosine,
/// weights folded with mz/1000 so high-mass peaks dominate. Self = 1.0, far = 0.0.
/// Pure-algo replication (akmalkhaniub Stein-Scott + neutral-loss hybrid): no chemistry,
/// no learning, one more two-pointer per candidate — standalone channel, NOT fused yet.
fn stein_row(q: &CSpectra, s: &CSpectra, tol: f64) -> f64 {
    if q.mz.is_empty() || s.mz.is_empty() {
        return 0.0;
    }
    let den = (q.snorm * s.snorm).sqrt();
    if !(den > 0.0) {
        return 0.0;
    }
    match_dot(&q.mz, &q.sw, &s.mz, &s.sw, 0.0, tol) / den
}

/// v8-mass-lite (romanrozen Version-22 numbers): graded neutral-mass error.
/// ppm = (cand_neutral - qnm)/qnm * 1e6. 1.0 inside |ppm-CENTER|<=FREE,
/// linear decay to 0.0 at FULL. Missing neutrals -> 0.0 for all (honest tie).
/// Standalone replication: ranked, NOT fused yet (existing fusions bit-identical).
fn mass_score(qnm: Option<f64>, neutral: Option<f64>) -> f64 {
    match (qnm, neutral) {
        (Some(q), Some(c)) if q > 0.0 => {
            let ppm = (c - q) / q * 1e6;
            let dev = (ppm - MASS_CENTER).abs();
            if dev <= MASS_FREE {
                1.0
            } else if dev >= MASS_FULL {
                0.0
            } else {
                1.0 - (dev - MASS_FREE) / (MASS_FULL - MASS_FREE)
            }
        }
        _ => 0.0,
    }
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn loss_set(mz: &[f64], prec: f64) -> Vec<f64> {
    let mut v: Vec<f64> = mz.iter().take(32).map(|m| r2(prec - m)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

fn overlap_count(a: &[f64], b: &[f64]) -> usize {
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            n += 1;
            i += 1;
            j += 1;
        }
    }
    n
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        Some(v[n / 2])
    } else {
        Some((v[n / 2 - 1] + v[n / 2]) / 2.0)
    }
}

/// pandas Series.std(): sample std, ddof=1; n<2 -> NaN.
fn sample_std(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 {
        return f64::NAN;
    }
    (xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// pandas rank(ascending=False, method="min"): 1 + #(strictly greater).
/// O(n log n) sort-based — bit-identical to the old O(n^2) count
/// (JS mirror: 30x at n=1500, same output incl. NaN + tie handling).
fn rank_min_desc(vals: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..vals.len()).filter(|&i| !vals[i].is_nan()).collect();
    idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap());
    let mut out = vec![f64::NAN; vals.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && vals[idx[j]] == vals[idx[i]] {
            j += 1;
        }
        let rank = (1 + i) as f64;
        for k in i..j {
            out[idx[k]] = rank;
        }
        i = j;
    }
    out
}

/// Strict "a ranks before b" (NaN last, ties false so equal scores keep
/// candidate order). Exactly the comparator inside order_desc, factored out.
fn less_desc(a: f64, b: f64) -> bool {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => false,
        (true, false) => false,
        (false, true) => true,
        (false, false) => a > b,
    }
}

/// Stable order by score desc, NaN last, ties keep candidate order
/// (documents the pandas-quicksort tie difference; gate absorbs).
fn order_desc(vals: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    if vals.len() <= 32 {
        // insertion sort: at tiny n it beats the general sort's setup cost,
        // and with a strict comparator it is stable — same order, bit for bit.
        for i in 1..idx.len() {
            let mut j = i;
            while j > 0 && less_desc(vals[idx[j]], vals[idx[j - 1]]) {
                idx.swap(j, j - 1);
                j -= 1;
            }
        }
        return idx;
    }
    idx.sort_by(|&a, &b| {
        let (va, vb) = (vals[a], vals[b]);
        match (va.is_nan(), vb.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => vb.partial_cmp(&va).unwrap(),
        }
    });
    idx
}

// ---------- structs mode ----------

#[derive(Deserialize)]
struct StructsIn {
    ktable: Vec<String>,
    kidx: Vec<usize>,
    precs: Vec<f64>,
    adducts: Vec<String>,
    queries: Vec<QNm>,
}

#[derive(Deserialize)]
struct QNm {
    qi: i64,
    prec: f64,
    adduct: String,
}

#[derive(Serialize)]
struct StructsOut {
    // M198: every ktable key gets an entry (None = no known-adduct spectra).
    // Omitting keys crashed the launcher with KeyError downstream.
    neutrals: HashMap<String, Option<f64>>,
    qnm: Vec<(i64, Option<f64>)>,
}

fn structs_main() {
    let stdin = std::io::stdin();
    let mut buf = String::new();
    stdin
        .read_line(&mut buf)
        .expect("structs: read stdin");
    let inp: StructsIn = serde_json::from_str(&buf).expect("structs: parse");
    let mut by_key: HashMap<usize, Vec<f64>> = HashMap::new();
    for (&ki, (&prec, ad)) in inp
        .kidx
        .iter()
        .zip(inp.precs.iter().zip(inp.adducts.iter()))
    {
        if let Some(sh) = adduct_shift(ad.as_str()) {
            let e = prec - sh;
            if e > 50.0 && e < 3000.0 {
                by_key.entry(ki).or_default().push(e);
            }
        }
    }
    let mut neutrals: HashMap<String, Option<f64>> = HashMap::new();
    for ki in 0..inp.ktable.len() {
        neutrals.insert(inp.ktable[ki].clone(), by_key.remove(&ki).and_then(median));
    }
    let qnm = inp
        .queries
        .iter()
        .map(|q| {
            (
                q.qi,
                adduct_shift(&q.adduct).map(|sh| q.prec - sh),
            )
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string(&StructsOut { neutrals, qnm }).expect("structs: serialize")
    );
}

// ---------- sengine mode ----------

fn def1() -> f64 {
    1.0
}
fn def0() -> f64 {
    0.0
}
fn def_cascz() -> f64 {
    3.0
}
fn def_rlib() -> f64 {
    0.5
}
fn def_rbin() -> f64 {
    0.4
}

/// BET-EVOLVE genome (replication queue #3): tunable fusion weights + gates.
/// Every default reproduces the old behavior bit-identically (neutral = 1.0,
/// casc_z = CASCADE_Z, rout_* = ROUTER_*): machinery lands MRR-neutral, the
/// canary fits the values on class2 n=50 (never class3). Missing keys in the
/// params JSON (old launchers) also default neutral — backward compatible.
#[derive(Deserialize)]
struct EngineParams {
    qpeak: f64,
    qloss: f64,
    #[serde(default = "def1")]
    wl: f64,
    #[serde(default = "def1")]
    wa: f64,
    #[serde(default = "def1")]
    wf: f64,
    #[serde(default = "def1")]
    rl: f64,
    #[serde(default = "def1")]
    ra: f64,
    #[serde(default = "def1")]
    rf: f64,
    #[serde(default = "def_cascz")]
    casc_z: f64,
    #[serde(default = "def_rlib")]
    rout_lib: f64,
    #[serde(default = "def_rbin")]
    rout_bin: f64,
    #[serde(default = "def_rbin")]
    rout_binw: f64,
    /// I6: v8 rank-1-by-mass replace on every ranked list (0 = off, 1 = on).
    #[serde(default = "def0")]
    mass_replace: f64,
    /// I6: funnel recall probe — echo first-K pool keys (0 = off).
    #[serde(default = "def0")]
    recall_topk: f64,
    /// I6: funnel A/B — disable the Jaccard prefilter (0 = funnel on, 1 = off).
    #[serde(default = "def0")]
    funnel_off: f64,
    /// R2: diagnostic-loss demotion — stable-partition ranked lists so
    /// candidates failing the loss/subformula gate sink below passing ones.
    /// 0 = off (bit-identical lists), 1 = on.
    #[serde(default = "def0")]
    lossgate: f64,
    /// E1: engine-owned store. Slim dir path; when set, the engine loads it
    /// at startup and resolves library + query spectra from it on demand —
    /// no preload chunks, no per-query spectra JSON. Python sends keys +
    /// row refs only. None = protocol v2 (preload chunks) as before.
    #[serde(default)]
    store: Option<String>,
    /// WIN-FIRST Phase 1: trained spectrum->Morgan MLP weights file
    /// (data/fp_mlp.bin layout, TRAIN_PLAN). None = fp channel stays
    /// quarantined at 0.0 (safe default until weights land).
    #[serde(default)]
    fp_weights: Option<String>,
}

impl EngineParams {
    fn neutral(qpeak: f64, qloss: f64) -> Self {
        Self {
            qpeak,
            qloss,
            wl: 1.0,
            wa: 1.0,
            wf: 1.0,
            rl: 1.0,
            ra: 1.0,
            rf: 1.0,
            casc_z: CASCADE_Z,
            rout_lib: ROUTER_LIB,
            rout_bin: ROUTER_BIN,
            rout_binw: ROUTER_BINW,
            mass_replace: 0.0,
            recall_topk: 0.0,
            funnel_off: 0.0,
            lossgate: 0.0,
            store: None,
            fp_weights: None,
        }
    }
}

/// WIN-FIRST Phase 1: trained spectrum->Morgan MLP (TRAIN_PLAN).
/// Bins must match scripts/train_fp.py featurize EXACTLY:
/// bins 0..1535 Da @1Da of max cleaned weight (clean stores sqrt of normalized
/// intensity — train_fp.py applies the same single sqrt), feat[1536]=prec/2000.
const FP_NBIN: usize = 1536;
const FP_INDIM: usize = 1537;
const FP_NWORDS: usize = 32;

struct FpModel {
    hid: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn bin_feats(mz: &[f64], w: &[f64], prec: f64) -> Vec<f32> {
    let mut x = vec![0f32; FP_INDIM];
    for (&m, &ww) in mz.iter().zip(w.iter()) {
        if m >= 0.0 && m < FP_NBIN as f64 && ww > 0.0 {
            let b = m as usize;
            let v = ww as f32; // w is already sqrt (clean); train_fp.py sqrts raw once
            if v > x[b] {
                x[b] = v;
            }
        }
    }
    x[FP_NBIN] = (prec as f32) / 2000.0;
    x
}

fn load_fp_model(path: &str) -> Option<FpModel> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 16 {
        return None;
    }
    let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    if u(0) != 0x314D5046 || u(4) != FP_INDIM {
        return None;
    }
    let (hid, out) = (u(8), u(12));
    if out != 2048 || hid == 0 || hid > 4096 {
        return None;
    }
    let f = |off: usize, n: usize| -> Option<Vec<f32>> {
        let end = off.checked_add(n.checked_mul(4)?)?;
        b.get(off..end).map(|s| {
            s.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
    };
    let mut off = 16;
    let w1 = f(off, FP_INDIM * hid)?;
    off += FP_INDIM * hid * 4;
    let b1 = f(off, hid)?;
    off += hid * 4;
    let w2 = f(off, hid * 2048)?;
    off += hid * 2048 * 4;
    let b2 = f(off, 2048)?;
    Some(FpModel { hid, w1, b1, w2, b2 })
}

fn fp_forward(m: &FpModel, x: &[f32]) -> [u64; FP_NWORDS] {
    let mut h = vec![0f32; m.hid];
    for j in 0..m.hid {
        let mut s = m.b1[j];
        for i in 0..x.len().min(FP_INDIM) {
            s += x[i] * m.w1[i * m.hid + j];
        }
        h[j] = if s > 0.0 { s } else { 0.0 };
    }
    let mut out = [0u64; FP_NWORDS];
    for k in 0..2048 {
        let mut s = m.b2[k];
        for j in 0..m.hid {
            s += h[j] * m.w2[j * 2048 + k];
        }
        if 1.0 / (1.0 + (-s).exp()) >= 0.5 {
            out[k / 64] |= 1u64 << (k % 64);
        }
    }
    out
}

/// dump-fp: stdin {"keys":[],"smiles":[]} -> stdout {"bits":{key:hex512},"n":N}.
/// Labels for train_fp.py — same rust-chem Morgan code that scores candidates.
fn dump_fp_main() {
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf).expect("dump-fp: read stdin");
    #[derive(serde::Deserialize)]
    struct In {
        keys: Vec<String>,
        smiles: Vec<String>,
    }
    let inp: In = serde_json::from_str(&buf).expect("dump-fp: parse");
    let mut bits: HashMap<String, String> = HashMap::with_capacity(inp.keys.len());
    for (k, s) in inp.keys.iter().zip(inp.smiles.iter()) {
        let w = chem::morgan_bits_2048(s);
        let mut hx = String::with_capacity(512);
        for word in w {
            hx.push_str(&format!("{:016x}", word));
        }
        bits.insert(k.clone(), hx);
    }
    let n = bits.len();
    println!("{}", serde_json::json!({"bits": bits, "n": n}));
}

#[derive(Deserialize)]
struct RawSpec {
    mzs: Vec<f64>,
    ints: Vec<f64>,
    prec: f64,
    /// Single collision energy in eV (None = unknown or merged ramp).
    #[serde(default)]
    ce: Option<f64>,
}

/// One library table chunk (preload protocol v2): per-query messages carry
/// keys only, so the ~1MB/query JSON transfer+parse+re-prepare — ~70% of
/// per-query time on the profile — happens exactly once per engine.
#[derive(Deserialize)]
struct PreloadChunk {
    keys: Vec<String>,
    smiles: Vec<String>,
    neutrals: Vec<Option<f64>>,
    specs: Vec<Vec<RawSpec>>,
}

#[derive(Deserialize)]
struct EngineIn {
    #[serde(default)]
    q: Option<RawSpec>,
    #[serde(default)]
    qs: Option<Vec<RawSpec>>,
    #[serde(default)]
    qnms: Option<Vec<Option<f64>>>,
    #[serde(default)]
    qis: Option<Vec<i64>>,
    #[serde(default)]
    cand_keys: Option<Vec<String>>,
    #[serde(default)]
    pool_keys: Option<Vec<String>>,
    #[serde(default)]
    qi: Option<i64>,
    #[serde(default)]
    qnm: Option<f64>,
    /// E1: query as a slim store row index. The engine resolves
    /// (mzs, ints, prec, ce) from its own store — zero query-spectra JSON.
    /// None = full `q` RawSpec as before.
    #[serde(default)]
    qrow: Option<usize>,
    #[serde(default)]
    preload: Option<PreloadChunk>,
    #[serde(default)]
    preload_done: Option<bool>,
    /// E2-pool parity: engine-side mirror of run.py select_task selection.
    /// Python sends one per selected query; the engine rebuilds the neutral
    /// window + analog pool from its slim store and returns both lists.
    /// qrows (class2/cross) or held keys (class3) define the library split.
    #[serde(default)]
    pool_check: Option<PoolCheck>,
}

#[derive(Deserialize)]
struct PoolCheck {
    qnm: Option<f64>,
    prec_q: f64,
    ppm: f64,
    mask: String,
    qrows: Vec<usize>,
    held: Vec<String>,
}

#[derive(Serialize)]
struct PoolCheckOut {
    cand_keys: Vec<String>,
    pool_keys: Vec<String>,
}

#[derive(Serialize)]
struct PoolCheckReply {
    pool_check: Option<PoolCheckOut>,
}

#[derive(Serialize)]
struct EngineReply {
    ranked: HashMap<String, Vec<String>>,
    pools: usize,
    /// v8 rank-1-by-mass: lib-leader |ppm-CENTER|>5.0 && lib<0.9 would swap. Info only.
    mass_swap: bool,
    /// I6 funnel recall probe: first-K pool keys in funnel output order.
    /// None unless recall_topk > 0. Python scores truth retention offline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prefilter_topk: Option<Vec<String>>,
    /// per-query stage timings ms (measure-first: worker stderr is DEVNULLed
    /// by the launcher, so timings ride home in the reply and land in logs
    /// via the Python heartbeat line). 8 stages: parse,qprep,cands,pool,
    /// funnel,analog,chan,fuse. Observation only — never touches scoring.
    timing: Vec<f64>,
}

#[derive(Serialize)]
struct Ready {
    ready: bool,
}

struct CandData {
    key: String,
    specs: Vec<CSpectra>,
    fp: Option<[u64; 32]>,
    neutral: Option<f64>,
    formula: Option<BTreeMap<String, u32>>,
}

/// Preloaded library table entry (protocol v2): raw spectra + smiles stored
/// at preload; prepared CSpectra + fingerprint computed on first use and
/// cached (M209). Pool uses specs[0]; candidates use all specs.
struct TableEntry {
    raw: Vec<RawSpec>,
    specs: Option<Vec<CSpectra>>,
    smiles: String,
    fp: Option<[u64; 32]>,
    neutral: Option<f64>,
    formula: Option<BTreeMap<String, u32>>,
}

/// M209: prepare-on-first-use. prepare() is a pure function of (raw spectra,
/// qpeak, qloss) and morgan_bits_2048 a pure function of smiles, so cached
/// values equal the old eager-preload values bit-for-bit — only the WHEN
/// changes. Raw is dropped after preparing to bound engine memory:
/// raw(untouched keys) + prepared(touched keys).
fn ensure_prepared(e: &mut TableEntry, qpeak: f64, qloss: f64) {
    if e.specs.is_none() {
        let specs: Vec<CSpectra> = e
            .raw
            .iter()
            .map(|s| prepare(&s.mzs, &s.ints, s.prec, qpeak, qloss, s.ce))
            .collect();
        e.specs = Some(specs);
        e.raw = Vec::new();
        // parsed once with the spectra (None = unparseable, not retried).
        e.formula = formula_counts(&e.smiles);
    }
    if e.fp.is_none() {
        e.fp = Some(morgan_bits_2048(&e.smiles));
    }
}

/// E1: key -> slim row indices + key position (ints only; spectra resolve
/// lazily per touched key, M209 pattern — untouched keys cost indices only).
fn build_key_maps(
    s: &slim::SlimStore,
) -> (HashMap<String, Vec<usize>>, HashMap<String, usize>) {
    let mut rows: HashMap<String, Vec<usize>> = HashMap::with_capacity(s.keys.len());
    let mut pos: HashMap<String, usize> = HashMap::with_capacity(s.keys.len());
    for (i, k) in s.keys.iter().enumerate() {
        pos.insert(k.clone(), i);
        rows.insert(k.clone(), Vec::new());
    }
    for (r, ki) in s.key_idx.iter().enumerate() {
        if let Some(v) = s.keys.get(*ki as usize).and_then(|k| rows.get_mut(k)) {
            v.push(r);
        }
    }
    (rows, pos)
}

/// E2-pool: neutral-sorted + prec-sorted library views, built once per split.
/// Mirrors run.py select_task inputs: neutral arrays cover ALL store keys
/// (byte-sorted, stable NaN-last = numpy stable argsort over sorted ukeys);
/// prec arrays cover LIBRARY keys only (first library-row prec per key,
/// stable sort by prec, ties in key-index order = Python key_idx grouping).
/// ASSUMPTIONS (bench-arbitrated, pool parity key-for-key): store keys ==
/// table keys; builder per-key neutral == structs median; slim row order ==
/// parquet row order; slim prec bit-identical to parquet prec.
struct PoolArrays {
    masses: Vec<f64>,
    skeys: Vec<String>,
    precs: Vec<f64>,
    pkeys: Vec<String>,
}

fn build_pool_arrays(
    st: &slim::SlimStore,
    pos: &std::collections::HashMap<String, usize>,
    _mask: &str,
    _qrows: &[usize],
    held: &[String],
) -> PoolArrays {
    // neutral view: all keys.
    let mut keys_sorted: Vec<String> = st.keys.clone();
    keys_sorted.sort();
    let mut neut: Vec<(f64, String)> = keys_sorted
        .into_iter()
        .map(|k| {
            let n = pos
                .get(&k)
                .and_then(|&p| st.neutrals[p])
                .filter(|v| *v == *v);
            (n.unwrap_or(f64::NAN), k)
        })
        .collect();
    neut.sort_by(|a, b| match (a.0.is_nan(), b.0.is_nan()) {
        (false, false) => a
            .0
            .partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal),
        (false, true) => std::cmp::Ordering::Less,
        (true, false) => std::cmp::Ordering::Greater,
        (true, true) => std::cmp::Ordering::Equal,
    });
    // stable: equal/NaN keep byte-sorted key order = numpy stable semantics.
    let masses: Vec<f64> = neut.iter().map(|(n, _)| *n).collect();
    let skeys: Vec<String> = neut.into_iter().map(|(_, k)| k).collect();
    // prec view: library keys only. ALL masks exclude held KEYS
    // (PHONEBOOK-STRIP 2026-11-24: every spectrum of a test molecule leaves
    // the library — mirrors validation.split_* exactly; query rows' keys are
    // in held, so row exclusion is subsumed. qrows stay in the cache key).
    let key_excl: std::collections::HashSet<&str> =
        held.iter().map(|x| x.as_str()).collect();
    let mut first: std::collections::HashMap<usize, (f64, String)> =
        std::collections::HashMap::new();
    for (r, ki) in st.key_idx.iter().enumerate() {
        let ki = *ki as usize;
        let k = match st.keys.get(ki) {
            Some(k) => k,
            None => continue,
        };
        let excluded = key_excl.contains(k.as_str());
        if excluded {
            continue;
        }
        first.entry(ki).or_insert((st.prec[r], k.clone()));
    }
    let mut pairs: Vec<(usize, f64, String)> = first
        .into_iter()
        .map(|(ki, (p, k))| (ki, p, k))
        .collect();
    pairs.sort_by_key(|(ki, _, _)| *ki);
    pairs.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // stable: prec ties keep key-index order = Python key_idx grouping order.
    let precs: Vec<f64> = pairs.iter().map(|(_, p, _)| *p).collect();
    let pkeys: Vec<String> = pairs.into_iter().map(|(_, _, k)| k).collect();
    PoolArrays {
        masses,
        skeys,
        precs,
        pkeys,
    }
}

/// E1: build one TableEntry straight from the slim store (same floats the
/// preload path would have carried — bit-identical input to
/// ensure_prepared, so scoring is unchanged).
fn entry_from_slim(
    s: &slim::SlimStore,
    pos: &HashMap<String, usize>,
    rows: &[usize],
    key: &str,
) -> TableEntry {
    let mut raw = Vec::with_capacity(rows.len());
    for &r in rows {
        let (m, ii) = slim::row_spec(s, r);
        raw.push(RawSpec {
            mzs: m.to_vec(),
            ints: ii.to_vec(),
            prec: s.prec[r],
            ce: slim::row_ce(s, r),
        });
    }
    TableEntry {
        raw,
        specs: None,
        smiles: s.smiles.get(key).cloned().unwrap_or_default(),
        fp: None,
        neutral: pos.get(key).and_then(|&p| s.neutrals[p]),
        formula: None,
    }
}

/// E1: resolve one key, filling the table from slim on first touch.
/// Unknown keys (absent from the store) stay absent — same skip semantics
/// as the preload path (M217), so coverage behavior is unchanged.
fn ensure_key(
    table: &mut HashMap<String, TableEntry>,
    store: Option<&slim::SlimStore>,
    rows: &HashMap<String, Vec<usize>>,
    pos: &HashMap<String, usize>,
    key: &str,
) {
    if table.contains_key(key) {
        return;
    }
    if let (Some(st), Some(rr)) = (store, rows.get(key)) {
        table.insert(key.to_string(), entry_from_slim(st, pos, rr, key));
    }
}

/// R2: collision-energy band. Query and library CE are single eV values
/// (merged ramps arrive as None from the launcher); bands split the three
/// acquisition energies the test set uses (20 / 40 / 60 eV).
fn ce_band(ce: f64) -> u8 {
    if ce <= 25.0 {
        0
    } else if ce <= 50.0 {
        1
    } else {
        2
    }
}

/// R2: diagnostic neutral losses (exact monoisotopic masses, verified by
/// arithmetic 2026-11-23): water, ammonia, CO, CO2, formic acid, hexose,
/// glucuronide, deoxyhexose. Match tolerance is absolute Da (fragment side).
const DIAG_LOSSES: [(f64, &[(&str, u32)]); 8] = [
    (18.010565, &[("H", 2), ("O", 1)]),
    (17.026549, &[("N", 1), ("H", 3)]),
    (27.994915, &[("C", 1), ("O", 1)]),
    (43.989829, &[("C", 1), ("O", 2)]),
    (46.005479, &[("C", 1), ("H", 2), ("O", 2)]),
    (162.052823, &[("C", 6), ("H", 10), ("O", 5)]),
    (176.032088, &[("C", 6), ("H", 8), ("O", 6)]),
    (146.057909, &[("C", 6), ("H", 10), ("O", 4)]),
];
const LOSS_TOL_DA: f64 = 0.02;

fn el_norm(el: &str) -> String {
    let mut c = el.chars();
    match c.next() {
        None => "C".to_string(),
        Some(f) => {
            let mut s = f.to_ascii_uppercase().to_string();
            for ch in c {
                s.push(ch.to_ascii_lowercase());
            }
            s
        }
    }
}

/// R2: atom counts from a SMILES via the chem parser. The parser stores
/// implicit H as 0 on unbracketed atoms, so implicit H is reconstructed from
/// standard organic valences minus bond-order sums (bracket/charged atoms
/// carry their own explicit h and are trusted as-is). The rule errs toward
/// OVERCOUNTING H, which is the fail-safe direction for a demotion gate
/// (extra H never wrongly demotes; missing H could).
fn formula_counts(smiles: &str) -> Option<BTreeMap<String, u32>> {
    let mol = parse_smiles(smiles)?;
    if mol.atoms.is_empty() {
        return None;
    }
    let mut bond_sum = vec![0u32; mol.atoms.len()];
    for &(a, b, o) in &mol.bonds {
        if a < bond_sum.len() && b < bond_sum.len() {
            bond_sum[a] += o as u32;
            bond_sum[b] += o as u32;
        }
    }
    let mut m = BTreeMap::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        let e = el_norm(&a.el);
        *m.entry(e.clone()).or_insert(0) += 1;
        let h = if a.h > 0 || a.charge != 0 {
            a.h
        } else if a.aromatic && e == "N" {
            0 // bare aromatic n is pyridine-like (pyrrole arrives bracketed)
        } else {
            let v: u32 = match e.as_str() {
                "C" => {
                    if a.aromatic {
                        3
                    } else {
                        4
                    }
                }
                "N" => 3,
                "O" | "S" => 2,
                "P" => 3,
                "F" | "Cl" | "Br" | "I" => 1,
                _ => 4,
            };
            v.saturating_sub(bond_sum[i])
        };
        if h > 0 {
            *m.entry("H".to_string()).or_insert(0) += h;
        }
    }
    Some(m)
}

/// R2: diagnostic-loss gate fraction for one candidate: of the diagnostic
/// neutral losses observed in the query (prec - frag within LOSS_TOL_DA),
/// how many have atoms ⊆ candidate formula. Vacuous (no losses observed or
/// no formula) scores 1.0 — the gate only demotes, never promotes.
fn gate_frac(qmz: &[f64], qprec: f64, formula: Option<&BTreeMap<String, u32>>) -> f64 {
    let mut observed = 0u32;
    let mut explained = 0u32;
    for &(lmass, atoms) in DIAG_LOSSES.iter() {
        let mut seen = false;
        for &m in qmz {
            if m < qprec && (qprec - m - lmass).abs() <= LOSS_TOL_DA {
                seen = true;
                break;
            }
        }
        if !seen {
            continue;
        }
        observed += 1;
        match formula {
            None => explained += 1,
            Some(f) => {
                let mut ok = true;
                for &(el, n) in atoms.iter() {
                    if f.get(el).copied().unwrap_or(0) < n {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    explained += 1;
                }
            }
        }
    }
    if observed == 0 {
        1.0
    } else {
        explained as f64 / observed as f64
    }
}

/// R2: loss-gate demotion — stable-partition one ranked list so gate-passing
/// keys keep their relative order ahead of failing ones. Passing = gate
/// fraction 1.0 (fully explained or vacuous).
fn partition_by_gate(list: &mut Vec<String>, gate_of: &HashMap<String, f64>) {
    if list.is_empty() {
        return;
    }
    let mut pass = Vec::with_capacity(list.len());
    let mut fail = Vec::with_capacity(list.len());
    for k in list.drain(..) {
        if gate_of.get(&k).copied().unwrap_or(1.0) >= 1.0 - 1e-12 {
            pass.push(k);
        } else {
            fail.push(k);
        }
    }
    list.extend(pass);
    list.extend(fail);
}

/// R2: dynamic spectral-entropy weight exponent (Li & Fiehn shape): sparse
/// spectra (S < 3 nats) boost minor peaks, busy spectra stay linear.
fn entropy_exp(rel: &[f64]) -> f64 {
    let sum: f64 = rel.iter().sum();
    if !(sum > 0.0) {
        return 1.0;
    }
    let mut s = 0.0;
    for &x in rel {
        if x > 0.0 {
            let p = x / sum;
            s -= p * p.ln();
        }
    }
    if s < 3.0 {
        0.25 + 0.25 * s
    } else {
        1.0
    }
}

/// R2: entropy-weighted cosine. Relative intensities recovered as w^2
/// (clean stores sqrt), raised to the dynamic exponent, cosine-normalized.
/// Self-match is exactly 1.0 (same accumulation order both sides).
/// Restored with lib (library-recall channel).
fn ent_row(qmz: &[f64], qw: &[f64], smz: &[f64], sw: &[f64], tol: f64) -> f64 {
    if qmz.is_empty() || smz.is_empty() {
        return 0.0;
    }
    let qe = entropy_exp(&qw.iter().map(|x| x * x).collect::<Vec<f64>>());
    let se = entropy_exp(&sw.iter().map(|x| x * x).collect::<Vec<f64>>());
    let qew: Vec<f64> = qw.iter().map(|x| (x * x).powf(qe)).collect();
    let sew: Vec<f64> = sw.iter().map(|x| (x * x).powf(se)).collect();
    let qn: f64 = qew.iter().map(|x| x * x).sum();
    let sn: f64 = sew.iter().map(|x| x * x).sum();
    let den = (qn * sn).sqrt();
    if !(den > 0.0) {
        return 0.0;
    }
    match_dot(qmz, &qew, smz, &sew, 0.0, tol) / den
}

/// S2 cleavage score for one candidate: explained-intensity fraction of the
/// query's cleaned peaks by the candidate's ≤2-bond cleavage fragments.
/// tol 0.02 Da (fragment side). No frags / no peaks -> 0.0 (no evidence).
fn cleavage_score(qmz: &[f64], qw: &[f64], smiles: &str) -> f64 {
    let frags = cleavage_masses(smiles);
    if frags.is_empty() {
        return 0.0;
    }
    explained_frac(qmz, qw, &frags, 0.02)
}

/// Neural-fp-lite score: QUARANTINED 2026-11-24 (Gemini audit, upheld).
/// Predicted bits hash mz/loss/nm/prec STRINGS into 2048 bits; candidate Morgan
/// bits hash 2D atom NEIGHBORHOODS. Same FNV function, different vocabularies:
/// cross-domain Tanimoto is uncalibrated overlap noise, NOT a learned mapping.
/// Kept as a function for reference; scoring path returns 0.0 so it cannot
/// contaminate stumps/router/fusion. Restore only behind a TRAINED
/// spectrum->fingerprint model (user override of WEIGHTS-PARKED required).
#[allow(dead_code)]
fn fp_score(_qmz: &[f64], _qprec: f64, _qnm: Option<f64>, _cand_fp: Option<&[u64; 32]>) -> f64 {
    0.0
}

/// Q6 query entropy (nats) over cleaned sqrt weights + exponent.
/// Sparse spectra (S<3) boost minor peaks; busy stay linear. Computed only.
fn query_entropy_info(qw: &[f64]) -> (f64, f64) {
    spectral_entropy_info(qw)
}

/// Entropy-router: per-query choice of computed evidence.
/// Sparse queries (exp<0.5, i.e. very few dominant peaks) trust neutral-mass
/// + cleavage (mass + cleav); busy queries trust loss-gate + fp + entropy
/// grade (gate + fp + entw). Returns fused desc order over candidate idx.
/// Pure computed, no library spectra, no weights.
fn entropy_router_order(
    mass: &[f64],
    gate: &[f64],
    cleav: &[f64],
    _fp: &[f64],
    entw: &[f64],
    qent_exp: f64,
) -> Vec<usize> {
    let n = mass.len();
    let sparse = qent_exp < 0.5;
    let mut sc = vec![0.0; n];
    if sparse {
        let mz = zscore(mass);
        let cz = zscore(cleav);
        for i in 0..n {
            let mut s = 0.0;
            if !mz[i].is_nan() {
                s += mz[i];
            }
            if !cz[i].is_nan() {
                s += cz[i];
            }
            sc[i] = s;
        }
    } else {
        let gz = zscore(gate);
        let ez = zscore(entw);
        for i in 0..n {
            let mut s = 0.0;
            for x in [gz[i], ez[i]] {
                if !x.is_nan() {
                    s += x;
                }
            }
            sc[i] = s;
        }
    }
    order_desc(&sc)
}

/// z-score with pandas semantics: sample std (ddof=1), +1e-9 guard.
/// Shared by fuse_rank and the molecule-batch per-spectrum columns so S=1
/// equals the single-query rank exactly.
fn zscore(xs: &[f64]) -> Vec<f64> {
    let mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
    let s = sample_std(xs, mean) + 1e-9;
    xs.iter().map(|&x| (x - mean) / s).collect()
}

fn fuse_rank(
    keys: &[String],
    cols: &HashMap<String, Vec<f64>>,
    fw: &EngineParams,
) -> HashMap<String, Vec<String>> {
    // LIB-RESTORED 2026-11-24 (user: money first): spectral-recall channels
    // back as evidence. Validation keeps the STRIP key-exclusion (held truths
    // never enter pools, so lib can't leak there); on the hidden test the
    // overlap share pays through lib, the novel share falls to computed.
    // MONEY GATE: lib cosine >= 0.9 hit-locks cascade+router to lib order
    // (explicit; the old cascade_z path usually agrees). fp stays quarantined
    // (cross-domain Tanimoto noise — see fp_score) and emits empty.
    let n = keys.len();
    // borrow channels in place. Missing channel == all-NaN fallback.
    let nan = vec![f64::NAN; n];
    let col = |c: &str| -> &[f64] { cols.get(c).map(Vec::as_slice).unwrap_or(&nan) };
    let lib = col("lib");
    let analog = col("analog");
    let frag = col("frag");
    let bin = col("bin");
    let binw = col("binw");
    let stein = col("stein"); // standalone (Loop-2 replication): ranked, NOT fused yet
    let mass = col("mass"); // standalone (Loop-3 v8-mass-lite): ranked, NOT fused yet
    let libce = col("libce"); // standalone (R2 CE band): ranked, NOT fused yet
    let ent = col("ent"); // standalone (R2 entropy-weighted cosine): ranked, NOT fused yet
    let gate = col("gate"); // standalone (R2 diagnostic-loss gate): ranked, NOT fused yet
    let cleav = col("cleav"); // S2 MetFrag-lite cleavage, computed
    let entw = col("entw"); // Q6 entropy-weighted gate, computed
    let lgbma = col("lgbma"); // hand-set stumps (NOT LightGBM), fp-free
    let lgbmb = col("lgbmb");
    let lgbmc = col("lgbmc");
    let z_of = |xs: &[f64]| -> Vec<f64> {
        let mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        let s = sample_std(xs, mean) + 1e-9;
        xs.iter().map(|&x| (x - mean) / s).collect()
    };
    let libz = z_of(lib);
    let analogz = z_of(analog);
    let fragz = z_of(frag);
    let steinz = z_of(stein);
    let massz = z_of(mass);
    // pure (pure-algo canary v0): equal-weight z of stein + frag + mass.
    let pure: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            if !steinz[i].is_nan() {
                s += steinz[i];
            }
            if !fragz[i].is_nan() {
                s += fragz[i];
            }
            if !massz[i].is_nan() {
                s += massz[i];
            }
            s
        })
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| {
            [libz[i], analogz[i], fragz[i]]
                .iter()
                .filter(|&&x| !x.is_nan())
                .sum()
        })
        .collect();
    // zw: weighted z (genome wl/wa/wf; neutral 1.0 == z bit-identically)
    let zw: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            if !libz[i].is_nan() {
                s += fw.wl * libz[i];
            }
            if !analogz[i].is_nan() {
                s += fw.wa * analogz[i];
            }
            if !fragz[i].is_nan() {
                s += fw.wf * fragz[i];
            }
            s
        })
        .collect();
    // ranks computed ONCE per channel and shared by rrf/rrfb/rrfw.
    let rlib = rank_min_desc(lib);
    let rana = rank_min_desc(analog);
    let rfrag = rank_min_desc(frag);
    let rbin = rank_min_desc(bin);
    let rrf_of = |ranks: &[&[f64]], ws: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                ranks
                    .iter()
                    .zip(ws.iter())
                    .map(|(r, w)| {
                        if r[i].is_nan() {
                            f64::NAN
                        } else {
                            w / (RRF_BASE + r[i])
                        }
                    })
                    .filter(|x| !x.is_nan())
                    .sum()
            })
            .collect()
    };
    // rrf restored to banked meaning (lib+analog+frag; computed fusion rides
    // cleav/entw/lgbm* standalone + router hit-bypass + single-path entropy).
    let rrf = rrf_of(&[&rlib, &rana, &rfrag], &[1.0, 1.0, 1.0]);
    let rrfb = rrf_of(&[&rlib, &rana, &rfrag, &rbin], &[1.0, 1.0, 1.0, 1.0]);
    // rrfw: weighted RRF (genome rl/ra/rf; neutral 1.0 == rrf bit-identically)
    let rrfw = rrf_of(&[&rlib, &rana, &rfrag], &[fw.rl, fw.ra, fw.rf]);
    let max_skipnan = |xs: &[f64]| -> f64 {
        xs.iter()
            .filter(|&&x| !x.is_nan())
            .cloned()
            .fold(f64::NAN, f64::max)
    };
    // MONEY GATE: lib cosine >= 0.9 hit-locks cascade to lib order.
    let lib_hit = !lib.is_empty() && max_skipnan(lib) >= 0.9;
    let cascade: Vec<f64> = if lib_hit {
        lib.to_vec()
    } else if max_skipnan(&libz) > fw.casc_z {
        libz.clone()
    } else {
        analogz.clone()
    };
    let overlap = max_skipnan(lib) > fw.rout_lib
        || max_skipnan(bin) > fw.rout_bin
        || max_skipnan(binw) > fw.rout_binw;
    let router = if overlap { cascade.clone() } else { analog.to_vec() };
    let mut out = HashMap::new();
    // borrows, not clones: every output here is read-only from here on.
    let named: Vec<(&str, &[f64])> = vec![
        ("lib", lib),
        ("analog", analog),
        ("frag", frag),
        ("bin", bin),
        ("binw", binw),
        ("stein", stein),
        ("mass", mass),
        ("cleav", cleav),
        ("entw", entw),
        ("lgbma", lgbma),
        ("lgbmb", lgbmb),
        ("lgbmc", lgbmc),
        ("z", &z),
        ("rrf", &rrf),
        ("rrfb", &rrfb),
        ("cascade", &cascade),
        ("router", &router),
        ("zw", &zw),
        ("rrfw", &rrfw),
        ("pure", &pure),
        ("libce", libce),
        ("ent", ent),
        ("gate", gate),
    ];
    for (name, vals) in named {
        let kv: Vec<String> = order_desc(vals)
            .into_iter()
            .take(TOP_K_OUT)
            .map(|i| keys[i].clone())
            .collect();
        out.insert(name.to_string(), kv);
    }
    // fp: fused IFF weights are loaded (trained MLP); otherwise quarantined
    // empty (cross-domain hash noise — see fp_score). No-weights default = safe.
    if fw.fp_weights.is_some() {
        if let Some(fp) = cols.get("fp") {
            let kv: Vec<String> = order_desc(fp)
                .into_iter()
                .take(TOP_K_OUT)
                .map(|i| keys[i].clone())
                .collect();
            out.insert("fp".to_string(), kv);
        } else {
            out.insert("fp".to_string(), Vec::new());
        }
    } else {
        out.insert("fp".to_string(), Vec::new());
    }
    out
}

/// v8 rank-1-by-mass (info only): lib-leader |ppm-CENTER|>LEAD_OFF && lib<TAU.
/// Restored with lib (the swap rule keys off the lib leader).
fn mass_swap_flag(lib: &[f64], _mass: &[f64], qnm: Option<f64>, neutrals: &[Option<f64>]) -> bool {
    if qnm.is_none() {
        return false;
    }
    let q = qnm.unwrap();
    if !(q > 0.0) || lib.is_empty() {
        return false;
    }
    let mut best = 0usize;
    for i in 1..lib.len() {
        let (a, b) = (lib[best], lib[i]);
        if b.is_nan() {
            continue;
        }
        if a.is_nan() || b > a {
            best = i;
        }
    }
    if lib[best].is_nan() || lib[best] >= MASS_LIB_TAU {
        return false;
    }
    match neutrals.get(best).copied().flatten() {
        Some(c) => ((c - q) / q * 1e6 - MASS_CENTER).abs() > MASS_LEAD_OFF,
        None => false,
    }
}

/// v8 rank-1-by-mass replace, engine follow-up (I6): per ranked list, if the
/// rank-1 key's |ppm-CENTER| exceeds LEAD_OFF, move the list's min-|ppm| key
/// to front. Deterministic (first-seen wins ties); a no-op without qnm/ppms
/// or when the leader is already best. Truth-absent lanes are unaffected.
fn apply_mass_replace(ranked: &mut HashMap<String, Vec<String>>, ppm_of: &HashMap<String, f64>) {
    for list in ranked.values_mut() {
        if list.is_empty() {
            continue;
        }
        let lead_d = match ppm_of.get(&list[0]) {
            Some(p) => (*p - MASS_CENTER).abs(),
            None => continue,
        };
        if lead_d <= MASS_LEAD_OFF {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, k) in list.iter().enumerate() {
            if let Some(p) = ppm_of.get(k) {
                let d = (*p - MASS_CENTER).abs();
                match best {
                    Some((_, bd)) if bd <= d => {}
                    _ => best = Some((i, d)),
                }
            }
        }
        if let Some((bi, bd)) = best {
            if bd < lead_d {
                let k = list.remove(bi);
                list.insert(0, k);
            }
        }
    }
}

/// funnel recall probe (I6): first-K keys of the funnel-ordered pool.
/// Pure slice — the engine echoes these so Python can score truth retention.
fn funnel_topk(ordered_keys: &[String], k: usize) -> Vec<String> {
    ordered_keys.iter().take(k).cloned().collect()
}

/// Molecule-batch fuse (microsec lane): S per-spectrum score columns over the
/// same candidate order -> one fused ranked list per variant via RRF across
/// spectra. No learning, no weights: rank_s -> 1/(60+rank_s), NaN ranks skip.
/// All-NaN across spectra stays NaN (order_desc puts it last). S=1 equals the
/// single-query rank exactly (one RRF term, monotonic in rank).
fn fuse_molecule_rrf(
    per_spec: &[HashMap<String, Vec<f64>>],
    keys: &[String],
    fp_live: bool,
) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    if per_spec.is_empty() || keys.is_empty() {
        for v in VARIANTS {
            out.insert(v.to_string(), Vec::new());
        }
        return out;
    }
    let n = keys.len();
    let nan = vec![f64::NAN; n];
    // fp fused IFF weights are loaded (trained MLP); otherwise quarantined
    // empty. Every other variant (lib-family restored + computed) fuses here.
    const DEAD: [&str; 1] = ["fp"];
    for v in VARIANTS {
        if !fp_live && DEAD.contains(&v) {
            out.insert(v.to_string(), Vec::new());
            continue;
        }
        let ranks: Vec<Vec<f64>> = per_spec
            .iter()
            .map(|cols| {
                let c: &[f64] = cols.get(v).map(Vec::as_slice).unwrap_or(&nan);
                rank_min_desc(c)
            })
            .collect();
        let mut fused = vec![f64::NAN; n];
        for i in 0..n {
            let mut s = 0.0;
            let mut hit = false;
            for r in &ranks {
                let rv = r[i];
                if !rv.is_nan() {
                    s += 1.0 / (RRF_BASE + rv);
                    hit = true;
                }
            }
            if hit {
                fused[i] = s;
            }
        }
        let kv: Vec<String> = order_desc(&fused)
            .into_iter()
            .take(TOP_K_OUT)
            .map(|i| keys[i].clone())
            .collect();
        out.insert(v.to_string(), kv);
    }
    out
}

fn sengine(params: EngineParams) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let lines = stdin.lock().lines();
    let e = std::io::stderr();
    let mut err = e.lock();
    // Preloaded library table (protocol v2): one engine, one copy.
    // Per-query messages carry keys only; entries prepare on first use,
    // so untouched keys cost parse+store only (M209).
    let mut table: HashMap<String, TableEntry> = HashMap::new();
    // E1: engine-owned store. When params.store is set, library + query
    // spectra resolve from slim on demand — preload chunks are unnecessary
    // (Python sends none) and per-query spectra JSON drops to zero.
    let mut slim_store: Option<slim::SlimStore> = None;
    let mut slim_rows: HashMap<String, Vec<usize>> = HashMap::new();
    let mut slim_pos: HashMap<String, usize> = HashMap::new();
    // E2-pool: cached library views, rebuilt when the split changes.
    let mut pool_cache: Option<(String, Vec<usize>, Vec<String>, PoolArrays)> = None;
    if let Some(dir) = params.store.clone() {
        let st = slim::load_slim(std::path::Path::new(&dir));
        let (rows, pos) = build_key_maps(&st);
        eprintln!(
            "sengine: slim store loaded rows={} keys={} ({})",
            st.n,
            st.keys.len(),
            dir
        );
        slim_rows = rows;
        slim_pos = pos;
        slim_store = Some(st);
    }
    // WIN-FIRST Phase 1: trained fp MLP. None = quarantined zeros (safe default).
    let fp_model: Option<FpModel> = params.fp_weights.as_ref().and_then(|p| {
        let m = load_fp_model(p);
        eprintln!(
            "sengine: fp model {} ({})",
            if m.is_some() { "loaded" } else { "MISSING-or-corrupt, fp=0" },
            p
        );
        m
    });
    let fp_live = fp_model.is_some();
    for line in lines {
        let line = line.expect("sengine: read");
        if line.trim().is_empty() {
            continue;
        }
        let t0 = std::time::Instant::now();
        let ipc_bytes = line.len();
        let m: EngineIn = serde_json::from_str(&line).expect("sengine: parse");
        // preload path: table chunks, then done -> ready.
        if let Some(ch) = m.preload {
            for (((k, sm), n), ss) in ch
                .keys
                .into_iter()
                .zip(ch.smiles.into_iter())
                .zip(ch.neutrals.into_iter())
                .zip(ch.specs.into_iter())
            {
                // M209: store raw; prepare on first use (see ensure_prepared).
                table.insert(k, TableEntry { raw: ss, specs: None, smiles: sm, fp: None, neutral: n, formula: None });
            }
            continue;
        }
        if m.preload_done.unwrap_or(false) {
            writeln!(h, "{}", serde_json::to_string(&Ready { ready: true }).unwrap()).unwrap();
            h.flush().unwrap();
            continue;
        }
        // E2-pool parity: engine rebuilds neutral window + analog pool from
        // its slim store and returns both lists. Python compares against its
        // own select_task output (pool_check flag); selection stays Python's.
        if let Some(pc) = m.pool_check {
            let reply = match slim_store.as_ref() {
                None => PoolCheckReply { pool_check: None },
                Some(st) => {
                    let same = match &pool_cache {
                        Some((km, kq, kh, _))
                            if *km == pc.mask && *kq == pc.qrows && *kh == pc.held =>
                        {
                            true
                        }
                        _ => false,
                    };
                    if !same {
                        let arr = build_pool_arrays(
                            st,
                            &slim_pos,
                            &pc.mask,
                            &pc.qrows,
                            &pc.held,
                        );
                        pool_cache = Some((
                            pc.mask.clone(),
                            pc.qrows.clone(),
                            pc.held.clone(),
                            arr,
                        ));
                    }
                    let arrays = &pool_cache.as_ref().expect("pool cache built").3;
                    // qnm None or empty window mirrors select_task's None.
                    let out = pc.qnm.and_then(|qnm| {
                        let (_, _, ck) = boundary::ppm_window(
                            &arrays.masses,
                            &arrays.skeys,
                            qnm,
                            pc.ppm,
                        );
                        if ck.is_empty() {
                            return None;
                        }
                        let cs: std::collections::HashSet<String> =
                            ck.iter().cloned().collect();
                        let pool = boundary::analog_window(
                            &arrays.precs,
                            &arrays.pkeys,
                            pc.prec_q,
                            &cs,
                            1500,
                        );
                        Some(PoolCheckOut {
                            cand_keys: ck,
                            pool_keys: pool,
                        })
                    });
                    PoolCheckReply { pool_check: out }
                }
            };
            writeln!(h, "{}", serde_json::to_string(&reply).unwrap()).unwrap();
            h.flush().unwrap();
            continue;
        }
        // Molecule-batch path: S spectra, one round trip, cands/pool prepped
        // once, per-spectrum channels fused by RRF. S=1 equals single path.
        if let Some(qs) = m.qs {
            let t_parse = t0.elapsed();
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            let qnms: Vec<Option<f64>> = m.qnms.unwrap_or_default();
            let qis: Vec<i64> = m.qis.unwrap_or_default();
            let cand_keys = m.cand_keys.unwrap_or_default();
            let mut cands: Vec<CandData> = Vec::with_capacity(cand_keys.len());
            for k in &cand_keys {
                ensure_key(&mut table, slim_store.as_ref(), &slim_rows, &slim_pos, k);
                if let Some(e) = table.get_mut(k) {
                    ensure_prepared(e, params.qpeak, params.qloss);
                    cands.push(CandData {
                        key: k.clone(),
                        specs: e.specs.clone().unwrap_or_default(),
                        fp: if e.smiles.is_empty() { None } else { e.fp },
                        neutral: e.neutral,
                        formula: e.formula.clone(),
                    });
                }
            }
            let t_qprep = t0.elapsed();
            let pool_keys = m.pool_keys.unwrap_or_default();
            // Pool entries carry all first-spectra (prec from specs[0]); per-
            // spectrum distance is computed against each qc below.
            let mut pool_first: Vec<(String, CSpectra, [u64; 32])> = Vec::with_capacity(pool_keys.len());
            for k in &pool_keys {
                ensure_key(&mut table, slim_store.as_ref(), &slim_rows, &slim_pos, k);
                if let Some(e) = table.get_mut(k) {
                    ensure_prepared(e, params.qpeak, params.qloss);
                    if let Some(s) = e.specs.as_ref().and_then(|v| v.first()) {
                        pool_first.push((k.clone(), s.clone(), e.fp.unwrap_or([0u64; 32])));
                    }
                }
            }
            let t_cands = t0.elapsed();
            // Per-spectrum: qc + funnel + hits + channels (same math as single).
            let mut per_spec: Vec<HashMap<String, Vec<f64>>> = Vec::with_capacity(qs.len());
            let mut pool_union: usize = 0;
            let mut mass_swap_any = false;
            let mut neutrals: Vec<Option<f64>> = cands.iter().map(|c| c.neutral).collect();
            let _ = &mut neutrals;
            for (si, q) in qs.iter().enumerate() {
                let qc = prepare(&q.mzs, &q.ints, q.prec, params.qpeak, params.qloss, q.ce);
                let qloss = loss_set(&qc.mz, qc.prec);
                let qn = qloss.len().max(1) as f64;
                let qnm: Option<f64> = qnms.get(si).copied().flatten();
                // WIN-FIRST Phase 1: trained fp prediction for this spectrum.
                // None (no weights) = quarantined zeros.
                let qfp: Option<[u64; FP_NWORDS]> = fp_model
                    .as_ref()
                    .map(|m| fp_forward(m, &bin_feats(&qc.mz, &qc.w, qc.prec)));
                let mut pool: Vec<(String, CSpectra, f64, [u64; 32])> = Vec::with_capacity(pool_first.len());
                for (k, s, fp) in &pool_first {
                    pool.push((k.clone(), s.clone(), (s.prec - qc.prec).abs(), *fp));
                }
                let mut pool_idx: Vec<usize> = (0..pool.len()).collect();
                if pool.len() > FUNNEL_MIN && params.funnel_off < 0.5 && qc.bits.iter().any(|&w| w != 0) {
                    let mut scored: Vec<(f64, f64, usize)> = pool
                        .iter()
                        .enumerate()
                        .map(|(i, (_, s, dist, _))| (jaccard(&qc.bits, &s.bits), *dist, i))
                        .collect();
                    scored.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
                    });
                    pool_idx = scored.into_iter().take(FUNNEL_K).map(|(_, _, i)| i).collect();
                }
                if pool_idx.len() > POOL_CAP {
                    pool_idx.truncate(POOL_CAP);
                }
                pool_union = pool_union.max(pool_idx.len());
                let mut hits: Vec<([u64; 32], f64)> = Vec::new();
                for &i in &pool_idx {
                    let (_, s, _, fp) = &pool[i];
                    if fp == &[0u64; 32] {
                        continue;
                    }
                    let msv = modified_row(&qc, s, TOL);
                    if msv > MS_THRESH {
                        hits.push((*fp, msv));
                    }
                }
                hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                hits.truncate(TOP_N_ANALOG);
                let mut vlib = Vec::with_capacity(cands.len());
                let mut vanalog = Vec::with_capacity(cands.len());
                let mut vfrag = Vec::with_capacity(cands.len());
                let mut vbin = Vec::with_capacity(cands.len());
                let mut vbinw = Vec::with_capacity(cands.len());
                let mut vstein = Vec::with_capacity(cands.len());
                let mut vmass = Vec::with_capacity(cands.len());
                let mut vlibce = Vec::with_capacity(cands.len());
                let mut vent = Vec::with_capacity(cands.len());
                let mut vgate = Vec::with_capacity(cands.len());
                for c in &cands {
                    let mut lib = 0.0;
                    let mut stein = 0.0;
                    let mut ent = 0.0;
                    for s in &c.specs {
                        let v = cosine_row(&qc, s, TOL);
                        if v > lib {
                            lib = v;
                        }
                    }
                    let mut libce = lib;
                    if let Some(qce) = qc.ce {
                        let qb = ce_band(qce);
                        let mut found = false;
                        let mut best = 0.0;
                        for s in &c.specs {
                            if s.ce.map(ce_band) == Some(qb) {
                                found = true;
                                let v = cosine_row(&qc, s, TOL);
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        if found {
                            libce = best;
                        }
                    }
                    let mut analog = 0.0;
                    if let Some(cfp) = c.fp {
                        for (ofp, msv) in &hits {
                            let v = msv * chem_tanimoto(&cfp, ofp);
                            if v > analog {
                                analog = v;
                            }
                        }
                    }
                    let mut frag = 0.0;
                    for s in c.specs.iter().take(2) {
                        let v = overlap_count(&qloss, &s.floss) as f64 / qn;
                        if v > frag {
                            frag = v;
                        }
                    }
                    let (mut bscore, mut bwscore) = (0.0, 0.0);
                    for s in &c.specs {
                        let j = jaccard(&qc.bits, &s.bits);
                        if j > bscore {
                            bscore = j;
                        }
                        let wj = wjaccard(&qc.wb, &s.wb);
                        if wj > bwscore {
                            bwscore = wj;
                        }
                        let st = stein_row(&qc, s, TOL);
                        if st > stein {
                            stein = st;
                        }
                        let en = ent_row(&qc.mz, &qc.w, &s.mz, &s.w, TOL);
                        if en > ent {
                            ent = en;
                        }
                    }
                    vlib.push(lib);
                    vanalog.push(analog);
                    vfrag.push(frag);
                    vbin.push(bscore);
                    vbinw.push(bwscore);
                    vstein.push(stein);
                    vmass.push(mass_score(qnm, c.neutral));
                    vlibce.push(libce);
                    vent.push(ent);
                    vgate.push(gate_frac(&qc.mz, qc.prec, c.formula.as_ref()));
                }
                let mut cols: HashMap<String, Vec<f64>> = HashMap::with_capacity(VARIANTS.len() * 2);
                cols.insert("lib".to_string(), vlib);
                cols.insert("analog".to_string(), vanalog);
                cols.insert("frag".to_string(), vfrag);
                cols.insert("bin".to_string(), vbin);
                cols.insert("binw".to_string(), vbinw);
                cols.insert("stein".to_string(), vstein);
                cols.insert("mass".to_string(), vmass.clone()); // cloned: COMPUTED block below reuses vmass
                cols.insert("libce".to_string(), vlibce);
                cols.insert("ent".to_string(), vent);
                cols.insert("gate".to_string(), vgate.clone()); // cloned: COMPUTED block below reuses vgate
                // S2 cleavage + Q6 entw + hand-set stumps (computed, additive).
                // fp LIVE when weights are loaded (trained MLP), else zeros.
                let (_qent_s, qent_exp) = query_entropy_info(&qc.w);
                let mut vcleav = Vec::with_capacity(cands.len());
                let mut vfp = Vec::with_capacity(cands.len());
                let mut ventw = Vec::with_capacity(cands.len());
                let mut vlgbma = Vec::with_capacity(cands.len());
                let mut vlgbmb = Vec::with_capacity(cands.len());
                let mut vlgbmc = Vec::with_capacity(cands.len());
                for (ci, c) in cands.iter().enumerate() {
                    let smi = table.get(&c.key).map(|e| e.smiles.as_str()).unwrap_or("");
                    let cs = cleavage_score(&qc.mz, &qc.w, smi);
                    let fs = match (&qfp, c.fp) {
                        (Some(q), Some(cf)) => chem_tanimoto(q, &cf),
                        _ => 0.0,
                    };
                    let es = entropy_grade(vgate[ci], qent_exp);
                    vcleav.push(cs);
                    vfp.push(fs);
                    ventw.push(es);
                    let (la, lb, lc) = lgbm_scores(vmass[ci], vgate[ci], cs, fs, es);
                    vlgbma.push(la);
                    vlgbmb.push(lb);
                    vlgbmc.push(lc);
                }
                cols.insert("cleav".to_string(), vcleav);
                cols.insert("fp".to_string(), vfp);
                cols.insert("entw".to_string(), ventw);
                cols.insert("lgbma".to_string(), vlgbma);
                cols.insert("lgbmb".to_string(), vlgbmb);
                cols.insert("lgbmc".to_string(), vlgbmc);
                // mass-replace + lossgate + swap are molecule-level (applied
                // after fuse); per-spectrum swap contributes OR.
                let neutrals_s: Vec<Option<f64>> = cands.iter().map(|c| c.neutral).collect();
                if mass_swap_flag(
                    cols.get("lib").map(|v| v.as_slice()).unwrap_or(&[]),
                    cols.get("mass").map(|v| v.as_slice()).unwrap_or(&[]),
                    qnm,
                    &neutrals_s,
                ) {
                    mass_swap_any = true;
                }
                per_spec.push(cols);
                let _ = qis.get(si);
            }
            let keys: Vec<String> = cands.iter().map(|c| c.key.clone()).collect();
            let mut ranked = fuse_molecule_rrf(&per_spec, &keys, fp_live);
            if params.mass_replace > 0.5 {
                // molecule qnm = median of available per-spectrum qnms.
                let mut qvs: Vec<f64> = qnms.iter().filter_map(|v| *v).filter(|v| *v > 0.0).collect();
                qvs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if !qvs.is_empty() {
                    let qmed = qvs[qvs.len() / 2];
                    let ppm_of: HashMap<String, f64> = cands
                        .iter()
                        .filter_map(|c| c.neutral.map(|n| (c.key.clone(), (n - qmed) / qmed * 1e6)))
                        .collect();
                    if !ppm_of.is_empty() {
                        apply_mass_replace(&mut ranked, &ppm_of);
                    }
                }
            }
            if params.lossgate > 0.5 {
                // gate = mean across spectra (pass-first); vacuous-pass preserved.
                let n = keys.len();
                let mut gmean = vec![1.0; n];
                for i in 0..n {
                    let mut s = 0.0;
                    let mut k = 0;
                    for cols in &per_spec {
                        if let Some(g) = cols.get("gate").and_then(|v| v.get(i)).copied() {
                            s += g;
                            k += 1;
                        }
                    }
                    if k > 0 {
                        gmean[i] = s / k as f64;
                    }
                }
                let gate_of: HashMap<String, f64> =
                    keys.iter().enumerate().map(|(i, kk)| (kk.clone(), gmean[i])).collect();
                for list in ranked.values_mut() {
                    partition_by_gate(list, &gate_of);
                }
            }
            let t_fuse = t0.elapsed();
            writeln!(
                h,
                "{}",
                serde_json::to_string(&EngineReply {
                    ranked,
                    pools: pool_union,
                    mass_swap: mass_swap_any,
                    prefilter_topk: None,
                    timing: vec![ms(t_parse), ms(t_qprep - t_parse), ms(t_cands - t_qprep), 0.0, 0.0, 0.0, 0.0, ms(t_fuse - t_cands)],
                })
                .unwrap()
            )
            .unwrap();
            h.flush().unwrap();
            let t_ser = t0.elapsed();
            writeln!(err, "sengine mol nspec={} cands={} pool={} ipc={} t_tot={:.1}", qs.len(), keys.len(), pool_union, ipc_bytes, ms(t_ser)).unwrap();
            continue;
        }
        // E1: qrow resolves the query spectrum from the engine-owned store
        // (no query-spectra JSON); q (full RawSpec) stays as the fallback.
        // qrow without params.store is a protocol error — fail loud.
        let q: RawSpec = match m.qrow {
            Some(r) => {
                let st = slim_store
                    .as_ref()
                    .expect("sengine: qrow without params.store");
                let (qm, qii) = slim::row_spec(st, r);
                RawSpec {
                    mzs: qm.to_vec(),
                    ints: qii.to_vec(),
                    prec: st.prec[r],
                    ce: slim::row_ce(st, r),
                }
            }
            None => m.q.expect("sengine: query without q"),
        };
        let t_parse = t0.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let qc = prepare(&q.mzs, &q.ints, q.prec, params.qpeak, params.qloss, q.ce);
        let qloss = loss_set(&qc.mz, qc.prec);
        let qn = qloss.len().max(1) as f64;
        let qi: i64 = m.qi.unwrap_or(0);
        let qnm: Option<f64> = m.qnm;
        // candidates (table lookups only — prepared once at preload)
        let t_qprep = t0.elapsed();
        let cand_keys = m.cand_keys.unwrap_or_default();
        let mut cands: Vec<CandData> = Vec::with_capacity(cand_keys.len());
        for k in &cand_keys {
            ensure_key(&mut table, slim_store.as_ref(), &slim_rows, &slim_pos, k);
            if let Some(e) = table.get_mut(k) {
                ensure_prepared(e, params.qpeak, params.qloss);
                cands.push(CandData {
                    key: k.clone(),
                    specs: e.specs.clone().unwrap_or_default(),
                    fp: if e.smiles.is_empty() { None } else { e.fp },
                    neutral: e.neutral,
                    formula: e.formula.clone(),
                });
            }
        }
        // pool (input order = distance order from the launcher; keep it).
        // Pool entry = first spectrum of the key — the same spectra the
        // launcher used to send per query as first-spectrum messages.
        let t_cands = t0.elapsed();
        let pool_keys = m.pool_keys.unwrap_or_default();
        let mut pool: Vec<(String, CSpectra, f64, [u64; 32])> = Vec::with_capacity(pool_keys.len());
        for k in &pool_keys {
            ensure_key(&mut table, slim_store.as_ref(), &slim_rows, &slim_pos, k);
            if let Some(e) = table.get_mut(k) {
                ensure_prepared(e, params.qpeak, params.qloss);
                if let Some(s) = e.specs.as_ref().and_then(|v| v.first()) {
                    let d = (s.prec - qc.prec).abs();
                    pool.push((k.clone(), s.clone(), d, e.fp.unwrap_or([0u64; 32])));
                }
            }
        }
        // funnel: jaccard over first-spectrum bits when the pool is big.
        // funnel_off = 1 disables it (I6 A/B: donor-retention question).
        let t_pool = t0.elapsed();
        let mut pool_idx: Vec<usize> = (0..pool.len()).collect();
        if pool.len() > FUNNEL_MIN && params.funnel_off < 0.5 && qc.bits.iter().any(|&w| w != 0) {
            let mut scored: Vec<(f64, f64, usize)> = pool
                .iter()
                .enumerate()
                .map(|(i, (_, s, dist, _))| (jaccard(&qc.bits, &s.bits), *dist, i))
                .collect();
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.1.partial_cmp(&b.1).unwrap())
            });
            pool_idx = scored.into_iter().take(FUNNEL_K).map(|(_, _, i)| i).collect();
        }
        if pool_idx.len() > POOL_CAP {
            pool_idx.truncate(POOL_CAP);
        }
        // I6 funnel recall probe: echo first-K pool keys in funnel order.
        let prefilter_topk: Option<Vec<String>> = if params.recall_topk > 0.0 {
            let ordered: Vec<String> = pool_idx.iter().map(|&i| pool[i].0.clone()).collect();
            Some(funnel_topk(&ordered, params.recall_topk as usize))
        } else {
            None
        };
        // analog hits: first-spectrum mass-shift matches, ms > thresh, top 10
        let t_funnel = t0.elapsed();
        let mut hits: Vec<([u64; 32], f64)> = Vec::new();
        for &i in &pool_idx {
            let (_, s, _, fp) = &pool[i];
            if fp == &[0u64; 32] {
                continue;
            }
            let ms = modified_row(&qc, s, TOL);
            if ms > MS_THRESH {
                hits.push((*fp, ms));
            }
        }
        hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        hits.truncate(TOP_N_ANALOG);
        // channels per candidate, accumulated in locals then inserted once
        // (was: a HashMap lookup per channel per candidate).
        let t_analog = t0.elapsed();
        let mut vlib: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vanalog: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vfrag: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vbin: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vbinw: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vstein: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vmass: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vlibce: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vent: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vgate: Vec<f64> = Vec::with_capacity(cands.len());
        for c in &cands {
            let mut lib = 0.0;
            let mut stein = 0.0;
            let mut ent = 0.0;
            for s in &c.specs {
                let v = cosine_row(&qc, s, TOL);
                if v > lib {
                    lib = v;
                }
            }
            // R2 CE band: same-band max; missing CE anywhere falls back to
            // lib, so the variant never scores below lib on missing data.
            let mut libce = lib;
            if let Some(qce) = qc.ce {
                let qb = ce_band(qce);
                let mut found = false;
                let mut best = 0.0;
                for s in &c.specs {
                    if s.ce.map(ce_band) == Some(qb) {
                        found = true;
                        let v = cosine_row(&qc, s, TOL);
                        if v > best {
                            best = v;
                        }
                    }
                }
                if found {
                    libce = best;
                }
            }
            let mut analog = 0.0;
            if let Some(cfp) = c.fp {
                for (ofp, ms) in &hits {
                    let v = ms * chem_tanimoto(&cfp, ofp);
                    if v > analog {
                        analog = v;
                    }
                }
            }
            let mut frag = 0.0;
            for s in c.specs.iter().take(2) {
                let v = overlap_count(&qloss, &s.floss) as f64 / qn;
                if v > frag {
                    frag = v;
                }
            }
            let (mut bscore, mut bwscore) = (0.0, 0.0);
            for s in &c.specs {
                let j = jaccard(&qc.bits, &s.bits);
                if j > bscore {
                    bscore = j;
                }
                let wj = wjaccard(&qc.wb, &s.wb);
                if wj > bwscore {
                    bwscore = wj;
                }
                let st = stein_row(&qc, s, TOL);
                if st > stein {
                    stein = st;
                }
                let en = ent_row(&qc.mz, &qc.w, &s.mz, &s.w, TOL);
                if en > ent {
                    ent = en;
                }
            }
            vlib.push(lib);
            vanalog.push(analog);
            vfrag.push(frag);
            vbin.push(bscore);
            vbinw.push(bwscore);
            vstein.push(stein);
            vmass.push(mass_score(qnm, c.neutral));
            vlibce.push(libce);
            vent.push(ent);
            vgate.push(gate_frac(&qc.mz, qc.prec, c.formula.as_ref()));
        }
        let mut cols: HashMap<String, Vec<f64>> = HashMap::with_capacity(VARIANTS.len() * 2);
        cols.insert("lib".to_string(), vlib);
        cols.insert("analog".to_string(), vanalog);
        cols.insert("frag".to_string(), vfrag);
        cols.insert("bin".to_string(), vbin);
        cols.insert("binw".to_string(), vbinw);
        cols.insert("stein".to_string(), vstein);
        cols.insert("mass".to_string(), vmass.clone()); // cloned: COMPUTED block below reuses vmass
        cols.insert("libce".to_string(), vlibce);
        cols.insert("ent".to_string(), vent);
        cols.insert("gate".to_string(), vgate.clone()); // cloned: COMPUTED block below reuses vgate
        // S2 cleavage + Q6 entw + hand-set stumps (computed, additive).
        // fp LIVE when weights are loaded (trained MLP), else zeros.
        // WIN-FIRST Phase 1: trained fp prediction for this query.
        let qfp: Option<[u64; FP_NWORDS]> = fp_model
            .as_ref()
            .map(|m| fp_forward(m, &bin_feats(&qc.mz, &qc.w, qc.prec)));
        let (_qent_s, qent_exp) = query_entropy_info(&qc.w);
        let mut vcleav = Vec::with_capacity(cands.len());
        let mut vfp = Vec::with_capacity(cands.len());
        let mut ventw = Vec::with_capacity(cands.len());
        let mut vlgbma = Vec::with_capacity(cands.len());
        let mut vlgbmb = Vec::with_capacity(cands.len());
        let mut vlgbmc = Vec::with_capacity(cands.len());
        for (ci, c) in cands.iter().enumerate() {
            let smi = table.get(&c.key).map(|e| e.smiles.as_str()).unwrap_or("");
            let cs = cleavage_score(&qc.mz, &qc.w, smi);
            let fs = match (&qfp, c.fp) {
                (Some(q), Some(cf)) => chem_tanimoto(q, &cf),
                _ => 0.0,
            };
            let es = entropy_grade(vgate[ci], qent_exp);
            vcleav.push(cs);
            vfp.push(fs);
            ventw.push(es);
            let (la, lb, lc) = lgbm_scores(vmass[ci], vgate[ci], cs, fs, es);
            vlgbma.push(la);
            vlgbmb.push(lb);
            vlgbmc.push(lc);
        }
        cols.insert("cleav".to_string(), vcleav);
        cols.insert("fp".to_string(), vfp);
        cols.insert("entw".to_string(), ventw);
        cols.insert("lgbma".to_string(), vlgbma);
        cols.insert("lgbmb".to_string(), vlgbmb);
        cols.insert("lgbmc".to_string(), vlgbmc);
        let keys: Vec<String> = cands.iter().map(|c| c.key.clone()).collect();
        let t_chan = t0.elapsed();
        let mut ranked = fuse_rank(&keys, &cols, &params);
        // Router: MONEY GATE first — strong library hit (>= 0.9 cosine)
        // locks router to lib order (fuse_rank already routed overlap to
        // cascade==lib on hits); otherwise the entropy choice over computed
        // evidence (sparse: mass+cleav; busy: gate+entw).
        {
            let libmax: f64 = cols
                .get("lib")
                .map(|v| {
                    v.iter()
                        .filter(|&&x| !x.is_nan())
                        .cloned()
                        .fold(f64::NAN, f64::max)
                })
                .unwrap_or(f64::NAN);
            if libmax.is_nan() || libmax < 0.9 {
                let empty_fp: &[f64] = &[];
                let order = entropy_router_order(
                    cols.get("mass").map(Vec::as_slice).unwrap_or(&[]),
                    cols.get("gate").map(Vec::as_slice).unwrap_or(&[]),
                    cols.get("cleav").map(Vec::as_slice).unwrap_or(&[]),
                    empty_fp,
                    cols.get("entw").map(Vec::as_slice).unwrap_or(&[]),
                    qent_exp,
                );
                let kv: Vec<String> = order
                    .into_iter()
                    .take(TOP_K_OUT)
                    .map(|i| keys[i].clone())
                    .collect();
                ranked.insert("router".to_string(), kv);
            }
        }
        // I6 mass-replace (engine follow-up): v8 rank-1-by-mass per ranked list.
        if params.mass_replace > 0.5 {
            if let Some(q) = qnm {
                if q > 0.0 {
                    let ppm_of: HashMap<String, f64> = cands
                        .iter()
                        .filter_map(|c| c.neutral.map(|n| (c.key.clone(), (n - q) / q * 1e6)))
                        .collect();
                    if !ppm_of.is_empty() {
                        apply_mass_replace(&mut ranked, &ppm_of);
                    }
                }
            }
        }
        // R2 loss-gate demotion: stable-partition each ranked list so
        // gate-passing keys keep order ahead of failing ones. Off = untouched.
        if params.lossgate > 0.5 {
            let gcol: &[f64] = cols.get("gate").map(Vec::as_slice).unwrap_or(&[]);
            let gate_of: HashMap<String, f64> = keys
                .iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), gcol.get(i).copied().unwrap_or(1.0)))
                .collect();
            for list in ranked.values_mut() {
                partition_by_gate(list, &gate_of);
            }
        }
        let neutrals: Vec<Option<f64>> = cands.iter().map(|c| c.neutral).collect();
        let mass_swap = mass_swap_flag(
            cols.get("lib").map(|v| v.as_slice()).unwrap_or(&[]),
            cols.get("mass").map(|v| v.as_slice()).unwrap_or(&[]),
            qnm,
            &neutrals,
        );
        let t_fuse = t0.elapsed();
        writeln!(
            h,
            "{}",
            serde_json::to_string(&EngineReply {
                ranked,
                pools: pool_idx.len(),
                mass_swap,
                prefilter_topk,
                // 8 stages, all known before serialization (ser+io stay on the
                // stderr line only). Order: parse,qprep,cands,pool,funnel,
                // analog,chan,fuse.
                timing: vec![
                    ms(t_parse),
                    ms(t_qprep - t_parse),
                    ms(t_cands - t_qprep),
                    ms(t_pool - t_cands),
                    ms(t_funnel - t_pool),
                    ms(t_analog - t_funnel),
                    ms(t_chan - t_analog),
                    ms(t_fuse - t_chan),
                ],
            })
            .unwrap()
        )
        .unwrap();
        h.flush().unwrap();
        let t_ser = t0.elapsed();
        writeln!(err, "sengine q={} cands={} pool={} ipc={} t_parse={:.1} t_qprep={:.1} t_cands={:.1} t_pool={:.1} t_funnel={:.1} t_analog={:.1} t_chan={:.1} t_fuse={:.1} t_ser={:.1} t_tot={:.1}", qi, keys.len(), pool_idx.len(), ipc_bytes, ms(t_parse), ms(t_qprep - t_parse), ms(t_cands - t_qprep), ms(t_pool - t_cands), ms(t_funnel - t_pool), ms(t_analog - t_funnel), ms(t_chan - t_analog), ms(t_fuse - t_chan), ms(t_ser - t_fuse), ms(t_ser))
            .unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "structs") {
        structs_main();
        return;
    }
    if args.iter().any(|a| a == "dump-fp") {
        dump_fp_main();
        return;
    }
    if args.iter().any(|a| a == "sengine") {
        let stdin = std::io::stdin();
        let mut buf = String::new();
        stdin
            .read_line(&mut buf)
            .expect("sengine: read params");
        #[derive(Deserialize)]
        struct ParamsIn {
            params: EngineParams,
        }
        let pin: ParamsIn = serde_json::from_str(&buf).expect("sengine: parse params");
        let stdout = std::io::stdout();
        writeln!(
            stdout.lock(),
            "{}",
            serde_json::to_string(&Ready { ready: true }).unwrap()
        )
        .unwrap();
        sengine(pin.params);
        return;
    }
    if let Some(pos) = args.iter().position(|a| a == "slimcheck") {
        // Equivalence probe for the slim port (test-then-swap): load the
        // store with the Rust loader and print the fingerprint + row bits.
        // `tests/test_slim_rust.py` compares every field against slim.py.
        let mut store: Option<String> = None;
        let mut row: Option<usize> = None;
        let mut it = args.iter().skip(pos + 1);
        while let Some(a) = it.next() {
            if a == "--store" {
                store = it.next().cloned();
            } else if a == "--row" {
                row = it.next().and_then(|v| v.parse().ok());
            }
        }
        let store = store.expect("slimcheck: missing --store DIR");
        let s = slim::load_slim(std::path::Path::new(&store));
        match row {
            Some(i) => println!("{}", slim::row_json(&s, i)),
            None => println!("{}", slim::slimcheck_json(&s)),
        }
        return;
    }
    eprintln!("usage: binvec <structs|sengine|dump-fp|slimcheck --store DIR [--row I]>");
    std::process::exit(2);
}

#[cfg(test)]
mod engine_tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn clean_cases() {
        let (m, w) = clean(&[100.0, 200.0, 300.0], &[1.0, 0.5, 0.25], 301.0);
        assert_eq!(m, vec![100.0, 200.0, 300.0]);
        assert!(approx(w[1], 0.7071067811865476));
        let (m, _) = clean(&[100.0, 200.0], &[1.0, 0.005], 201.0);
        assert_eq!(m, vec![100.0]);
        let (m, _) = clean(&[100.0, 500.0], &[1.0, 1.0], 201.0);
        assert_eq!(m, vec![100.0]);
        assert!(clean(&[], &[], 100.0).0.is_empty());
        assert!(clean(&[100.0], &[0.0], 101.0).0.is_empty());
        let big: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let ones = vec![1.0; 200];
        assert_eq!(clean(&big, &ones, 500.0).0.len(), 64);
    }

    #[test]
    fn quantize_matches_python() {
        // int(round(mz/0.05)) & 0xFFFFFFFF from the banked reference battery
        assert_eq!(qbin(100.0, 0.05), 2000);
        assert_eq!(qbin(200.5, 0.05), 4010);
        assert_eq!(qbin(350.25, 0.05), 7005);
        assert_eq!(qbin(400.0 - 100.0, 0.01), 30000);
        assert_eq!(qbin(400.0 - 200.5, 0.01), 19950);
        assert_eq!(qbin(400.0 - 350.25, 0.01), 4975);
    }

    #[test]
    fn adduct_table_matches_python() {
        assert!(approx(adduct_shift("[M+H]+").unwrap(), 1.007276));
        assert!(approx(adduct_shift("[M+Na]+").unwrap(), 22.989218));
        assert!(approx(adduct_shift("[M+NH4]+").unwrap(), 18.033823));
        assert!(approx(adduct_shift("[M+K]+").unwrap(), 38.963158));
        assert!(approx(adduct_shift("[M-H]-").unwrap(), -1.007276));
        assert!(approx(adduct_shift("[M+Cl]-").unwrap(), 34.968853));
        assert!(approx(adduct_shift("[M-H2O+H]+").unwrap(), -17.003289));
        assert!(approx(adduct_shift("[M-2H2O+H]+").unwrap(), -35.013854));
        assert!(approx(adduct_shift("[M-H2O-H]-").unwrap(), -19.017841));
        // R2: formate was 43.990901 (double-subtracted proton, ~1 Da off).
        assert!(approx(adduct_shift("[M+CH2O2-H]-").unwrap(), 44.998203));
        // Q1: HCOO alias resolves to the same shift.
        assert!(approx(adduct_shift("[M+HCOO]-").unwrap(), 44.998203));
        assert!(approx(adduct_shift("[M]+").unwrap(), 0.0));
        assert!(approx(500.0 - adduct_shift("[M+H]+").unwrap(), 498.992724));
        assert!(adduct_shift("[NOPE]").is_none());
    }

    #[test]
    fn ce_band_edges() {
        assert_eq!(ce_band(20.0), 0);
        assert_eq!(ce_band(25.0), 0);
        assert_eq!(ce_band(25.1), 1);
        assert_eq!(ce_band(40.0), 1);
        assert_eq!(ce_band(50.0), 1);
        assert_eq!(ce_band(50.1), 2);
        assert_eq!(ce_band(60.0), 2);
    }

    #[test]
    fn entropy_single_peak_quarter_and_self_is_one() {
        // single peak: S = 0 -> exponent 0.25.
        assert!(approx(entropy_exp(&[3.0]), 0.25));
        // empty/zero -> linear fallback.
        assert!(approx(entropy_exp(&[]), 1.0));
        // self-match is exactly 1.0 (same accumulation order both sides).
        let mz = vec![100.0, 200.0, 300.0];
        let w = vec![1.0, 0.5, 0.25];
        assert!(approx(ent_row(&mz, &w, &mz, &w, 0.01), 1.0));
        assert_eq!(ent_row(&[], &[], &mz, &w, 0.01), 0.0);
    }

    #[test]
    fn gate_vacuous_passes_and_subset_logic() {
        // no peaks -> vacuous 1.0, even without a formula.
        assert!(approx(gate_frac(&[], 200.0, None), 1.0));
        // hexose loss observed at prec - 162.052823.
        let qmz = vec![37.947177, 150.0];
        assert!(approx(gate_frac(&qmz, 200.0, None), 1.0)); // unknown = no gate
        let mut full = BTreeMap::new();
        full.insert("C".to_string(), 6);
        full.insert("H".to_string(), 12);
        full.insert("O".to_string(), 6);
        assert!(approx(gate_frac(&qmz, 200.0, Some(&full)), 1.0));
        let mut short = BTreeMap::new();
        short.insert("C".to_string(), 6);
        short.insert("H".to_string(), 12);
        short.insert("O".to_string(), 4); // hexose needs O5
        assert!(approx(gate_frac(&qmz, 200.0, Some(&short)), 0.0));
        // formula parser: ethanol counts heavy atoms + implicit H (3+2+1).
        let fc = formula_counts("CCO").unwrap();
        assert_eq!(fc.get("C"), Some(&2));
        assert_eq!(fc.get("O"), Some(&1));
        assert_eq!(fc.get("H"), Some(&6));
        assert_eq!(formula_counts(""), None);
    }

    #[test]
    fn lossgate_partition_stable_pass_first() {
        let mut list = vec!["b".to_string(), "a".to_string(), "c".to_string()];
        let mut g = HashMap::new();
        g.insert("a".to_string(), 1.0);
        g.insert("b".to_string(), 0.0);
        g.insert("c".to_string(), 1.0);
        partition_by_gate(&mut list, &g);
        assert_eq!(list, vec!["a".to_string(), "c".to_string(), "b".to_string()]);
        // missing key passes vacuously; empty list safe.
        let mut list2 = vec!["z".to_string()];
        partition_by_gate(&mut list2, &HashMap::new());
        assert_eq!(list2, vec!["z".to_string()]);
        let mut empty: Vec<String> = vec![];
        partition_by_gate(&mut empty, &g);
        assert!(empty.is_empty());
    }

    #[test]
    fn median_std_rank_match_pandas() {
        assert!(approx(median(vec![3.0, 1.0, 2.0]).unwrap(), 2.0));
        assert!(approx(median(vec![4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5));
        assert!(median(vec![]).is_none());
        let m = 2.0;
        assert!(approx(sample_std(&[3.0, 2.0, 1.0], m), 1.0));
        assert!(sample_std(&[5.0], 5.0).is_nan());
        assert_eq!(rank_min_desc(&[0.5, 0.9, 0.9, 0.1]), vec![3.0, 1.0, 1.0, 4.0]);
    }

    #[test]
    fn channels_match_python_reference() {
        // banked reference: q=(100,200) vs k1 identical, k2 far, k3 near-miss
        let q = prepare(&[100.0, 200.0], &[1.0, 1.0], 201.0, 0.05, 0.01, None);
        let k1 = prepare(&[100.0, 200.0], &[1.0, 1.0], 201.0, 0.05, 0.01, None);
        let k2 = prepare(&[150.0, 250.0], &[1.0, 1.0], 251.0, 0.05, 0.01, None);
        let k3 = prepare(&[100.05, 300.0], &[0.9, 0.8], 301.0, 0.05, 0.01, None);
        assert!(approx(cosine_row(&q, &k1, TOL), 1.0));
        assert!(approx(cosine_row(&q, &k2, TOL), 0.0));
        let ql = loss_set(&q.mz, q.prec);
        let f1 = loss_set(&k1.mz, k1.prec);
        let f3 = loss_set(&k3.mz, k3.prec);
        let qn = ql.len().max(1) as f64;
        assert!(approx(overlap_count(&ql, &f1) as f64 / qn, 1.0));
        // frag k3 == 0.5 in the banked reference
        assert!(approx(overlap_count(&ql, &f3) as f64 / qn, 0.5));
        // bin self-channel is exact Jaccard 1.0
        assert!(approx(jaccard(&q.bits, &k1.bits), 1.0));
        assert!(approx(wjaccard(&q.wb, &k1.wb), 1.0));
    }

    #[test]
    fn z_rrf_match_pandas() {
        // banked: lib [0.9,0.5,0.0] etc. pandas rank(ascending=False, method="min"),
        // RRF base 60: 1/(60+rank) summed over 3 channels.
        // a: ranks 1,2,2 -> 1/61+1/62+1/62 = 0.048651507139079855
        // b: ranks 2,1,2 -> same 0.048651507139079855
        // c: ranks 3,3,1 -> 1/63+1/63+1/61 = 0.04813947436898257
        let lib = vec![0.9, 0.5, 0.0];
        let analog = vec![0.2, 0.8, 0.1];
        let frag = vec![0.3, 0.3, 0.9];
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("lib".to_string(), lib.clone());
        cols.insert("analog".to_string(), analog.clone());
        cols.insert("frag".to_string(), frag.clone());
        cols.insert("bin".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("binw".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("stein".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        // exact RRF scores via the same rank helper (locks base + min-rank)
        let rrf_of = |chs: &[Vec<f64>]| -> Vec<f64> {
            let rs: Vec<Vec<f64>> = chs.iter().map(|c| rank_min_desc(c)).collect();
            (0..3)
                .map(|i| {
                    rs.iter()
                        .map(|r| 1.0 / (RRF_BASE + r[i]))
                        .sum()
                })
                .collect()
        };
        let rrf = rrf_of(&[lib, analog, frag]);
        assert!((rrf[0] - 0.048651507139079855).abs() < 1e-12);
        assert!((rrf[1] - 0.048651507139079855).abs() < 1e-12);
        assert!((rrf[2] - 0.04813947436898257).abs() < 1e-12);
        let fw = EngineParams::neutral(0.05, 0.01);
        let ranked = fuse_rank(&keys, &cols, &fw);
        // neutral genome reproduces legacy fusions bit-identically
        assert_eq!(ranked["zw"], ranked["z"]);
        assert_eq!(ranked["rrfw"], ranked["rrf"]);
        let z = &ranked["z"];
        assert_eq!(z[0], "b"); // banked z: b=0.6412 > a=-0.0566 > c=-0.5846
        // rrf top-25 contains all three, ties keep candidate order
        assert_eq!(ranked["rrf"].len(), 3);
        // rrf order: a,b tie on 0.04865... -> stable candidate order keeps a first
        assert_eq!(ranked["rrf"][0], "a");
        assert_eq!(ranked["rrf"][1], "b");
        assert_eq!(ranked["rrf"][2], "c");
    }

    #[test]
    fn pure_fusion_sums_channels() {
        // pure = z(stein) + z(frag) + z(mass), equal weights, NaN-skipped.
        // stein [0.9,0.5,0.0] -> z [0.9610,0.0739,-1.0349]
        // frag  [0.3,0.3,0.9] -> z [-0.5774,-0.5774,1.1547]
        // mass  [1.0,0.5,0.0] -> z [1.0,0.0,-1.0]
        // pure = [1.3836,-0.5035,-0.8802] -> order a,b,c.
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("lib".to_string(), vec![0.9, 0.5, 0.0]);
        cols.insert("analog".to_string(), vec![0.2, 0.8, 0.1]);
        cols.insert("frag".to_string(), vec![0.3, 0.3, 0.9]);
        cols.insert("bin".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("binw".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("stein".to_string(), vec![0.9, 0.5, 0.0]);
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        let fw = EngineParams::neutral(0.05, 0.01);
        let ranked = fuse_rank(&keys, &cols, &fw);
        assert_eq!(ranked["pure"], vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn lazy_prep_matches_eager_and_caches() {
        // M209: prepare-on-first-use == eager values bit-for-bit; raw dropped,
        // second call a no-op; empty-smiles fp is zeros (pool skips it).
        let raws = vec![
            RawSpec { mzs: vec![100.0, 200.0], ints: vec![1.0, 0.5], prec: 201.0, ce: None },
            RawSpec { mzs: vec![150.0], ints: vec![1.0], prec: 251.0, ce: None },
        ];
        let eager: Vec<CSpectra> = raws
            .iter()
            .map(|s| prepare(&s.mzs, &s.ints, s.prec, 0.05, 0.01, None))
            .collect();
        let mut e = TableEntry {
            raw: raws, specs: None, smiles: "CCO".to_string(), fp: None,
            neutral: Some(200.0), formula: None,
        };
        ensure_prepared(&mut e, 0.05, 0.01);
        assert!(e.raw.is_empty(), "raw dropped after prepare");
        let got = e.specs.clone().unwrap();
        assert_eq!(got.len(), eager.len());
        for (a, b) in got.iter().zip(eager.iter()) {
            assert_eq!(a.mz, b.mz);
            assert_eq!(a.w, b.w);
            assert_eq!(a.prec, b.prec);
            assert_eq!(a.norm, b.norm);
            assert_eq!(a.sw, b.sw);
            assert_eq!(a.snorm, b.snorm);
            assert_eq!(a.bits, b.bits);
            assert_eq!(a.wb, b.wb);
            assert_eq!(a.floss, b.floss);
        }
        assert_eq!(e.fp, Some(morgan_bits_2048("CCO")));
        let fp_before = e.fp;
        ensure_prepared(&mut e, 0.05, 0.01);
        assert_eq!(e.fp, fp_before, "second call is a no-op");
        assert_eq!(e.specs.clone().unwrap().len(), eager.len());
        let mut e2 = TableEntry {
            raw: vec![], specs: None, smiles: String::new(), fp: None,
            neutral: None, formula: None,
        };
        ensure_prepared(&mut e2, 0.05, 0.01);
        assert_eq!(e2.fp, Some([0u64; 32]));
        assert!(e2.specs.unwrap().is_empty());
    }

    #[test]
    fn weighted_genome_moves_rankings() {
        // genome: per-channel z + RRF weights, cascade/router gates (defaults neutral).
        // lib max 0.8 stays below the MONEY GATE (0.9) so the cascade-gate leg
        // flip is exercised; 0.9 would hit-lock cascade to lib on both legs.
        let lib = vec![0.8, 0.5, 0.0];
        let analog = vec![0.2, 0.8, 0.1];
        let frag = vec![0.3, 0.3, 0.9];
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("lib".to_string(), lib);
        cols.insert("analog".to_string(), analog);
        cols.insert("frag".to_string(), frag);
        cols.insert("bin".to_string(), vec![0.1, 0.2, 0.3]);
        cols.insert("binw".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("stein".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        // analog-heavy genome (novel-lane shape): zw top flips from b to a... verify known flip:
        // neutral zw top is b (banked z: b=0.6412 first). lib-heavy genome keeps b.
        let mut fw = EngineParams::neutral(0.05, 0.01);
        assert_eq!(fuse_rank(&keys, &cols, &fw)["zw"][0], "b");
        fw.wl = 3.0;
        fw.wa = 0.1;
        fw.wf = 0.1;
        assert_eq!(fuse_rank(&keys, &cols, &fw)["zw"][0], "a");
        // cascade gate: default 3.0 -> analog leg (max libz ~1.0 < 3); gate 0.0 -> lib leg
        let mut fw2 = EngineParams::neutral(0.05, 0.01);
        let leg_default = fuse_rank(&keys, &cols, &fw2)["cascade"].clone();
        fw2.casc_z = 0.0;
        let leg_zero = fuse_rank(&keys, &cols, &fw2)["cascade"].clone();
        assert_ne!(leg_default, leg_zero);
        // rrfw neutral == rrf; analog-heavy rrfw flips c (frag-best) down
        let fw3 = EngineParams::neutral(0.05, 0.01);
        assert_eq!(fuse_rank(&keys, &cols, &fw3)["rrfw"], fuse_rank(&keys, &cols, &fw3)["rrf"]);
    }

    #[test]
    fn stein_mass_weighted_self_and_far() {
        // Stein-Scott-lite: self 1.0, far 0.0, and high-mass matches outweigh low-mass.
        let q = prepare(&[100.0, 900.0], &[1.0, 1.0], 901.0, 0.05, 0.01, None);
        let same = prepare(&[100.0, 900.0], &[1.0, 1.0], 901.0, 0.05, 0.01, None);
        let far = prepare(&[150.0, 250.0], &[1.0, 1.0], 251.0, 0.05, 0.01, None);
        assert!((stein_row(&q, &same, TOL) - 1.0).abs() < 1e-9);
        assert_eq!(stein_row(&q, &far, TOL), 0.0);
        // low-only vs high-only match: high-mass partial scores higher than low-mass partial
        let low = prepare(&[100.0, 500.0], &[1.0, 1.0], 501.0, 0.05, 0.01, None);
        let high = prepare(&[900.0, 500.0], &[1.0, 1.0], 901.0, 0.05, 0.01, None);
        assert!(stein_row(&q, &high, TOL) > stein_row(&q, &low, TOL));
        // cosine ties single-peak matches (both 0.5); stein separates by mass
        assert!((cosine_row(&q, &high, TOL) - cosine_row(&q, &low, TOL)).abs() < 1e-9);
        assert!(stein_row(&q, &high, TOL) > stein_row(&q, &low, TOL));
    }

    #[test]
    fn mass_grading_v8_numbers() {
        // v8 Version-22: CENTER -1.5, FREE 3.0, FULL 8.0.
        assert_eq!(mass_score(Some(500.0), Some(500.0)), 1.0); // -0.0ppm... within free
        assert_eq!(mass_score(Some(1000.0), Some(1000.0)), 1.0);
        // +8ppm (1000.008) => dev |8-(-1.5)|=9.5 > FULL 8 => 0.0
        assert_eq!(mass_score(Some(1000.0), Some(1000.008)), 0.0);
        // +5ppm (1000.005) => dev 6.5 -> 1-(6.5-3)/5 = 0.3
        assert!((mass_score(Some(1000.0), Some(1000.005)) - 0.3).abs() < 1e-9);
        assert_eq!(mass_score(None, Some(500.0)), 0.0);
        assert_eq!(mass_score(Some(500.0), None), 0.0);
        // swap flag: lib-leader off-mass + weak lib => true; strong lib => false
        assert!(mass_swap_flag(&[0.5], &[0.0], Some(1000.0), &[Some(1020.0)]));
        assert!(!mass_swap_flag(&[0.95], &[0.0], Some(1000.0), &[Some(1020.0)]));
        assert!(!mass_swap_flag(&[0.5], &[1.0], Some(1000.0), &[Some(1000.0)]));
        assert!(!mass_swap_flag(&[0.5], &[0.0], None, &[Some(1020.0)]));
    }

    #[test]
    fn mass_replace_moves_bad_leader() {
        // I6: rank-1 |ppm-CENTER| = |20-(-1.5)| > 5 -> min-|ppm| key fronts.
        let mut ranked: HashMap<String, Vec<String>> = HashMap::new();
        ranked.insert("z".to_string(), vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let ppm: HashMap<String, f64> =
            [("a".to_string(), 20.0), ("b".to_string(), 1.0), ("c".to_string(), 2.0)]
                .into_iter()
                .collect();
        apply_mass_replace(&mut ranked, &ppm);
        assert_eq!(ranked["z"], vec!["b".to_string(), "a".to_string(), "c".to_string()]);
        // leader already best -> untouched.
        let mut ranked2: HashMap<String, Vec<String>> = HashMap::new();
        ranked2.insert("z".to_string(), vec!["b".to_string(), "a".to_string()]);
        apply_mass_replace(&mut ranked2, &ppm);
        assert_eq!(ranked2["z"], vec!["b".to_string(), "a".to_string()]);
        // no ppms -> no-op, never panics.
        let mut ranked3: HashMap<String, Vec<String>> = HashMap::new();
        ranked3.insert("z".to_string(), vec!["a".to_string()]);
        apply_mass_replace(&mut ranked3, &HashMap::new());
        assert_eq!(ranked3["z"], vec!["a".to_string()]);
    }

    #[test]
    fn funnel_topk_slices_in_order() {
        let keys: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(funnel_topk(&keys, 2), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(funnel_topk(&keys, 0), Vec::<String>::new());
        assert_eq!(funnel_topk(&keys, 99).len(), 3);
    }

    #[test]
    fn analog_empty_pool_is_zero() {
        // TRULY far pool (100 Da away): neither direct nor shifted matches.
        // (An earlier revision of this test used prec 251 for `far` and
        // expected modified==0 — wrong: 251 is 50 Da away, i.e. INSIDE the
        // analog pool, and shifting it by (201-251)=-50 aligns it exactly,
        // so modified==1.0 is the CORRECT analog signal, not a false hit.
        // The runner caught the bad expectation; this is the corrected one.)
        let q = prepare(&[100.0, 200.0], &[1.0, 1.0], 201.0, 0.05, 0.01, None);
        let far = prepare(&[150.0, 250.0], &[1.0, 1.0], 301.0, 0.05, 0.01, None);
        assert_eq!(modified_row(&q, &far, TOL), 0.0);
        assert_eq!(cosine_row(&q, &far, TOL), 0.0);
        // in-pool shifted matches DO hit — that is the analog signal itself.
        let near = prepare(&[150.0, 250.0], &[1.0, 1.0], 251.0, 0.05, 0.01, None);
        assert_eq!(modified_row(&q, &near, TOL), 1.0);
        assert_eq!(cosine_row(&q, &near, TOL), 0.0);
    }

    #[test]
    fn molecule_rrf_single_equals_rank_and_fuses() {
        // S=1: RRF over one spectrum is monotonic in rank -> same order as
        // order_desc on the raw scores (NaN last in both).
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("lib".to_string(), vec![0.9, 0.5, 0.0]);
        cols.insert("analog".to_string(), vec![0.2, 0.8, 0.1]);
        cols.insert("frag".to_string(), vec![0.3, 0.3, 0.9]);
        cols.insert("bin".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("binw".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("stein".to_string(), vec![0.0, 0.0, 0.0]);
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        cols.insert("libce".to_string(), vec![0.9, 0.5, 0.0]);
        cols.insert("ent".to_string(), vec![0.9, 0.5, 0.0]);
        cols.insert("gate".to_string(), vec![1.0, 1.0, 1.0]);
        let fused = fuse_molecule_rrf(&[cols], &keys, false);
        // lib single: a(0.9) b(0.5) c(0.0) -> a,b,c.
        assert_eq!(fused["lib"], vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        // S=2 agreeing spectra keep the winner: both rank a first.
        let mut c2 = HashMap::new();
        for v in VARIANTS {
            c2.insert(v.to_string(), vec![0.9, 0.1, 0.0]);
        }
        let mut c3 = HashMap::new();
        for v in VARIANTS {
            c3.insert(v.to_string(), vec![0.8, 0.2, 0.0]);
        }
        let fused2 = fuse_molecule_rrf(&[c2, c3], &keys, false);
        assert_eq!(fused2["lib"][0], "a");
        // fp quarantined: empty, never fused.
        assert!(fused["fp"].is_empty());
        assert!(fused2["fp"].is_empty());
        // All-NaN stays empty-order (NaN last, stable).
        let mut cn = HashMap::new();
        for v in VARIANTS {
            cn.insert(v.to_string(), vec![f64::NAN; 3]);
        }
        let fusedn = fuse_molecule_rrf(&[cn], &keys, false);
        assert_eq!(fusedn["lib"].len(), 3);
    }

    #[test]
    fn new_channels_helpers() {
        // cleavage on ethanol non-empty; fp deterministic; entropy sane; lgbm monotone.
        let mz = vec![100.0, 150.0];
        let w = vec![1.0, 1.0];
        assert!(cleavage_score(&mz, &w, "CCO") >= 0.0);
        assert_eq!(cleavage_score(&mz, &w, ""), 0.0);
        let cf = Some(chem::morgan_bits_2048("CCO"));
        assert!(fp_score(&mz, 200.0, Some(199.0), cf.as_ref()) >= 0.0);
        assert_eq!(fp_score(&[], 200.0, None, cf.as_ref()), 0.0);
        let (s, e) = query_entropy_info(&[1.0, 1.0]);
        assert!(s > 0.0 && e > 0.25 && e <= 1.0);
        // router sparse vs busy both return full order
        let mass = vec![1.0, 0.5, 0.0];
        let gate = vec![0.0, 1.0, 0.5];
        let ord_s = entropy_router_order(&mass, &gate, &mass, &gate, &gate, 0.3);
        let ord_b = entropy_router_order(&mass, &gate, &mass, &gate, &gate, 0.9);
        assert_eq!(ord_s.len(), 3);
        assert_eq!(ord_b.len(), 3);
    }

    #[test]
    fn fp_bins_match_train_features() {
        // bin_feats == train_fp.py featurize: max cleaned-w per 1Da bin.
        // (clean stores sqrt; both sides apply exactly one sqrt.)
        let x = bin_feats(&[100.0, 100.4, 2000.0], &[1.0, 0.25, 1.0], 1000.0);
        assert_eq!(x.len(), FP_INDIM);
        assert!((x[100] - 1.0).abs() < 1e-6); // max w at mz 100
        assert!(x[0] == 0.0);
        assert!(x[FP_NBIN - 1] == 0.0); // mz 2000 out of range
        assert!((x[FP_NBIN] - 0.5).abs() < 1e-9); // 1000/2000
    }

    #[test]
    fn fp_forward_toy_and_loader_rejects_garbage() {
        // toy: in=FP_INDIM, hid=1, out=2048: h=relu(x0), y=sigmoid(+-h).
        let mut w1 = vec![0f32; FP_INDIM];
        w1[0] = 1.0;
        let mut w2 = vec![0f32; 2048];
        w2[0] = 1.0;
        w2[1] = -1.0;
        let m = FpModel { hid: 1, w1, b1: vec![0.0], w2, b2: vec![0f32; 2048] };
        let mut x = vec![0f32; FP_INDIM];
        x[0] = 2.0;
        let out = fp_forward(&m, &x);
        // h=relu(2)=2; y0=sigmoid(2)>0.5 -> bit0 set; y1=sigmoid(-2)<0.5 clear.
        assert_eq!(out[0] & 1, 1);
        assert_eq!((out[0] >> 1) & 1, 0);
        assert!(out[1..].iter().all(|&w| w == 0));
        // loader rejects missing files and bad magic.
        assert!(load_fp_model("/nonexistent/fp_mlp.bin").is_none());
    }
}
