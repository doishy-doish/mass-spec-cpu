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

use chem::{morgan_bits_2048, tanimoto as chem_tanimoto};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};

#[allow(dead_code)]
mod boundary;

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
/// Cross-query prepared-spectra cache: same library spectra repeat across
/// queries (pools overlap heavily), but each query used to re-prepare every
/// candidate + pool spectrum from scratch. Key (key, pos) is a sound
/// identity: per-key spec order is deterministic within a run (stable index
/// build), and the pool takes position 0. Bounded: spectra are ~2KB, so the
/// cap is ~100MB per worker; on overflow the cache clears (correct, just
/// recompute churn — tuning owed per the MASS-DEATHS lesson).
const PREP_CACHE_CAP: usize = 50_000;const VARIANTS: [&str; 15] = [
    "lib", "analog", "frag", "bin", "binw", "stein", "mass", "z", "rrf", "rrfb", "cascade", "router",
    "zw", "rrfw", "pure",
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
        "[M+CH2O2-H]-" => Some(43.990901),
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
}

fn prepare(mzs: &[f64], ints: &[f64], prec: f64, qpeak: f64, qloss: f64) -> CSpectra {
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
    CSpectra { mz, w, prec, norm, sw, snorm, bits, wb, floss }
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
        }
    }
}

#[derive(Deserialize)]
struct RawSpec {
    mzs: Vec<f64>,
    ints: Vec<f64>,
    prec: f64,
}

#[derive(Deserialize)]
struct CandSpec {
    key: String,
    smiles: String,
    specs: Vec<RawSpec>,
    #[serde(default)]
    neutral: Option<f64>,
}

#[derive(Deserialize)]
struct PoolSpec {
    key: String,
    smiles: String,
    mzs: Vec<f64>,
    ints: Vec<f64>,
    prec: f64,
}

#[derive(Deserialize)]
struct EngineQuery {
    q: RawSpec,
    cands: Vec<CandSpec>,
    pool: Vec<PoolSpec>,
    #[serde(default)]
    qi: i64,
    #[serde(default)]
    qnm: Option<f64>,
}

#[derive(Serialize)]
struct EngineReply {
    ranked: HashMap<String, Vec<String>>,
    pools: usize,
    /// v8 rank-1-by-mass: lib-leader |ppm-CENTER|>5.0 && lib<0.9 would swap. Info only.
    mass_swap: bool,
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
}

fn fuse_rank(
    keys: &[String],
    cols: &HashMap<String, Vec<f64>>,
    fw: &EngineParams,
) -> HashMap<String, Vec<String>> {
    let n = keys.len();
    // borrow channels in place (was: one Vec clone per channel here, plus
    // another per rrf/rrfb/rrfw call below). Missing channel == all-NaN,
    // exactly like the old fallback.
    let nan = vec![f64::NAN; n];
    let col = |c: &str| -> &[f64] { cols.get(c).map(Vec::as_slice).unwrap_or(&nan) };
    let lib = col("lib");
    let analog = col("analog");
    let frag = col("frag");
    let bin = col("bin");
    let binw = col("binw");
    let stein = col("stein"); // standalone (Loop-2 replication): ranked, NOT fused yet
    let mass = col("mass"); // standalone (Loop-3 v8-mass-lite): ranked, NOT fused yet
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
    // All three already exist as standalone channels; this only fuses them.
    // Predicted class2 N=10: lands between frag and stein (~0.7-0.8); measures
    // whether the combo beats stein alone. N=10 is smoke + direction, fit nothing.
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
    // ranks computed ONCE per channel and shared by rrf/rrfb/rrfw (the old
    // code re-ranked lib/analog/frag three times each — same inputs, so the
    // shared ranks are bit-identical, at ~1/3 the rank cost).
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
    let cascade: Vec<f64> = if max_skipnan(&libz) > fw.casc_z {
        libz.clone()
    } else {
        analogz.clone()
    };
    let overlap = max_skipnan(&lib) > fw.rout_lib
        || max_skipnan(&bin) > fw.rout_bin
        || max_skipnan(&binw) > fw.rout_binw;
    let router = if overlap { cascade.clone() } else { analog.to_vec() };
    let mut out = HashMap::new();
    // borrows, not clones: every output here is read-only from here on
    // (was: a Vec clone per channel, including a redundant mass.clone()).
    let named: Vec<(&str, &[f64])> = vec![
        ("lib", lib),
        ("analog", analog),
        ("frag", frag),
        ("bin", bin),
        ("binw", binw),
        ("stein", stein),
        ("mass", mass),
        ("z", &z),
        ("rrf", &rrf),
        ("rrfb", &rrfb),
        ("cascade", &cascade),
        ("router", &router),
        ("zw", &zw),
        ("rrfw", &rrfw),
        ("pure", &pure),
    ];
    for (name, vals) in named {
        let kv: Vec<String> = order_desc(vals)
            .into_iter()
            .take(TOP_K_OUT)
            .map(|i| keys[i].clone())
            .collect();
        out.insert(name.to_string(), kv);
    }
    out
}

/// v8 rank-1-by-mass (info only): lib-leader |ppm-CENTER|>LEAD_OFF && lib<TAU.
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

fn sengine(params: EngineParams) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let mut lines = stdin.lock().lines();
    let e = std::io::stderr();
    let mut err = e.lock();
    // cross-query fingerprint cache: same structure keys repeat across queries
    // (pool overlap); Morgan is ~10us with format!/allocs per key, so caching
    // across the whole run beats the old per-query rebuild by ~5-10x.
    // Correctness: fp depends only on smiles, which is fixed per key.
    // If a key ever arrives with a different smiles, first-wins (data keys are stable).
    let mut fp_cache: HashMap<String, [u64; 32]> = HashMap::new();
    // prepared spectra are covered by PREP_CACHE_CAP (see const): same
    // (key, pos) pairs repeat across queries, so hits skip prepare() whole.
    let mut prep_cache: HashMap<(String, usize), CSpectra> = HashMap::new();
    for line in lines {
        let line = line.expect("sengine: read");
        if line.trim().is_empty() {
            continue;
        }
        let t0 = std::time::Instant::now();
        let ipc_bytes = line.len();
        let q: EngineQuery = serde_json::from_str(&line).expect("sengine: parse query");
        let t_parse = t0.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let qc = prepare(&q.q.mzs, &q.q.ints, q.q.prec, params.qpeak, params.qloss);
        let qloss = loss_set(&qc.mz, qc.prec);
        let qn = qloss.len().max(1) as f64;
        // candidates
        let t_qprep = t0.elapsed();
        let mut cands: Vec<CandData> = Vec::with_capacity(q.cands.len());
        for c in &q.cands {
            let fp = fp_cache
                .entry(c.key.clone())
                .or_insert_with(|| morgan_bits_2048(&c.smiles));
            let specs = c
                .specs
                .iter()
                .enumerate()
                .map(|(pi, s)| {
                    let ck = (c.key.clone(), pi);
                    if let Some(hit) = prep_cache.get(&ck) {
                        return hit.clone();
                    }
                    let cs = prepare(&s.mzs, &s.ints, s.prec, params.qpeak, params.qloss);
                    if prep_cache.len() > PREP_CACHE_CAP {
                        prep_cache.clear();
                    }
                    prep_cache.insert(ck, cs.clone());
                    cs
                })
                .collect();
            cands.push(CandData {
                key: c.key.clone(),
                specs,
                fp: if c.smiles.is_empty() { None } else { Some(*fp) },
                neutral: c.neutral,
            });
        }
        // pool (input order = distance order from the launcher; keep it)
        let t_cands = t0.elapsed();
        let mut pool: Vec<(String, CSpectra, f64, [u64; 32])> = Vec::with_capacity(q.pool.len());
        for p in &q.pool {
            // pool first-spectrum == candidate position 0 for the same key,
            // so pool hits share the candidate cache entries.
            let ck = (p.key.clone(), 0usize);
            let s = if let Some(hit) = prep_cache.get(&ck) {
                hit.clone()
            } else {
                let cs = prepare(&p.mzs, &p.ints, p.prec, params.qpeak, params.qloss);
                if prep_cache.len() > PREP_CACHE_CAP {
                    prep_cache.clear();
                }
                prep_cache.insert(ck, cs.clone());
                cs
            };
            let fp = *fp_cache
                .entry(p.key.clone())
                .or_insert_with(|| morgan_bits_2048(&p.smiles));
            pool.push((p.key.clone(), s, (p.prec - qc.prec).abs(), fp));
        }
        // funnel: jaccard over first-spectrum bits when the pool is big
        let t_pool = t0.elapsed();
        let mut pool_idx: Vec<usize> = (0..pool.len()).collect();
        if pool.len() > FUNNEL_MIN && qc.bits.iter().any(|&w| w != 0) {
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
        for c in &cands {
            let mut lib = 0.0;
            let mut stein = 0.0;
            for s in &c.specs {
                let v = cosine_row(&qc, s, TOL);
                if v > lib {
                    lib = v;
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
            }
            vlib.push(lib);
            vanalog.push(analog);
            vfrag.push(frag);
            vbin.push(bscore);
            vbinw.push(bwscore);
            vstein.push(stein);
            vmass.push(mass_score(q.qnm, c.neutral));
        }
        let mut cols: HashMap<String, Vec<f64>> = HashMap::with_capacity(VARIANTS.len() * 2);
        cols.insert("lib".to_string(), vlib);
        cols.insert("analog".to_string(), vanalog);
        cols.insert("frag".to_string(), vfrag);
        cols.insert("bin".to_string(), vbin);
        cols.insert("binw".to_string(), vbinw);
        cols.insert("stein".to_string(), vstein);
        cols.insert("mass".to_string(), vmass);
        let keys: Vec<String> = cands.iter().map(|c| c.key.clone()).collect();
        let t_chan = t0.elapsed();
        let ranked = fuse_rank(&keys, &cols, &params);
        let neutrals: Vec<Option<f64>> = q.cands.iter().map(|c| c.neutral).collect();
        let mass_swap = mass_swap_flag(
            cols.get("lib").map(|v| v.as_slice()).unwrap_or(&[]),
            cols.get("mass").map(|v| v.as_slice()).unwrap_or(&[]),
            q.qnm,
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
            })
            .unwrap()
        )
        .unwrap();
        h.flush().unwrap();
        let t_ser = t0.elapsed();
        writeln!(err, "sengine q={} cands={} pool={} ipc={} t_parse={:.1} t_qprep={:.1} t_cands={:.1} t_pool={:.1} t_funnel={:.1} t_analog={:.1} t_chan={:.1} t_fuse={:.1} t_ser={:.1} t_tot={:.1}", q.qi, keys.len(), pool_idx.len(), ipc_bytes, ms(t_parse), ms(t_qprep - t_parse), ms(t_cands - t_qprep), ms(t_pool - t_cands), ms(t_funnel - t_pool), ms(t_analog - t_funnel), ms(t_chan - t_analog), ms(t_fuse - t_chan), ms(t_ser - t_fuse), ms(t_ser))
            .unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "structs") {
        structs_main();
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
    eprintln!("usage: binvec <structs|sengine>");
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
        assert!(approx(adduct_shift("[M-2H2O+H]+").unwrap(), -35.013854));
        assert!(approx(adduct_shift("[M-H2O-H]-").unwrap(), -19.017841));
        assert!(approx(500.0 - adduct_shift("[M+H]+").unwrap(), 498.992724));
        assert!(adduct_shift("[NOPE]").is_none());
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
        let q = prepare(&[100.0, 200.0], &[1.0, 1.0], 201.0, 0.05, 0.01);
        let k1 = prepare(&[100.0, 200.0], &[1.0, 1.0], 201.0, 0.05, 0.01);
        let k2 = prepare(&[150.0, 250.0], &[1.0, 1.0], 251.0, 0.05, 0.01);
        let k3 = prepare(&[100.05, 300.0], &[0.9, 0.8], 301.0, 0.05, 0.01);
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
    fn weighted_genome_moves_rankings() {
        // genome: per-channel z + RRF weights, cascade/router gates (defaults neutral).
        let lib = vec![0.9, 0.5, 0.0];
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
        let q = prepare(&[100.0, 900.0], &[1.0, 1.0], 901.0, 0.05, 0.01);
        let same = prepare(&[100.0, 900.0], &[1.0, 1.0], 901.0, 0.05, 0.01);
        let far = prepare(&[150.0, 250.0], &[1.0, 1.0], 251.0, 0.05, 0.01);
        assert!((stein_row(&q, &same, TOL) - 1.0).abs() < 1e-9);
        assert_eq!(stein_row(&q, &far, TOL), 0.0);
        // low-only vs high-only match: high-mass partial scores higher than low-mass partial
        let low = prepare(&[100.0, 500.0], &[1.0, 1.0], 501.0, 0.05, 0.01);
        let high = prepare(&[900.0, 500.0], &[1.0, 1.0], 901.0, 0.05, 0.01);
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
    fn analog_empty_pool_is_zero() {
        // TRULY far pool (100 Da away): neither direct nor shifted matches.
        // (An earlier revision of this test used prec 251 for `far` and
        // expected modified==0 — wrong: 251 is 50 Da away, i.e. INSIDE the
        // analog pool, and shifting it by (201-251)=-50 aligns it exactly,
        // so modified==1.0 is the CORRECT analog signal, not a false hit.
        // The runner caught the bad expectation; this is the corrected one.)
        let q = prepare(&[100.0, 200.0], &[1.0, 1.0], 201.0, 0.05, 0.01);
        let far = prepare(&[150.0, 250.0], &[1.0, 1.0], 301.0, 0.05, 0.01);
        assert_eq!(modified_row(&q, &far, TOL), 0.0);
        assert_eq!(cosine_row(&q, &far, TOL), 0.0);
        // in-pool shifted matches DO hit — that is the analog signal itself.
        let near = prepare(&[150.0, 250.0], &[1.0, 1.0], 251.0, 0.05, 0.01);
        assert_eq!(modified_row(&q, &near, TOL), 1.0);
        assert_eq!(cosine_row(&q, &near, TOL), 0.0);
    }
}
