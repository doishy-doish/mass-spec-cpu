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

use chem::{morgan_bits_2048, parse_smiles};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};

#[allow(dead_code)]
mod boundary;
#[allow(dead_code)]
mod slim;

// (phonebook bit-channels deleted 2026-11-24: binned spectral Jaccard was library recall)

// (phonebook channel constants deleted 2026-11-24 with their channels)
const RRF_BASE: f64 = 60.0;
const MASS_CENTER: f64 = -1.5;
const MASS_FREE: f64 = 3.0;
const MASS_FULL: f64 = 8.0;
const MASS_LEAD_OFF: f64 = 5.0;
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
const VARIANTS: [&str; 18] = [
    "lib", "analog", "frag", "bin", "binw", "stein", "mass", "z", "rrf", "rrfb", "cascade", "router",
    "zw", "rrfw", "pure", "libce", "ent", "gate",
];

// (mix/jaccard deleted 2026-11-24 with the bit-channels)

// (build_w/wjaccard deleted 2026-11-24 with the bit-channels)

// (wjaccard body deleted 2026-11-24 with the bit-channels)

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

/// Cleaned spectrum: mz + sqrt weights + precursor. Per-spectrum library
/// fingerprints (bits/wb/mass-weights/losses) deleted 2026-11-24 with the
/// phonebook channels; query losses recompute from mz on use (S2).
#[derive(Clone)]
struct CSpectra {
    mz: Vec<f64>,
    w: Vec<f64>,
    prec: f64,
}

fn prepare(mzs: &[f64], ints: &[f64], prec: f64) -> CSpectra {
    let (mz, w) = clean(mzs, ints, prec);
    CSpectra { mz, w, prec }
}

// (match_dot/cosine_row/modified_row/stein_row deleted 2026-11-24: library spectral recall)

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

#[allow(dead_code)]
fn loss_set(mz: &[f64], prec: f64) -> Vec<f64> {
    let mut v: Vec<f64> = mz.iter().take(32).map(|m| r2(prec - m)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
        }
    }
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

#[allow(dead_code)]
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

/// M209: prepare-on-first-use. prepare() is a pure function of the raw spectra
/// (and morgan_bits_2048 of the smiles), so cached values equal the old eager-preload
/// values bit-for-bit — only the WHEN changes. Raw is dropped after preparing
/// to bound engine memory: raw(untouched keys) + prepared(touched keys).
fn ensure_prepared(e: &mut TableEntry, _qpeak: f64, _qloss: f64) {
    if e.specs.is_none() {
        let specs: Vec<CSpectra> = e
            .raw
            .iter()
            .map(|s| prepare(&s.mzs, &s.ints, s.prec))
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
// (ce_band deleted 2026-11-24 with libce: same-band library recall)

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
#[allow(dead_code)]
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

// (ent_row deleted 2026-11-24: entropy-weighted cosine vs library spectra)

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
    _fw: &EngineParams,
) -> HashMap<String, Vec<String>> {
    // Post-phonebook (2026-11-24): only computed columns survive (mass =
    // neutral grading, gate = loss/subformula gate). Deleted spectral-recall
    // channels emit empty lists (RR 0, honest) — never a NaN-tie lottery.
    let n = keys.len();
    let nan = vec![f64::NAN; n];
    let col = |c: &str| -> &[f64] { cols.get(c).map(Vec::as_slice).unwrap_or(&nan) };
    let mass = col("mass");
    let gate = col("gate");
    let massz = zscore(mass);
    let gatez = zscore(gate);
    // z = computed fusion: z-scored mass + z-scored gate. pure keeps the
    // canary name on computed evidence only.
    let z: Vec<f64> = (0..n)
        .map(|i| {
            [massz[i], gatez[i]]
                .iter()
                .filter(|&&x| !x.is_nan())
                .sum()
        })
        .collect();
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
    // ranks computed ONCE and shared (the old code re-ranked lib/analog/frag
    // three times each — same inputs, same outputs, at ~1/3 the rank cost).
    let rrank = rank_min_desc(mass);
    let grank = rank_min_desc(gate);
    let rrf = rrf_of(&[&rrank, &grank], &[1.0, 1.0]);
    let mut out = HashMap::new();
    // computed lists, ranked desc.
    let named: Vec<(&str, &[f64])> = vec![
        ("mass", mass),
        ("gate", gate),
        ("z", &z),
        ("pure", &z),
        ("zw", &z),
        ("cascade", &z),
        ("router", &z),
        ("rrf", &rrf),
        ("rrfb", &rrf),
        ("rrfw", &rrf),
    ];
    for (name, vals) in named {
        let kv: Vec<String> = order_desc(vals)
            .into_iter()
            .take(TOP_K_OUT)
            .map(|i| keys[i].clone())
            .collect();
        out.insert(name.to_string(), kv);
    }
    // deleted spectral-recall channels: empty lists, never ranked.
    for dead in ["lib", "analog", "frag", "bin", "binw", "stein", "libce", "ent"] {
        out.insert(dead.to_string(), Vec::new());
    }
    out
}

// (mass_swap_flag deleted 2026-11-24: the swap rule keyed off the lib leader)

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

// (funnel_topk deleted 2026-11-24: the funnel filtered library spectra)

/// Molecule-batch fuse (microsec lane): S per-spectrum score columns over the
/// same candidate order -> one fused ranked list per variant via RRF across
/// spectra. No learning, no weights: rank_s -> 1/(60+rank_s), NaN ranks skip.
/// All-NaN across spectra stays NaN (order_desc puts it last). S=1 equals the
/// single-query rank exactly (one RRF term, monotonic in rank).
fn fuse_molecule_rrf(
    per_spec: &[HashMap<String, Vec<f64>>],
    keys: &[String],
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
    // Deleted spectral-recall channels emit empty lists (honest RR 0) —
    // same contract as fuse_rank; only computed variants fuse here.
    const DEAD: [&str; 8] = ["lib", "analog", "frag", "bin", "binw", "stein", "libce", "ent"];
    for v in VARIANTS {
        if DEAD.contains(&v) {
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
            // (library pool deleted 2026-11-24: pool spectra fed analog/funnel/bin)
            let t_cands = t0.elapsed();
            // Per-spectrum: qc + funnel + hits + channels (same math as single).
            let mut per_spec: Vec<HashMap<String, Vec<f64>>> = Vec::with_capacity(qs.len());
            let pool_union: usize = 0;
            let mass_swap_any = false;
            let mut neutrals: Vec<Option<f64>> = cands.iter().map(|c| c.neutral).collect();
            let _ = &mut neutrals;
            for (si, q) in qs.iter().enumerate() {
                let qc = prepare(&q.mzs, &q.ints, q.prec);
                let qnm: Option<f64> = qnms.get(si).copied().flatten();
                // (library pool + funnel + analog hits deleted 2026-11-24)
                // computed channels only: neutral-mass grading + loss/gate.
                let mut vmass = Vec::with_capacity(cands.len());
                let mut vgate = Vec::with_capacity(cands.len());
                for c in &cands {
                    vmass.push(mass_score(qnm, c.neutral));
                    vgate.push(gate_frac(&qc.mz, qc.prec, c.formula.as_ref()));
                }
                // Computed fusions per spectrum (same math as fuse_rank, so S=1
                // equals the single-query rank exactly; RRF fuses spectra).
                let zm = zscore(&vmass);
                let zg = zscore(&vgate);
                let z: Vec<f64> = (0..cands.len())
                    .map(|i| {
                        let mut s = 0.0;
                        for x in [zm[i], zg[i]] {
                            if !x.is_nan() {
                                s += x;
                            }
                        }
                        s
                    })
                    .collect();
                let rm = rank_min_desc(&vmass);
                let rg = rank_min_desc(&vgate);
                let rrf: Vec<f64> = (0..cands.len())
                    .map(|i| {
                        let mut s = 0.0;
                        for r in [rm[i], rg[i]] {
                            if !r.is_nan() {
                                s += 1.0 / (RRF_BASE + r);
                            }
                        }
                        s
                    })
                    .collect();
                let mut cols: HashMap<String, Vec<f64>> = HashMap::with_capacity(10);
                cols.insert("mass".to_string(), vmass);
                cols.insert("gate".to_string(), vgate);
                cols.insert("z".to_string(), z.clone());
                cols.insert("pure".to_string(), z.clone());
                cols.insert("zw".to_string(), z.clone());
                cols.insert("cascade".to_string(), z.clone());
                cols.insert("router".to_string(), z);
                cols.insert("rrf".to_string(), rrf.clone());
                cols.insert("rrfb".to_string(), rrf.clone());
                cols.insert("rrfw".to_string(), rrf);
                per_spec.push(cols);
                let _ = qis.get(si);
            }
            let keys: Vec<String> = cands.iter().map(|c| c.key.clone()).collect();
            let mut ranked = fuse_molecule_rrf(&per_spec, &keys);
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
        let qc = prepare(&q.mzs, &q.ints, q.prec);
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
        // (library pool deleted 2026-11-24)
        let t_cands = t0.elapsed();
        // (pool vec deleted 2026-11-24)
        // (funnel + recall probe deleted 2026-11-24: funnel filtered library spectra)
        let t_pool = t0.elapsed();
        let t_funnel = t_pool;
        let t_analog = t_pool;
        let prefilter_topk: Option<Vec<String>> = None;
        // computed channels only: neutral-mass grading + loss/gate.
        let mut vmass: Vec<f64> = Vec::with_capacity(cands.len());
        let mut vgate: Vec<f64> = Vec::with_capacity(cands.len());
        for c in &cands {
            vmass.push(mass_score(qnm, c.neutral));
            vgate.push(gate_frac(&qc.mz, qc.prec, c.formula.as_ref()));
        }
        let mut cols: HashMap<String, Vec<f64>> = HashMap::with_capacity(4);
        cols.insert("mass".to_string(), vmass);
        cols.insert("gate".to_string(), vgate);
        let keys: Vec<String> = cands.iter().map(|c| c.key.clone()).collect();
        let t_chan = t0.elapsed();
        let mut ranked = fuse_rank(&keys, &cols, &params);
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
        // (mass_swap deleted 2026-11-24: the swap rule keyed off the lib leader)
        let mass_swap = false;
        let t_fuse = t0.elapsed();
        writeln!(
            h,
            "{}",
            serde_json::to_string(&EngineReply {
                ranked,
                pools: 0,
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
        writeln!(err, "sengine q={} cands={} pool={} ipc={} t_parse={:.1} t_qprep={:.1} t_cands={:.1} t_pool={:.1} t_funnel={:.1} t_analog={:.1} t_chan={:.1} t_fuse={:.1} t_ser={:.1} t_tot={:.1}", qi, keys.len(), 0, ipc_bytes, ms(t_parse), ms(t_qprep - t_parse), ms(t_cands - t_qprep), ms(t_pool - t_cands), ms(t_funnel - t_pool), ms(t_analog - t_funnel), ms(t_chan - t_analog), ms(t_fuse - t_chan), ms(t_ser - t_fuse), ms(t_ser))
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
    eprintln!("usage: binvec <structs|sengine|slimcheck --store DIR [--row I]>");
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

    // (quantize_matches_python deleted 2026-11-24 with qbin: binned spectral
    // Jaccard was library recall)

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

    // (ce_band_edges deleted 2026-11-24 with ce_band: same-band library recall)

    #[test]
    fn entropy_single_peak_quarter_and_empty_linear() {
        // single peak: S = 0 -> exponent 0.25.
        assert!(approx(entropy_exp(&[3.0]), 0.25));
        // empty/zero -> linear fallback.
        assert!(approx(entropy_exp(&[]), 1.0));
        assert!(approx(entropy_exp(&[0.0, 0.0]), 1.0));
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

    // (channels_match_python_reference deleted 2026-11-24 with cosine_row /
    // modified_row / jaccard / wjaccard: spectral-recall channel parity)

    #[test]
    fn z_rrf_computed_fusion() {
        // Post-phonebook (2026-11-24): mass (neutral grading) + gate
        // (loss/subformula) only. mass a>b>c, gate b>c>a.
        // z = z(mass)+z(gate): mass z [1,0,-1], gate z [-1,1,0] -> [0,1,-1].
        // rrf over mass ranks [1,2,3] + gate ranks [3,1,2] -> b wins.
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        cols.insert("gate".to_string(), vec![0.0, 1.0, 0.5]);
        // exact RRF scores via the same rank helper (locks base + min-rank).
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
        let rrf = rrf_of([vec![1.0, 0.5, 0.0], vec![0.0, 1.0, 0.5]].as_slice());
        assert!((rrf[1] - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-12);
        assert!(rrf[1] > rrf[0] && rrf[1] > rrf[2]);
        let fw = EngineParams::neutral(0.05, 0.01);
        let ranked = fuse_rank(&keys, &cols, &fw);
        // one evidence source, several names: all computed aliases agree.
        assert_eq!(ranked["zw"], ranked["z"]);
        assert_eq!(ranked["pure"], ranked["z"]);
        assert_eq!(ranked["cascade"], ranked["z"]);
        assert_eq!(ranked["router"], ranked["z"]);
        assert_eq!(ranked["rrfw"], ranked["rrf"]);
        assert_eq!(ranked["rrfb"], ranked["rrf"]);
        assert_eq!(ranked["z"], vec!["b".to_string(), "a".to_string(), "c".to_string()]);
        assert_eq!(ranked["rrf"][0], "b");
        assert_eq!(ranked["rrf"].len(), 3);
        // deleted spectral-recall channels emit empty lists, never ranked.
        for dead in ["lib", "analog", "frag", "bin", "binw", "stein", "libce", "ent"] {
            assert!(ranked[dead].is_empty());
        }
    }

    #[test]
    fn pure_is_computed_z() {
        // pure keeps the canary name on computed evidence only: pure == z.
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        cols.insert("gate".to_string(), vec![0.0, 1.0, 0.5]);
        let fw = EngineParams::neutral(0.05, 0.01);
        let ranked = fuse_rank(&keys, &cols, &fw);
        assert_eq!(ranked["pure"], ranked["z"]);
        assert_eq!(ranked["pure"], vec!["b".to_string(), "a".to_string(), "c".to_string()]);
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
            .map(|s| prepare(&s.mzs, &s.ints, s.prec))
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
    fn genome_weights_ignored_post_phonebook() {
        // Weights are parked (user order): the engine ignores _fw, so genome
        // variations must NOT move rankings. Locks the no-fitting rule.
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        cols.insert("gate".to_string(), vec![0.0, 1.0, 0.5]);
        let fw = EngineParams::neutral(0.05, 0.01);
        let base = fuse_rank(&keys, &cols, &fw);
        let mut heavy = EngineParams::neutral(0.05, 0.01);
        heavy.wl = 3.0;
        heavy.wa = 0.1;
        heavy.wf = 0.1;
        heavy.casc_z = 0.0;
        assert_eq!(fuse_rank(&keys, &cols, &heavy)["z"], base["z"]);
        assert_eq!(fuse_rank(&keys, &cols, &heavy)["rrf"], base["rrf"]);
        assert_eq!(fuse_rank(&keys, &cols, &heavy)["cascade"], base["cascade"]);
    }

    // (stein_mass_weighted_self_and_far deleted 2026-11-24 with stein_row:
    // mass-weighted library recall)

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
        // (swap asserts deleted 2026-11-24 with mass_swap_flag: the swap rule
        // keyed off the lib leader; mass_score grading above is computed)
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

    // (funnel_topk_slices_in_order + analog_empty_pool_is_zero deleted 2026-11-24
    // with funnel_topk / modified_row / cosine_row: library-recall probes)

    #[test]
    fn molecule_rrf_single_equals_rank_and_fuses() {
        // S=1: per-spectrum z/rrf use the same math as the single path, and RRF
        // over one spectrum is monotonic in rank -> same order as order_desc
        // on the raw scores (NaN last in both).
        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut cols = HashMap::new();
        cols.insert("mass".to_string(), vec![1.0, 0.5, 0.0]);
        cols.insert("gate".to_string(), vec![1.0, 1.0, 1.0]);
        cols.insert("z".to_string(), vec![1.0, 0.0, -1.0]);
        cols.insert("pure".to_string(), vec![1.0, 0.0, -1.0]);
        cols.insert("zw".to_string(), vec![1.0, 0.0, -1.0]);
        cols.insert("cascade".to_string(), vec![1.0, 0.0, -1.0]);
        cols.insert("router".to_string(), vec![1.0, 0.0, -1.0]);
        cols.insert("rrf".to_string(), vec![1.0 / 61.0, 1.0 / 62.0, 1.0 / 63.0]);
        cols.insert("rrfb".to_string(), vec![1.0 / 61.0, 1.0 / 62.0, 1.0 / 63.0]);
        cols.insert("rrfw".to_string(), vec![1.0 / 61.0, 1.0 / 62.0, 1.0 / 63.0]);
        let fused = fuse_molecule_rrf(&[cols], &keys);
        // mass single: a(1.0) b(0.5) c(0.0) -> a,b,c.
        assert_eq!(fused["mass"], vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(fused["z"], vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        // S=2 agreeing spectra keep the winner: both rank a first.
        let mut c2 = HashMap::new();
        for v in ["mass", "gate", "z", "pure", "zw", "cascade", "router", "rrf", "rrfb", "rrfw"] {
            c2.insert(v.to_string(), vec![0.9, 0.1, 0.0]);
        }
        let mut c3 = HashMap::new();
        for v in ["mass", "gate", "z", "pure", "zw", "cascade", "router", "rrf", "rrfb", "rrfw"] {
            c3.insert(v.to_string(), vec![0.8, 0.2, 0.0]);
        }
        let fused2 = fuse_molecule_rrf(&[c2, c3], &keys);
        assert_eq!(fused2["mass"][0], "a");
        assert_eq!(fused2["z"][0], "a");
        // deleted spectral-recall channels emit empty lists, never fused.
        assert!(fused["lib"].is_empty());
        assert!(fused2["analog"].is_empty());
        // All-NaN computed stays candidate order (NaN last, stable).
        let mut cn = HashMap::new();
        for v in ["mass", "gate", "z", "pure", "zw", "cascade", "router", "rrf", "rrfb", "rrfw"] {
            cn.insert(v.to_string(), vec![f64::NAN; 3]);
        }
        let fusedn = fuse_molecule_rrf(&[cn], &keys);
        assert_eq!(fusedn["mass"], vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }
}
