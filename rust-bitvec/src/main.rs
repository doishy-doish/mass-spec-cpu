//! binvec — Rust inference core for the binary bitset channels.
//!
//! Two protocols:
//!   single-shot (stdin JSON, one query): for smoke tests.
//!     {"q": {"p": [u64...], "l": [...]},
//!      "refs": [{"key": "<key14>#<idx>", "p": [...], "l": [...]}, ...]}
//!     -> {"scores": [["<key>", jaccard], ...]}
//!   worker (argv "worker"): preload the whole library ONCE, then one
//!   JSON line per query. Serves the plain AND the weighted (Ruzicka) channel.
//!     preload: {"lib": {"<key14>": [{"p":..,"l":..,"pw":[f32..],"lw":[f32..]}, ...]}}
//!     query:   {"q": {"p":..,"l":..,"pw":..,"lw":..}, "cands": ["<key14>", ...]}
//!     reply:   {"scores": [...], "wscores": [...]}
//! Quantization is done by the Python driver (peak m/z / 0.05, loss / 0.01,
//! w<=0 dropped); bit placement matches the Python reference bit-for-bit.
//! Weights: f32 store (Python float64 ~1e-7), f64 accumulation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};

const NBITS: u64 = 4096;
const KNUTH: u64 = 2654435761;
const SALT_K: u64 = 0x9E3779B9;

fn mix(x: u64, salt: u64) -> u64 {
    x.wrapping_add(salt.wrapping_mul(SALT_K))
        .wrapping_mul(KNUTH)
        & 0xFFFF_FFFF
}

#[derive(Deserialize)]
struct Spec {
    p: Vec<u64>,
    l: Vec<u64>,
}

#[derive(Deserialize)]
struct RefSpec {
    key: String,
    p: Vec<u64>,
    l: Vec<u64>,
}

#[derive(Deserialize)]
struct Job {
    q: Spec,
    refs: Vec<RefSpec>,
}

#[derive(Serialize)]
struct Out {
    scores: Vec<(String, f64)>,
}

fn encode(p: &[u64], l: &[u64]) -> [u64; 64] {
    let mut w = [0u64; 64];
    for &b in p {
        let i = (mix(b, 0) % NBITS) as usize;
        w[i / 64] |= 1u64 << (i % 64);
    }
    for &b in l {
        let i = (mix(b, 1) % NBITS) as usize;
        w[i / 64] |= 1u64 << (i % 64);
    }
    w
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "worker") {
        worker();
        return;
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .expect("binvec: read stdin");
    let job: Job = serde_json::from_str(&buf).expect("binvec: parse job");
    let qb = encode(&job.q.p, &job.q.l);
    let mut scores = Vec::with_capacity(job.refs.len());
    for r in &job.refs {
        let rb = encode(&r.p, &r.l);
        scores.push((r.key.clone(), jaccard(&qb, &rb)));
    }
    let out = Out { scores };
    println!("{}", serde_json::to_string(&out).expect("binvec: serialize"));
}

// ---------- worker v2: binary + WEIGHTED-binary, preloaded store ----------

#[derive(Deserialize)]
struct WSpec {
    p: Vec<u64>,
    l: Vec<u64>,
    #[serde(default)]
    pw: Vec<f32>,
    #[serde(default)]
    lw: Vec<f32>,
}

#[derive(Deserialize)]
struct LibMsg {
    lib: HashMap<String, Vec<WSpec>>,
}

#[derive(Deserialize)]
struct Query {
    q: WSpec,
    cands: Vec<String>,
}

#[derive(Serialize)]
struct Out2 {
    scores: Vec<(String, f64)>,
    wscores: Vec<(String, f64)>,
}

#[derive(Serialize)]
struct Ready {
    ready: bool,
}

struct Entry {
    bits: [u64; 64],
    w: Vec<(u16, f32)>, // sorted by bit; max weight per bit
}

fn build_w(p: &[u64], pw: &[f32], l: &[u64], lw: &[f32]) -> Vec<(u16, f32)> {
    // max weight per mixed bit — exactly Python encode_weighted (max is order-free).
    let mut m: HashMap<u64, f32> = HashMap::new();
    for (i, &b) in p.iter().enumerate() {
        let bit = mix(b, 0) % NBITS;
        let w = pw.get(i).copied().unwrap_or(0.0);
        if w > 0.0 {
            let e = m.entry(bit).or_insert(0.0);
            if w > *e {
                *e = w;
            }
        }
    }
    for (i, &b) in l.iter().enumerate() {
        let bit = mix(b, 1) % NBITS;
        let w = lw.get(i).copied().unwrap_or(0.0);
        if w > 0.0 {
            let e = m.entry(bit).or_insert(0.0);
            if w > *e {
                *e = w;
            }
        }
    }
    let mut v: Vec<(u16, f32)> = m.into_iter().map(|(b, w)| (b as u16, w)).collect();
    v.sort_by_key(|e| e.0);
    v
}

fn wjaccard(a: &[(u16, f32)], b: &[(u16, f32)]) -> f64 {
    // Ruzicka over sorted (bit, weight) lists; f64 accumulation.
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

fn worker() {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let first = lines.next().expect("binvec: eof").expect("binvec: read");
    let libmsg: LibMsg = serde_json::from_str(&first).expect("binvec: parse lib");
    let mut lib: HashMap<String, Vec<Entry>> = HashMap::with_capacity(libmsg.lib.len());
    for (k, specs) in &libmsg.lib {
        lib.insert(
            k.clone(),
            specs
                .iter()
                .map(|s| Entry {
                    bits: encode(&s.p, &s.l),
                    w: build_w(&s.p, &s.pw, &s.l, &s.lw),
                })
                .collect(),
        );
    }
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    writeln!(
        h,
        "{}",
        serde_json::to_string(&Ready { ready: true }).unwrap()
    )
    .unwrap();
    h.flush().unwrap();
    for line in lines {
        let line = line.expect("binvec: read");
        if line.trim().is_empty() {
            continue;
        }
        let q: Query = serde_json::from_str(&line).expect("binvec: parse query");
        let qb = encode(&q.q.p, &q.q.l);
        let qw = build_w(&q.q.p, &q.q.pw, &q.q.l, &q.q.lw);
        let mut scores = Vec::with_capacity(q.cands.len());
        let mut wscores = Vec::with_capacity(q.cands.len());
        for key in &q.cands {
            let (mut best, mut wbest) = (0.0, 0.0);
            if let Some(specs) = lib.get(key) {
                for e in specs {
                    let j = jaccard(&qb, &e.bits);
                    if j > best {
                        best = j;
                    }
                    let wj = wjaccard(&qw, &e.w);
                    if wj > wbest {
                        wbest = wj;
                    }
                }
            }
            scores.push((key.clone(), best));
            wscores.push((key.clone(), wbest));
        }
        writeln!(
            h,
            "{}",
            serde_json::to_string(&Out2 { scores, wscores }).unwrap()
        )
        .unwrap();
        h.flush().unwrap();
    }
}
