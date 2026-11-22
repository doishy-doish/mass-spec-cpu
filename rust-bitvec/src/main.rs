//! binvec — Rust inference core for the binary bitset channel.
//!
//! Contract (stdin JSON, single query):
//!   {"q": {"p": [u32...], "l": [u32...]},
//!    "refs": [{"key": "<key14>#<idx>", "p": [...], "l": [...]}, ...]}
//! Bins are pre-quantized by the Python driver (peak m/z / 0.05,
//! neutral-loss / 0.01, masked to u32), so bit placement matches the
//! Python reference bit-for-bit. Stdout:
//!   {"scores": [["<key>", jaccard], ...]}

use serde::{Deserialize, Serialize};
use std::io::Read;

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
