//! slim.rs — Rust mirror of `src/mscpu/slim.py` (the reference, untouched).
//!
//! PROTOCOL (user order, test-then-swap): Python stays the reference until the
//! equivalence gate is GREEN on a runner — `tests/test_slim_rust.py` builds a
//! tiny store with the Python builder, runs `binvec slimcheck`, and compares
//! sha + row bits. Until then `run.py` keeps calling `slim.py`. No swap here.
//!
//! Codec twins of `scripts/build_slim.py` (`enc_*`): same math (int64-bit
//! deltas with WRAPPING arithmetic — numpy `diff`/`cumsum` on int64 wraps
//! modulo 2^64, so Rust must `wrapping_*` or 1e300-class jumps drift), same
//! zstd level 3, same manifest CODEC. Anything else fails loud.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Must equal build_slim.CODEC and slim.CODEC. Anything else fails loud.
pub const CODEC: &str = "f64bitdelta+zstd3+ce1";
/// Must equal build_slim.LEVEL.
const ZSTD_LEVEL: i32 = 3;

#[derive(serde::Deserialize)]
struct Manifest {
    rows: usize,
    keys: usize,
    codec: String,
    mz_values: usize,
    files: HashMap<String, String>,
}

/// Flat store, row order == parquet order (same contract as `SL.load_slim`).
/// `neutrals` is parallel to `keys` (`None` = NaN in the store).
pub struct SlimStore {
    pub n: usize,
    pub keys: Vec<String>,
    pub smiles: HashMap<String, String>,
    pub adducts: Vec<String>,
    pub instruments: Vec<String>,
    pub key_idx: Vec<i32>,
    pub key_inst: Vec<i32>,
    pub key_first: Vec<i32>,
    pub prec: Vec<f64>,
    pub ad: Vec<u8>,
    pub inst: Vec<u8>,
    pub ce: Vec<f64>,
    pub mz_off: Vec<i64>,
    pub mz_all: Vec<f64>,
    pub int_all: Vec<f64>,
    pub neutrals: Vec<Option<f64>>,
}

fn sha_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn read_file(store: &Path, name: &str) -> Vec<u8> {
    let p: PathBuf = store.join(name);
    fs::read(&p).unwrap_or_else(|e| panic!("slim store missing {}: {}", p.display(), e))
}

fn verify(files: &HashMap<String, String>, store: &Path) {
    for (name, digest) in files {
        let b = read_file(store, name);
        if sha_hex(&b) != *digest {
            panic!(
                "slim store corrupt/tampered: {} sha mismatch (delete {} and rebuild)",
                name,
                store.display()
            );
        }
    }
}

fn zdec(b: &[u8], what: &str) -> Vec<u8> {
    zstd::decode_all(b).unwrap_or_else(|e| panic!("slim {} zstd decode failed: {}", what, e))
}

fn zenc(b: &[u8]) -> Vec<u8> {
    zstd::encode_all(&b[..], ZSTD_LEVEL).expect("slim zstd encode failed")
}

fn dec_i32(b: &[u8], what: &str) -> Vec<i32> {
    let raw = zdec(b, what);
    assert!(
        raw.len() % 4 == 0,
        "slim {} len {} not a multiple of 4",
        what,
        raw.len()
    );
    raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn dec_u8(b: &[u8], _what: &str) -> Vec<u8> {
    zdec(b, _what)
}

fn dec_i64(b: &[u8], what: &str) -> Vec<i64> {
    let raw = zdec(b, what);
    assert!(
        raw.len() % 8 == 0,
        "slim {} len {} not a multiple of 8",
        what,
        raw.len()
    );
    raw.chunks_exact(8)
        .map(|c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        })
        .collect()
}

fn dec_f64(b: &[u8], what: &str) -> Vec<f64> {
    let raw = zdec(b, what);
    assert!(
        raw.len() % 8 == 0,
        "slim {} len {} not a multiple of 8",
        what,
        raw.len()
    );
    raw.chunks_exact(8)
        .map(|c| {
            f64::from_bits(u64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            ]))
        })
        .collect()
}

/// Builder twin of `build_slim.enc_deltas_f64` (bit-exact: wrapping deltas).
pub fn enc_deltas_f64(x: &[f64]) -> Vec<u8> {
    let bits: Vec<i64> = x.iter().map(|v| v.to_bits() as i64).collect();
    let mut d = Vec::with_capacity(bits.len());
    if !bits.is_empty() {
        d.push(bits[0]);
        for i in 1..bits.len() {
            d.push(bits[i].wrapping_sub(bits[i - 1]));
        }
    }
    let mut raw = Vec::with_capacity(d.len() * 8);
    for v in &d {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    zenc(&raw)
}

/// Loader twin of `slim._dec_deltas_f64` (wrapping cumsum — must match numpy).
pub fn dec_deltas_f64(b: &[u8], n: usize) -> Vec<f64> {
    let d = dec_i64(b, "mz_data");
    assert!(
        d.len() == n,
        "slim mz_data len {} != manifest mz_values {}",
        d.len(),
        n
    );
    let mut out = Vec::with_capacity(n);
    let mut acc: i64 = 0;
    for (i, v) in d.iter().enumerate() {
        acc = if i == 0 { *v } else { acc.wrapping_add(*v) };
        out.push(f64::from_bits(acc as u64));
    }
    out
}

/// Builder twin of `build_slim.enc_raw` (LE bytes + zstd; same for all dtypes).
pub fn enc_raw_f64(x: &[f64]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(x.len() * 8);
    for v in x {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    zenc(&raw)
}

/// Load the slim store (mirror of `slim.load_slim`; same checks, same order).
pub fn load_slim(store: &Path) -> SlimStore {
    let manifest_raw = read_file(store, "manifest.json");
    let manifest: Manifest = serde_json::from_slice(&manifest_raw)
        .expect("slim manifest.json is not valid JSON");
    if manifest.codec != CODEC {
        panic!(
            "slim codec '{}' != loader '{}': rebuild the store (scripts/build_slim.py) — refusing to mix codecs",
            manifest.codec, CODEC
        );
    }
    verify(&manifest.files, store);

    let keys: Vec<String> = serde_json::from_slice(&read_file(store, "keys.json"))
        .expect("slim keys.json invalid");
    let smiles: HashMap<String, String> =
        serde_json::from_slice(&read_file(store, "smiles.json"))
            .expect("slim smiles.json invalid");
    let adducts: Vec<String> = serde_json::from_slice(&read_file(store, "adducts.json"))
        .expect("slim adducts.json invalid");
    let instruments: Vec<String> =
        serde_json::from_slice(&read_file(store, "instruments.json"))
            .expect("slim instruments.json invalid");
    let n = manifest.rows;
    if keys.len() != manifest.keys {
        panic!("slim keys.json len != manifest keys");
    }

    let key_idx = dec_i32(&read_file(store, "key_idx.int32.zst"), "key_idx");
    let prec = dec_f64(&read_file(store, "prec.f64.zst"), "prec");
    let ad = dec_u8(&read_file(store, "adduct.u8.zst"), "adduct");
    let inst = dec_u8(&read_file(store, "inst.u8.zst"), "inst");
    let mz_off = dec_i64(&read_file(store, "mz_off.int64.zst"), "mz_off");
    let ce = dec_f64(&read_file(store, "ce.f64.zst"), "ce");
    let key_inst = dec_i32(&read_file(store, "key_inst.int32.zst"), "key_inst");
    let key_first = dec_i32(&read_file(store, "key_first.int32.zst"), "key_first");
    let neutrals_arr = dec_f64(&read_file(store, "neutrals.f64.zst"), "neutrals");
    let mz_all = dec_deltas_f64(&read_file(store, "mz_data.zst"), manifest.mz_values);
    let int_all = dec_f64(&read_file(store, "int_data.zst"), "int_all");

    if !(key_idx.len() == n
        && prec.len() == n
        && ad.len() == n
        && inst.len() == n
        && ce.len() == n)
    {
        panic!("slim row-array len mismatch vs manifest rows");
    }
    if mz_off.len() != n + 1 {
        panic!("slim mz_off len != rows+1");
    }
    if !(key_inst.len() == keys.len() && key_first.len() == keys.len()) {
        panic!("slim per-key array len != keys");
    }
    if neutrals_arr.len() != keys.len() {
        panic!("slim neutrals len != keys");
    }

    let neutrals: Vec<Option<f64>> = neutrals_arr
        .into_iter()
        .map(|v| if v.is_nan() { None } else { Some(v) })
        .collect();

    SlimStore {
        n,
        keys,
        smiles,
        adducts,
        instruments,
        key_idx,
        key_inst,
        key_first,
        prec,
        ad,
        inst,
        ce,
        mz_off,
        mz_all,
        int_all,
        neutrals,
    }
}

/// One row's (m/z slice, intensity slice). Slices, not copies (mirror `row_spec`).
pub fn row_spec(s: &SlimStore, i: usize) -> (&[f64], &[f64]) {
    assert!(i < s.n, "slim row {} out of range {}", i, s.n);
    let a = s.mz_off[i] as usize;
    let b = s.mz_off[i + 1] as usize;
    (&s.mz_all[a..b], &s.int_all[a..b])
}

/// One row's CE in eV, or None when unknown/merged (mirror `row_ce`).
pub fn row_ce(s: &SlimStore, i: usize) -> Option<f64> {
    assert!(i < s.n, "slim row {} out of range {}", i, s.n);
    let v = s.ce[i];
    if v.is_nan() {
        None
    } else {
        Some(v)
    }
}

fn le_bytes_f64(x: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(x.len() * 8);
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn le_bytes_i32(x: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(x.len() * 4);
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

#[derive(serde::Serialize)]
pub struct SlimCheck {
    pub n: usize,
    pub keys: usize,
    pub codec: String,
    pub sha: HashMap<String, String>,
    pub row0_mzs_bits: Vec<i64>,
    pub row0_ints_bits: Vec<i64>,
    pub ce0: Option<f64>,
    pub keys0: Vec<String>,
}

/// Compact equivalence fingerprint: sha over the same LE bytes numpy's
/// `.tobytes()` yields on x86_64, plus full row-0 bits. `test_slim_rust.py`
/// compares every field against the Python loader on the same store bytes.
pub fn slimcheck_json(s: &SlimStore) -> String {
    let mut sha = HashMap::new();
    sha.insert("prec".to_string(), sha_hex(&le_bytes_f64(&s.prec)));
    sha.insert("mz".to_string(), sha_hex(&le_bytes_f64(&s.mz_all)));
    sha.insert("ints".to_string(), sha_hex(&le_bytes_f64(&s.int_all)));
    sha.insert("ce".to_string(), sha_hex(&le_bytes_f64(&s.ce)));
    sha.insert("key_idx".to_string(), sha_hex(&le_bytes_i32(&s.key_idx)));
    let (m0, i0) = if s.n > 0 { row_spec(s, 0) } else { (&[][..], &[][..]) };
    let ce0 = if s.n > 0 { row_ce(s, 0) } else { None };
    let keys0: Vec<String> = s.keys.iter().take(5).cloned().collect();
    serde_json::to_string(&SlimCheck {
        n: s.n,
        keys: s.keys.len(),
        codec: CODEC.to_string(),
        sha,
        row0_mzs_bits: m0.iter().map(|v| v.to_bits() as i64).collect(),
        row0_ints_bits: i0.iter().map(|v| v.to_bits() as i64).collect(),
        ce0,
        keys0,
    })
    .unwrap()
}

#[derive(serde::Serialize)]
struct SlimRow {
    i: usize,
    mzs_bits: Vec<i64>,
    ints_bits: Vec<i64>,
    ce: Option<f64>,
}

/// Full per-row bits (pytest compares every row on the tiny test store).
pub fn row_json(s: &SlimStore, i: usize) -> String {
    let (m, ii) = row_spec(s, i);
    serde_json::to_string(&SlimRow {
        i,
        mzs_bits: m.iter().map(|v| v.to_bits() as i64).collect(),
        ints_bits: ii.iter().map(|v| v.to_bits() as i64).collect(),
        ce: row_ce(s, i),
    })
    .unwrap()
}

#[cfg(test)]
mod slim_tests {
    use super::*;

    fn tricky() -> Vec<f64> {
        // 0.1+0.2-style, subnormal-adjacent, huge, tiny, dup-adjacent jumps
        // (1e300 -> 5e-320 wraps int64 deltas; the wrapping path is the test).
        vec![0.1 + 0.2, 500.25, 1e300, 5e-320, 999.999, 999.999, 150.0]
    }

    fn bits(x: &[f64]) -> Vec<u64> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn delta_codec_bit_exact_tricky() {
        let x = tricky();
        let rt = dec_deltas_f64(&enc_deltas_f64(&x), x.len());
        assert_eq!(bits(&rt), bits(&x), "bit drift on tricky floats");
    }

    #[test]
    fn delta_codec_single_and_empty() {
        let x = vec![150.0];
        let rt = dec_deltas_f64(&enc_deltas_f64(&x), x.len());
        assert_eq!(bits(&rt), bits(&x));
        let e: Vec<f64> = vec![];
        let rt = dec_deltas_f64(&enc_deltas_f64(&e), 0);
        assert!(rt.is_empty());
    }

    #[test]
    fn delta_codec_descending_wraps() {
        // Descending jumps go negative-wrapping; must still round-trip.
        let x = vec![2000.0, 999.999, 5e-320, 1e300, 0.1 + 0.2];
        let rt = dec_deltas_f64(&enc_deltas_f64(&x), x.len());
        assert_eq!(bits(&rt), bits(&x));
    }

    #[test]
    fn delta_len_mismatch_fails_loud() {
        let x = tricky();
        let b = enc_deltas_f64(&x);
        let r = std::panic::catch_unwind(|| dec_deltas_f64(&b, x.len() + 1));
        assert!(r.is_err(), "mz len != manifest must fail loud");
    }

    #[test]
    fn raw_round_trip_preserves_bytes() {
        let x = vec![1i32, -2, 0, i32::MAX, i32::MIN];
        let mut raw = Vec::new();
        for v in &x {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let back = dec_i32(&zenc(&raw), "t");
        assert_eq!(back, x);
        let u = vec![0u8, 1, 255, 7];
        assert_eq!(dec_u8(&zenc(&u), "t"), u);
    }

    fn write_store(dir: &Path, codec: &str, flip_byte: bool) {
        // Minimal 3-row store built with the Rust enc twins (no Python needed).
        let keys = vec!["K1".to_string(), "K2".to_string()];
        let key_idx = vec![0i32, 1, 0];
        let prec = vec![100.0001f64, 200.5, 100.0001];
        let ad = vec![0u8, 1, 0];
        let inst = vec![1u8, 0, 1];
        let ce = vec![20.0f64, f64::NAN, 40.0];
        let mz_off = vec![0i64, 2, 3, 3];
        let mz_all = vec![0.1 + 0.2, 500.25, 1e300];
        let int_all = vec![1.0f64, 0.5, 1.0];
        let neutrals = vec![90.0f64, f64::NAN];
        let mut files: HashMap<String, String> = HashMap::new();
        fn wfile(dir: &Path, files: &mut HashMap<String, String>, flip_byte: bool, name: &str, b: Vec<u8>) {
            let mut bb = b;
            if flip_byte && name == "prec.f64.zst" {
                bb[10] ^= 1;
            }
            files.insert(name.to_string(), sha_hex(&bb));
            fs::write(dir.join(name), bb).unwrap();
        }
        fn ri32(dir: &Path, files: &mut HashMap<String, String>, flip_byte: bool, n: &str, x: &[i32]) {
            let mut b = Vec::new();
            for v in x {
                b.extend_from_slice(&v.to_le_bytes());
            }
            wfile(dir, files, flip_byte, n, zenc(&b));
        }
        fn rf64(dir: &Path, files: &mut HashMap<String, String>, flip_byte: bool, n: &str, x: &[f64]) {
            let mut b = Vec::new();
            for v in x {
                b.extend_from_slice(&v.to_le_bytes());
            }
            wfile(dir, files, flip_byte, n, zenc(&b));
        }
        ri32(dir, &mut files, flip_byte, "key_idx.int32.zst", &key_idx);
        rf64(dir, &mut files, flip_byte, "prec.f64.zst", &prec);
        wfile(dir, &mut files, flip_byte, "adduct.u8.zst", zenc(&ad));
        wfile(dir, &mut files, flip_byte, "inst.u8.zst", zenc(&inst));
        {
            let mut b = Vec::new();
            for v in &mz_off {
                b.extend_from_slice(&v.to_le_bytes());
            }
            wfile(dir, &mut files, flip_byte, "mz_off.int64.zst", zenc(&b));
        }
        wfile(dir, &mut files, flip_byte, "mz_data.zst", enc_deltas_f64(&mz_all));
        rf64(dir, &mut files, flip_byte, "int_data.zst", &int_all);
        rf64(dir, &mut files, flip_byte, "ce.f64.zst", &ce);
        ri32(dir, &mut files, flip_byte, "key_inst.int32.zst", &[1, 0]);
        ri32(dir, &mut files, flip_byte, "key_first.int32.zst", &[0, 1]);
        rf64(dir, &mut files, flip_byte, "neutrals.f64.zst", &neutrals);
        for (name, obj) in [
            ("keys.json", serde_json::to_string(&keys).unwrap()),
            (
                "smiles.json",
                "{\"K1\":\"CCO\",\"K2\":\"CCC\"}".to_string(),
            ),
            ("adducts.json", "[\"[M+H]+\",\"[M+Na]+\"]".to_string()),
            ("instruments.json", "[\"Orbitrap\",\"timsTOF\"]".to_string()),
        ] {
            wfile(dir, &mut files, flip_byte, name, obj.into_bytes());
        }
        // NOTE: sha recorded AFTER the flip, so tamper test flips again below.
        let manifest = serde_json::json!({
            "rows": 3, "keys": 2, "codec": codec,
            "mz_values": mz_all.len(), "files": files,
        });
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "slimtest-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn loader_round_trip_and_accessors() {
        let d = tmpdir("roundtrip");
        write_store(&d, CODEC, false);
        let s = load_slim(&d);
        assert_eq!(s.n, 3);
        assert_eq!(s.keys, vec!["K1".to_string(), "K2".to_string()]);
        assert_eq!(bits(&s.prec), bits(&vec![100.0001f64, 200.5, 100.0001]));
        assert_eq!(
            bits(&s.mz_all),
            bits(&vec![0.1 + 0.2, 500.25, 1e300])
        );
        assert_eq!(s.key_idx, vec![0, 1, 0]);
        assert_eq!(s.neutrals, vec![Some(90.0), None]);
        let (m0, i0) = row_spec(&s, 0);
        assert_eq!(bits(m0), bits(&vec![0.1 + 0.2, 500.25]));
        assert_eq!(bits(i0), bits(&vec![1.0f64, 0.5]));
        let (m2, i2) = row_spec(&s, 2);
        assert!(m2.is_empty() && i2.is_empty(), "empty row must slice empty");
        assert_eq!(row_ce(&s, 0), Some(20.0));
        assert_eq!(row_ce(&s, 1), None);
        assert_eq!(row_ce(&s, 2), Some(40.0));
        // slimcheck fingerprint parses and carries the codec + counts.
        let v: serde_json::Value =
            serde_json::from_str(&slimcheck_json(&s)).unwrap();
        assert_eq!(v["n"].as_u64(), Some(3));
        assert_eq!(v["codec"].as_str(), Some(CODEC));
        assert_eq!(v["keys0"], serde_json::json!(["K1", "K2"]));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn loader_rejects_bad_codec() {
        let d = tmpdir("badcodec");
        write_store(&d, "bogus", false);
        let r = std::panic::catch_unwind(|| load_slim(&d));
        assert!(r.is_err(), "bad codec must fail loud");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn loader_detects_tamper() {
        let d = tmpdir("tamper");
        write_store(&d, CODEC, false);
        // Flip a byte AFTER the manifest recorded its sha.
        let p = d.join("prec.f64.zst");
        let mut b = fs::read(&p).unwrap();
        b[10] ^= 1;
        fs::write(&p, b).unwrap();
        let r = std::panic::catch_unwind(|| load_slim(&d));
        assert!(r.is_err(), "tampered file must fail loud");
        let _ = fs::remove_dir_all(&d);
    }
}
