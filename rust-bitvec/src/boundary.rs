//! boundary — deterministic launcher/index/output helpers, Rust-owned.
//!
//! This module ports the DETERMINISTIC numpy logic from validation.py,
//! run.py (index builders, pool windows, query-assembly inputs) and
//! predict.py (aggregation, submission CSV) into dependency-free Rust
//! (std only — no new crates, no new build inputs).
//!
//! Explicitly NOT ported here (measured change boundaries — see the stub
//! report at scratch/mass-spec/stubs/boundary_report.json):
//! - parquet decoding: needs a parquet crate; pyarrow stays the reader.
//! - seeded RNG sampling (numpy default_rng identity): split parity depends on
//!   the exact RNG stream, so sampling stays in Python until an exact RNG port
//!   is proven bit-identical. The deterministic *apply* of a held set IS
//!   ported (apply_class2/3/cross).
//!
//! Lossless law: every function mirrors current Python semantics, including
//! the orders that matter: first_smiles_sorted follows
//! np.unique(return_index=True) = sorted-code order, and every sort that
//! Python does stably is a stable sort here.

use std::collections::{HashMap, HashSet};

/// First-occurrence unique keys (mirrors pandas .unique() order).
pub fn uniq_first(keys: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for k in keys {
        if seen.insert(k.clone()) {
            out.push(k.clone());
        }
    }
    out
}

/// Case-insensitive substring match (mirrors np.char lower + find != -1).
/// Fixture scope is ASCII instrument strings; exotic-Unicode lowering parity
/// is outside the data contract and not asserted.
pub fn contains_ci(hay: &str, needle: &str) -> bool {
    hay.to_lowercase().contains(&needle.to_lowercase())
}

/// Pool mask: instrument substring + inclusive precursor mz window.
/// Mirrors validation.query_pool (row order == table order).
pub fn query_pool(
    instruments: &[String],
    precs: &[f64],
    regex: &str,
    lo: f64,
    hi: f64,
) -> Vec<usize> {
    assert_eq!(instruments.len(), precs.len(), "boundary: table ragged");
    (0..instruments.len())
        .filter(|&i| contains_ci(&instruments[i], regex) && precs[i] >= lo && precs[i] <= hi)
        .collect()
}

/// Sorted unique codes + inverse row map (mirrors np.unique: codes sorted).
pub fn codes_inv(all_keys: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut codes: Vec<String> = all_keys.to_vec();
    codes.sort();
    codes.dedup();
    let inv: Vec<usize> = all_keys
        .iter()
        .map(|k| {
            codes
                .binary_search(k)
                .expect("boundary: key missing from codes")
        })
        .collect();
    (codes, inv)
}

/// Library index: stable sort of library rows by code, then contiguous groups
/// (mirrors the vectorized bounds grouping in run.py main).
pub fn lib_index(
    codes: &[String],
    inv: &[usize],
    library: &[usize],
) -> Vec<(String, Vec<usize>)> {
    let mut rows: Vec<usize> = library.to_vec();
    rows.sort_by_key(|&i| inv[i]); // stable: ties keep library input order
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    for i in rows {
        let key = codes[inv[i]].clone();
        let same = out.last().map(|(k, _)| k == &key).unwrap_or(false);
        if same {
            out.last_mut().expect("boundary: group vanished").1.push(i);
        } else {
            out.push((key, vec![i]));
        }
    }
    out
}

/// First SMILES per code, emitted in sorted-code order (mirrors
/// np.unique(inv, return_index=True) + groupby().iloc[0]).
pub fn first_smiles_sorted(
    codes: &[String],
    inv: &[usize],
    smiles: &[String],
) -> Vec<(String, String)> {
    let mut first: HashMap<usize, usize> = HashMap::new();
    for (idx, &c) in inv.iter().enumerate() {
        first.entry(c).or_insert(idx);
    }
    codes
        .iter()
        .enumerate()
        .map(|(c, key)| {
            let idx = *first.get(&c).expect("boundary: code has no row");
            (key.clone(), smiles[idx].clone())
        })
        .collect()
}

/// Stable neutral-mass sort (mirrors np.argsort(kind="stable"): equal-neutral
/// ties keep input order, deterministically).
pub fn neutral_sort(ukeys: &[String], neutrals: &[f64]) -> (Vec<String>, Vec<f64>) {
    assert_eq!(ukeys.len(), neutrals.len(), "boundary: struct arrays ragged");
    let mut idx: Vec<usize> = (0..ukeys.len()).collect();
    idx.sort_by(|&a, &b| {
        neutrals[a]
            .partial_cmp(&neutrals[b])
            .expect("boundary: non-finite neutral")
    });
    (
        idx.iter().map(|&i| ukeys[i].clone()).collect(),
        idx.iter().map(|&i| neutrals[i]).collect(),
    )
}

/// Precursor-sorted first-spectrum-per-key lookup (mirrors _prec_pairs sort).
pub fn prec_lookup(
    lib: &[(String, Vec<usize>)],
    precs: &[f64],
) -> (Vec<f64>, Vec<String>) {
    let mut pairs: Vec<(f64, String)> = lib
        .iter()
        .map(|(k, rows)| {
            let j = *rows.first().expect("boundary: key with no rows");
            (precs[j], k.clone())
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("boundary: non-finite prec"));
    (pairs.iter().map(|(p, _)| *p).collect(), pairs.iter().map(|(_, k)| k.clone()).collect())
}

/// First index with xs[i] >= x (mirrors searchsorted side="left").
pub fn lower_bound(xs: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = xs.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if xs[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// First index with xs[i] > x (mirrors searchsorted side="right").
pub fn upper_bound(xs: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = xs.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if xs[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// ppm candidate window over neutral-sorted (masses, keys).
/// Mirrors run.py _run_query searchsorted slice (lo inclusive, hi exclusive).
pub fn ppm_window(
    masses: &[f64],
    keys: &[String],
    qnm: f64,
    ppm: f64,
) -> (usize, usize, Vec<String>) {
    let tol = qnm * ppm * 1e-6;
    let lo = lower_bound(masses, qnm - tol);
    let hi = upper_bound(masses, qnm + tol);
    (lo, hi, keys[lo..hi].to_vec())
}

/// Analog pool: precursor +-50 Da window excluding candidates, nearest-first,
/// capped (mirrors run.py _run_query; stable sort keeps precursor order ties).
pub fn analog_window(
    lib_precs: &[f64],
    lib_pkeys: &[String],
    prec_q: f64,
    cand: &HashSet<String>,
    cap: usize,
) -> Vec<String> {
    let lo = lower_bound(lib_precs, prec_q - 50.0);
    let hi = upper_bound(lib_precs, prec_q + 50.0);
    let mut near: Vec<(f64, usize)> = Vec::new();
    for i in lo..hi {
        if !cand.contains(&lib_pkeys[i]) {
            near.push(((lib_precs[i] - prec_q).abs(), i));
        }
    }
    near.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("boundary: non-finite dist"));
    near.into_iter()
        .take(cap)
        .map(|(_, i)| lib_pkeys[i].clone())
        .collect()
}

/// Top-k keys by summed reciprocal rank (mirrors predict.aggregate: ranks
/// 0..24 score, per-list dedupe, empty keys skipped, ties keep first-seen
/// order via the stable sort over insertion order).
pub fn aggregate(ranked: &[Vec<String>], k: usize) -> Vec<String> {
    let mut score: HashMap<String, f64> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for rl in ranked {
        let mut seen = HashSet::new();
        for (i, key) in rl.iter().enumerate() {
            if i >= 25 {
                break;
            }
            if key.is_empty() || !seen.insert(key.clone()) {
                continue;
            }
            if !score.contains_key(key) {
                order.push(key.clone());
            }
            *score.entry(key.clone()).or_insert(0.0) += 1.0 / (i + 1) as f64;
        }
    }
    let mut idx: Vec<usize> = (0..order.len()).collect();
    idx.sort_by(|&a, &b| {
        score[&order[b]]
            .partial_cmp(&score[&order[a]])
            .expect("boundary: non-finite score")
    });
    idx.into_iter().take(k).map(|i| order[i].clone()).collect()
}

/// Split a submission cell on ";" (mirrors metric.parse_cell).
pub fn parse_cell(cell: &str) -> Vec<String> {
    cell.split(';')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Submission rule check (mirrors metric.validate_submission messages).
pub fn validate_submission(rows: &[(String, String)]) -> Vec<String> {
    let mut errs = Vec::new();
    let mut seen = HashSet::new();
    for (i, (mid, cell)) in rows.iter().enumerate() {
        if mid.is_empty() {
            errs.push(format!("row{}: missing column", i));
            continue;
        }
        if !seen.insert(mid.clone()) {
            errs.push(format!("row{}: repeated {}", i, mid));
        }
        if parse_cell(cell).len() > 25 {
            errs.push(format!("row{}: >25 guesses", i));
        }
    }
    errs
}

/// Exact submission.csv body (mirrors predict.py: header + mid,cell lines,
/// no quoting — cells are semicolon-joined SMILES, never contain commas).
pub fn format_csv(rows: &[(String, String)]) -> String {
    let mut out = String::from("molecule_id,smiles\n");
    for (mid, cell) in rows {
        out.push_str(mid);
        out.push(',');
        out.push_str(cell);
        out.push('\n');
    }
    out
}

/// class2 apply: one query row per held key (first pool occurrence), library =
/// every row outside the query set. Sampling of `held` stays in Python.
pub fn apply_class2(
    pool_idx: &[usize],
    all_keys: &[String],
    held: &HashSet<String>,
) -> (Vec<usize>, Vec<usize>, Vec<String>) {
    let mut seen = HashSet::new();
    let mut qrows = Vec::new();
    for &i in pool_idx {
        let k = &all_keys[i];
        if held.contains(k) && seen.insert(k.clone()) {
            qrows.push(i);
        }
    }
    let qset: HashSet<usize> = qrows.iter().copied().collect();
    let library: Vec<usize> = (0..all_keys.len()).filter(|i| !qset.contains(i)).collect();
    let mut held_sorted: Vec<String> = held.iter().cloned().collect();
    held_sorted.sort();
    (library, qrows, held_sorted)
}

/// class3 apply: every held spectrum leaves the library.
pub fn apply_class3(
    pool_idx: &[usize],
    all_keys: &[String],
    held: &HashSet<String>,
) -> (Vec<usize>, Vec<usize>, Vec<String>) {
    let qrows: Vec<usize> = pool_idx
        .iter()
        .copied()
        .filter(|&i| held.contains(&all_keys[i]))
        .collect();
    let library: Vec<usize> = (0..all_keys.len())
        .filter(|&i| !held.contains(&all_keys[i]))
        .collect();
    let mut held_sorted: Vec<String> = held.iter().cloned().collect();
    held_sorted.sort();
    (library, qrows, held_sorted)
}

/// cross-instrument apply: queries are pool rows of held keys, but non-pool
/// rows of held keys (other instruments) STAY in the library.
pub fn apply_cross(
    pool_idx: &[usize],
    all_keys: &[String],
    held: &HashSet<String>,
) -> (Vec<usize>, Vec<usize>, Vec<String>) {
    let qrows: Vec<usize> = pool_idx
        .iter()
        .copied()
        .filter(|&i| held.contains(&all_keys[i]))
        .collect();
    let qset: HashSet<usize> = qrows.iter().copied().collect();
    let library: Vec<usize> = (0..all_keys.len()).filter(|i| !qset.contains(i)).collect();
    let mut held_sorted: Vec<String> = held.iter().cloned().collect();
    held_sorted.sort();
    (library, qrows, held_sorted)
}

/// Per-query ppm-window candidate count (snake-draft costing input).
/// Mirrors run.py: tol = qnm*ppm*1e-6 over neutral-sorted masses.
pub fn window_count(masses: &[f64], qnm: f64, ppm: f64) -> usize {
    let tol = qnm * ppm * 1e-6;
    upper_bound(masses, qnm + tol) - lower_bound(masses, qnm - tol)
}

/// Snake-draft shard partition (mirrors run.py main: stable cost-desc order,
/// snake across shards, winning bucket sorted by row id).
pub fn snake_partition(
    query_ids: &[usize],
    costs: &[usize],
    shards: usize,
    shard: usize,
) -> Vec<usize> {
    assert_eq!(query_ids.len(), costs.len(), "boundary: costs ragged");
    let sh = shards.max(1);
    let mut order: Vec<usize> = (0..query_ids.len()).collect();
    order.sort_by(|&a, &b| costs[b].cmp(&costs[a])); // stable: ties keep query order
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); sh];
    for (rank, &pos) in order.iter().enumerate() {
        let p = rank % sh;
        let b = if (rank / sh) % 2 == 0 { p } else { sh - 1 - p };
        buckets[b].push(query_ids[pos]);
    }
    let mut out = buckets[shard.min(sh - 1)].clone();
    out.sort();
    out
}

/// Sorted-unique id groups with row positions (mirrors predict.py:
/// `for mid in np.unique(...)` + flatnonzero).
pub fn group_positions(ids: &[String]) -> Vec<(String, Vec<usize>)> {
    let mut uniq = uniq_first(ids);
    uniq.sort();
    uniq.into_iter()
        .map(|m| {
            let pos = ids
                .iter()
                .enumerate()
                .filter(|(_, x)| *x == &m)
                .map(|(i, _)| i)
                .collect();
            (m, pos)
        })
        .collect()
}

/// One structs query row (mirrors a build_structs_payload sq dict).
pub struct StructsQuery {
    pub qi: usize,
    pub prec: f64,
    pub adduct: String,
}

/// Structs round-trip payload (mirrors run.py build_structs_payload:
/// sorted-unique keys, per-row key index, query rows).
pub fn structs_payload(
    key14: &[String],
    precs: &[f64],
    adducts: &[String],
    queries: &[usize],
) -> (Vec<String>, Vec<usize>, Vec<f64>, Vec<String>, Vec<StructsQuery>) {
    assert_eq!(key14.len(), precs.len(), "boundary: precs ragged");
    assert_eq!(key14.len(), adducts.len(), "boundary: adducts ragged");
    let mut ukeys = uniq_first(key14);
    ukeys.sort();
    let kindex: HashMap<String, usize> =
        ukeys.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
    let kidx = key14.iter().map(|k| kindex[k]).collect();
    let sq = queries
        .iter()
        .map(|&qi| StructsQuery { qi, prec: precs[qi], adduct: adducts[qi].clone() })
        .collect();
    (ukeys, kidx, precs.to_vec(), adducts.to_vec(), sq)
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn set(v: &[&str]) -> HashSet<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn uniq_first_order() {
        assert_eq!(uniq_first(&s(&["b", "a", "b", "c", "a"])), s(&["b", "a", "c"]));
    }

    #[test]
    fn pool_mask_ci_and_window() {
        let inst = s(&["timsTOF", "TIMSTOF", "Orbitrap", "None"]);
        let prec = vec![300.0, 240.0, 300.0, 300.0];
        assert_eq!(query_pool(&inst, &prec, "timsTOF", 240.0, 465.0), vec![0, 1]);
    }

    #[test]
    fn codes_inv_sorted_inverse() {
        let (codes, inv) = codes_inv(&s(&["k2", "k1", "k2", "k3"]));
        assert_eq!(codes, s(&["k1", "k2", "k3"]));
        assert_eq!(inv, vec![1, 0, 1, 2]);
    }

    #[test]
    fn lib_index_stable_groups() {
        let (codes, inv) = codes_inv(&s(&["k1", "k1", "k2", "k2"]));
        // library input shuffled: stable sort by code keeps input ties ordered
        let lib = lib_index(&codes, &inv, &[3, 0, 2, 1]);
        assert_eq!(lib[0], ("k1".to_string(), vec![0, 1]));
        assert_eq!(lib[1], ("k2".to_string(), vec![3, 2]));
    }

    #[test]
    fn first_smiles_follows_sorted_codes() {
        let (codes, inv) = codes_inv(&s(&["k2", "k1", "k2"]));
        let got = first_smiles_sorted(&codes, &inv, &s(&["CCC", "CCO", "CCC-DUP"]));
        assert_eq!(got, vec![("k1".to_string(), "CCO".to_string()), ("k2".to_string(), "CCC".to_string())]);
    }

    #[test]
    fn neutral_sort_stable_ties() {
        let (sk, m) = neutral_sort(&s(&["x", "y", "z"]), &[5.0, 5.0, 3.0]);
        assert_eq!(sk, s(&["z", "x", "y"]));
        assert_eq!(m, vec![3.0, 5.0, 5.0]);
    }

    #[test]
    fn searchsorted_edges() {
        let xs = vec![1.0, 2.0, 2.0, 3.0];
        assert_eq!(lower_bound(&xs, 2.0), 1);
        assert_eq!(upper_bound(&xs, 2.0), 3);
        assert_eq!(lower_bound(&xs, 0.0), 0);
        assert_eq!(upper_bound(&xs, 3.0), 4);
    }

    #[test]
    fn ppm_window_lo_inclusive_hi_exclusive() {
        let (lo, hi, ck) = ppm_window(&[100.0, 200.0, 300.0], &s(&["a", "b", "c"]), 200.0, 0.0);
        assert_eq!((lo, hi), (1, 2));
        assert_eq!(ck, s(&["b"]));
    }

    #[test]
    fn analog_window_excludes_caps_stable() {
        let (lp, lk) = (vec![200.0, 210.0, 220.0], s(&["a", "b", "c"]));
        let got = analog_window(&lp, &lk, 210.0, &set(&["b"]), 1500);
        assert_eq!(got, s(&["a", "c"])); // tie 10 Da keeps precursor order
        let got1 = analog_window(&lp, &lk, 210.0, &set(&["b"]), 1);
        assert_eq!(got1, s(&["a"]));
        let empty = analog_window(&lp, &lk, 210.0, &set(&["a", "b", "c"]), 1500);
        assert!(empty.is_empty());
    }

    #[test]
    fn aggregate_tie_keeps_first_seen() {
        let got = aggregate(&[vec![s(&["a", "b"])[0].clone(), s(&["a", "b"])[1].clone()], vec!["b".to_string(), "a".to_string()]], 2);
        assert_eq!(got, s(&["a", "b"])); // 1.5 == 1.5, insertion order wins
    }

    #[test]
    fn submission_rules_and_csv() {
        let ok = vec![("m1".to_string(), "CCO;CCC".to_string()), ("m2".to_string(), "CCC".to_string())];
        assert!(validate_submission(&ok).is_empty());
        assert_eq!(format_csv(&ok), "molecule_id,smiles\nm1,CCO;CCC\nm2,CCC\n");
        let many = vec![("m1".to_string(), vec!["CCO"; 26].join(";"))];
        assert!(validate_submission(&many).iter().any(|e| e.contains(">25")));
        let dup = vec![("m1".to_string(), "CCO".to_string()), ("m1".to_string(), "CCC".to_string())];
        assert!(validate_submission(&dup).iter().any(|e| e.contains("repeated")));
    }

    #[test]
    fn holdouts_match_python_semantics() {
        let keys = s(&["k1", "k1", "k2", "k3"]);
        let (lib, q, held) = apply_class2(&[0, 1, 2, 3], &keys, &set(&["k1", "k2"]));
        assert_eq!((lib, q, held), (vec![1, 3], vec![0, 2], s(&["k1", "k2"])));
        let (lib3, q3, _) = apply_class3(&[0, 1, 2, 3], &keys, &set(&["k1"]));
        assert_eq!((lib3, q3), (vec![2, 3], vec![0, 1]));
        // cross keeps the non-pool k1 row (idx 1) in the library
        let (libx, qx, _) = apply_cross(&[0, 2], &keys, &set(&["k1"]));
        assert_eq!((libx, qx), (vec![1, 2, 3], vec![0]));
        let _ = HashMap::<String, f64>::new(); // keep HashMap import live in tests
    }

    #[test]
    fn window_count_edges() {
        assert_eq!(window_count(&[100.0, 200.0, 300.0], 200.0, 0.0), 1);
        assert_eq!(window_count(&[], 200.0, 8.5), 0);
    }

    #[test]
    fn snake_partition_balances() {
        let costs = [5, 1, 4, 2];
        assert_eq!(snake_partition(&[10, 20, 30, 40], &costs, 2, 0), vec![10, 20]);
        assert_eq!(snake_partition(&[10, 20, 30, 40], &costs, 2, 1), vec![30, 40]);
    }

    #[test]
    fn snake_ties_keep_query_order() {
        // equal costs: stable desc keeps input order; round 1 reverses
        // (rank2: rnd=1, pos=0 -> b=sh-1-pos=1, exactly like run.py)
        assert_eq!(snake_partition(&[7, 8, 9], &[3, 3, 3], 2, 0), vec![7]);
        assert_eq!(snake_partition(&[7, 8, 9], &[3, 3, 3], 2, 1), vec![8, 9]);
    }

    #[test]
    fn group_positions_sorted_unique() {
        let g = group_positions(&s(&["m2", "m1", "m2"]));
        assert_eq!(g, vec![("m1".to_string(), vec![1]), ("m2".to_string(), vec![0, 2])]);
    }

    #[test]
    fn structs_payload_schema() {
        let (uk, ki, pr, ad, sq) = structs_payload(
            &s(&["k1", "k1", "k2"]),
            &[201.0, 201.5, 251.0],
            &s(&["[M+H]+", "[M+H]+", "[M+H]+"]),
            &[0, 2],
        );
        assert_eq!(uk, s(&["k1", "k2"]));
        assert_eq!(ki, vec![0, 0, 1]);
        assert_eq!(pr, vec![201.0, 201.5, 251.0]);
        assert_eq!(ad, s(&["[M+H]+", "[M+H]+", "[M+H]+"]));
        assert_eq!(sq.len(), 2);
        assert_eq!((sq[0].qi, sq[0].prec), (0, 201.0));
        assert_eq!(sq[0].adduct, "[M+H]+");
        assert_eq!((sq[1].qi, sq[1].prec), (2, 251.0));
    }
}
