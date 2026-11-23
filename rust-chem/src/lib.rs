//! chem — Rust-native chemistry elements for the ALL-RUST law.
//!
//! Scope: SMILES parsing (organic subset) + Morgan ECFP radius-2 2048-bit +
//! Tanimoto + connectivity `key14` surrogate. This is INTENTIONALLY not RDKit:
//! bit-identity with `AllChem.GetMorganFingerprintAsBitVect` is NOT promised
//! (RDKit invariants + hashing are their own codebase). What IS promised:
//! determinism, self-Tanimoto 1.0, and MRR-parity when the analog channel's
//! `tanimoto_fp` call swaps to this crate (bench-gated per REGISTRY A3/A9).
//!
//! The `connectivity_key14` is a stable 14-char block over formula + degree
//! sequence + aromatic count. It is NOT an InChIKey and must never be
//! presented as one; data grouping keeps using the parquet `inchikey14`
//! column. The surrogate exists for novel structures with no data key.
//!
//! Algorithm mirror: the JS reference is the Gate-0 stub (r2_stub.js:
//! formulaCounts/gateFrac/ent, run in-worker) plus this crate's own unit
//! tests. Both use FNV-1a-32 over the same invariant strings, so stub bit
//! sets and these `[u64; 32]` words agree bit-for-bit on the same SMILES.
//! (There is intentionally no src/mscpu/chem.py — the single-engine guard
//! forbids Python compute backups; see tests/test_single_engine.py.)

pub const NBITS: usize = 2048;
pub const NWORDS: usize = 32;
pub const RADIUS: u32 = 2;

#[derive(Clone, Debug)]
pub struct Atom {
    pub el: String,
    pub aromatic: bool,
    pub charge: i32,
    pub h: u32,
}

#[derive(Clone, Debug)]
pub struct Mol {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<(usize, usize, u8)>,
}

fn fnv1a32(s: &str) -> u32 {
    let mut x: u32 = 2166136261;
    for b in s.bytes() {
        x ^= b as u32;
        x = x.wrapping_mul(16777619);
    }
    x
}

fn atomic_num(el: &str) -> u32 {
    match el {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Br" => 35,
        "I" => 53,
        _ => 6,
    }
}

/// Minimal SMILES parser: organic C N O S P F Cl Br I, aromatic c n o s,
/// `[...]` brackets (element + H-count + charge, isotope/`@` ignored),
/// bonds `-=#:/\`, branches `()`, rings `1-9` + `%nn`, `.` disconnect.
/// Unknown chars (e.g. `@`) are skipped. Returns None on empty.
pub fn parse_smiles(smi: &str) -> Option<Mol> {
    let b = smi.as_bytes();
    let n = b.len();
    let mut atoms: Vec<Atom> = Vec::new();
    let mut bonds: Vec<(usize, usize, u8)> = Vec::new();
    let mut i = 0usize;
    let mut prev: Option<usize> = None;
    let mut order: u8 = 1;
    let mut ring_open: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut stack: Vec<Option<usize>> = Vec::new();
    let link = |atoms_len: usize,
                prev: &mut Option<usize>,
                order: &mut u8,
                bonds: &mut Vec<(usize, usize, u8)>| {
        if let Some(p) = *prev {
            bonds.push((p, atoms_len, *order));
        }
        *prev = Some(atoms_len);
        *order = 1;
    };
    while i < n {
        let c = b[i] as char;
        match c {
            '(' => {
                stack.push(prev);
                i += 1;
            }
            ')' => {
                prev = stack.pop().unwrap_or(None);
                i += 1;
            }
            '-' | '=' | '#' | ':' | '/' | '\\' => {
                order = match c {
                    '-' => 1,
                    '=' => 2,
                    '#' => 3,
                    _ => 1,
                };
                i += 1;
            }
            '[' => {
                let mut j = i + 1;
                while j < n && b[j] as char != ']' {
                    j += 1;
                }
                if j >= n {
                    return None;
                }
                let inside = &smi[i + 1..j];
                i = j + 1;
                // element: first 1-2 letter symbol inside
                let mut el = "C".to_string();
                let mut aromatic = false;
                let mut charge: i32 = 0;
                let mut h: u32 = 0;
                let chars: Vec<char> = inside.chars().collect();
                let mut k = 0;
                while k < chars.len() && chars[k].is_ascii_digit() {
                    k += 1;
                }
                if k < chars.len() {
                    let c1 = chars[k];
                    if c1.is_ascii_alphabetic() {
                        if k + 1 < chars.len() && chars[k + 1].is_ascii_lowercase() {
                            el = chars[k..k + 2].iter().collect();
                            k += 2;
                        } else {
                            el = c1.to_string();
                            if "cnos".contains(c1) && el.len() == 1 {
                                aromatic = true;
                                el = el.to_uppercase();
                            }
                            k += 1;
                        }
                    }
                }
                // H count
                if let Some(pos) = inside.find('H') {
                    let rest = &inside[pos + 1..];
                    let digits: String =
                        rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    h = if digits.is_empty() { 1 } else { digits.parse().unwrap_or(1) };
                }
                // charge: count trailing +/- runs
                let plus = inside.chars().filter(|&c| c == '+').count() as i32;
                let minus = inside.chars().filter(|&c| c == '-').count() as i32;
                // explicit signed number like +2 / -1 wins over counts
                let mut explicit: Option<i32> = None;
                let ic: Vec<char> = inside.chars().collect();
                for t in 0..ic.len() {
                    if (ic[t] == '+' || ic[t] == '-')
                        && t + 1 < ic.len()
                        && ic[t + 1].is_ascii_digit()
                    {
                        let mut u = t + 1;
                        while u < ic.len() && ic[u].is_ascii_digit() {
                            u += 1;
                        }
                        let v: i32 =
                            ic[t + 1..u].iter().collect::<String>().parse().unwrap_or(0);
                        explicit = Some(if ic[t] == '+' { v } else { -v });
                    }
                }
                charge = explicit.unwrap_or(plus - minus);
                let idx = atoms.len();
                atoms.push(Atom { el, aromatic, charge, h });
                link(idx, &mut prev, &mut order, &mut bonds);
            }
            '%' => {
                if i + 2 < n {
                    if let Ok(nn) = smi[i + 1..i + 3].parse::<u32>() {
                        if let Some(p) = prev {
                            if let Some(o) = ring_open.remove(&nn) {
                                bonds.push((o, p, order));
                                order = 1;
                            } else {
                                ring_open.insert(nn, p);
                            }
                        }
                        i += 3;
                        continue;
                    }
                }
                i += 1;
            }
            '0'..='9' => {
                let nn = (c as u8 - b'0') as u32;
                if let Some(p) = prev {
                    if let Some(o) = ring_open.remove(&nn) {
                        bonds.push((o, p, order));
                        order = 1;
                    } else {
                        ring_open.insert(nn, p);
                    }
                }
                i += 1;
            }
            '.' => {
                prev = None;
                order = 1;
                i += 1;
            }
            '+' => {
                if let Some(p) = prev {
                    if let Some(a) = atoms.get_mut(p) {
                        a.charge += 1;
                    }
                }
                i += 1;
            }
            _ => {
                // two-letter organics first
                if i + 1 < n {
                    let two = &smi[i..i + 2];
                    if two == "Cl" || two == "Br" {
                        let idx = atoms.len();
                        atoms.push(Atom {
                            el: two.to_string(),
                            aromatic: false,
                            charge: 0,
                            h: 0,
                        });
                        link(idx, &mut prev, &mut order, &mut bonds);
                        i += 2;
                        continue;
                    }
                }
                if matches!(c, 'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'I') {
                    let idx = atoms.len();
                    atoms.push(Atom {
                        el: c.to_string(),
                        aromatic: false,
                        charge: 0,
                        h: 0,
                    });
                    link(idx, &mut prev, &mut order, &mut bonds);
                    i += 1;
                    continue;
                }
                if matches!(c, 'c' | 'n' | 'o' | 's') {
                    let idx = atoms.len();
                    atoms.push(Atom {
                        el: c.to_uppercase().to_string(),
                        aromatic: true,
                        charge: 0,
                        h: 0,
                    });
                    link(idx, &mut prev, &mut order, &mut bonds);
                    i += 1;
                    continue;
                }
                // skip stereochemistry / unknown markers
                i += 1;
            }
        }
    }
    if atoms.is_empty() {
        return None;
    }
    Some(Mol { atoms, bonds })
}

fn adjacency(mol: &Mol) -> Vec<Vec<(usize, u8)>> {
    let mut adj = vec![Vec::new(); mol.atoms.len()];
    for &(a, b, o) in &mol.bonds {
        if a < adj.len() && b < adj.len() {
            adj[a].push((b, o));
            adj[b].push((a, o));
        }
    }
    adj
}

/// Morgan ECFP radius-2 over `NBITS` bits. Empty/None -> all-zero words.
pub fn morgan_bits_2048(smi: &str) -> [u64; NWORDS] {
    let mut out = [0u64; NWORDS];
    let mol = match parse_smiles(smi) {
        Some(m) => m,
        None => return out,
    };
    let adj = adjacency(&mol);
    let n = mol.atoms.len();
    let mut cur: Vec<u32> = Vec::with_capacity(n);
    for (idx, a) in mol.atoms.iter().enumerate() {
        let deg = adj[idx].len() as u32;
        let inv = format!(
            "{}:{}:{}:{}:{}",
            atomic_num(&a.el),
            deg,
            a.h,
            a.charge,
            if a.aromatic { 1 } else { 0 }
        );
        cur.push(fnv1a32(&inv));
    }
    let mut bit = |x: u32| {
        let b = (x as usize) % NBITS;
        out[b / 64] |= 1u64 << (b % 64);
    };
    for &x in &cur {
        bit(x);
    }
    for _ in 0..RADIUS {
        let mut nxt = vec![0u32; n];
        for idx in 0..n {
            let mut nbrs: Vec<String> = adj[idx]
                .iter()
                .map(|&(j, o)| format!("{}x{}", cur[j], o))
                .collect();
            nbrs.sort();
            let h = format!("{}|{}", cur[idx], nbrs.join(","));
            nxt[idx] = fnv1a32(&h);
        }
        cur = nxt;
        for &x in &cur {
            bit(x);
        }
    }
    out
}

pub fn tanimoto(a: &[u64; NWORDS], b: &[u64; NWORDS]) -> f64 {
    let (mut inter, mut union) = (0u32, 0u32);
    for k in 0..NWORDS {
        inter += (a[k] & b[k]).count_ones();
        union += (a[k] | b[k]).count_ones();
    }
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

fn fnv1a64(s: &str) -> u64 {
    let mut x: u64 = 14695981039346656037;
    for b in s.bytes() {
        x ^= b as u64;
        x = x.wrapping_mul(1099511628211);
    }
    x
}

/// Stable 14-char connectivity block. NOT an InChIKey — never label it one.
/// Data grouping keeps using parquet `inchikey14`; this covers novel
/// structures with no data key.
pub fn connectivity_key14(smi: &str) -> Option<String> {
    let mol = parse_smiles(smi)?;
    let mut counts: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();
    let mut arom = 0usize;
    for a in &mol.atoms {
        *counts.entry(a.el.as_str()).or_insert(0) += 1;
        if a.aromatic {
            arom += 1;
        }
    }
    let formula: String = counts.iter().map(|(e, c)| format!("{}{}", e, c)).collect();
    let adj = adjacency(&mol);
    let mut degs: Vec<usize> = adj.iter().map(|v| v.len()).collect();
    degs.sort_unstable();
    let degstr: String = degs.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("");
    let s = format!(
        "{}|{}|{}|{}|{}",
        formula,
        degstr,
        mol.atoms.len(),
        mol.bonds.len(),
        arom
    );
    let mut x = fnv1a64(&s);
    const ALPH: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut out = String::with_capacity(14);
    for k in 0..14 {
        out.push(ALPH[(x % 36) as usize] as char);
        x /= 36;
        if x == 0 && k < 13 {
            x = (fnv1a64(&s) >> (k + 1)) | 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_tanimoto_is_one() {
        let a = morgan_bits_2048("CCO");
        let b = morgan_bits_2048("CCO");
        assert_eq!(a, b);
        assert!((tanimoto(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_is_zero() {
        let z = morgan_bits_2048("");
        assert_eq!(z, [0u64; NWORDS]);
        assert_eq!(tanimoto(&z, &z), 0.0);
    }

    #[test]
    fn similar_outscores_distant() {
        let cco = morgan_bits_2048("CCO");
        let ccc = morgan_bits_2048("CCC");
        let benz = morgan_bits_2048("c1ccccc1");
        assert!(tanimoto(&cco, &ccc) > tanimoto(&cco, &benz));
    }

    #[test]
    fn key14_stable_and_sized() {
        let a = connectivity_key14("CCO").unwrap();
        let b = connectivity_key14("CCO").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 14);
        assert_ne!(a, connectivity_key14("CCC").unwrap());
        assert!(connectivity_key14("").is_none());
    }
}
