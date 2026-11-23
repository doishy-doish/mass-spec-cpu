# mass-spec-cpu — CPU retrieval baseline for Enveda CASMI 2026 (Molecule ID from Mass Spectra)

Single Rust engine (`rust-bitvec` `sengine`/`structs` modes + `rust-chem`):
all cleaning, channels, fusion, ranking, and structure neutrals live in Rust.
Python (`src/mscpu/run.py`, `predict.py`) is a thin launcher: parquet load,
leakage-safe masks, pool shortlists, output assembly, submission CSV.
No Python backup of any channel math exists (user order: one code, no backups).

## Layout

- `rust-bitvec/src/main.rs` — THE engine: clean/quantize/encode(+weighted),
  Morgan/Tanimoto (via `rust-chem`), cosine/modified_cosine, all five evidence
  channels, funnel, z/rrf/rrfb/cascade/router fusion + ranking, structs.
- `rust-chem/` — SMILES parser + Morgan ECFP r2-2048 + Tanimoto +
  `connectivity_key14` surrogate (NOT an InChIKey; data grouping keeps the
  parquet `inchikey14` column). Zero deps.
- `src/mscpu/run.py` — thin validate launcher (masks, snake-draft, engine
  spawn, heartbeats, MRR output). No compute.
- `src/mscpu/predict.py` — thin inference launcher (narrow+wide pools,
  per-molecule aggregation, submission.csv). No compute.
- `src/mscpu/metric.py` — competition-spec helpers: MRR@25, submission
  validation. No RDKit.
- `src/mscpu/validation.py` — leakage-safe three-mask splits grouped by key14.
- `scripts/scramble.mjs` — AES-256-GCM sealing for `results/*.json` (public repo, private numbers).
- `.github/workflows/cpu.yml` (+ `cpu-fleet.yml`) — pytest + cargo test gates,
  then run; results scrambled + committed. No `--engine` flag: one engine.

## Setup (operator)

1. Create public repo, push this tree.
2. Secrets: `KAGGLE_USERNAME` + `KAGGLE_KEY` (competition data download),
   `SCRAMBLE_KEY` (64 hex chars; same value as the vault copy).
3. Dispatch: Actions → cpu → mask + n → results land scrambled in `results/`.
4. Decrypt locally: `node scripts/scramble.mjs decrypt results/class2.json --key-file <vault key>`.

## License

MIT (this repo's code). Competition data is Kaggle's. No NC-licensed weights or
datasets are used or shipped here.
