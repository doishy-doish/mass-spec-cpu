# mass-spec-cpu — CPU retrieval baseline for Enveda CASMI 2026 (Molecule ID from Mass Spectra)

All-CPU pipeline (RDKit fingerprints, spectral similarity, mass-filtered pools, z-fused
evidence channels) + leakage-safe three-mask validation. No GPU, no proprietary weights,
no NC-licensed data. Free GitHub runners only.

## Layout

- `src/mscpu/metric.py` — clean-room MRR@25 on InChIKey14 (RDKit only for keying).
- `src/mscpu/validation.py` — class2 / class3 / cross-instrument splits grouped by key14.
- `src/mscpu/spectra.py` — curation (floor, top-N, sqrt) + cosine + mass-shift analog cosine.
- `src/mscpu/fingerprints.py` — RDKit key14 + Morgan-2048 + Tanimoto.
- `src/mscpu/run.py` — pools (8.5ppm neutral mass) + 3 evidence channels + fused rank + MRR.
- `scripts/scramble.mjs` — AES-256-GCM sealing for `results/*.json` (public repo, private numbers).
- `.github/workflows/cpu.yml` — dispatch with mask + n; pytest, run, scramble, commit.

## Setup (operator)

1. Create public repo, push this tree.
2. Secrets: `KAGGLE_USERNAME` + `KAGGLE_KEY` (competition data download),
   `SCRAMBLE_KEY` (64 hex chars; same value as the vault copy).
3. Dispatch: Actions → cpu → mask + n → results land scrambled in `results/`.
4. Decrypt locally: `node scripts/scramble.mjs decrypt results/class2.json --key-file <vault key>`.

## License

MIT (this repo's code). Competition data is Kaggle's. No NC-licensed weights or
datasets are used or shipped here.
