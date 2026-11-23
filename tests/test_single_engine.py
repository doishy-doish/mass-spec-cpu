"""Single-engine guard: fails if Python compute ever returns.

STANDARD (ENGINE-SWAP 2026-11-22, user order: one code, no backups):
Rust (rust-bitvec sengine/structs + rust-chem) owns ALL compute
(clean/quantize/encode/Morgan/channels/funnel/fusion/rank/neutrals).
Python keeps ONLY parquet load, mask splits, pool shortlists (bisect),
snake-draft sharding, output assembly, submission-spec (metric.py).
No RDKit, no channel math, no fusion/rank duplicates in Python.

Run the same way every session:
  GH runner:  pip install -r requirements.txt
              cargo build+test (both crates) + pytest (this file gates)
  Local:      same three files in Pyodide (test_sengine/test_predict/test_metric)
              + JS mirror of rank/fuse for exact RRF values.
A green push requires all of the above in the SAME session (A9).
"""
import os
import sys
sys.path.insert(0, "src")

FORBIDDEN_FILES = [
    "src/mscpu/spectra.py",
    "src/mscpu/binary.py",
    "src/mscpu/chem.py",
    "src/mscpu/fingerprints.py",
    "src/mscpu/rust_score.py",
]

FORBIDDEN_SNIPPETS = [
    "import rdkit", "from rdkit", "AllChem", "GetMorganFingerprint",
    "from .spectra", "from .binary", "from .chem", "from .fingerprints",
    "import spectra", "import binary",
    "def cosine", "def modified_cosine", "def jaccard", "def wjaccard",
    "def morgan", "def tanimoto", "def rrf", "def rrf_fuse", "def encode",
    "def clean", "def score_query", "def fuse",
]

SCANNED = [
    "src/mscpu/run.py",
    "src/mscpu/predict.py",
    "src/mscpu/metric.py",
    "src/mscpu/validation.py",
]


def _repo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def test_no_python_compute_files():
    root = _repo_root()
    for rel in FORBIDDEN_FILES:
        assert not os.path.exists(os.path.join(root, rel)), f"forbidden file returned: {rel}"


def test_no_python_compute_code():
    root = _repo_root()
    for rel in SCANNED:
        with open(os.path.join(root, rel)) as f:
            src = f.read()
        for snip in FORBIDDEN_SNIPPETS:
            assert snip not in src, f"{rel} contains forbidden {snip!r} (compute belongs in Rust)"


def test_no_rdkit_dependency():
    root = _repo_root()
    with open(os.path.join(root, "requirements.txt")) as f:
        # comments may mention the removal; only real dependency lines count
        req_lines = [ln.strip() for ln in f
                     if ln.strip() and not ln.strip().startswith("#")]
    req = "\n".join(req_lines).lower()
    assert "rdkit" not in req, "requirements.txt must not depend on rdkit (Rust owns chemistry)"


def test_rust_engine_present():
    root = _repo_root()
    assert os.path.exists(os.path.join(root, "rust-bitvec", "src", "main.rs"))
    assert os.path.exists(os.path.join(root, "rust-chem", "src", "lib.rs"))
    with open(os.path.join(root, "rust-bitvec", "Cargo.toml")) as f:
        assert 'chem = { path = "../rust-chem" }' in f.read()


def test_variants_complete():
    from mscpu import run as R
    assert set(R.VARIANTS) == {"z", "lib", "analog", "frag", "rrf", "cascade",
                               "bin", "rrfb", "binw", "router", "stein", "mass", "zw", "rrfw"}

def test_fusion_defaults_neutral():
    from mscpu import run as R
    assert R.FUSION_DEFAULTS == {"wl": 1.0, "wa": 1.0, "wf": 1.0, "rl": 1.0,
                                 "ra": 1.0, "rf": 1.0, "casc_z": 3.0,
                                 "rout_lib": 0.5, "rout_bin": 0.4,
                                 "rout_binw": 0.4}


def test_good_probe_plumbing():
    # GOOD-PROBE regression: the mass-swap signal must reach the output JSON,
    # or probes silently measure nothing (the bug the user called out).
    import pathlib
    root = _repo_root()
    run_src = pathlib.Path(root, "src/mscpu/run.py").read_text()
    assert "mass_swap_rate" in run_src, "run.py must record mass_swap_rate in output JSON"
    assert 'out.get("mass_swap", False)' in run_src, "run.py must read sengine mass_swap flag"
    yml = pathlib.Path(root, ".github/workflows/cpu.yml").read_text()
    assert "probe assertions (GOOD PROBE" in yml, "cpu.yml must gate on GOOD-PROBE assertions"
    assert "mass_swap_rate" in yml, "GOOD-PROBE must check mass_swap_rate"
    assert "non-neutral genome left zw==z" in yml, "GOOD-PROBE must check genome actually moved"
