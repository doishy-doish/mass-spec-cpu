"""RDKit structure keys: key14 + Morgan fingerprints (CPU)."""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_ENUM = rdMolStandardize.TautomerEnumerator()
NBITS = 2048


@lru_cache(maxsize=2_000_000)
def key14(smiles: str | None) -> str | None:
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _ENUM.Canonicalize(mol)
    except Exception:
        pass
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return ik[:14] if ik else None


@lru_cache(maxsize=2_000_000)
def morgan(smiles: str | None):
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=NBITS)
    except Exception:
        return None


def tanimoto_fp(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(a, b)
