"""
V6b data preparation — OpenNMT format with char-level tokenization and Kekulé SMILES.

Outputs:
  data/processed/v6b/src-train.txt, tgt-train.txt
  data/processed/v6b/src-val.txt, tgt-val.txt
  data/processed/v6b/src-test.txt, tgt-test.txt

Format: space-separated SMILES chars + mass token (char-level)
  src:  C 1 = C C = C C = C 1 O | M A S S _ 1 5
  tgt:  C 1 = C C = C C = C 1 = O
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem

from src.model.v6b.config import V6bConfig
from src.model.v5.dataset import chemical_whitespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6b_prepare_data")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _kekulize(smi: str) -> str:
    """Convert SMILES to Kekulé form. Returns canonical SMILES on failure."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    except Exception:
        return smi


def _space_chars(smi: str) -> str:
    """Insert spaces between characters of a SMILES string (char-level tokenization).

    Respects two-letter elements and bracketed groups via chemical_whitespace,
    then re-splits each token into individual characters.
    """
    cw = chemical_whitespace(smi)
    # chemical_whitespace already handles two-letter elements, brackets, etc.
    # Now split each space-delimited token into individual characters
    tokens = cw.split()
    result: list[str] = []
    for tok in tokens:
        # Two-letter element symbols stay together
        if len(tok) == 2 and tok[0].isupper() and tok[1].islower():
            result.append(tok)
        elif len(tok) > 1 and tok[0] == "%" and tok[1:].isdigit():
            result.append(tok)
        else:
            result.extend(list(tok))
    return " ".join(result)


def prepare_v6b_data():
    """Load V6a generated CSVs, convert to OpenNMT format."""
    config = V6bConfig()
    data_dir = REPO_ROOT / config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_path = REPO_ROOT / f"data/processed/v6_{split}.csv"
        if not csv_path.exists():
            logger.warning("V6 CSV not found: %s, skipping", csv_path)
            continue

        df = pd.read_csv(csv_path)
        logger.info("Processing %s: %d rows", split, len(df))

        src_lines: list[str] = []
        tgt_lines: list[str] = []

        for _, row in df.iterrows():
            parent = str(row["parent_smiles"])
            product = str(row["product_smiles"])
            delta_mz = float(row["delta_mz"])

            # Kekulize
            parent_kek = _kekulize(parent)
            product_kek = _kekulize(product)

            # Char-level spacing
            parent_spaced = _space_chars(parent_kek)
            product_spaced = _space_chars(product_kek)

            # Mass token
            mass_tok = config.mass_token(delta_mz)
            src_line = f"{parent_spaced} | {' '.join(list(mass_tok))}"
            tgt_line = product_spaced

            src_lines.append(src_line)
            tgt_lines.append(tgt_line)

        # Write
        src_path = data_dir / f"src-{split}.txt"
        tgt_path = data_dir / f"tgt-{split}.txt"

        src_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8")
        tgt_path.write_text("\n".join(tgt_lines) + "\n", encoding="utf-8")
        logger.info("  %s: %d lines", src_path.name, len(src_lines))
        logger.info("  %s: %d lines", tgt_path.name, len(tgt_lines))

    logger.info("V6b data preparation complete.")


if __name__ == "__main__":
    prepare_v6b_data()
