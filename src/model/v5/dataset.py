"""
V5Dataset — chemically-aware tokenization with configurable SMILES augmentation.

Key improvements over V4B:
  - Chemical whitespace: 'C(=O)O' stays coherent for BPE subword learning
  - Configurable SMILES randomization (default 5x)
  - Reaction ID output (integer index) instead of inline token sandwich
  - Pad-to-max_length tokenization consistent with Seq2SeqTrainer
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Two-letter elements that must be matched before single-letter
_TWO_LETTER = {"Cl", "Br", "Si", "Na", "Mg", "Al", "Ca", "Ti", "Cr", "Mn",
               "Fe", "Co", "Ni", "Cu", "Zn", "Ge", "As", "Se", "Zr", "Nb",
               "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
               "Te", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
               "Pb", "Bi", "Po", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
               "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
               "Lu", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
               "Cf", "Es", "Fm", "Md", "No", "Lr", "Be", "Sr", "Ra", "Li",
               "Rb", "Y", "K", "Ne", "Ar", "Kr", "Rn"}


def chemical_whitespace(smi: str) -> str:
    """Insert spaces around chemical tokens so SentencePiece BPE sees meaningful units.

    Before: C C ( = O ) O C 1 = C C = C C = C 1 C ( = O ) O
    After:  C C ( = O ) O C 1 = C C = C C = C 1 C ( = O ) O

    The spacing lets BPE learn subwords like 'C(=O)O' (carboxyl) naturally.
    """
    if not smi or not isinstance(smi, str):
        return ""

    tokens: List[str] = []
    i = 0
    n = len(smi)

    while i < n:
        # Two-letter elements (match longest first)
        if i + 1 < n and smi[i : i + 2] in _TWO_LETTER:
            tokens.append(smi[i : i + 2])
            i += 2
            continue

        c = smi[i]

        # Brackets: capture everything inside [...] as one token
        if c == "[":
            j = smi.index("]", i) + 1 if "]" in smi[i:] else n
            tokens.append(smi[i:j])
            i = j
            continue

        # Stereochemistry markers
        if c in "@/\\":
            tokens.append(c)
            i += 1
            continue

        # Extended ring closure indices (OpenSMILES): %12 => single token
        if c == "%" and i + 2 < n and smi[i + 1].isdigit() and smi[i + 2].isdigit():
            tokens.append(smi[i : i + 3])
            i += 3
            continue

        # Digits (ring closures, isotope counts) — attach to preceding element context
        if c.isdigit():
            tokens.append(c)
            i += 1
            continue

        # Bond symbols, parentheses, dot
        tokens.append(c)
        i += 1

    return " ".join(tokens)


def randomize_smiles(smi: str) -> str:
    """Generate a non-canonical SMILES via random atom renumbering."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        atoms = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atoms)
        new_mol = Chem.RenumberAtoms(mol, atoms)
        return Chem.MolToSmiles(new_mol, canonical=False)
    except Exception:
        return smi


class V5Dataset(Dataset):
    """V5 PyTorch Dataset with chemical whitespace and configurable augmentation.

    Each sample returns:
      - input_ids: tokenized parent SMILES (with chemical whitespace)
      - attention_mask: padding mask
      - labels: tokenized product SMILES (pad tokens → -100)
      - reaction_id: integer reaction class index
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        token_to_idx: Dict[str, int],
        max_input_len: int = 512,
        max_output_len: int = 512,
        augment: bool = False,
        augment_n: int = 5,
        use_chemical_whitespace: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.token_to_idx = token_to_idx
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.augment = augment
        self.augment_n = augment_n
        self.use_chemical_whitespace = use_chemical_whitespace
        self._csv_path = str(csv_path)

        # Expand dataset: repeat each row augment_n times for per-epoch variation
        self._n_base = len(self.df)
        self._n_effective = self._n_base * augment_n if augment else self._n_base
        logger.info(
            "V5Dataset: %s (%d base × %d aug = %d effective)",
            csv_path, self._n_base, augment_n if augment else 1, self._n_effective,
        )

    def __len__(self) -> int:
        return self._n_effective

    def __getitem__(self, idx: int):
        row_idx = idx % self._n_base
        row = self.df.iloc[row_idx]
        parent = str(row["parent_smiles"])
        product = str(row["product_smiles"])
        token = str(row["token"])

        # SMILES randomization
        if self.augment:
            parent = randomize_smiles(parent)
            product = randomize_smiles(product)

        # Chemical whitespace
        if self.use_chemical_whitespace:
            parent = chemical_whitespace(parent)

        # Tokenize input (parent SMILES only — no token sandwich)
        input_enc = self.tokenizer(
            parent,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize output (product SMILES)
        output_enc = self.tokenizer(
            product,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = output_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Reaction ID
        if token not in self.token_to_idx:
            raise ValueError(
                f"Unknown reaction token {token!r} (row {row_idx}). "
                f"Extend V5Config.token_to_delta or fix {self._csv_path}."
            )
        reaction_id = self.token_to_idx[token]

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "reaction_ids": torch.tensor(reaction_id, dtype=torch.long),
        }
