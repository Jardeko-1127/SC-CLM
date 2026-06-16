"""
V6Dataset — mass-conditioned dataset with chemical whitespace and SMILES augmentation.

Returns: input_ids, attention_mask, labels, delta_mz (float), bin_idx (long)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.model.v5.dataset import chemical_whitespace, randomize_smiles
from src.model.v6.config import V6Config

logger = logging.getLogger(__name__)


class V6Dataset(Dataset):
    """V6 PyTorch Dataset for mass-conditioned CLM.

    Each sample returns:
      - input_ids: tokenized parent SMILES (with chemical whitespace)
      - attention_mask: padding mask
      - labels: tokenized product SMILES (pad → -100)
      - delta_mz: float mass difference
      - bin_idx: quantized bin index (long)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        mass_config: V6Config,
        max_input_len: int = 256,
        max_output_len: int = 128,
        augment: bool = False,
        augment_n: int = 5,
        use_chemical_whitespace: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.mass_config = mass_config
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.augment = augment
        self.augment_n = augment_n
        self.use_chemical_whitespace = use_chemical_whitespace
        self._csv_path = str(csv_path)

        self._n_base = len(self.df)
        self._n_effective = self._n_base * augment_n if augment else self._n_base
        logger.info(
            "V6Dataset: %s (%d base × %d aug = %d effective)",
            csv_path, self._n_base, augment_n if augment else 1, self._n_effective,
        )

    def __len__(self) -> int:
        return self._n_effective

    def _compute_bin_idx(self, delta_mz: float) -> int:
        return self.mass_config.mass_to_bin(delta_mz)

    def __getitem__(self, idx: int):
        row_idx = idx % self._n_base
        row = self.df.iloc[row_idx]
        parent = str(row["parent_smiles"])
        product = str(row["product_smiles"])
        delta_mz = float(row["delta_mz"])
        bin_idx = self._compute_bin_idx(delta_mz)

        # SMILES randomization
        if self.augment:
            parent = randomize_smiles(parent)
            product = randomize_smiles(product)

        # Chemical whitespace for input (parent only)
        if self.use_chemical_whitespace:
            parent = chemical_whitespace(parent)

        # Tokenize input
        input_enc = self.tokenizer(
            parent,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize output
        output_enc = self.tokenizer(
            product,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = output_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "delta_mz": torch.tensor(delta_mz, dtype=torch.float32),
            "bin_idx": torch.tensor(bin_idx, dtype=torch.long),
        }
