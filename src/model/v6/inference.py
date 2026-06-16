"""
V6a inference pipeline — continuous mass-delta conditioning.

API:
  predict(parent_smiles, product_mz, model, tokenizer, config) → List[str]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from transformers import T5Tokenizer

from src.model.v5.dataset import chemical_whitespace
from src.model.v6.config import V6Config
from src.model.v6.model import V6SCLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6a_inference")


def _get_mass(smi: str) -> Optional[float]:
    try:
        m = Chem.MolFromSmiles(smi)
        return Descriptors.ExactMolWt(m) if m else None
    except Exception:
        return None


def _check_valid(smi: str) -> bool:
    try:
        m = Chem.MolFromSmiles(smi)
        return m is not None
    except Exception:
        return False


class V6Inference:
    """V6a inference with CFG generation via mass-delta conditioning."""

    def __init__(
        self,
        checkpoint_path: str,
        guidance_scale: float = 1.5,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.guidance_scale = guidance_scale
        ckpt_dir = Path(checkpoint_path)

        # Tokenizer: try checkpoint first, then HF
        logger.info("Loading tokenizer...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(str(ckpt_dir))
        except OSError:
            self.tokenizer = T5Tokenizer.from_pretrained("sagawa/ReactionT5v2-forward")

        # Model
        logger.info("Loading V6SCLM from %s", checkpoint_path)
        self.model = V6SCLM.load(checkpoint_path, tokenizer=self.tokenizer)
        self.model.to(self.device)
        self.model.eval()
        self.config = self.model.v6_config

        logger.info("Loaded. %d mass bins, device=%s", self.config.num_mass_bins, self.device)

    def _prepare_input(self, parent_smi: str) -> tuple:
        spaced = chemical_whitespace(parent_smi)
        enc = self.tokenizer(
            spaced,
            max_length=self.config.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    def predict(
        self,
        parent_smi: str,
        product_mz: float,
        num_beams: int = 10,
        num_return: int = 5,
        use_cfg: bool = True,
    ) -> List[str]:
        """Predict product SMILES from parent SMILES and product m/z.

        Args:
            parent_smi: parent molecule SMILES
            product_mz: product exact mass (m/z)
            num_beams: beam search width
            num_return: number of candidate sequences to return
            use_cfg: enable classifier-free guidance

        Returns:
            List of candidate product SMILES strings
        """
        parent_mass = _get_mass(parent_smi)
        if parent_mass is None:
            logger.error("Cannot compute parent mass for: %s", parent_smi)
            return []

        delta_mz = product_mz - parent_mass
        delta_tensor = torch.tensor([delta_mz], dtype=torch.float32, device=self.device)

        input_ids, attention_mask = self._prepare_input(parent_smi)

        gen_kwargs = {
            "max_length": self.config.max_output_len,
            "num_beams": num_beams,
            "num_return_sequences": min(num_return, num_beams),
        }

        if use_cfg:
            output_ids = self.model.generate_with_cfg(
                input_ids=input_ids,
                attention_mask=attention_mask,
                delta_mz=delta_tensor,
                guidance_scale=self.guidance_scale,
                **gen_kwargs,
            )
        else:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                delta_mz=delta_tensor,
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)
        # Filter to valid SMILES
        valid = [s.strip() for s in decoded if _check_valid(s.strip())]
        return valid


def predict(parent_smi: str, product_mz: float, model, tokenizer, config: V6Config) -> List[str]:
    """Convenience function for programmatic use.

    Args:
        parent_smi: parent molecule SMILES
        product_mz: product exact mass (m/z)
        model: V6SCLM instance
        tokenizer: T5Tokenizer instance
        config: V6Config instance

    Returns:
        List of candidate product SMILES
    """
    parent_mass = _get_mass(parent_smi)
    if parent_mass is None:
        return []

    delta_mz = product_mz - parent_mass
    delta_tensor = torch.tensor([delta_mz], dtype=torch.float32, device=model.device)

    spaced = chemical_whitespace(parent_smi)
    enc = tokenizer(
        spaced,
        max_length=config.max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate_with_cfg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            delta_mz=delta_tensor,
            guidance_scale=config.cfg_guidance_scale,
            max_length=config.max_output_len,
            num_beams=5,
            num_return_sequences=3,
        )

    decoded = tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)
    return [s.strip() for s in decoded if _check_valid(s.strip())]
