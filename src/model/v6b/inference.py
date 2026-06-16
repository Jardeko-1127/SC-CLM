"""
V6b inference — OpenNMT Transformer with mass token conditioning.

Two modes:
  1. onmt_translate (subprocess): for trained models
  2. Direct predict: format input, run translation, parse output
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.model.v5.dataset import chemical_whitespace
from src.model.v6b.config import V6bConfig
from src.model.v6b.prepare_data import _kekulize, _space_chars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6b_inference")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_venv_python() -> str:
    venv_python = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def _check_valid(smi: str) -> bool:
    try:
        m = Chem.MolFromSmiles(smi)
        return m is not None
    except Exception:
        return False


def _canon(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _get_mass(smi: str) -> Optional[float]:
    try:
        m = Chem.MolFromSmiles(smi)
        return Descriptors.ExactMolWt(m) if m else None
    except Exception:
        return None


def format_input(parent_smi: str, delta_mz: float, config: V6bConfig) -> str:
    """Format parent SMILES + mass token as char-level input."""
    parent_kek = _kekulize(parent_smi)
    parent_spaced = _space_chars(parent_kek)
    mass_tok = config.mass_token(delta_mz)
    return f"{parent_spaced} | {' '.join(list(mass_tok))}"


def onmt_translate(
    src_lines: List[str],
    model_path: str,
    config: V6bConfig,
    beam_size: int = 20,
    n_best: int = 20,
    gpu: int = 0,
) -> List[str]:
    """Run onmt_translate via subprocess and return decoded outputs."""
    import tempfile

    # Write source to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(src_lines) + "\n")
        src_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        pred_path = f.name

    try:
        cmd = [
            _resolve_venv_python(), "-m", "onmt.bin.translate",
            "-model", model_path,
            "-src", src_path,
            "-output", pred_path,
            "-beam_size", str(beam_size),
            "-n_best", str(n_best),
            "-batch_size", "32",
            "-gpu", str(gpu),
            "-replace_unk",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(pred_path, encoding="utf-8") as f:
            results = [line.strip() for line in f]

        return results
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(pred_path).unlink(missing_ok=True)


def predict(
    parent_smi: str,
    product_mz: float,
    model_path: str,
    config: Optional[V6bConfig] = None,
    beam_size: int = 20,
    topk: int = 5,
) -> List[str]:
    """Predict product SMILES from parent SMILES and product m/z.

    Args:
        parent_smi: parent molecule SMILES
        product_mz: product exact mass
        model_path: path to trained OpenNMT model (.pt file)
        config: V6bConfig instance
        beam_size: beam search width
        topk: number of top candidates to return

    Returns:
        List of candidate product SMILES (deduplicated, valid only, top-k)
    """
    if config is None:
        config = V6bConfig()

    parent_mass = _get_mass(parent_smi)
    if parent_mass is None:
        return []

    delta_mz = product_mz - parent_mass
    src_line = format_input(parent_smi, delta_mz, config)

    raw_outputs = onmt_translate(
        [src_line], model_path, config, beam_size=beam_size, n_best=beam_size
    )

    # Post-process: despace, canonicalize, dedup, filter valid
    seen = set()
    candidates = []
    for line in raw_outputs:
        smi = line.replace(" ", "").strip()
        if not smi or smi in seen:
            continue
        can = _canon(smi)
        if can and can not in seen and _check_valid(can):
            seen.add(can)
            candidates.append(can)

    return candidates[:topk]
