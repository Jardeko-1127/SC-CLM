"""
SC-CLM V5.0 Inference Pipeline
------------------------------
CFG-guided generation with Test-Time Augmentation (TTA) and PPM reranking.

Usage:
  python src/model/v5/inference.py \
      --checkpoint results/checkpoints/v5/best \
      --input data/processed/test.csv \
      --output results/v5_predictions.csv \
      --tta 10 \
      --guidance 1.5
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

from transformers import T5Tokenizer

from src.model.v5.config import V5Config
from src.model.v5.model import V5SCLM
from src.model.v5.dataset import chemical_whitespace, randomize_smiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v5_inference")


def _resolve_tokenizer_path(checkpoint_path: Path, tokenizer_source: Optional[str] = None) -> str:
    """Use tokenizer in checkpoint dir when present; else HF/local backbone path."""
    if tokenizer_source:
        return tokenizer_source
    if (checkpoint_path / "tokenizer.json").is_file():
        return str(checkpoint_path)
    cfg_path = checkpoint_path / "config.json"
    d_model = 512
    if cfg_path.is_file():
        d_model = int(json.loads(cfg_path.read_text(encoding="utf-8")).get("d_model", 512))
    if d_model == 768:
        local_rt = Path("models/ReactionT5v2-forward")
        return str(local_rt) if local_rt.is_dir() else "sagawa/ReactionT5v2-forward"
    return V5Config().base_model


# ── Helpers ────────────────────────────────────────────────────────────────
def get_inchikey_14(smi: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        ik = InchiToInchiKey(MolToInchi(mol))
        return ik[:14] if ik else None
    except Exception:
        return None


def get_mass(smi: str) -> Optional[float]:
    try:
        mol = Chem.MolFromSmiles(smi)
        return Descriptors.ExactMolWt(mol) if mol else None
    except Exception:
        return None


def check_ppm(parent_smi: str, pred_smi: str, theoretical_delta: float, threshold: float = 5.0) -> bool:
    """PPM check with target_mass denominator (correct formula)."""
    try:
        p_mol = Chem.MolFromSmiles(parent_smi)
        pr_mol = Chem.MolFromSmiles(pred_smi)
        if not (p_mol and pr_mol):
            return False
        p_mass = Descriptors.ExactMolWt(p_mol)
        pr_mass = Descriptors.ExactMolWt(pr_mol)
        actual_delta = pr_mass - p_mass
        target_mass = p_mass + theoretical_delta
        if target_mass <= 0:
            return False
        ppm_error = abs(actual_delta - theoretical_delta) / target_mass * 1e6
        return ppm_error <= threshold
    except Exception:
        return False


# ── V5 Inference ───────────────────────────────────────────────────────────
class V5Inference:
    """Production inference with CFG + TTA + PPM reranking."""

    def __init__(
        self,
        checkpoint_path: str,
        guidance_scale: float = 1.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer_source: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.guidance_scale = guidance_scale
        ckpt_dir = Path(checkpoint_path)

        tok_path = _resolve_tokenizer_path(ckpt_dir, tokenizer_source)
        logger.info("Loading tokenizer: %s", tok_path)
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tok_path, local_files_only=True)
        except OSError:
            self.tokenizer = T5Tokenizer.from_pretrained(tok_path, local_files_only=False)
        if not (ckpt_dir / "tokenizer.json").is_file():
            extra = [t for t in V5Config().special_tokens if t not in self.tokenizer.get_vocab()]
            if extra:
                self.tokenizer.add_special_tokens({"additional_special_tokens": extra})

        logger.info("Loading V5SCLM from %s", checkpoint_path)
        self.model = V5SCLM.load(checkpoint_path, tokenizer=self.tokenizer)
        self.model.to(self.device)
        self.model.eval()

        self.token_to_idx = self.model.token_to_idx
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.token_to_delta = V5Config().token_to_delta

        logger.info("Model loaded. %d reaction tokens, device=%s",
                     len(self.token_to_idx), self.device)

    def _prepare_input(self, parent_smi: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chemical whitespace + tokenize."""
        spaced = chemical_whitespace(parent_smi)
        enc = self.tokenizer(
            spaced,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    def predict_single(
        self,
        parent_smi: str,
        token: str,
        num_beams: int = 10,
        num_return: int = 10,
        use_cfg: bool = True,
    ) -> List[str]:
        """Generate candidates for one parent molecule."""
        input_ids, attention_mask = self._prepare_input(parent_smi)
        reaction_id = torch.tensor(
            [self.token_to_idx.get(token, 0)], device=self.device
        )

        gen_kwargs = {
            "max_length": 512,
            "num_beams": num_beams,
            "num_return_sequences": min(num_return, num_beams),
        }

        if use_cfg:
            output_ids = self.model.generate_with_cfg(
                input_ids=input_ids,
                attention_mask=attention_mask,
                reaction_ids=reaction_id,
                guidance_scale=self.guidance_scale,
                **gen_kwargs,
            )
        else:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                reaction_ids=reaction_id,
                **gen_kwargs,
            )

        return self.tokenizer.batch_decode(
            output_ids[:, 1:], skip_special_tokens=True
        )

    def predict_with_tta(
        self,
        parent_smi: str,
        token: str,
        n_variants: int = 10,
        use_cfg: bool = True,
    ) -> Tuple[str, float]:
        """
        Test-Time Augmentation: generate from n randomized SMILES variants,
        return consensus prediction (most frequent InChIKey-14) with confidence.
        """
        mol = Chem.MolFromSmiles(parent_smi)
        if mol is None:
            return "", 0.0

        all_candidates: List[str] = []
        for _ in range(n_variants):
            r_smi = randomize_smiles(parent_smi)
            candidates = self.predict_single(r_smi, token, num_beams=5, num_return=5, use_cfg=use_cfg)
            all_candidates.extend(candidates)

        if not all_candidates:
            return "", 0.0

        # Vote by InChIKey-14 connectivity
        ik_counter: Counter = Counter()
        ik_to_smi: Dict[str, str] = {}
        for c in all_candidates:
            c = c.strip()
            if not c:
                continue
            ik14 = get_inchikey_14(c)
            if ik14:
                ik_counter[ik14] += 1
                ik_to_smi.setdefault(ik14, c)

        if not ik_counter:
            return all_candidates[0], 0.0

        top_ik, top_count = ik_counter.most_common(1)[0]
        confidence = top_count / len(all_candidates)
        return ik_to_smi[top_ik], confidence

    def predict_batch_with_rerank(
        self,
        df: pd.DataFrame,
        use_cfg: bool = True,
        tta: int = 10,
        num_beams: int = 10,
    ) -> pd.DataFrame:
        """
        Batch prediction with TTA + PPM reranking.

        For each row: TTA → collect candidates → rank by PPM fidelity → best.
        """
        results = []
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            parent = str(row["parent_smiles"])
            token = str(row["token"])
            delta = float(row["delta_mz"])

            # TTA → consensus
            consensus, conf = self.predict_with_tta(
                parent, token, n_variants=tta, use_cfg=use_cfg
            )

            # PPM rerank: also generate beam directly and pick best PPM
            direct_candidates = self.predict_single(
                parent, token, num_beams=num_beams, num_return=num_beams, use_cfg=use_cfg
            )

            # Merge TTA consensus and direct candidates; pick best by PPM
            all_cands = [consensus] + direct_candidates if consensus else direct_candidates
            best_pred = ""
            best_ppm = float("inf")
            for c in all_cands:
                c = c.strip()
                if not c:
                    continue
                if not check_ppm(parent, c, delta, threshold=5.0):
                    continue
                # Compute exact PPM
                p_mass = get_mass(parent)
                pr_mass = get_mass(c)
                if p_mass and pr_mass and p_mass > 0:
                    target_mass = p_mass + delta
                    ppm_val = abs((pr_mass - p_mass) - delta) / target_mass * 1e6
                    if ppm_val < best_ppm:
                        best_ppm = ppm_val
                        best_pred = c

            # Fallback: no PPM-passing candidate → use TTA consensus or top-1
            if not best_pred:
                best_pred = consensus if consensus else (
                    direct_candidates[0] if direct_candidates else ""
                )

            results.append({
                "parent_smiles": parent,
                "token": token,
                "delta_mz": delta,
                "target_smiles": row.get("product_smiles", ""),
                "prediction": best_pred,
                "tta_confidence": round(conf, 4),
                "ppm_error": round(best_ppm, 2) if best_ppm < float("inf") else None,
            })

            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d", i + 1, n)

        return pd.DataFrame(results)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SC-CLM V5.0 Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to V5 checkpoint dir")
    parser.add_argument(
        "--tokenizer-source",
        default=None,
        help="HF/local tokenizer path when checkpoint has no tokenizer files",
    )
    parser.add_argument("--input", required=True, help="CSV with parent_smiles, token, delta_mz columns")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--tta", type=int, default=10, help="TTA variants per sample")
    parser.add_argument("--guidance", type=float, default=1.5, help="CFG guidance scale")
    parser.add_argument("--beams", type=int, default=10, help="Number of beam search beams")
    parser.add_argument("--no-cfg", action="store_true", help="Disable CFG")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    logger.info("Input: %d samples", len(df))

    infer = V5Inference(
        checkpoint_path=args.checkpoint,
        guidance_scale=args.guidance,
        tokenizer_source=args.tokenizer_source,
    )

    result_df = infer.predict_batch_with_rerank(
        df,
        use_cfg=not args.no_cfg,
        tta=args.tta,
        num_beams=args.beams,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s (%d rows)", args.output, len(result_df))


if __name__ == "__main__":
    main()
