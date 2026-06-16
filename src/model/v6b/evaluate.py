"""
V6b evaluation — compare V6a vs V6b on the same test set.

Measures: validity, exact_match (InChIKey-14), PPM pass rate, Tanimoto.

Usage:
  python src/model/v6b/evaluate.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import AllChem as _AllChem
from rdkit.DataStructs import TanimotoSimilarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6b_evaluate")

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Metrics (consistent with src/eval/metrics.py) ──────────────────────────
def _canon(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m else None
    except Exception:
        return None


def _get_inchikey14(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return AllChem.InchiToInchiKey(AllChem.MolToInchi(m))[:14]
    except Exception:
        return None


def _get_mass(smi: str) -> Optional[float]:
    try:
        m = Chem.MolFromSmiles(smi)
        return Descriptors.ExactMolWt(m) if m else None
    except Exception:
        return None


def _check_ppm(parent_smi: str, pred_smi: str, delta_mz: float, threshold=5.0) -> bool:
    p_m = _get_mass(parent_smi)
    pr_m = _get_mass(pred_smi)
    if p_m is None or pr_m is None:
        return False
    actual_delta = pr_m - p_m
    target_mass = p_m + delta_mz
    if target_mass <= 0:
        return False
    ppm_error = abs(actual_delta - delta_mz) / target_mass * 1e6
    return ppm_error <= threshold


def _calc_tanimoto(smi1: str, smi2: str) -> float:
    try:
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    """Compute metrics for a DataFrame with columns:
    parent_smiles, target_smiles, prediction, delta_mz
    """
    n = len(df)
    if n == 0:
        return {"validity": 0.0, "exact_match": 0.0, "ppm_pass_rate": 0.0, "tanimoto": 0.0}

    valid = em = ppm = 0
    tanimoto_sum = 0.0
    valid_for_tan = 0

    for _, row in df.iterrows():
        p = str(row["parent_smiles"])
        t = str(row["target_smiles"])
        pred = str(row.get("prediction", ""))
        d = float(row["delta_mz"])

        pred_valid = _check_valid(pred)
        if not pred_valid:
            continue
        valid += 1

        pik = _get_inchikey14(pred)
        tik = _get_inchikey14(t)
        if pik and tik and pik == tik:
            em += 1

        if _check_ppm(p, pred, d):
            ppm += 1

        tani = _calc_tanimoto(pred, t)
        tanimoto_sum += tani
        valid_for_tan += 1

    return {
        "validity": round(valid / n * 100, 2),
        "exact_match": round(em / n * 100, 2),
        "ppm_pass_rate": round(ppm / n * 100, 2),
        "tanimoto": round(tanimoto_sum / max(1, valid_for_tan), 4),
    }


def _check_valid(smi: str) -> bool:
    try:
        m = Chem.MolFromSmiles(smi)
        return m is not None
    except Exception:
        return False


# ── V6a inference wrapper ──────────────────────────────────────────────────
def run_v6a_inference(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run V6a inference on test set (requires trained V6a checkpoint)."""
    from src.model.v6.config import V6Config
    from src.model.v6.model import V6SCLM
    from transformers import T5Tokenizer
    from src.model.v6.inference import predict as v6a_predict

    checkpoint = REPO_ROOT / "results" / "checkpoints" / "v6a" / "best"
    if not checkpoint.is_dir():
        logger.warning("V6a checkpoint not found at %s, trying checkpoint dirs...", checkpoint)
        # Try to find any V6a checkpoint
        ckpt_dir = REPO_ROOT / "results" / "checkpoints" / "v6a"
        if ckpt_dir.is_dir():
            subdirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")])
            if subdirs:
                checkpoint = subdirs[-1]
                logger.info("Using checkpoint: %s", checkpoint)
            else:
                logger.error("No V6a checkpoints found. Run V6a training first.")
                raise FileNotFoundError(f"No V6a checkpoints in {ckpt_dir}")

    config = V6Config()
    config.base_model = str(REPO_ROOT / "models" / "ReactionT5v2-forward")
    tokenizer = T5Tokenizer.from_pretrained(config.base_model)
    model = V6SCLM.load(str(checkpoint))
    model.eval()

    results = []
    for _, row in test_df.iterrows():
        parent = str(row["parent_smiles"])
        target = str(row["product_smiles"])
        delta = float(row["delta_mz"])
        parent_mass = _get_mass(parent)
        if parent_mass is None:
            continue
        product_mz = parent_mass + delta
        candidates = v6a_predict(parent, product_mz, model, tokenizer, config)
        pred = candidates[0] if candidates else ""
        results.append({
            "parent_smiles": parent,
            "target_smiles": target,
            "prediction": pred,
            "delta_mz": delta,
            "model": "V6a",
        })

    return pd.DataFrame(results)


# ── V6b inference wrapper ──────────────────────────────────────────────────
def run_v6b_inference(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run V6b inference on test set (requires trained V6b model)."""
    from src.model.v6b.config import V6bConfig
    from src.model.v6b.inference import predict as v6b_predict

    config = V6bConfig()
    model_dir = REPO_ROOT / config.model_dir
    model_path = str(model_dir / "model_step_50000.pt")

    if not Path(model_path).exists():
        # Try to find latest checkpoint
        checkpoints = sorted(model_dir.glob("model_step_*.pt"))
        if checkpoints:
            model_path = str(checkpoints[-1])
            logger.info("Using checkpoint: %s", model_path)
        else:
            logger.error("No V6b model found. Run V6b training first.")
            raise FileNotFoundError(f"No model in {model_dir}")

    results = []
    for _, row in test_df.iterrows():
        parent = str(row["parent_smiles"])
        target = str(row["product_smiles"])
        delta = float(row["delta_mz"])
        parent_mass = _get_mass(parent)
        if parent_mass is None:
            continue
        product_mz = parent_mass + delta
        candidates = v6b_predict(parent, product_mz, model_path, config, beam_size=20, topk=5)
        pred = candidates[0] if candidates else ""
        results.append({
            "parent_smiles": parent,
            "target_smiles": target,
            "prediction": pred,
            "delta_mz": delta,
            "model": "V6b",
        })

    return pd.DataFrame(results)


# ── Comparison ─────────────────────────────────────────────────────────────
def compare_v6a_v6b():
    """Run V6a and V6b inference on the test set and compare metrics."""
    test_csv = REPO_ROOT / "data" / "processed" / "v6_test.csv"
    if not test_csv.exists():
        logger.error("Test CSV not found: %s", test_csv)
        return

    test_df = pd.read_csv(test_csv)
    logger.info("Test set: %d samples", len(test_df))

    output_dir = REPO_ROOT / "results" / "benchmark" / "v6"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # V6a
    try:
        v6a_df = run_v6a_inference(test_df)
        v6a_metrics = evaluate_predictions(v6a_df)
        logger.info("V6a metrics: %s", v6a_metrics)
        v6a_metrics["model"] = "V6a"
        all_results.append(v6a_metrics)
        v6a_df.to_csv(output_dir / "v6a_predictions.csv", index=False)
    except (FileNotFoundError, Exception) as e:
        logger.warning("V6a evaluation skipped: %s", e)

    # V6b
    try:
        v6b_df = run_v6b_inference(test_df)
        v6b_metrics = evaluate_predictions(v6b_df)
        logger.info("V6b metrics: %s", v6b_metrics)
        v6b_metrics["model"] = "V6b"
        all_results.append(v6b_metrics)
        v6b_df.to_csv(output_dir / "v6b_predictions.csv", index=False)
    except (FileNotFoundError, Exception) as e:
        logger.warning("V6b evaluation skipped: %s", e)

    # Summary table
    if all_results:
        summary = pd.DataFrame(all_results)
        summary = summary[["model", "validity", "exact_match", "ppm_pass_rate", "tanimoto"]]
        summary.to_csv(output_dir / "v6_comparison.csv", index=False)
        logger.info("\n%s", summary.to_string(index=False))
    else:
        logger.warning("No models evaluated. Train V6a and/or V6b first.")


if __name__ == "__main__":
    compare_v6a_v6b()
